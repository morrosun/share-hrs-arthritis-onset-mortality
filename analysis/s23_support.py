# -*- coding: utf-8 -*-
"""Step 23: Table 5 的年龄支持审计

评审核心指控：新发病例「仅诊断后前5年」分析中，
  <60 起病者的观察年龄上限约 65 岁，≥75 起病者的观察年龄下限 75 岁，
  两组在达到年龄上没有任何重叠 → 0.38 只能是依赖 PH 假定的间接比较。

本脚本对 Table 5 的每个设计：
  (1) 各组 n / 死亡 / 人年 / 达到年龄 min-median-max / 随访长度
  (2) 5 岁年龄段 × 组的风险人年与死亡数（真实风险集）
  (3) 两两比较的共同支持年龄段（双方人年均 >= 阈值）
  (4) 限定共同支持后重估 HR；无共同支持则标记 not estimable
"""
import os
import sys, io, math, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010,
        11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

GRPS = ['<60', '60-74', '>=75']
CUTS = [0, 59.999, 74.999, 200]

def to_grp(v):
    return pd.cut(v, CUTS, labels=GRPS)

def split_bands(entry, exit, w=5, lo=40, hi=110):
    """把 [entry, exit] 按 w 岁年龄段切分，返回 [(band_start, py)]"""
    out = []
    a = math.floor(entry / w) * w
    a = max(a, lo)
    while a < exit and a < hi:
        s = max(entry, a); e = min(exit, a + w)
        if e > s + 1e-9:
            out.append((int(a), e - s))
        a += w
    return out

def band_table(df, entry_col, exit_col, death_col='event', label=''):
    """返回 (band, grp) -> py, deaths 的表"""
    rows = []
    for r in df.itertuples(index=False):
        e = getattr(r, entry_col); x = getattr(r, exit_col)
        if not np.isfinite(e) or not np.isfinite(x) or x <= e:
            continue
        g = getattr(r, 'grp')
        for b, py in split_bands(e, x):
            rows.append((b, g, py))
    B = pd.DataFrame(rows, columns=['band', 'grp', 'py'])
    T = B.pivot_table(index='band', columns='grp', values='py', aggfunc='sum', fill_value=0.0)
    for g in GRPS:
        if g not in T.columns:
            T[g] = 0.0
    return T[GRPS].sort_index()

# ==================== 1. SHARE ====================
s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s = s.sort_values(['pid', 'wave'])

# 1a prevalent（主队列，来自 analysis_share.csv）
A = pd.read_csv(os.path.join(OUT, "analysis_share.csv"))
A = A.rename(columns={'entry_age': 'entry', 'exit_age': 'exit'})
A['grp'] = to_grp(A['onset_age'])
A['male'] = (A['sex'] == 1).astype(float)
A = A.dropna(subset=['grp', 'entry', 'exit'])
A = A[A['exit'] > A['entry']]

# 1b incident
rows = []
for pid, x in s.groupby('pid'):
    a = x.dropna(subset=['arthre'])
    if a.empty: continue
    fp = a[a['arthre'] == 1]
    if fp.empty: continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    if neg.empty: continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=a['age'].iloc[-1], dead=a['radyear'].max(),
                     byear=a['rabyear'].iloc[0], sex=a['ragender'].iloc[0],
                     edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['width'] = I['pos'] - I['lastneg']
I['event'] = I['dead'].notna().astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['male'] = (I['sex'] == 1).astype(float)
I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
I['entry'] = I['onset']
I['grp'] = to_grp(I['onset'])
I = I[(I['exit'] > I['entry']) & (I['onset'] >= 40)].dropna(subset=['grp'])

# 1c incident, censored at onset+5
IW = I.copy()
IW['exit_full'] = IW['exit']
IW['exit'] = np.minimum(IW['exit'], IW['onset'] + 5)
IW['event'] = np.where(IW['exit'] < IW['exit_full'] - 1e-9, 0, IW['event'])
IW = IW[IW['exit'] > IW['entry']]

# ==================== 2. HRS ====================
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    if neg.empty:
        continue                              # prevalent：起病年龄未知，不进 onset 分析
    rows.append(dict(pid=pid,
                     onset=(neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2,
                     pos=fp['age'].iloc[0], lastneg=neg['age'].iloc[-1],
                     last=x['age'].iloc[-1], dead=x['radyear'].max(),
                     byear=x['rabyear'].iloc[0], sex=x['ragender'].iloc[0],
                     edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
H = pd.DataFrame(rows)
H['event'] = H['dead'].notna().astype(int)
H['exit'] = np.where(H['event'] == 1, H['dead'] - H['byear'], H['last'])
H['male'] = (H['sex'] == 1).astype(float)
H['smoke_ever'] = (H['smoke'] >= 1).astype(float)
H['entry'] = H['onset']
H['grp'] = to_grp(H['onset'])
H = H[(H['exit'] > H['entry']) & (H['onset'] >= 40)].dropna(subset=['grp'])

HW = H.copy()
HW['exit_full'] = HW['exit']
HW['exit'] = np.minimum(HW['exit'], HW['onset'] + 5)
HW['event'] = np.where(HW['exit'] < HW['exit_full'] - 1e-9, 0, HW['event'])
HW = HW[HW['exit'] > HW['entry']]

DESIGNS = {
    'SHARE prevalent (main)': A,
    'SHARE incident, full follow-up': I,
    'SHARE incident, first 5 y': IW,
    'HRS incident, full follow-up': H,
    'HRS incident, first 5 y': HW,
}

# ==================== 3. 描述 + 风险集 ====================
summary_rows = []
for name, d in DESIGNS.items():
    P("\n" + "=" * 88)
    P(f"# {name}")
    P("=" * 88)
    desc = d.groupby('grp', observed=True).agg(
        n=('entry', 'size'), deaths=('event', 'sum'),
        py=('entry', lambda x: 0.0)).reset_index()
    desc = desc.drop(columns='py')
    desc['py'] = [float((d[d['grp'] == g]['exit'] - d[d['grp'] == g]['entry']).sum()) for g in desc['grp']]
    desc['fu_med'] = [float((d[d['grp'] == g]['exit'] - d[d['grp'] == g]['entry']).median()) for g in desc['grp']]
    desc['ent_min'] = [float(d[d['grp'] == g]['entry'].min()) for g in desc['grp']]
    desc['ent_med'] = [float(d[d['grp'] == g]['entry'].median()) for g in desc['grp']]
    desc['exit_min'] = [float(d[d['grp'] == g]['exit'].min()) for g in desc['grp']]
    desc['exit_med'] = [float(d[d['grp'] == g]['exit'].median()) for g in desc['grp']]
    desc['exit_max'] = [float(d[d['grp'] == g]['exit'].max()) for g in desc['grp']]
    P(desc.round(1).to_string(index=False))
    P("\n  [达到年龄范围 —— 这是判断共同支持的唯一依据]")
    for g in desc['grp']:
        q = desc[desc['grp'] == g].iloc[0]
        P(f"    {g:6s} entry {q['ent_min']:.0f}–{q['ent_med']:.0f}(med)–{q['ent_med']:.0f}"
          f" | exit {q['exit_min']:.0f}–{q['exit_max']:.0f} (med {q['exit_med']:.0f})")
    T = band_table(d, 'entry', 'exit')
    P("\n  [5 岁年龄段的风险人年]")
    P(T.round(0).to_string())
    for _, r in desc.iterrows():
        summary_rows.append(dict(design=name, grp=r['grp'], n=int(r['n']), deaths=int(r['deaths']),
                                 py=round(r['py'], 0), fu_med=round(r['fu_med'], 1),
                                 entry_min=round(r['ent_min'], 0), entry_med=round(r['ent_med'], 0),
                                 exit_min=round(r['exit_min'], 0), exit_med=round(r['exit_med'], 0),
                                 exit_max=round(r['exit_max'], 0),
                                 band_lo=int(T.index.min()), band_hi=int(T.index.max())))

pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "table5_support.csv"), index=False, encoding='utf-8-sig')

# ==================== 4. 共同支持 + 限定重估 ====================
P("\n\n" + "=" * 88)
P("# 两两共同支持审计")
P("=" * 88)
PY_MIN = 20.0   # 一个年龄段至少 20 人年才算"有风险集"

def common_support(T, g1, g2, ymin=PY_MIN):
    ok = T[(T[g1] >= ymin) & (T[g2] >= ymin)].index.tolist()
    if not ok:
        return []
    ok = sorted(ok)
    segs, cur = [], [ok[0]]
    for b in ok[1:]:
        if b - cur[-1] <= 5:
            cur.append(b)
        else:
            segs.append((cur[0], cur[-1] + 5)); cur = [b]
    segs.append((cur[0], cur[-1] + 5))
    return segs

def restrict(df, lo, hi):
    """把每个人-时间的观察窗口与 [lo, hi) 求交，得到限定样本"""
    df = df.reset_index(drop=True)
    e = df['entry'].clip(lower=lo)
    x = df['exit'].clip(upper=hi)
    keep = (x > e + 1e-9).values
    ev = ((df['event'].values == 1) & (df['exit'].values <= hi + 1e-9)).astype(int)
    out = df.loc[keep, ['grp', 'male', 'edu', 'smoke_ever']].copy().reset_index(drop=True)
    out['entry'] = e[keep].values
    out['exit'] = x[keep].values
    out['event'] = ev[keep]
    return out

fit_rows = []
for name, d in DESIGNS.items():
    T = band_table(d, 'entry', 'exit')
    P(f"\n--- {name} ---")
    for g1, g2 in [('<60', '60-74'), ('60-74', '>=75'), ('<60', '>=75')]:
        segs = common_support(T, g1, g2)
        if not segs:
            P(f"  {g1} vs {g2}: 无共同支持段（双方均无人年 ≥ {PY_MIN:.0f}）→ NOT ESTIMABLE")
            fit_rows.append(dict(design=name, contrast=f'{g2} vs {g1}', support='none',
                                 lo=np.nan, hi=np.nan, n=0, deaths=0, hr=np.nan, lo_ci=np.nan,
                                 hi_ci=np.nan, p=np.nan, note='no common support'))
            continue
        P(f"  {g1} vs {g2}: 共同支持段 {segs}")
        for lo, hi in segs:
            sub = d[(d['grp'].isin([g1, g2]))]
            R = restrict(sub, lo, hi)
            if len(R) == 0:
                continue
            R['g'] = (R['grp'] == g2).astype(float)
            D = R[['entry', 'exit', 'event', 'g', 'male', 'edu', 'smoke_ever']].dropna()
            if D['g'].sum() < 20 or int(D['event'].sum()) < 10:
                P(f"      [{lo},{hi}) n={len(D)} 死亡 {int(D['event'].sum())} → 事件太少，不估计")
                fit_rows.append(dict(design=name, contrast=f'{g2} vs {g1}', support=f'[{lo},{hi})',
                                     lo=lo, hi=hi, n=len(D), deaths=int(D['event'].sum()),
                                     hr=np.nan, lo_ci=np.nan, hi_ci=np.nan, p=np.nan,
                                     note='too few events'))
                continue
            try:
                m = CoxPHFitter()
                m.fit(D, duration_col='exit', event_col='event', entry_col='entry')
                q = m.summary.loc['g']
                P(f"      [{lo},{hi}) n={len(D):,} 死亡 {int(D['event'].sum()):,} "
                  f"HR={q['exp(coef)']:.2f} ({q['exp(coef) lower 95%']:.2f}-{q['exp(coef) upper 95%']:.2f}) p={q['p']:.3g}")
                fit_rows.append(dict(design=name, contrast=f'{g2} vs {g1}', support=f'[{lo},{hi})',
                                     lo=lo, hi=hi, n=len(D), deaths=int(D['event'].sum()),
                                     hr=q['exp(coef)'], lo_ci=q['exp(coef) lower 95%'],
                                     hi_ci=q['exp(coef) upper 95%'], p=q['p'], note=''))
            except Exception as ex:
                P(f"      [{lo},{hi}) 拟合失败: {str(ex)[:60]}")

pd.DataFrame(fit_rows).to_csv(os.path.join(OUT, "table5_common_support.csv"), index=False, encoding='utf-8-sig')

# ==================== 5. 汇总对照 ====================
P("\n\n" + "=" * 88)
P("# 汇总：全体估计 vs 共同支持内估计")
P("=" * 88)
old = pd.read_csv(os.path.join(OUT, "table5_designs.csv"))
P(old[['design', 'group', 'hr', 'lo', 'hi']].to_string(index=False))
P("")
cs = pd.DataFrame(fit_rows)
P(cs[['design', 'contrast', 'support', 'n', 'deaths', 'hr', 'lo_ci', 'hi_ci', 'note']].to_string(index=False))

io.open(os.path.join(OUT, "s23_support.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE s23_support.txt / table5_support.csv / table5_common_support.csv")
