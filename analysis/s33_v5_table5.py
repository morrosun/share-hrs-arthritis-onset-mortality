# -*- coding: utf-8 -*-
"""Step 33: v5 口径 Table 5（年龄支持审计 + 两两 Cox）。
v5 规则：
  - 分层：区间中点（与 v4 相同）
  - 入组：首次阳性报告年龄（pos），不再用中点入组
  - 窗口限定：入组 max(entry, lo)，退出 min(exit, hi)；事件仅在 exit<=hi 时计入
  - 5 年窗口：exit = min(exit, pos + 5)
  - 共同支持：按 5 岁年龄段，双方每段人年 >= 20 且两组在险
输出 out/table5_v5.csv、out/s33_v5_table5.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])

def build_incident(df, wave_map):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map); df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rows = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty: continue
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                         sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
    I = pd.DataFrame(rows)
    I['onset'] = (I['lastneg'] + I['pos']) / 2
    I['event'] = I['dead'].notna().astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['male'] = (I['sex'] == 1).astype(float)
    I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
    I['grp'] = grp(I['onset'])
    I = I[(I['onset'] >= 40)].dropna(subset=['grp', 'edu'])
    I['entry'] = I['pos']                              # v5：入组 = 首次阳性
    # v5 主口径：exit<=entry 且已死亡者保留（exit=entry+0.5，见 s31 方案 B）；存活者零时间自然剔除
    fix = (I['exit'] <= I['entry']) & (I['event'] == 1)
    I.loc[fix, 'exit'] = I.loc[fix, 'entry'] + 0.5
    return I[I['exit'] > I['entry']].copy()

def prevalent_share(s):
    # v18 口径（2026-11-07 作者确认）：同年死亡统一插补 0.5 年；剔除 2020 入组波（w0=8）
    agg = s.groupby('pid').agg(entry=('age', 'min'), last=('age', 'max'), sex=('ragender', 'first'),
                               byear=('rabyear', 'first'), edu=('raeducl', 'first'), smoke=('smoken', 'max'),
                               w0=('wave', 'min'))
    dg = s.dropna(subset=['radiagarthr']).groupby('pid')['radiagarthr'].first().rename('onset')
    dd = s.groupby('pid')['radyear'].max().rename('dead')
    D = agg.join(dg).join(dd).dropna(subset=['onset'])
    D = D[D['onset'] >= 5]
    D['event'] = D['dead'].notna().astype(int)
    D['exit'] = np.where(D['event'] == 1, D['dead'] - D['byear'], D['last'])
    zero = D['exit'] <= D['entry']
    D.loc[zero & (D['event'] == 1), 'exit'] = D.loc[zero & (D['event'] == 1), 'entry'] + 0.5
    D = D[(D['exit'] > D['entry']) & (D['entry'] >= 40) & (D['exit'] <= 110) & (D['w0'] != 8)].copy()
    D['male'] = (D['sex'] == 1).astype(float)
    D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
    D['grp'] = grp(D['onset'])
    D = D.dropna(subset=['grp', 'edu']).reset_index()
    P(f"[SHARE prevalent 口径核对] n={len(D):,} deaths={int(D['event'].sum()):,}（应为 18,587 / 4,932）")
    return D

def band_py(D):
    out = {}
    for g, d in D.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0 = max(e, a); e0 = min(x, a + 5)
                if e0 > s0: bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g] = bands
    return pd.DataFrame(out).fillna(0).sort_index()

def support(D, a, b):
    T = band_py(D)
    segs = []; cur = None
    for band in T.index:
        ok = (T.loc[band, a] >= 20) and (T.loc[band, b] >= 20)
        if ok and cur is None: cur = [band, band + 5]
        elif ok: cur[1] = band + 5
        elif cur is not None: segs.append(tuple(cur)); cur = None
    if cur is not None: segs.append(tuple(cur))
    return segs

def pair(D, a, b, lo, hi):
    X = D.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'] == a].empty or X[X['grp'] == b].empty:
        return None
    X['cmp'] = (X['grp'] == a).astype(float)
    X = X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']]
    m = CoxPHFitter()
    m.fit(X, duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=q['exp(coef)'],
                lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'])

def window5(I):
    W = I.copy()
    W['exit'] = np.minimum(W['exit'], W['pos'] + 5)
    W['event'] = np.where(W['exit'] < I['exit'] - 1e-9, 0, W['event'])
    return W[W['exit'] > W['entry']].copy()

rows = []
desc = []
def describe(D, label):
    for g, d in D.groupby('grp', observed=True):
        desc.append(dict(design=label, grp=str(g), n=len(d), deaths=int(d['event'].sum()),
                         py=round(float((d['exit'] - d['entry']).sum()), 0),
                         exit_med=float(d['exit'].median()), exit_min=float(d['exit'].min()),
                         exit_max=float(d['exit'].max())))
def audit(D, label):
    describe(D, label)
    P("=" * 78 + f"\n# {label}\n" + "=" * 78)
    P(f"n={len(D):,} deaths={int(D['event'].sum()):,} py={(D['exit']-D['entry']).sum():,.0f}")
    P(band_py(D).round(0).to_string())
    for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
        segs = support(D, a, b)
        if not segs:
            P(f"  {a} vs {b}: 无共同支持 -> not estimable")
            rows.append(dict(design=label, contrast=f"{a} vs {b}", support='none', n=None, deaths=None,
                             hr=None, lo=None, hi=None))
            continue
        lo, hi = segs[0][0], segs[-1][1]
        r = pair(D, a, b, lo, hi)
        full = pair(D, a, b, 0, 1000)
        allhr = (round(full['hr'], 2), round(full['lo'], 2), round(full['hi'], 2)) if full else (None, None, None)
        if r is None:
            P(f"  {a} vs {b}: 支持段 {segs} 内无两组人数 -> not estimable")
            rows.append(dict(design=label, contrast=f"{a} vs {b}", support=f"{lo}-{hi}", n=None, deaths=None,
                             hr=None, lo=None, hi=None)); continue
        P(f"  {a} vs {b}: 共同支持 [{lo},{hi}) n={r['n']:,} deaths={r['deaths']:,} HR {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})")
        rows.append(dict(design=label, contrast=f"{a} vs {b}", support=f"{lo}-{hi}", n=r['n'], deaths=r['deaths'],
                         hr=round(r['hr'], 2), lo=round(r['lo'], 2), hi=round(r['hi'], 2),
                         all_hr=allhr[0], all_lo=allhr[1], all_hi=allhr[2]))
    P("")

s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']

h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})

audit(prevalent_share(s), 'SHARE prevalent (main, unchanged)')
I_s = build_incident(s, WY_S)
audit(I_s, 'SHARE incident, v5 entry=first positive')
audit(window5(I_s), 'SHARE incident, first 5 y after first positive, v5')
I_h = build_incident(h, WY_H)
audit(I_h, 'HRS incident, v5 entry=first positive')
audit(window5(I_h), 'HRS incident, first 5 y after first positive, v5')

pd.DataFrame(rows).to_csv(f"{OUT}\\table5_v5.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(desc).to_csv(f"{OUT}\\table5_v5_desc.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s33_v5_table5.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("写出 table5_v5.csv / s33_v5_table5.txt")
