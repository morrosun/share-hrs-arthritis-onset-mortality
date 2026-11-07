# -*- coding: utf-8 -*-
"""Step 25: 按第二轮评审重做 Table 2（70-79 内加连续年龄）与 Figure 3（新暴露五态）

改动依据：
  · Table 2 原 70-79 列只用 C(state)+教育+性别，子集内无年龄项 → 同龄段不等于同年龄（评审 4）
  · Figure 3 原用「首次阳性报告年龄」分组，含 15,586 名 prevalent 误分类（评审 5.5）
"""
import os
import sys, io, math, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010,
        11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

COM = ['hibpe', 'diabe', 'hearte', 'stroke', 'cancre', 'lunge']
BINS = [45, 60, 70, 80, 90, 106]
BLAB = ['45-59', '60-69', '70-79', '80-89', '90+']

def comorb(df):
    x = df[COM]
    return x.fillna(0).sum(axis=1)

# ==================== HRS ====================
h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = comorb(h)
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        rows.append(dict(pid=pid, state='never', onset=np.nan)); continue
    ip = x.index.get_loc(fp.index[0]); neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty:
        rows.append(dict(pid=pid, state='prevalent', onset=np.nan))
    else:
        o = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2
        g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
        rows.append(dict(pid=pid, state='on_' + g, onset=o))
S = pd.DataFrame(rows)
hh = h.merge(S, on='pid', how='left')
hh = hh[(hh['age'] >= 45) & (hh['age'] <= 105)]
hh = hh[(~hh['state'].str.startswith('on_')) | (hh['age'] >= hh['onset'])]
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
hh['band'] = pd.cut(hh['age'], BINS, labels=BLAB, right=False)
hh = hh.dropna(subset=['band'])
hh['state'] = pd.Categorical(hh['state'], ['never', 'prevalent', 'on_<60', 'on_60-74', 'on_>=75'])
P(f"HRS person-waves={len(hh):,}  persons={hh['pid'].nunique():,}")

# ==================== SHARE ====================
s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = comorb(s)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
rows = []
for pid, x in s.groupby('pid'):
    dg = x['radiagarthr'].dropna(); dg = dg[dg >= 5]
    if dg.empty:
        rows.append(dict(pid=pid, state='never', onset=np.nan)); continue
    o = dg.iloc[0]
    g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
    rows.append(dict(pid=pid, state='on_' + g, onset=o))
S2 = pd.DataFrame(rows)
ss = s.merge(S2, on='pid', how='left')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)]
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss['band'] = pd.cut(ss['age'], BINS, labels=BLAB, right=False)
ss = ss.dropna(subset=['band'])
ss['state'] = pd.Categorical(ss['state'], ['never', 'on_<60', 'on_60-74', 'on_>=75'])
P(f"SHARE person-waves={len(ss):,}  persons={ss['pid'].nunique():,}")

# ==================== Table 2 重估 ====================
rows_t2 = []

def fit_block(data, outcome, lib, kind, label):
    for sup in ['all', '70-79']:
        d = data.dropna(subset=[outcome]).copy()
        if sup == '70-79':
            d = d[(d['age'] >= 70) & (d['age'] < 80)]
        if len(d) < 200:
            continue
        # 年龄段 + 连续年龄；70-79 子集内 band 为常数，故只留连续年龄
        rhs = f"C(state) + C(edu_c) + male + age"
        if sup == 'all':
            rhs = f"C(state) + C(band) + C(edu_c) + male + age"
        try:
            if kind == 'beta':
                m = smf.mixedlm(f"{outcome} ~ {rhs}", d, groups=d['pid'].values).fit(
                    reml=False, method='lbfgs')
            else:
                m = smf.gee(f"{outcome} ~ {rhs}", groups='pid', data=d,
                            family=sm.families.Poisson(),
                            cov_struct=sm.cov_struct.Independence()).fit()
        except Exception as ex:
            P(f"  {lib} {label} [{sup}] 失败: {str(ex)[:60]}")
            continue
        P(f"\n--- {lib} | {label} | support={sup}: waves={len(d):,} persons={d['pid'].nunique():,} ---")
        for k in m.params.index:
            if not k.startswith('C(state)'):
                continue
            c, se, p = m.params[k], m.bse[k], m.pvalues[k]
            if kind == 'beta':
                est, lo, hi = c, c - 1.96 * se, c + 1.96 * se
            else:
                est, lo, hi = np.exp(c), np.exp(c - 1.96 * se), np.exp(c + 1.96 * se)
            st = k.split('[T.')[1].rstrip(']')
            P(f"    {st:10s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={p:.3g}")
            rows_t2.append(dict(lib=lib, outcome=label, term=st, kind=kind, est=est, lo=lo, hi=hi,
                                p=p, n=len(d), persons=d['pid'].nunique(), support=sup))

fit_block(hh, 'adl6a', 'HRS', 'beta', 'ADL limitations (0-6)')
fit_block(hh, 'iadl5a', 'HRS', 'beta', 'IADL limitations (0-5)')
fit_block(hh, 'comorb', 'HRS', 'IRR', 'Comorbidity count (0-6)')
fit_block(ss, 'mob', 'SHARE', 'IRR', 'Mobility limitation count (0-2)')
fit_block(ss, 'comorb', 'SHARE', 'IRR', 'Comorbidity count (0-6)')

T2 = pd.DataFrame(rows_t2)
T2.to_csv(f"{OUT}\\table2_states_v2.csv", index=False, encoding='utf-8-sig')
P(f"\n写出 table2_states_v2.csv  {len(T2)} 行")

# ==================== Figure 3：新暴露五态，分年龄别死亡率 ====================
P("\n\n" + "=" * 84 + "\n# Figure 3：分年龄别死亡率（新暴露定义）\n" + "=" * 84)

def split_py(start, stop, grp, dead, w=5, lo=45, hi=110):
    out = []
    a = max(math.floor(start / w) * w, lo)
    while a < stop and a < hi:
        s0 = max(start, a); e0 = min(stop, a + w)
        if e0 > s0 + 1e-9:
            out.append((grp, a, e0 - s0, 0))
        a += w
    if out and dead:
        out[-1] = (out[-1][0], out[-1][1], out[-1][2], 1)
    return out

# HRS 时依分段：状态在首次阳性报告处切换
h2 = h.copy()
rows3 = []
for pid, x in h2.groupby('pid'):
    x = x.dropna(subset=['age'])
    if x.empty: continue
    e0 = x['age'].iloc[0]
    lastyr = x['yr'].max()
    dead = x['radyear'].max()
    ex = (dead - x['rabyear'].iloc[0]) if pd.notna(dead) else (lastyr - x['rabyear'].iloc[0])
    if not np.isfinite(ex) or ex <= e0: continue
    ev = int(pd.notna(dead))
    st = S[S['pid'] == pid]
    state = st['state'].iloc[0] if len(st) else 'never'
    onset = st['onset'].iloc[0] if len(st) else np.nan
    if state.startswith('on_') and np.isfinite(onset) and onset > e0:
        # 起病前算未患病，起病后算该层
        rows3 += split_py(e0, min(onset, ex), 'never', False)
        if ex > onset:
            seg = split_py(onset, ex, state, ev)
            rows3 += seg
    else:
        rows3 += split_py(e0, ex, state, ev)

F3 = pd.DataFrame(rows3, columns=['grp', 'band', 'py', 'death'])
g3 = F3.groupby(['grp', 'band'], observed=True).agg(py=('py', 'sum'), death=('death', 'sum')).reset_index()
g3['rate'] = 1000 * g3['death'] / g3['py']
g3 = g3[g3['py'] >= 200]
P(g3.pivot(index='band', columns='grp', values='rate').round(1).to_string())
g3.to_csv(f"{OUT}\\fig3_rates_v2.csv", index=False, encoding='utf-8-sig')
P(f"\n写出 fig3_rates_v2.csv  {len(g3)} 行")
P("\n[各组人年与死亡（Figure 3 口径）]")
P(F3.groupby('grp', observed=True).agg(py=('py', 'sum'), deaths=('death', 'sum')).round(0).to_string())

# Figure 1B：两库起病年龄分布（HRS 用 incident 中点）
d1 = []
for lib, v in [('SHARE', pd.read_csv(f"{OUT}\\analysis_share.csv")['onset_age'].dropna()),
               ('HRS', S[S['state'].str.startswith('on_')]['onset'].dropna())]:
    v = v[(v >= 15) & (v < 100)]
    for lo in range(15, 100, 5):
        d1.append(dict(lib=lib, age=lo + 2.5, n=int(((v >= lo) & (v < lo + 5)).sum())))
D1 = pd.DataFrame(d1)
D1['pct'] = D1.groupby('lib')['n'].transform(lambda x: 100 * x / x.sum())
D1.to_csv(f"{OUT}\\fig1_onset_v2.csv", index=False, encoding='utf-8-sig')
P("\n[Figure 1B 数据已写 fig1_onset_v2.csv]")
P(D1.groupby('lib')['n'].sum().to_string())

io.open(f"{OUT}\\s25_v2.txt", 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE s25_v2.txt")
