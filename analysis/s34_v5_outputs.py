# -*- coding: utf-8 -*-
"""Step 34: v5 口径重跑 Table 2、图 3、图 4 数据。
v5 规则（与 s29、s33 一致）：
  - HRS 发病层分层 = 区间中点（与 v4 相同）
  - HRS 暴露切换点 = 首次阳性报告年龄：首次阳性之前的所有访谈记为 never，之后记为发病层
  - SHARE 仍为基线自报 radiagarthr 患病队列（无随访起病年龄，切换规则不适用）
输出：
  out/table2_v5.csv         Table 2（70–79，连续年龄；HRS 与 SHARE）
  out/fig3_rates_v5.csv     图 3（HRS 五态分年龄死亡率）
  out/fig4_observed_v5.csv  图 4（HRS ADL/IADL、SHARE 移动受限的观测均值）
  out/s34_v5_outputs.txt
"""
import os
import sys, io, math, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
COM = ['hibpe', 'diabe', 'hearte', 'stroke', 'cancre', 'lunge']
BINS = [45, 60, 70, 80, 90, 106]
BLAB = ['45-59', '60-69', '70-79', '80-89', '90+']
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def gof(o): return '<60' if o < 60 else ('60-74' if o < 75 else '>=75')

# ---------------- HRS：v5 暴露 ----------------
h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = h[COM].fillna(0).sum(axis=1)
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

pers = {}
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        pers[pid] = ('never', np.nan, np.nan); continue
    ip = x.index.get_loc(fp.index[0]); neg = x.iloc[:ip]; neg = neg[neg['arthre'] == 0]
    if neg.empty:
        pers[pid] = ('prevalent', np.nan, np.nan); continue
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2
    pers[pid] = ('on', mid, fp['age'].iloc[0])

def hrs_state(row):
    pat, mid, pos = pers[row['pid']]
    if pat == 'never': return 'never'
    if pat == 'prevalent': return 'prevalent'
    return 'never' if row['age'] < pos else 'on_' + gof(mid)

h['state'] = h.apply(hrs_state, axis=1)
hh = h[(h['age'] >= 45) & (h['age'] <= 105)].copy()
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
hh['band'] = pd.cut(hh['age'], BINS, labels=BLAB, right=False)
hh = hh.dropna(subset=['band'])
hh['state'] = pd.Categorical(hh['state'], ['never', 'prevalent', 'on_<60', 'on_60-74', 'on_>=75'])
P(f"HRS v5 person-waves={len(hh):,} persons={hh['pid'].nunique():,}")

# ---------------- SHARE（不变：基线自报诊断年龄）----------------
s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = s[COM].fillna(0).sum(axis=1)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
sd = {}
for pid, x in s.groupby('pid'):
    dg = x['radiagarthr'].dropna(); dg = dg[dg >= 5]
    sd[pid] = ('on_' + gof(dg.iloc[0])) if not dg.empty else 'never'
ss = s.copy(); ss['state'] = ss['pid'].map(sd).fillna('never')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)].copy()
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss['band'] = pd.cut(ss['age'], BINS, labels=BLAB, right=False)
ss = ss.dropna(subset=['band'])
ss['state'] = pd.Categorical(ss['state'], ['never', 'on_<60', 'on_60-74', 'on_>=75'])
P(f"SHARE person-waves={len(ss):,} persons={ss['pid'].nunique():,}")

# ---------------- Table 2（70–79，连续年龄）----------------
def difftest(res, k1, k2, kind):
    b = res.params[k1] - res.params[k2]
    V = res.cov_params()
    se = np.sqrt(V.loc[k1, k1] + V.loc[k2, k2] - 2 * V.loc[k1, k2])
    p = 2 * (1 - st.norm.cdf(abs(b / se)))
    if kind == 'beta':
        return b, b - 1.96 * se, b + 1.96 * se, p
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p

STATES = {'HRS': ['prevalent', 'on_<60', 'on_60-74', 'on_>=75'], 'SHARE': ['on_<60', 'on_60-74', 'on_>=75']}
out_t2 = []
def run_t2(data, outcome, lib, kind, label):
    d = data[(data['age'] >= 70) & (data['age'] < 80)].dropna(subset=[outcome]).copy()
    rhs = "C(state) + C(edu_c) + male + age"
    if kind == 'beta':
        m = smf.mixedlm(f"{outcome} ~ {rhs}", d, groups=d['pid'].values).fit(reml=False, method='lbfgs')
    else:
        m = smf.gee(f"{outcome} ~ {rhs}", groups='pid', data=d,
                    family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
    P(f"\n--- {lib} | {label} | 70-79: waves={len(d):,} persons={d['pid'].nunique():,} ---")
    for stt in STATES[lib]:
        k = f"C(state)[T.{stt}]"
        c, se, p = m.params[k], m.bse[k], m.pvalues[k]
        est, lo, hi = (c, c - 1.96 * se, c + 1.96 * se) if kind == 'beta' else (np.exp(c), np.exp(c - 1.96 * se), np.exp(c + 1.96 * se))
        P(f"  vs never  {stt:11s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={p:.3g}")
        out_t2.append(dict(panel='A', lib=lib, outcome=label, kind=kind, term=stt, est=est, lo=lo, hi=hi,
                           p=p, n=len(d), persons=d['pid'].nunique()))
    for a, b in [('on_60-74', 'on_<60'), ('on_>=75', 'on_<60'), ('on_>=75', 'on_60-74')]:
        try:
            est, lo, hi, p = difftest(m, f"C(state)[T.{a}]", f"C(state)[T.{b}]", kind)
            lab = f"{a.replace('on_','')} vs {b.replace('on_','')}"
            P(f"  diff      {lab:18s} {est:+.3f} ({lo:+.3f} to {hi:+.3f})  p={p:.3g}")
            out_t2.append(dict(panel='B', lib=lib, outcome=label, kind=kind, term=lab, est=est, lo=lo, hi=hi,
                               p=p, n=len(d), persons=d['pid'].nunique()))
        except Exception as ex:
            P(f"  diff {a} vs {b} 失败: {str(ex)[:60]}")

run_t2(hh, 'adl6a', 'HRS', 'beta', 'ADL limitations (0-6)')
run_t2(hh, 'iadl5a', 'HRS', 'beta', 'IADL limitations (0-5)')
run_t2(hh, 'comorb', 'HRS', 'ratio', 'Comorbidity count (0-6)')
run_t2(ss, 'mob', 'SHARE', 'ratio', 'Mobility limitations (0-2)')
run_t2(ss, 'comorb', 'SHARE', 'ratio', 'Comorbidity count (0-6)')
pd.DataFrame(out_t2).to_csv(f"{OUT}\\table2_v5.csv", index=False, encoding='utf-8-sig')

# ---------------- Figure 3：HRS 五态分年龄死亡率（v5 切换）----------------
P("\n" + "=" * 84 + "\n# Figure 3 (v5)\n" + "=" * 84)
def split_py(start, stop, grp, dead, w=5, lo=45, hi=110):
    res = []
    a = max(math.floor(start / w) * w, lo)
    while a < stop and a < hi:
        s0 = max(start, a); e0 = min(stop, a + w)
        if e0 > s0 + 1e-9: res.append((grp, a, e0 - s0, 0))
        a += w
    if res and dead: res[-1] = (res[-1][0], res[-1][1], res[-1][2], 1)
    return res

rows3 = []
for pid, x in h.groupby('pid'):
    e0 = x['age'].iloc[0]; dead = x['radyear'].max(); lastyr = x['yr'].max()
    ex = (dead - x['rabyear'].iloc[0]) if pd.notna(dead) else (lastyr - x['rabyear'].iloc[0])
    if not np.isfinite(ex) or ex <= e0: continue
    ev = int(pd.notna(dead))
    pat, mid, pos = pers[pid]
    if pat == 'never':
        rows3 += split_py(e0, ex, 'never', ev)
    elif pat == 'prevalent':
        rows3 += split_py(e0, ex, 'prevalent', ev)
    else:
        g = 'on_' + gof(mid)
        ex_inc = max(ex, pos + 0.5) if ev else ex      # 保留同区间死亡事件（同 s29）
        if pos > e0:
            rows3 += split_py(e0, min(pos, ex), 'never', False)
        rows3 += split_py(max(pos, e0), ex_inc, g, ev)
F3 = pd.DataFrame(rows3, columns=['grp', 'band', 'py', 'death'])
g3 = F3.groupby(['grp', 'band'], observed=True).agg(py=('py', 'sum'), death=('death', 'sum')).reset_index()
g3['rate'] = 1000 * g3['death'] / g3['py']
g3 = g3[g3['py'] >= 200]
P(g3.pivot(index='band', columns='grp', values='rate').round(1).to_string())
g3.to_csv(f"{OUT}\\fig3_rates_v5.csv", index=False, encoding='utf-8-sig')
P("\n[各组人年与死亡（v5 口径）]")
P(F3.groupby('grp', observed=True).agg(py=('py', 'sum'), deaths=('death', 'sum')).round(0).to_string())

# ---------------- Figure 4：观测均值（v5）----------------
def obs(df, outcome, lib, label, lab_map):
    d = df.dropna(subset=[outcome]).copy()
    d = d[d['state'].astype(str) != 'prevalent']
    g = d.groupby(['state', 'band'], observed=True)[outcome].agg(['size', 'mean', 'std']).reset_index()
    g['se'] = g['std'] / np.sqrt(g['size'])
    g['lo'] = g['mean'] - 1.96 * g['se']; g['hi'] = g['mean'] + 1.96 * g['se']
    g['state'] = g['state'].astype(str).map(lab_map)
    g['lib'] = lib; g['outcome'] = label
    return g.rename(columns={'size': 'n'}).dropna(subset=['state'])

LAB_HRS = {'never': 'never', 'on_<60': 'inc_<60', 'on_60-74': 'inc_60-74', 'on_>=75': 'inc_>=75'}
LAB_SH = {'never': 'never', 'on_<60': 'inc_<60', 'on_60-74': 'inc_60-74', 'on_>=75': 'inc_>=75'}
f4 = pd.concat([obs(hh, 'adl6a', 'HRS', 'ADL count', LAB_HRS),
                obs(hh, 'iadl5a', 'HRS', 'IADL count', LAB_HRS),
                obs(ss, 'mob', 'SHARE', 'Mobility limitation count', LAB_SH)], ignore_index=True)
f4.to_csv(f"{OUT}\\fig4_observed_v5.csv", index=False, encoding='utf-8-sig')
P(f"\n写出 fig4_observed_v5.csv ({len(f4)} 行)")

io.open(f"{OUT}\\s34_v5_outputs.txt", 'w', encoding='utf-8').write("\n".join(lines))
print("DONE")
