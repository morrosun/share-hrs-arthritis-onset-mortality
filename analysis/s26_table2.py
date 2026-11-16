# -*- coding: utf-8 -*-
"""Step 26: Table 2 v2 —— 70-79 共同支持域 + 单岁年龄调整 + 显式组间差异

第二轮评审的三点：
  (1) 70-79 内未调整具体年龄 → 加连续 age
  (2) 「各组相对无关节炎相近」不等于「组间不可区分」→ 直接估层间差值及其 CI
  (3) SHARE 功能分析的分组来源需说明（该队列无 prevalent 类，全部为基线已患病者）
"""
import os
import sys, io, warnings
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
COM = ['hibpe', 'diabe', 'hearte', 'stroke', 'cancre', 'lunge']
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

# ---------- 数据 ----------
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = h[COM].fillna(0).sum(axis=1)
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
hh = h.merge(pd.DataFrame(rows), on='pid', how='left')
hh = hh[(hh['age'] >= 45) & (hh['age'] <= 105)]
hh = hh[(~hh['state'].str.startswith('on_')) | (hh['age'] >= hh['onset'])]
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
hh['band'] = pd.cut(hh['age'], [45, 60, 70, 80, 90, 106],
                    labels=['45-59', '60-69', '70-79', '80-89', '90+'], right=False)
hh = hh.dropna(subset=['band'])
hh['state'] = pd.Categorical(hh['state'], ['never', 'prevalent', 'on_<60', 'on_60-74', 'on_>=75'])

s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = s[COM].fillna(0).sum(axis=1)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
rows = []
for pid, x in s.groupby('pid'):
    dg = x['radiagarthr'].dropna(); dg = dg[dg >= 5]
    if dg.empty:
        rows.append(dict(pid=pid, state='never', onset=np.nan)); continue
    o = dg.iloc[0]
    g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
    rows.append(dict(pid=pid, state='on_' + g, onset=o))
ss = s.merge(pd.DataFrame(rows), on='pid', how='left')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)]
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss['band'] = pd.cut(ss['age'], [45, 60, 70, 80, 90, 106],
                    labels=['45-59', '60-69', '70-79', '80-89', '90+'], right=False)
ss = ss.dropna(subset=['band'])
ss['state'] = pd.Categorical(ss['state'], ['never', 'on_<60', 'on_60-74', 'on_>=75'])
P(f"HRS waves={len(hh):,}  SHARE waves={len(ss):,}")

# ---------- 工具 ----------
def difftest(res, k1, k2, kind):
    b = res.params[k1] - res.params[k2]
    V = res.cov_params()
    se = np.sqrt(V.loc[k1, k1] + V.loc[k2, k2] - 2 * V.loc[k1, k2])
    z = b / se
    from scipy import stats as st
    p = 2 * (1 - st.norm.cdf(abs(z)))
    if kind == 'beta':
        return b, b - 1.96 * se, b + 1.96 * se, p
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p

STATES = {
    'HRS': ['prevalent', 'on_<60', 'on_60-74', 'on_>=75'],
    'SHARE': ['on_<60', 'on_60-74', 'on_>=75'],
}

def run(data, outcome, lib, kind, label):
    d = data[(data['age'] >= 70) & (data['age'] < 80)].dropna(subset=[outcome]).copy()
    rhs = "C(state) + C(edu_c) + male + age"
    if kind == 'beta':
        m = smf.mixedlm(f"{outcome} ~ {rhs}", d, groups=d['pid'].values).fit(reml=False, method='lbfgs')
    else:
        m = smf.gee(f"{outcome} ~ {rhs}", groups='pid', data=d,
                    family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
    P(f"\n--- {lib} | {label} | 70-79: waves={len(d):,} persons={d['pid'].nunique():,} ---")
    out = []
    for st in STATES[lib]:
        k = f"C(state)[T.{st}]"
        c, se, p = m.params[k], m.bse[k], m.pvalues[k]
        if kind == 'beta':
            est, lo, hi = c, c - 1.96 * se, c + 1.96 * se
        else:
            est, lo, hi = np.exp(c), np.exp(c - 1.96 * se), np.exp(c + 1.96 * se)
        P(f"  vs never  {st:11s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={p:.3g}")
        out.append(dict(panel='A', lib=lib, outcome=label, kind=kind, term=st,
                        est=est, lo=lo, hi=hi, p=p, n=len(d), persons=d['pid'].nunique()))
    pairs = [('on_60-74', 'on_<60'), ('on_>=75', 'on_<60'), ('on_>=75', 'on_60-74')]
    for a, b in pairs:
        try:
            est, lo, hi, p = difftest(m, f"C(state)[T.{a}]", f"C(state)[T.{b}]", kind)
            lab = f"{a.replace('on_','')} vs {b.replace('on_','')}"
            P(f"  diff      {lab:18s} {est:+.3f} ({lo:+.3f} to {hi:+.3f})  p={p:.3g}")
            out.append(dict(panel='B', lib=lib, outcome=label, kind=kind, term=lab,
                            est=est, lo=lo, hi=hi, p=p, n=len(d), persons=d['pid'].nunique()))
        except Exception as ex:
            P(f"  diff {a} vs {b} 失败: {str(ex)[:60]}")
    return out

res = []
res += run(hh, 'adl6a', 'HRS', 'beta', 'ADL limitations (0\u20136)')
res += run(hh, 'iadl5a', 'HRS', 'beta', 'IADL limitations (0\u20135)')
res += run(hh, 'comorb', 'HRS', 'ratio', 'Comorbidity count (0\u20136)')
res += run(ss, 'mob', 'SHARE', 'ratio', 'Mobility limitations (0\u20132)')
res += run(ss, 'comorb', 'SHARE', 'ratio', 'Comorbidity count (0\u20136)')

pd.DataFrame(res).to_csv(os.path.join(OUT, "table2_v2.csv"), index=False, encoding='utf-8-sig')
P(f"\n写出 table2_v2.csv  {len(res)} 行")
io.open(os.path.join(OUT, "s26_table2.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("DONE")
