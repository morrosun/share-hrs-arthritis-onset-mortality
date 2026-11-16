# -*- coding: utf-8 -*-
"""Step 21: 失能与共病重做（回应评审第 3 条）
1) 暴露与 s20 一致：never / prevalent / incident 起病层（HRS），SHARE 用 radiagarthr 分层。
2) 组间对比只在与共同年龄支持域内做（HRS 70-79；SHARE 60-79 与 70-79）。
3) 另给年龄标准化预测均值（把各组放到同一 attained age 分布上）。
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
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
COMORB = ['hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre']
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def comorb_count(df):
    sub = df[COMORB]
    return sub.fillna(0).sum(axis=1).where(sub.notna().sum(axis=1) >= 4)

def classify(df, WY, use_radiag=False):
    """返回 person-level 的 state: never / prevalent / inc_<60 / inc_60-74 / inc_>=75"""
    df = df.sort_values(['pid', 'wave'])
    out = []
    for pid, a in df.groupby('pid'):
        a = a.copy()
        if use_radiag:
            dg = a['radiagarthr'].dropna()
            dg = dg[dg >= 5]
            fp = a[a['arthre'] == 1]
            if fp.empty or dg.empty:
                out.append((pid, 'never', np.nan)); continue
            onset = dg.iloc[0]
            g = '<60' if onset < 60 else ('60-74' if onset < 75 else '>=75')
            out.append((pid, 'inc_' + g, onset))
            continue
        fp = a[a['arthre'] == 1]
        if fp.empty:
            out.append((pid, 'never', np.nan)); continue
        ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty:
            out.append((pid, 'prevalent', np.nan))
        else:
            onset = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2
            g = '<60' if onset < 60 else ('60-74' if onset < 75 else '>=75')
            out.append((pid, 'inc_' + g, onset))
    return pd.DataFrame(out, columns=['pid', 'state', 'onset'])

# ==================== HRS ====================
P("="*78 + "\n# HRS 失能（暴露 5 态）\n" + "="*78)
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = comorb_count(h)
cls = classify(h, WY_H, use_radiag=False)
hh = h.merge(cls, on='pid', how='left')
hh = hh[(hh['age'] >= 45) & (hh['age'] <= 105)]
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
# 患病组只用起病后的波次
hh = hh[(~hh['state'].str.startswith('inc_')) | (hh['age'] >= hh['onset'])]
hh['state'] = pd.Categorical(hh['state'],
                             ['never', 'prevalent', 'inc_<60', 'inc_60-74', 'inc_>=75'])
P(f"person-waves={len(hh):,} persons={hh['pid'].nunique():,}")
P(hh.groupby('state', observed=True).size().to_string())

def gee(data, outcome, fam, label, keys=None):
    d = data.dropna(subset=[outcome]).copy()
    m = smf.gee(f"{outcome} ~ C(state) + C(band) + C(edu_c) + male", groups='pid', data=d,
                family=fam, cov_struct=sm.cov_struct.Independence()).fit()
    t = pd.DataFrame({'coef': m.params, 'se': m.bse, 'p': m.pvalues})
    t['lo'] = t['coef'] - 1.96 * t['se']; t['hi'] = t['coef'] + 1.96 * t['se']
    t = t[t.index.str.startswith('C(state)')]
    P(f"\n--- {label}  n={len(d):,} persons={d['pid'].nunique():,} ---")
    if fam is None:
        for k, r in t.iterrows():
            P(f"  {k:34s} {r['coef']:+.3f} ({r['lo']:+.3f} to {r['hi']:+.3f})  p={r['p']:.3g}")
    else:
        for k, r in t.iterrows():
            P(f"  {k:34s} IRR {np.exp(r['coef']):.3f} ({np.exp(r['lo']):.3f}-{np.exp(r['hi']):.3f})  p={r['p']:.3g}")
    return m, t

hh['band'] = pd.cut(hh['age'], [45, 60, 70, 80, 90, 106],
                    labels=['45-59', '60-69', '70-79', '80-89', '90+'], right=False)
mm = smf.mixedlm("adl6a ~ C(state) + C(band) + C(edu_c) + male", hh.dropna(subset=['adl6a']),
                 groups=hh.dropna(subset=['adl6a'])['pid'].values).fit(reml=False, method='lbfgs')
P("\n--- HRS ADL (0-6), 混合模型, 全年龄段 ---")
for k in mm.params.index:
    if k.startswith('C(state)'):
        c, se = mm.params[k], mm.bse[k]
        P(f"  {k:34s} {c:+.3f} ({c-1.96*se:+.3f} to {c+1.96*se:+.3f})  p={mm.pvalues[k]:.3g}")

gee(hh, 'comorb', sm.families.Poisson(), 'HRS 共病数 (IRR), 全年龄段')

# 共同支持域 70-79
sub = hh[(hh['age'] >= 70) & (hh['age'] < 80)].copy()
P(f"\n[共同支持域 70-79] person-waves={len(sub):,} persons={sub['pid'].nunique():,}")
P(sub.groupby('state', observed=True).size().to_string())
mm2 = smf.mixedlm("adl6a ~ C(state) + C(edu_c) + male", sub.dropna(subset=['adl6a']),
                  groups=sub.dropna(subset=['adl6a'])['pid'].values).fit(reml=False, method='lbfgs')
P("\n--- HRS ADL 70-79 内 ---")
for k in mm2.params.index:
    if k.startswith('C(state)'):
        c, se = mm2.params[k], mm2.bse[k]
        P(f"  {k:34s} {c:+.3f} ({c-1.96*se:+.3f} to {c+1.96*se:+.3f})  p={mm2.pvalues[k]:.3g}")
drop = ['band']
d2 = sub.dropna(subset=['comorb']).copy()
g2 = smf.gee("comorb ~ C(state) + C(edu_c) + male", groups='pid', data=d2,
             family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
P("\n--- HRS 共病 70-79 内 (IRR) ---")
for k in g2.params.index:
    if k.startswith('C(state)'):
        c, se = g2.params[k], g2.bse[k]
        P(f"  {k:34s} IRR {np.exp(c):.3f} ({np.exp(c-1.96*se):.3f}-{np.exp(c+1.96*se):.3f})  p={g2.pvalues[k]:.3g}")

# ==================== SHARE ====================
P("\n\n" + "="*78 + "\n# SHARE 失能（radiagarthr 分层）\n" + "="*78)
s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = comorb_count(s)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
cls_s = classify(s, WY_S, use_radiag=True)
ss = s.merge(cls_s, on='pid', how='left')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)]
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss = ss[(~ss['state'].str.startswith('inc_')) | (ss['age'] >= ss['onset'])]
ss['band'] = pd.cut(ss['age'], [45, 60, 70, 80, 90, 106],
                    labels=['45-59', '60-69', '70-79', '80-89', '90+'], right=False)
ss['state'] = pd.Categorical(ss['state'], ['never', 'inc_<60', 'inc_60-74', 'inc_>=75'])
P(f"person-waves={len(ss):,} persons={ss['pid'].nunique():,}")
P(ss.groupby('state', observed=True).size().to_string())
mob = ss.dropna(subset=['mob'])
g = smf.gee("mob ~ C(state) + C(band) + C(edu_c) + male", groups='pid', data=mob,
            family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
P("\n--- SHARE 移动受限 (IRR), 全年龄段 ---")
for k in g.params.index:
    if k.startswith('C(state)'):
        c, se = g.params[k], g.params[k] * 0 + g.bse[k]
        P(f"  {k:34s} IRR {np.exp(c):.3f} ({np.exp(c-1.96*se):.3f}-{np.exp(c+1.96*se):.3f})  p={g.pvalues[k]:.3g}")

sub_s = ss[(ss['age'] >= 70) & (ss['age'] < 80)].copy()
P(f"\n[SHARE 共同支持域 70-79] person-waves={len(sub_s):,} persons={sub_s['pid'].nunique():,}")
gs = smf.gee("mob ~ C(state) + C(edu_c) + male", groups='pid', data=sub_s.dropna(subset=['mob']),
             family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
P("\n--- SHARE 移动受限 70-79 内 (IRR) ---")
for k in gs.params.index:
    if k.startswith('C(state)'):
        c, se = gs.params[k], gs.bse[k]
        P(f"  {k:34s} IRR {np.exp(c):.3f} ({np.exp(c-1.96*se):.3f}-{np.exp(c+1.96*se):.3f})  p={gs.pvalues[k]:.3g}")

io.open(os.path.join(OUT, "s21_common_support.txt"), 'w', encoding='utf-8').write("\n".join(lines))
