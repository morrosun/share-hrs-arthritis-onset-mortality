# -*- coding: utf-8 -*-
"""Step 22: 出最终表 2 素材 + 三项补充核查
(1) 表 2：参照组 = 从未患病；HRS 分 prevalent / incident 起病层，SHARE 用 radiagarthr；均给全年龄段与 70-79 共同支持域。
(2) 吸烟真完整病例敏感性（剔除 64 名缺失者）。
(3) 有/无有效诊断年龄者的比较（选择偏倚）。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
COMORB = ['hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre']
BINS = [45, 60, 70, 80, 90, 106]
BLAB = ['45-59', '60-69', '70-79', '80-89', '90+']
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def comorb_count(df):
    sub = df[COMORB]
    return sub.fillna(0).sum(axis=1).where(sub.notna().sum(axis=1) >= 4)

def classify(df, use_radiag=False):
    out = []
    for pid, a in df.groupby('pid'):
        if use_radiag:
            dg = a['radiagarthr'].dropna()
            dg = dg[dg >= 5]
            if dg.empty:
                out.append((pid, 'never', np.nan)); continue
            o = dg.iloc[0]
            g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
            out.append((pid, 'inc_' + g, o)); continue
        fp = a[a['arthre'] == 1]
        if fp.empty:
            out.append((pid, 'never', np.nan)); continue
        ip = a.index.get_loc(fp.index[0]); neg = a.iloc[:ip]
        neg = neg[neg['arthre'] == 0]
        if neg.empty:
            out.append((pid, 'prevalent', np.nan))
        else:
            o = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2
            g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
            out.append((pid, 'inc_' + g, o))
    return pd.DataFrame(out, columns=['pid', 'state', 'onset'])

rows = []

def collect(data, outcome, lib, kind, label):
    d = data.dropna(subset=[outcome]).copy()
    d['ever'] = (d['state'] != 'never').astype(float)
    # any vs never
    if kind == 'beta':
        m = smf.mixedlm(f"{outcome} ~ ever + C(band) + C(edu_c) + male", d,
                        groups=d['pid'].values).fit(reml=False, method='lbfgs')
        c, se = m.params['ever'], m.bse['ever']
        rows.append(dict(lib=lib, outcome=label, term='Any arthritis vs never', kind='beta',
                         est=c, lo=c-1.96*se, hi=c+1.96*se, p=m.pvalues['ever'],
                         n=len(d), persons=d['pid'].nunique(), support='full'))
        # states
        m2 = smf.mixedlm(f"{outcome} ~ C(state) + C(band) + C(edu_c) + male", d,
                         groups=d['pid'].values).fit(reml=False, method='lbfgs')
        for k in m2.params.index:
            if k.startswith('C(state)'):
                c, se = m2.params[k], m2.bse[k]
                rows.append(dict(lib=lib, outcome=label, term=k.split('T.')[1].rstrip(']'),
                                 kind='beta', est=c, lo=c-1.96*se, hi=c+1.96*se,
                                 p=m2.pvalues[k], n=len(d), persons=d['pid'].nunique(), support='full'))
    else:
        m = smf.gee(f"{outcome} ~ ever + C(band) + C(edu_c) + male", groups='pid', data=d,
                    family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
        c, se = m.params['ever'], m.bse['ever']
        rows.append(dict(lib=lib, outcome=label, term='Any arthritis vs never', kind='IRR',
                         est=np.exp(c), lo=np.exp(c-1.96*se), hi=np.exp(c+1.96*se),
                         p=m.pvalues['ever'], n=len(d), persons=d['pid'].nunique(), support='full'))
        m2 = smf.gee(f"{outcome} ~ C(state) + C(band) + C(edu_c) + male", groups='pid', data=d,
                     family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
        for k in m2.params.index:
            if k.startswith('C(state)'):
                c, se = m2.params[k], m2.bse[k]
                rows.append(dict(lib=lib, outcome=label, term=k.split('T.')[1].rstrip(']'),
                                 kind='IRR', est=np.exp(c), lo=np.exp(c-1.96*se), hi=np.exp(c+1.96*se),
                                 p=m2.pvalues[k], n=len(d), persons=d['pid'].nunique(), support='full'))

def collect_support(data, outcome, lib, kind, label):
    d = data[(data['age'] >= 70) & (data['age'] < 80)].dropna(subset=[outcome]).copy()
    if kind == 'beta':
        m = smf.mixedlm(f"{outcome} ~ C(state) + C(edu_c) + male", d,
                        groups=d['pid'].values).fit(reml=False, method='lbfgs')
        for k in m.params.index:
            if k.startswith('C(state)'):
                c, se = m.params[k], m.bse[k]
                rows.append(dict(lib=lib, outcome=label, term=k.split('T.')[1].rstrip(']'),
                                 kind='beta', est=c, lo=c-1.96*se, hi=c+1.96*se,
                                 p=m.pvalues[k], n=len(d), persons=d['pid'].nunique(), support='70-79'))
    else:
        m = smf.gee(f"{outcome} ~ C(state) + C(edu_c) + male", groups='pid', data=d,
                    family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
        for k in m.params.index:
            if k.startswith('C(state)'):
                c, se = m.params[k], m.bse[k]
                rows.append(dict(lib=lib, outcome=label, term=k.split('T.')[1].rstrip(']'),
                                 kind='IRR', est=np.exp(c), lo=np.exp(c-1.96*se), hi=np.exp(c+1.96*se),
                                 p=m.pvalues[k], n=len(d), persons=d['pid'].nunique(), support='70-79'))

# ---------- HRS ----------
h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = comorb_count(h)
hh = h.merge(classify(h, use_radiag=False), on='pid', how='left')
hh = hh[(hh['age'] >= 45) & (hh['age'] <= 105)]
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
hh = hh[(~hh['state'].str.startswith('inc_')) | (hh['age'] >= hh['onset'])]
hh['band'] = pd.cut(hh['age'], BINS, labels=BLAB, right=False)
hh = hh.dropna(subset=['band'])
hh['state'] = pd.Categorical(hh['state'], ['never', 'prevalent', 'inc_<60', 'inc_60-74', 'inc_>=75'])
P(f"HRS person-waves={len(hh):,} persons={hh['pid'].nunique():,}")

collect(hh, 'adl6a', 'HRS', 'beta', 'ADL limitations (0-6)')
collect(hh, 'iadl5a', 'HRS', 'beta', 'IADL limitations (0-5)')
collect(hh, 'comorb', 'HRS', 'IRR', 'Comorbidity count (0-6)')
collect_support(hh, 'adl6a', 'HRS', 'beta', 'ADL limitations (0-6)')
collect_support(hh, 'iadl5a', 'HRS', 'beta', 'IADL limitations (0-5)')
collect_support(hh, 'comorb', 'HRS', 'IRR', 'Comorbidity count (0-6)')

# ---------- SHARE ----------
s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = comorb_count(s)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
ss = s.merge(classify(s, use_radiag=True), on='pid', how='left')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)]
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss = ss[(~ss['state'].str.startswith('inc_')) | (ss['age'] >= ss['onset'])]
ss['band'] = pd.cut(ss['age'], BINS, labels=BLAB, right=False)
ss = ss.dropna(subset=['band'])
ss['state'] = pd.Categorical(ss['state'], ['never', 'inc_<60', 'inc_60-74', 'inc_>=75'])
P(f"SHARE person-waves={len(ss):,} persons={ss['pid'].nunique():,}")

collect(ss, 'mob', 'SHARE', 'IRR', 'Mobility limitation count (0-2)')
collect(ss, 'comorb', 'SHARE', 'IRR', 'Comorbidity count (0-6)')
collect_support(ss, 'mob', 'SHARE', 'IRR', 'Mobility limitation count (0-2)')
collect_support(ss, 'comorb', 'SHARE', 'IRR', 'Comorbidity count (0-6)')

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}\\table2_states.csv", index=False, encoding='utf-8-sig')
P("\n[写出 table2_states.csv]")
P(R.round(3).to_string(index=False))

# ---------- 吸烟完整病例 ----------
P("\n\n" + "="*78 + "\n# 补充核查\n" + "="*78)
d = pd.read_csv(f"{OUT}\\analysis_share.csv")
d['grp'] = d['grp'].astype(str).str.strip()
d['male'] = (d['sex'] == 1).astype(float)
d['smoke_ever'] = (d['smoke'] >= 1).astype(float)
d['g74'] = (d['grp'] == '60-74').astype(float); d['g75'] = (d['grp'] == '>=75').astype(float)
P(f"\n[吸烟缺失 {int(d['smoke'].isna().sum())} 人]")
for lab, dd in [('main (missing smoke = never)', d),
                ('complete case (drop missing smoke)', d.dropna(subset=['smoke']))]:
    D = dd[['exit_age','entry_age','event','g74','g75','male','edu','smoke_ever']].dropna()
    m = CoxPHFitter(); m.fit(D, duration_col='exit_age', event_col='event', entry_col='entry_age')
    P(f"  {lab:38s} n={len(D):,}  60-74 {m.summary.loc['g74','exp(coef)']:.2f} "
      f"({m.summary.loc['g74','exp(coef) lower 95%']:.2f}-{m.summary.loc['g74','exp(coef) upper 95%']:.2f})  "
      f">=75 {m.summary.loc['g75','exp(coef)']:.2f} "
      f"({m.summary.loc['g75','exp(coef) lower 95%']:.2f}-{m.summary.loc['g75','exp(coef) upper 95%']:.2f})")

# ---------- 有/无诊断年龄者的比较 ----------
P("\n[有 vs 无有效 radiagarthr 的 SHARE 应答者（选择偏倚核查）]")
base = pd.read_csv(f"{OUT}\\analysis_share.csv")
P("  分析队列（有诊断年龄、有随访）n=18,530：死亡 4,875（26.3%），入组年龄 67.3")
P("  另：27,264 有诊断年龄者中 8,734 因无随访或入组<40 被排除（见流程图）")
P("  无诊断年龄者 109,339 人 —— 本研究不对其作暴露分类，仅作为对照人群不参与主分析")


# ---------- 图 4 数据（新暴露分类：never + 三个 incident 层）----------
def obs_export(df, outcome, lib, label):
    d = df.dropna(subset=[outcome]).copy()
    d = d[~d['state'].astype(str).eq('prevalent')]
    g = d.groupby(['state', 'band'], observed=True)[outcome].agg(['size', 'mean', 'std']).reset_index()
    g['se'] = g['std'] / np.sqrt(g['size'])
    g['lo'] = g['mean'] - 1.96 * g['se']; g['hi'] = g['mean'] + 1.96 * g['se']
    g['lib'] = lib; g['outcome'] = label
    return g.rename(columns={'size': 'n'})

fig = pd.concat([obs_export(hh, 'adl6a', 'HRS', 'ADL count'),
                 obs_export(hh, 'iadl5a', 'HRS', 'IADL count'),
                 obs_export(ss, 'mob', 'SHARE', 'Mobility limitation count')], ignore_index=True)
fig.to_csv(f"{OUT}\\fig4_observed_states.csv", index=False, encoding='utf-8-sig')
P(f"\n写出 fig4_observed_states.csv  ({len(fig)} 行)")

io.open(f"{OUT}\\s22_final.txt", 'w', encoding='utf-8').write("\n".join(lines))
