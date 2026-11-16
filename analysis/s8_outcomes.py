# -*- coding: utf-8 -*-
"""Step 8 (v2): 次结局 —— 失能轨迹 + 共病累积
关键修正：起病年龄组决定其年龄观测范围，组别×年龄 交互不可识别（已验证发散到 4e8）。
故：模型用主效应（平行曲线假设，可解释），图用分年龄段经验均值（不施加函数形式）。
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, pyreadr
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)
def load(db): return pyreadr.read_r(os.path.join(BASE, f"{db}.RData"))[db]

WY_H = {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020}
WY_S = {1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020}
COMORB = ['hibpe','diabe','hearte','stroke','lunge','cancre']
BINS  = [45,60,70,80,90,106]
BLAB  = ['45-59','60-69','70-79','80-89','90+']

def comorb_count(df):
    sub = df[COMORB]
    return sub.fillna(0).sum(axis=1).where(sub.notna().sum(axis=1) >= 4)

def tidy(res, pat):
    t = pd.DataFrame({'coef':res.params, 'se':res.bse, 'p':res.pvalues})
    t['lo'] = t['coef']-1.96*t['se']; t['hi'] = t['coef']+1.96*t['se']
    return t[t.index.str.match(pat)]

# ==================== HRS ====================
P("="*74+"\n# HRS\n"+"="*74)
h = load('hrs').copy()
h = h.rename(columns={h.columns[0]:'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = comorb_count(h)
h = h.sort_values(['pid','wave'])

pos  = h[h['arthre']==1].groupby('pid').agg(onset_yr=('yr','min'), onset_age=('age','min'))
ever = h.groupby('pid')['arthre'].max().rename('ever')
hp = pd.DataFrame({'pid': h['pid'].unique()}).set_index('pid').join(pos).join(ever).reset_index()
def gh(r):
    if r['ever'] != 1 or pd.isna(r['onset_age']): return 'never'
    a = r['onset_age']
    return '<60' if a < 60 else ('60-74' if a < 75 else '>=75')
hp['og'] = hp.apply(gh, axis=1)
P("人级分组:\n" + hp['og'].value_counts().to_string())

hh = h.merge(hp[['pid','og','onset_yr']], on='pid', how='left')
hh = hh[((hh['og']=='never') | (hh['yr'] >= hh['onset_yr'])) & hh['adl6a'].notna()].copy()
hh = hh[(hh['age']>=45) & (hh['age']<=105)].reset_index(drop=True)
hh['og'] = pd.Categorical(hh['og'], ['<60','60-74','>=75','never'], ordered=False)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1,2,3], ordered=True)
hh['male'] = (hh['ragender']==1).astype(float)
hh = hh.dropna(subset=['age','adl6a','edu_c','male']).reset_index(drop=True)
hh['band'] = pd.cut(hh['age'], BINS, labels=BLAB, right=False)
hh = hh.dropna(subset=['band']).reset_index(drop=True)
P(f"行={len(hh)}  人={hh['pid'].nunique()}")

P("\n[分年龄段经验均值 — ADL]")
tab = hh.groupby(['og','band'], observed=True)['adl6a'].agg(n='size', mean='mean').reset_index()
P(tab.pivot(index='band', columns='og', values='mean').round(2).to_string())
P("\n[格子 n]")
P(tab.pivot(index='band', columns='og', values='n').to_string())

P("\n--- HRS ADL (0-6): 线性混合模型，主效应（ref=<60）---")
m_adl = smf.mixedlm("adl6a ~ C(og) + C(band) + C(edu_c) + male",
                    hh, groups=hh['pid'].values).fit(reml=False, method='lbfgs')
P(tidy(m_adl, r'^C\(og\)').round(3).to_string())

P("\n--- HRS IADL (0-5) ---")
hh_i = hh.dropna(subset=['iadl5a']).reset_index(drop=True)
m_iadl = smf.mixedlm("iadl5a ~ C(og) + C(band) + C(edu_c) + male",
                     hh_i, groups=hh_i['pid'].values).fit(reml=False, method='lbfgs')
P(tidy(m_iadl, r'^C\(og\)').round(3).to_string())

P("\n--- HRS 共病数 (0-6): GEE Poisson ---")
hhc = hh.dropna(subset=['comorb']).reset_index(drop=True)
g_com = smf.gee("comorb ~ C(og) + C(band) + C(edu_c) + male", groups='pid',
                data=hhc, family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Independence()).fit()
P(tidy(g_com, r'^C\(og\)').round(3).to_string())

# ==================== SHARE ====================
P("\n\n"+"="*74+"\n# SHARE\n"+"="*74)
s = load('share').copy()
s = s.rename(columns={s.columns[0]:'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = comorb_count(s)
# dadliv 在本构建中不可用（随年龄下降、93% 跨波恒定），改用移动功能条目
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
s = s.sort_values(['pid','wave'])

dg = s.dropna(subset=['radiagarthr']).assign(
        radiagarthr=lambda x: x['radiagarthr'].where(x['radiagarthr']>=5)).dropna(subset=['radiagarthr'])
dg = dg.groupby('pid').agg(onset_age=('radiagarthr','first'))
ever_s = s.groupby('pid')['arthre'].max().rename('ever')
sp = pd.DataFrame({'pid': s['pid'].unique()}).set_index('pid').join(dg).join(ever_s).reset_index()
def gs(r):
    if pd.isna(r['onset_age']) or r['ever'] != 1: return 'never'
    a = r['onset_age']
    return '<60' if a < 60 else ('60-74' if a < 75 else '>=75')
sp['og'] = sp.apply(gs, axis=1)
P("人级分组:\n" + sp['og'].value_counts().to_string())

ss = s.merge(sp[['pid','og','onset_age']], on='pid', how='left')
ss['onset_yr'] = ss['rabyear'] + ss['onset_age']
ss = ss[((ss['og']=='never') | (ss['yr'] >= ss['onset_yr'])) & ss['mob'].notna()].copy()
ss = ss[(ss['age']>=45) & (ss['age']<=105)].reset_index(drop=True)
ss['og'] = pd.Categorical(ss['og'], ['<60','60-74','>=75','never'], ordered=False)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1,2,3], ordered=True)
ss['male'] = (ss['ragender']==1).astype(float)
ss = ss.dropna(subset=['age','mob','edu_c','male']).reset_index(drop=True)
ss['band'] = pd.cut(ss['age'], BINS, labels=BLAB, right=False)
ss = ss.dropna(subset=['band']).reset_index(drop=True)
P(f"行={len(ss)}  人={ss['pid'].nunique()}")

P("\n[分年龄段 移动功能受限数 均值]")
tab2 = ss.groupby(['og','band'], observed=True)['mob'].agg(n='size', mean='mean').reset_index()
P(tab2.pivot(index='band', columns='og', values='mean').round(3).to_string())
P("\n[格子 n]")
P(tab2.pivot(index='band', columns='og', values='n').to_string())

P("\n--- SHARE 移动功能受限数 (0-2): GEE Poisson（ref=<60）---")
g_adl = smf.gee("mob ~ C(og) + C(band) + C(edu_c) + male", groups='pid',
                data=ss, family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Independence()).fit()
ta = tidy(g_adl, r'^C\(og\)'); ta['IRR']=np.exp(ta['coef'])
ta['IRR_lo']=np.exp(ta['lo']); ta['IRR_hi']=np.exp(ta['hi'])
P(ta[['IRR','IRR_lo','IRR_hi','p']].round(3).to_string())

P("\n--- SHARE 共病数: GEE Poisson ---")
ssc = ss.dropna(subset=['comorb']).reset_index(drop=True)
g_com_s = smf.gee("comorb ~ C(og) + C(band) + C(edu_c) + male", groups='pid',
                  data=ssc, family=sm.families.Poisson(),
                  cov_struct=sm.cov_struct.Independence()).fit()
tb = tidy(g_com_s, r'^C\(og\)'); tb['IRR']=np.exp(tb['coef'])
tb['IRR_lo']=np.exp(tb['lo']); tb['IRR_hi']=np.exp(tb['hi'])
P(tb[['IRR','IRR_lo','IRR_hi','p']].round(3).to_string())

# ==================== 主对比：有/无关节炎 ====================
P("\n\n"+"="*74+"\n# 主对比：any arthritis vs none\n"+"="*74)
hh['ever'] = (hh['og'] != 'never').astype(float)
hh_i['ever'] = (hh_i['og'] != 'never').astype(float)
hhc['ever'] = (hhc['og'] != 'never').astype(float)
m_e_adl = smf.mixedlm("adl6a ~ ever + C(band) + C(edu_c) + male", hh,
                      groups=hh['pid'].values).fit(reml=False, method='lbfgs')
m_e_iadl = smf.mixedlm("iadl5a ~ ever + C(band) + C(edu_c) + male", hh_i,
                       groups=hh_i['pid'].values).fit(reml=False, method='lbfgs')
g_e_com = smf.gee("comorb ~ ever + C(band) + C(edu_c) + male", groups='pid', data=hhc,
                  family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
ss['ever'] = (ss['og'] != 'never').astype(float)
ssc['ever'] = (ssc['og'] != 'never').astype(float)
g_e_mob = smf.gee("mob ~ ever + C(band) + C(edu_c) + male", groups='pid', data=ss,
                  family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
g_e_com_s = smf.gee("comorb ~ ever + C(band) + C(edu_c) + male", groups='pid', data=ssc,
                    family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
any_rows = []
for lib, name, res, kind in [
    ('HRS','ADL count (0-6)', m_e_adl, 'beta'),
    ('HRS','IADL count (0-5)', m_e_iadl, 'beta'),
    ('HRS','Comorbidity count (0-6)', g_e_com, 'IRR'),
    ('SHARE','Mobility limitation count (0-2)', g_e_mob, 'IRR'),
    ('SHARE','Comorbidity count (0-6)', g_e_com_s, 'IRR')]:
    c, se, pv = res.params['ever'], res.bse['ever'], res.pvalues['ever']
    if kind == 'beta':
        est, lo, hi = c, c-1.96*se, c+1.96*se
    else:
        est, lo, hi = np.exp(c), np.exp(c-1.96*se), np.exp(c+1.96*se)
    any_rows.append(dict(lib=lib, outcome=name, est=est, lo=lo, hi=hi, p=pv, kind=kind))
    P(f"{lib:6s} {name:32s} {kind:4s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={pv:.2e}")
pd.DataFrame(any_rows).to_csv(os.path.join(OUT, "table2_any.csv"), index=False, encoding='utf-8-sig')

# ==================== 落盘 ====================
def obs(df, ycol, lib, outname):
    g = df.groupby(['og','band'], observed=True)[ycol].agg(['size','mean','std']).reset_index()
    g = g[g['size'] >= 30]
    g['se'] = g['std']/np.sqrt(g['size'])
    g['lo'] = g['mean'] - 1.96*g['se']; g['hi'] = g['mean'] + 1.96*g['se']
    g['lib'] = lib; g['outcome'] = outname
    return g.rename(columns={'size':'n'})

FIG = pd.concat([
    obs(hh, 'adl6a', 'HRS', 'ADL count'),
    obs(hh_i, 'iadl5a', 'HRS', 'IADL count'),
    obs(ss, 'mob', 'SHARE', 'Mobility limitation count'),
], ignore_index=True)
FIG.to_csv(os.path.join(OUT, "fig2_observed.csv"), index=False, encoding='utf-8-sig')

def grab(res, outcome, lib, kind):
    t = tidy(res, r'^C\(og\)')
    rows = []
    for idx, row in t.iterrows():
        g = idx.split('T.')[1].rstrip(']')
        if kind == 'beta':
            rows.append(dict(lib=lib, outcome=outcome, grp=g, est=row['coef'],
                             lo=row['lo'], hi=row['hi'], p=row['p'], kind='beta'))
        else:
            rows.append(dict(lib=lib, outcome=outcome, grp=g, est=np.exp(row['coef']),
                             lo=np.exp(row['lo']), hi=np.exp(row['hi']), p=row['p'], kind=kind))
    return rows
t2  = grab(m_adl, 'ADL count (0-6)', 'HRS', 'beta')
t2 += grab(m_iadl, 'IADL count (0-5)', 'HRS', 'beta')
t2 += grab(g_com, 'Comorbidity count (0-6)', 'HRS', 'IRR')
t2 += grab(g_adl, 'Mobility limitation count (0-2)', 'SHARE', 'IRR')
t2 += grab(g_com_s, 'Comorbidity count (0-6)', 'SHARE', 'IRR')
T2 = pd.DataFrame(t2)
T2.to_csv(os.path.join(OUT, "table2_est.csv"), index=False, encoding='utf-8-sig')
P("\n[表2 估计值]\n" + T2.round(3).to_string(index=False))

with open(os.path.join(OUT, "s8_outcomes.txt"),'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s8_outcomes.txt")
