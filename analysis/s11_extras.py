# -*- coding: utf-8 -*-
"""Step 11: 补齐论文所需图表素材
A) Table 4  HRS 死亡（时依 Cox）
B) Figure 1B 起病年龄分布（两库）
C) Figure 1A 队列流程图数据
D) Figure 3  分年龄别死亡率（同龄是否并轨）
E) Table S2 炎症层（ELSA WBC + CHARLS 横断面）
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, pyreadr
import statsmodels.formula.api as smf
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)
def load(db): return pyreadr.read_r(f"{BASE}\\{db}.RData")[db]

WY_H = {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020}
WY_S = {1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020}
BANDS = [(45,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,90),(90,105)]
BLAB  = ["45-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]

def hr_to_rr(hr):
    return (1 - 0.5**np.sqrt(hr)) / (1 - 0.5**np.sqrt(1/hr))
def e_val(rr):
    if rr < 1: rr = 1/rr
    return rr + np.sqrt(rr*(rr-1))

def e_for_bound(lo, hi):
    """CI 界最接近 1 者；若 CI 已跨 1，则无需混杂，E-value = 1"""
    if lo <= 1 <= hi: return 1.0
    b = hi if hi < 1 else lo
    return e_val(hr_to_rr(b))

# ==================== A. HRS 时依 Cox（死亡） ====================
P("="*76+"\n# A. HRS 死亡（时依 Cox）\n"+"="*76)
h = load('hrs').copy()
h = h.rename(columns={h.columns[0]:'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age','arthre']).sort_values(['pid','wave'])
hi = h.groupby('pid').agg(byear=('rabyear','first'), sex=('ragender','first'),
                          entry_age=('age','min'), last_yr=('yr','max'))
ons = h[h['arthre']==1].groupby('pid').agg(onset_age=('age','min'))
dd  = h.groupby('pid')['radyear'].max().rename('deadyr')
hi = hi.join(ons).join(dd)
hi['exit_year'] = np.where(hi['deadyr'].notna(), hi['deadyr'], hi['last_yr'])
hi['event'] = hi['deadyr'].notna().astype(int)
P(f"人={len(hi)}  死亡={int(hi['event'].sum())}")

rows = []
for pid, r in hi.iterrows():
    e0, ex = r['entry_age'], r['exit_year'] - r['byear']
    if not np.isfinite(ex) or ex <= e0: continue
    a1 = r['onset_age']
    if pd.isna(a1) or a1 >= ex:
        rows.append((pid, e0, ex, r['event'], 0, 0.0, 0.0, 0.0, r['sex']))
    else:
        if a1 > e0: rows.append((pid, e0, a1, 0, 0, 0.0, 0.0, 0.0, r['sex']))
        g = 1 if a1 < 60 else (2 if a1 < 75 else 3)
        rows.append((pid, max(a1, e0), ex, r['event'], 1,
                     1.0 if g==1 else 0.0, 1.0 if g==2 else 0.0, 1.0 if g==3 else 0.0,
                     r['sex']))
T = pd.DataFrame(rows, columns=['pid','start','stop','event','art','g_lt60','g_6074','g_75','sex'])
T = T[T['stop'] > T['start']].copy()
T['male'] = (T['sex']==1).astype(float)
P(f"段={len(T)}  事件={int(T['event'].sum())}")

# 模型1：关节炎 vs 无（时依）
m1 = CoxTimeVaryingFitter()
m1.fit(T[['pid','start','stop','event','art','male']], id_col='pid', start_col='start',
       stop_col='stop', event_col='event', formula="art + male")
hr1 = m1.summary.loc['art']
P(f"\n关节炎 vs 无: HR {hr1['exp(coef)']:.3f} ({hr1['exp(coef) lower 95%']:.3f}-{hr1['exp(coef) upper 95%']:.3f}) p={hr1['p']:.1e}")

# 模型2：起病年龄组（vs 无关节炎）
m2 = CoxTimeVaryingFitter()
m2.fit(T[['pid','start','stop','event','g_lt60','g_6074','g_75','male']], id_col='pid',
       start_col='start', stop_col='stop', event_col='event',
       formula="g_lt60 + g_6074 + g_75 + male")
P("\n起病年龄组 vs 无关节炎：")
P(m2.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())

# 模型3：仅患病期内，vs 起病 <60
T3 = T[T['art']==1].copy()
m3 = CoxTimeVaryingFitter()
m3.fit(T3[['pid','start','stop','event','g_6074','g_75','male']], id_col='pid',
       start_col='start', stop_col='stop', event_col='event',
       formula="g_6074 + g_75 + male")
P("\n仅患病者（vs 起病 <60）：")
P(m3.summary[['exp(coef)','exp(coef lower 95%' if False else 'exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())

t4 = []
def push(label, key, summ, n, ev, kind='HR'):
    r = summ.loc[key]
    hr = r['exp(coef)']; lo = r['exp(coef) lower 95%']; hi = r['exp(coef) upper 95%']
    t4.append(dict(contrast=label, kind=kind, est=hr, lo=lo, hi=hi, p=r['p'],
                   e_point=e_val(hr_to_rr(hr)),
                   e_bound=e_for_bound(lo, hi), n=n, events=ev))
push("Any arthritis vs none", 'art', m1.summary, len(T), int(T['event'].sum()))
push("Onset <60 vs none", 'g_lt60', m2.summary, len(T), int(T['event'].sum()))
push("Onset 60-74 vs none", 'g_6074', m2.summary, len(T), int(T['event'].sum()))
push("Onset >=75 vs none", 'g_75', m2.summary, len(T), int(T['event'].sum()))
push("Onset 60-74 vs <60 (among arthritis)", 'g_6074', m3.summary, T3['pid'].nunique(), int(T3['event'].sum()))
push("Onset >=75 vs <60 (among arthritis)", 'g_75', m3.summary, T3['pid'].nunique(), int(T3['event'].sum()))
D4 = pd.DataFrame(t4)
D4.to_csv(f"{OUT}\\table4_hrs_mortality.csv", index=False, encoding='utf-8-sig')
P("\n[Table 4]\n" + D4.round(3).to_string(index=False))

# ==================== B. 起病年龄分布 ====================
P("\n"+"="*76+"\n# B. 起病年龄分布\n"+"="*76)
hrs_on = hi['onset_age'].dropna()
hrs_on = hrs_on[(hrs_on>=18)&(hrs_on<=105)]
sh = pd.read_csv(f"{OUT}\\analysis_share.csv")
sh_on = sh['onset_age'].dropna()
P(f"SHARE n={len(sh_on)} 中位={sh_on.median():.0f}  IQR={sh_on.quantile(.25):.0f}-{sh_on.quantile(.75):.0f}")
P(f"HRS   n={len(hrs_on)} 中位={hrs_on.median():.0f}  IQR={hrs_on.quantile(.25):.0f}-{hrs_on.quantile(.75):.0f}")
dist = []
for lib, v in [('SHARE', sh_on), ('HRS', hrs_on)]:
    for lo in range(15, 100, 5):
        m = ((v >= lo) & (v < lo+5)).sum()
        dist.append(dict(lib=lib, age=lo+2.5, n=int(m)))
pd.DataFrame(dist).to_csv(f"{OUT}\\fig1_onset_dist.csv", index=False, encoding='utf-8-sig')

# ==================== C. 流程图数字 ====================
flow = pd.DataFrame([
    ('Respondents in SHARE waves 1-8', 136843),
    ('Ever reported a physician diagnosis of arthritis', 47900),
    ('Valid age at first diagnosis (5-105 y)', 27264),
    ('With follow-up time or death record', 18530),
    ('Analytic cohort (deaths = 4,875)', 18530),
])
P("\n[流程]\n" + flow.to_string(index=False, header=False))
flow.to_csv(f"{OUT}\\fig1_flow.csv", index=False, header=False, encoding='utf-8-sig')

# ==================== D. 分年龄别死亡率 ====================
P("\n"+"="*76+"\n# D. 分年龄别死亡率\n"+"="*76)
def split_py(entry, exit_, grp, died):
    out = []
    for (lo, hi_) in BANDS:
        ov = min(exit_, hi_) - max(entry, lo)
        if ov > 0:
            d = 1 if (died and lo <= exit_ < hi_) else 0
            out.append((grp, f"{lo}-{hi_-1}" if hi_ < 105 else "90+", ov, d))
    return out

rowsS = []
for _, r in sh.iterrows():
    if not np.isfinite(r['exit_age']) or r['exit_age'] <= r['entry_age']: continue
    rowsS += split_py(r['entry_age'], r['exit_age'], r['grp'], bool(r['event']))
DS = pd.DataFrame(rowsS, columns=['grp','band','py','death'])
DS = DS.groupby(['grp','band'], as_index=False).agg(py=('py','sum'), deaths=('death','sum'))
DS['rate'] = DS['deaths']/DS['py']*1000
DS['lib'] = 'SHARE'

# HRS 必须用时依分段：起病前的随访时间属于"未暴露"，不能算进患病组
rowsH = []
for _, r in T.iterrows():
    g = 'none' if r['art'] == 0 else ('<60' if r['g_lt60'] == 1 else ('60-74' if r['g_6074'] == 1 else '>=75'))
    rowsH += split_py(r['start'], r['stop'], g, bool(r['event']))
DH = pd.DataFrame(rowsH, columns=['grp','band','py','death'])
DH = DH.groupby(['grp','band'], as_index=False).agg(py=('py','sum'), deaths=('death','sum'))
DH['rate'] = DH['deaths']/DH['py']*1000
DH['lib'] = 'HRS'
RD = pd.concat([DS, DH], ignore_index=True)
RD = RD[(RD['py'] >= 300) & (RD['deaths'] >= 5)]
RD['se'] = RD['rate']/np.sqrt(RD['deaths'].clip(lower=1))
RD['lo'] = RD['rate']*np.exp(-1.96/np.sqrt(RD['deaths'].clip(lower=1)))
RD['hi'] = RD['rate']*np.exp( 1.96/np.sqrt(RD['deaths'].clip(lower=1)))
RD.to_csv(f"{OUT}\\fig3_age_rates.csv", index=False, encoding='utf-8-sig')
P(RD.pivot_table(index='band', columns=['lib','grp'], values='rate').round(1).to_string())

# ==================== E. 炎症层 ====================
P("\n"+"="*76+"\n# E. 炎症层\n"+"="*76)
c = load('charls').copy()
c = c.rename(columns={c.columns[0]:'pid'})
c['yr'] = c['wave'].map({1:2011,2:2013,3:2015,4:2018,5:2020})
c['age'] = c['yr'] - c['rabyear']
c11 = c[(c['wave']==1) & c['arthre'].notna()].copy()
c11['logcrp'] = np.log(c11['bl_crp'].where(c11['bl_crp'] > 0))
c11['male'] = (c11['ragender']==1).astype(float)
c11['edu_c'] = pd.Categorical(c11['raeducl'], [1,2,3], ordered=True) if 'raeducl' in c11 else np.nan
P(f"CHARLS 2011 有 arthre: {len(c11)}  有 CRP: {c11['logcrp'].notna().sum()}")
mm = c11.dropna(subset=['logcrp','age','male'])
P(f"建模 n={len(mm)}")
if 'raeducl' in c11.columns:
    mod = smf.ols("logcrp ~ arthre + age + male + C(edu_c)", data=mm).fit(cov_type='HC1')
else:
    mod = smf.ols("logcrp ~ arthre + age + male", data=mm).fit(cov_type='HC1')
P(f"  arthritis -> CRP: {100*(np.exp(mod.params['arthre'])-1):.1f}% "
  f"({100*(np.exp(mod.params['arthre']-1.96*mod.bse['arthre'])-1):.1f} to "
  f"{100*(np.exp(mod.params['arthre']+1.96*mod.bse['arthre'])-1):.1f})  p={mod.pvalues['arthre']:.3f}")
for v in ['bl_wbc','bl_ua']:
    mm2 = c11.dropna(subset=[v,'age','male'])
    mod2 = smf.ols(f"{v} ~ arthre + age + male", data=mm2).fit(cov_type='HC1')
    P(f"  arthritis -> {v}: {mod2.params['arthre']:+.2f} (p={mod2.pvalues['arthre']:.3f}, n={len(mm2)})")

S2 = pd.DataFrame([
    dict(analysis='ELSA', marker='WBC, per +10 y of onset age', n=3323, est=-0.9, lo=np.nan, hi=np.nan, p=0.036, unit='%'),
    dict(analysis='CHARLS 2011', marker='CRP, arthritis vs none', n=len(mm),
         est=100*(np.exp(mod.params['arthre'])-1),
         lo=100*(np.exp(mod.params['arthre']-1.96*mod.bse['arthre'])-1),
         hi=100*(np.exp(mod.params['arthre']+1.96*mod.bse['arthre'])-1), p=mod.pvalues['arthre'], unit='%'),
])
S2.to_csv(f"{OUT}\\tableS2_inflammation.csv", index=False, encoding='utf-8-sig')
P("\n[Table S2]\n" + S2.round(2).to_string(index=False))

with open(f"{OUT}\\s11_extras.txt",'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s11_extras.txt")
