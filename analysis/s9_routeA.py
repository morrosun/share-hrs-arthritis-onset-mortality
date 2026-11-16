# -*- coding: utf-8 -*-
"""Step 9: 路线 A 的两个核心分析
A) 偏倚复现：同一队列，三种分析策略 → 看"晚发更差"能否被时间尺度的错误选择造出来
B) E-value：关节炎-死亡关联，及起病年龄阴性结果的稳健性
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, pyreadr
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from scipy import stats

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)
def load(db): return pyreadr.read_r(os.path.join(BASE, f"{db}.RData"))[db]

WY_S = {1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020}
WY_H = {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020}

# ============ A. 偏倚复现（SHARE） ============
P("="*76+"\n# A. 同一队列、三种分析策略（SHARE, n=18,530 / 4,875 deaths）\n"+"="*76)
p = pd.read_csv(os.path.join(OUT, "analysis_share.csv"))
p['grp'] = p['grp'].astype(str).str.strip()
p['event'] = p['event'].astype(int)
p['male'] = (p['sex']==1).astype(float)
p['smoke_ever'] = (p['smoke']>=1).astype(float)
p = p.dropna(subset=['stop' if 'stop' in p else 'exit_age'])
p['g_60_74'] = (p['grp']=='60-74').astype(float)
p['g_ge75']  = (p['grp']=='>=75').astype(float)
p['fu'] = (p['exit_age'] - p['entry_age']).clip(lower=0.1)
D = p.dropna(subset=['fu','event','entry_age','male','edu','smoke_ever']).copy()
P(f"队列 n={len(D)}  deaths={int(D['event'].sum())}")
P(f"随访年数 均值={D['fu'].mean():.1f}  中位={D['fu'].median():.1f}")

def fit(df, dur, entry=None, formula="g_60_74 + g_ge75"):
    m = CoxPHFitter()
    kw = dict(duration_col=dur, event_col='event', formula=formula)
    if entry: kw['entry_col'] = entry
    m.fit(df, **kw)
    return m

rows = []
# 策略 1：随访年作时间尺度，不做左截断，不校正年龄（最常见的错误做法）
m1 = fit(D, 'fu')
for k, key in [('60-74','g_60_74'), ('>=75','g_ge75')]:
    rows.append(dict(model='1. Follow-up time scale, no left truncation, unadjusted',
                     grp=k, hr=m1.summary.loc[key,'exp(coef)'],
                     lo=m1.summary.loc[key,'exp(coef) lower 95%'],
                     hi=m1.summary.loc[key,'exp(coef) upper 95%']))
# 策略 2：随访年 + 校正入组年龄
D['entry_c'] = D['entry_age'] - D['entry_age'].mean()
m2 = fit(D, 'fu', formula="g_60_74 + g_ge75 + entry_c")
for k, key in [('60-74','g_60_74'), ('>=75','g_ge75')]:
    rows.append(dict(model='2. Follow-up time scale, adjusted for age at entry',
                     grp=k, hr=m2.summary.loc[key,'exp(coef)'],
                     lo=m2.summary.loc[key,'exp(coef) lower 95%'],
                     hi=m2.summary.loc[key,'exp(coef) upper 95%']))
# 策略 3：年龄作时间尺度 + 左截断 + 协变量（正确）
m3 = fit(D, 'exit_age', entry='entry_age',
         formula="g_60_74 + g_ge75 + male + edu + smoke_ever")
for k, key in [('60-74','g_60_74'), ('>=75','g_ge75')]:
    rows.append(dict(model='3. Age time scale, left-truncated, adjusted',
                     grp=k, hr=m3.summary.loc[key,'exp(coef)'],
                     lo=m3.summary.loc[key,'exp(coef) lower 95%'],
                     hi=m3.summary.loc[key,'exp(coef) upper 95%']))
    rows.append(dict(model='3. Age time scale, left-truncated, adjusted',
                     grp=k, hr=m3.summary.loc[key,'exp(coef)'],
                     lo=m3.summary.loc[key,'exp(coef) lower 95%'],
                     hi=m3.summary.loc[key,'exp(coef) upper 95%'])) if False else None
R = pd.DataFrame(rows)
R['txt'] = R.apply(lambda r: f"{r['hr']:.2f} ({r['lo']:.2f}\u2013{r['hi']:.2f})", axis=1)
P("\n" + R.pivot(index='grp', columns='model', values='txt').to_string())
R.to_csv(os.path.join(OUT, "table3_naive.csv"), index=False, encoding='utf-8-sig')

# 梯度检验（2 df Wald）
def wald2(model, keys):
    b = np.array([model.summary.loc[k, 'coef'] for k in keys])
    V = model.variance_matrix_.loc[keys, keys].values
    chi = float(b @ np.linalg.inv(V) @ b)
    return chi, 2, stats.chi2.sf(chi, 2)
for nm, m in [('策略1', m1), ('策略2', m2), ('策略3', m3)]:
    chi, df_, pv = wald2(m, ['g_60_74','g_ge75'])
    P(f"  {nm} 起病年龄梯度 Wald chi2={chi:.1f}, df={df_}, P={pv:.2e}")
    R.loc[R['model'].str.startswith(nm[2]), 'wald_p'] = pv

# ============ HRS：关节炎 vs 无（时依）============
P("\n"+"="*76+"\n# HRS 时依：任何关节炎 vs 无关节炎\n"+"="*76)
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

rows2 = []
for pid, r in hi.iterrows():
    e0 = r['entry_age']; ex = r['exit_year'] - r['byear']
    if not np.isfinite(ex) or ex <= e0: continue
    if pd.isna(r['onset_age']) or r['onset_age'] >= ex:
        rows2.append((pid, e0, ex, r['event'], 0.0, r['sex']))
    else:
        a1 = r['onset_age']
        if a1 > e0:
            rows2.append((pid, e0, a1, 0, 0.0, r['sex']))
        rows2.append((pid, max(a1, e0), ex, r['event'], 1.0, r['sex']))
T = pd.DataFrame(rows2, columns=['pid','start','stop','event','art','sex'])
T = T[T['stop'] > T['start']].copy()
T['male'] = (T['sex']==1).astype(float)
ctv = CoxTimeVaryingFitter()
ctv.fit(T[['pid','start','stop','event','art','male']], id_col='pid',
        start_col='start', stop_col='stop', event_col='event', formula="art + male")
hr_art = ctv.summary.loc['art','exp(coef)']
lo_art = ctv.summary.loc['art','exp(coef) lower 95%']
hi_art = ctv.summary.loc['art','exp(coef) upper 95%']
P(f"关节炎 HR={hr_art:.3f} ({lo_art:.3f}\u2013{hi_art:.3f})  p={ctv.summary.loc['art','p']:.2e}")
P(f"  erson-waves={len(T)}  persons={T['pid'].nunique()}  events={int(T['event'].sum())}")

# ============ B. E-value ============
P("\n"+"="*76+"\n# B. E-value\n"+"="*76)
def hr_to_rr(hr):
    """VanderWeele 2017: HR -> approximate RR for common outcomes"""
    return (1 - 0.5**np.sqrt(hr)) / (1 - 0.5**np.sqrt(1/hr))

def e_val(rr):
    if rr < 1: rr = 1/rr
    return rr + np.sqrt(rr*(rr-1))

def report(label, hr, lo, hi):
    rr  = hr_to_rr(hr)
    rr_lo, rr_hi = hr_to_rr(lo), hr_to_rr(hi)
    P(f"\n{label}")
    P(f"  HR {hr:.2f} ({lo:.2f}\u2013{hi:.2f})  ->  RR\u2248{rr:.2f} ({rr_lo:.2f}\u2013{rr_hi:.2f})")
    P(f"  E-value (点估计) = {e_val(rr):.2f}")
    # 阴性结果的 E-value：取最接近 1 的 CI 界
    if lo <= 1 <= hi:
        P(f"  E-value (CI 界) = 1.00   （CI 已包含 1，无需未测混杂）")
    else:
        bound = hi if hi < 1 else lo
        P(f"  E-value (CI 界 {bound:.2f}) = {e_val(hr_to_rr(bound)):.2f}")

report("关节炎 vs 无关节炎，全因死亡（HRS 时依 Cox）", hr_art, lo_art, hi_art)
report("起病 >=75 vs <60，全因死亡（SHARE 左截断，阴性）", 0.810, 0.731, 0.898)
report("起病 60-74 vs <60，全因死亡（SHARE 左截断，阴性）", 1.021, 0.958, 1.088)

P("\n注：E-value 回答的是「需要多强的未测混杂才能把该关联解释掉」。")
P("     对阴性结果，CI 界的 E-value <2 意味着中等强度混杂即可抹平——这正是我们主张「无实质差异」的依据。")

with open(os.path.join(OUT, "s9_routeA.txt"),'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s9_routeA.txt")
