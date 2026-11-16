# -*- coding: utf-8 -*-
"""Step 14: HRS 死亡表（表 4）改为完整校正版（性别 + 教育 + 吸烟）
产出 table4_hrs_mortality.csv 的替换版，含 E-value。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, pyreadr
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020}
lines=[]
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def hr_to_rr(hr): return (1 - 0.5**np.sqrt(hr)) / (1 - 0.5**np.sqrt(1/hr))
def e_val(rr):
    if rr < 1: rr = 1/rr
    return rr + np.sqrt(rr*(rr-1))
def e_for_bound(lo, hi):
    if lo <= 1 <= hi: return 1.0
    b = hi if hi < 1 else lo
    return e_val(hr_to_rr(b))

h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]:'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age','arthre']).sort_values(['pid','wave'])
base = h.groupby('pid').agg(byear=('rabyear','first'), sex=('ragender','first'),
                            entry_age=('age','min'), last_yr=('yr','max'),
                            edu=('raeducl','first'), smoke=('smoken','first'))
pos = h[h['arthre']==1].groupby('pid').agg(onset_age=('age','min'))
ddh = h.groupby('pid')['radyear'].max().rename('deadyr')
base = base.join(pos).join(ddh)
base['exit_year'] = np.where(base['deadyr'].notna(), base['deadyr'], base['last_yr'])
base['event'] = base['deadyr'].notna().astype(int)
base = base[base['exit_year'].notna()]

rows=[]
for pid, r in base.iterrows():
    e0, ex = r['entry_age'], r['exit_year'] - r['byear']
    if not np.isfinite(ex) or ex <= e0: continue
    if pd.isna(r['onset_age']) or r['onset_age'] >= ex:
        rows.append((pid, e0, ex, r['event'], 0,0,0, r['sex'], r['edu'], r['smoke'], 0))
    else:
        a1 = r['onset_age']
        if a1 > e0:
            rows.append((pid, e0, a1, 0, 0,0,0, r['sex'], r['edu'], r['smoke'], 0))
        g = '<60' if a1 < 60 else ('60-74' if a1 < 75 else '>=75')
        rows.append((pid, max(a1,e0), ex, r['event'],
                     1 if g=='<60' else 0, 1 if g=='60-74' else 0, 1 if g=='>=75' else 0,
                     r['sex'], r['edu'], r['smoke'], 1))
T = pd.DataFrame(rows, columns=['pid','start','stop','event','g_lt60','g_6074','g_75','sex','edu','smoke','art'])
T = T[T['stop'] > T['start']].reset_index(drop=True)
T['male'] = (T['sex']==1).astype(float)
T['smoke_ever'] = (T['smoke']>=1).astype(float)
T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
T['ever'] = T['art'].astype(float)
P(f"时依数据: {len(T)} 段, {T['pid'].nunique()} 人, 事件={int(T.groupby('pid')['event'].max().sum())}")

def fit(df, form):
    D = df[['pid','start','stop','event','male','edu','smoke_ever','ever',
            'g_lt60','g_6074','g_75']].dropna()
    m = CoxTimeVaryingFitter()
    m.fit(D, id_col='pid', start_col='start', stop_col='stop', event_col='event', formula=form)
    return m, len(D)

out=[]
specs = [
    ('Any arthritis vs none', 'ever',    'ever + male + edu + smoke_ever'),
    ('Onset <60 vs none',     'g_lt60',  'g_lt60 + male + edu + smoke_ever'),
    ('Onset 60-74 vs none',   'g_6074',  'g_6074 + male + edu + smoke_ever'),
    ('Onset >=75 vs none',    'g_75',    'g_75 + male + edu + smoke_ever'),
]
P("\n" + "="*72 + "\n# 表 4（完整校正：性别 + 教育 + 吸烟）\n" + "="*72)
for name, key, form in specs:
    m, n = fit(T, form)
    s = m.summary.loc[key]
    est, lo, hi, pv = s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']
    ev = int(m.event_observed.sum())
    out.append(dict(contrast=name, kind='HR', est=est, lo=lo, hi=hi, p=pv,
                    e_point=e_val(hr_to_rr(est)), e_bound=e_for_bound(lo, hi), n=n, events=ev))
    P(f"{name:22s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={pv:.2e}  E={e_val(hr_to_rr(est)):.2f}/{e_for_bound(lo,hi):.2f}  n={n} events={ev}")

Ta = T[T['art']==1].reset_index(drop=True)
P(f"\n患病者子集: {len(Ta)} 段, {Ta['pid'].nunique()} 人, 事件={int(Ta.groupby('pid')['event'].max().sum())}")
for name, key, form in [
    ('Onset 60-74 vs <60 (among arthritis)', 'g_6074', 'g_6074 + g_75 + male + edu + smoke_ever'),
    ('Onset >=75 vs <60 (among arthritis)',  'g_75',   'g_6074 + g_75 + male + edu + smoke_ever')]:
    m, n = fit(Ta, form)
    s = m.summary.loc[key]
    est, lo, hi, pv = s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']
    ev = int(m.event_observed.sum())
    out.append(dict(contrast=name, kind='HR', est=est, lo=lo, hi=hi, p=pv,
                    e_point=e_val(hr_to_rr(est)), e_bound=e_for_bound(lo, hi), n=n, events=ev))
    P(f"{name:38s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={pv:.2e}  E={e_val(hr_to_rr(est)):.2f}/{e_for_bound(lo,hi):.2f}  n={n} events={ev}")

df = pd.DataFrame(out)
df.to_csv(os.path.join(OUT, "table4_hrs_mortality.csv"), index=False, encoding='utf-8-sig')
P("\n写出 table4_hrs_mortality.csv（已覆盖为完整校正版）")
io.open(os.path.join(OUT, "s14_table4.txt"), 'w', encoding='utf-8').write("\n".join(lines))
