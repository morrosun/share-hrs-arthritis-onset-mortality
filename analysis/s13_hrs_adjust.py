# -*- coding: utf-8 -*-
"""Step 13: HRS 时依 Cox 增加教育/吸烟校正（原文只校正性别），并复核起病年龄组内对比。"""
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

h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]:'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age','arthre']).sort_values(['pid','wave'])
print('smoke-like cols:', [c for c in h.columns if 'smok' in c.lower()])
print('edu cols:', [c for c in h.columns if 'edu' in c.lower()])

# 人级基线协变量
base = h.groupby('pid').agg(byear=('rabyear','first'), sex=('ragender','first'),
                            entry_age=('age','min'), last_yr=('yr','max'),
                            edu=('raeducl','first'), smoke=('smoken','first') if 'smoken' in h else ('raeducl','first'))
pos = h[h['arthre']==1].groupby('pid').agg(onset_age=('age','min'), onset_yr=('yr','min'))
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
        rows.append((pid, e0, ex, r['event'], 0.0,0.0,0.0, r['sex'], r['edu'], r['smoke'], 0))
    else:
        a1 = r['onset_age']
        if a1 > e0:
            rows.append((pid, e0, a1, 0, 0.0,0.0,0.0, r['sex'], r['edu'], r['smoke'], 0))
        g = '<60' if a1 < 60 else ('60-74' if a1 < 75 else '>=75')
        rows.append((pid, max(a1,e0), ex, r['event'],
                     1.0 if g=='<60' else 0.0, 1.0 if g=='60-74' else 0.0,
                     1.0 if g=='>=75' else 0.0, r['sex'], r['edu'], r['smoke'], 1))
T = pd.DataFrame(rows, columns=['pid','start','stop','event','g_lt60','g_6074','g_75','sex','edu','smoke','art'])
T = T[T['stop'] > T['start']]
T['male'] = (T['sex']==1).astype(float)
T['smoke_ever'] = (T['smoke']>=1).astype(float)
T['edu'] = pd.to_numeric(T['edu'], errors='coerce')

P("="*72 + "\n# HRS 时依 Cox：仅性别 vs +教育+吸烟\n" + "="*72)
for lab, covs in [("sex only", "g_lt60 + g_6074 + g_75 + male"),
                  ("sex + edu + smoking", "g_lt60 + g_6074 + g_75 + male + edu + smoke_ever")]:
    D = T[['pid','start','stop','event','g_lt60','g_6074','g_75','male','edu','smoke_ever']].dropna()
    m = CoxTimeVaryingFitter()
    m.fit(D, id_col='pid', start_col='start', stop_col='stop', event_col='event', formula=covs)
    P(f"\n--- {lab}  n_seg={len(D)} persons={D['pid'].nunique()} events={int(m.event_observed.sum())} ---")
    s = m.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']]
    P(s.round(3).to_string())
    s.to_csv(os.path.join(OUT, f"hrs_cox_{lab.split()[0]}.csv"), encoding='utf-8-sig')

io.open(os.path.join(OUT, "s13_hrs_adjust.txt"), 'w', encoding='utf-8').write("\n".join(lines))
