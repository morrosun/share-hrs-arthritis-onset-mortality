# -*- coding: utf-8 -*-
"""Step 15: 比例风险假定检验（左截断下用 起病组 x 随访期 交互项）
SHARE：随访 5 年分界；HRS：同样以首次访视后 5 年为界。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, pyreadr
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020}
lines=[]
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

# ---------------- SHARE ----------------
d = pd.read_csv(os.path.join(OUT, "analysis_share.csv"))
d['grp'] = d['grp'].astype(str).str.strip()
d['male'] = (d['sex'] == 1).astype(float)
d['smoke_ever'] = (d['smoke'] >= 1).astype(float)
d['g74'] = (d['grp'] == '60-74').astype(float)
d['g75'] = (d['grp'] == '>=75').astype(float)
d = d.dropna(subset=['exit_age','entry_age','event','male','edu','smoke_ever'])

rows = []
for r in d.itertuples():
    a, b, ev = r.entry_age, r.exit_age, r.event
    cut = a + 5
    if b <= cut:
        rows.append((r.pid, a, b, ev, 0, r.g74, r.g75, r.male, r.edu, r.smoke_ever))
    else:
        rows.append((r.pid, a, cut, 0, 0, r.g74, r.g75, r.male, r.edu, r.smoke_ever))
        rows.append((r.pid, cut, b, ev, 1, r.g74, r.g75, r.male, r.edu, r.smoke_ever))
T = pd.DataFrame(rows, columns=['pid','start','stop','event','late','g74','g75','male','edu','smoke'])
T['g74_late'] = T['g74'] * T['late']
T['g75_late'] = T['g75'] * T['late']
P(f"SHARE 分段数据: {len(T)} 段, {T['pid'].nunique()} 人, 事件={int(T.groupby('pid')['event'].max().sum())}")

P("\n" + "="*72 + "\n# SHARE：起病组 x 随访期（<5 年 / >=5 年）\n" + "="*72)
m = CoxTimeVaryingFitter()
m.fit(T, id_col='pid', start_col='start', stop_col='stop', event_col='event',
      formula='g74 + g75 + g74_late + g75_late + male + edu + smoke')
P(m.summary[['exp(coef)','se(coef)','p']].round(3).to_string())
b = m.params_
P(f"\n前 5 年 HR:  60-74 {np.exp(b['g74']):.2f}   >=75 {np.exp(b['g75']):.2f}")
P(f"5 年后  HR:  60-74 {np.exp(b['g74']+b['g74_late']):.2f}   >=75 {np.exp(b['g75']+b['g75_late']):.2f}")
P(f"交互 P: 60-74 {m.summary.loc['g74_late','p']:.3f}   >=75 {m.summary.loc['g75_late','p']:.3f}")

# ---------------- HRS ----------------
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

rws = []
for pid, r in base.iterrows():
    e0, ex = r['entry_age'], r['exit_year'] - r['byear']
    if not np.isfinite(ex) or ex <= e0: continue
    def add(s0, s1, ev, grp):
        rws.append((pid, s0, s1, ev, grp, r['sex'], r['edu'], r['smoke'], e0))
    if pd.isna(r['onset_age']) or r['onset_age'] >= ex:
        add(e0, ex, r['event'], 'none')
    else:
        a1 = r['onset_age']
        if a1 > e0: add(e0, a1, 0, 'none')
        add(max(a1, e0), ex, r['event'],
            '<60' if a1 < 60 else ('60-74' if a1 < 75 else '>=75'))
H = pd.DataFrame(rws, columns=['pid','start','stop','event','grp','sex','edu','smoke','entry'])
H = H[H['stop'] > H['start']].reset_index(drop=True)
# 以首次访视后 5 年为界拆分时期
H['late'] = (H['start'] >= H['entry'] + 5 - 1e-9).astype(float)
H['art'] = (H['grp'] != 'none').astype(float)
H['g74'] = (H['grp'] == '60-74').astype(float)
H['g75'] = (H['grp'] == '>=75').astype(float)
H['male'] = (H['sex'] == 1).astype(float)
H['smoke_ever'] = (H['smoke'] >= 1).astype(float)
H['edu'] = pd.to_numeric(H['edu'], errors='coerce')
H = H.dropna(subset=['edu','male','smoke_ever'])
H['g74_late'] = H['g74'] * H['late']
H['g75_late'] = H['g75'] * H['late']
P(f"\nHRS 分段数据: {len(H)} 段, {H['pid'].nunique()} 人, 事件={int(H.groupby('pid')['event'].max().sum())}")

P("\n" + "="*72 + "\n# HRS：起病组 x 时期（首次访视后 <5 年 / >=5 年）\n" + "="*72)
for lab, form in [("any arthritis", 'art + art_late + male + edu + smoke_ever'),
                  ("onset strata", 'g74 + g75 + g74_late + g75_late + male + edu + smoke_ever')]:
    HH = H.copy(); HH['art_late'] = HH['art'] * HH['late']
    mm = CoxTimeVaryingFitter()
    mm.fit(HH, id_col='pid', start_col='start', stop_col='stop', event_col='event', formula=form)
    P(f"\n--- {lab} ---")
    P(mm.summary[['exp(coef)','se(coef)','p']].round(3).to_string())

io.open(os.path.join(OUT, "s15_ph.txt"), 'w', encoding='utf-8').write("\n".join(lines))
