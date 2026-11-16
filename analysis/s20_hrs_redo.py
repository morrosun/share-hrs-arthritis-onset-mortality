# -*- coding: utf-8 -*-
"""Step 20: HRS 死亡表重做（回应评审第 2 条）
暴露分 5 态：
  never（参照）        —— 至首次阳性报告前的全部人-年
  prevalent at entry  —— 基线访谈时已患病，起病年龄未知（左删失），不再强行分配起病年龄
  incident <60 / 60-74 / >=75 —— 基线阴性、随访新报告者，起病年龄取区间中点
时依 Cox，年龄时间尺度，延迟进入=首次访谈年龄，校正性别/教育/吸烟。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def hr_to_rr(hr): return (1 - 0.5**np.sqrt(hr)) / (1 - 0.5**np.sqrt(1/hr))
def e_val(rr):
    if rr < 1: rr = 1/rr
    return rr + np.sqrt(rr*(rr-1))
def e_for_bound(lo, hi):
    if lo <= 1 <= hi: return 1.0
    return e_val(hr_to_rr(hi if hi < 1 else lo))

h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

persons = []
for pid, a in h.groupby('pid'):
    fp = a[a['arthre'] == 1]
    if fp.empty:
        persons.append(dict(pid=pid, pattern='never', onset=np.nan, first_pos=np.nan,
                            entry=a['age'].iloc[0], last=a['age'].iloc[-1],
                            dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                            sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
        continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    persons.append(dict(pid=pid, pattern=pat,
                        onset=((neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2) if pat == 'incident' else np.nan,
                        first_pos=fp['age'].iloc[0],
                        entry=a['age'].iloc[0], last=a['age'].iloc[-1],
                        dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                        sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
P0 = pd.DataFrame(persons)
P0['event'] = P0['dead'].notna().astype(int)
P0['exit'] = np.where(P0['event'] == 1, P0['dead'] - P0['byear'], P0['last'])
P0 = P0[P0['exit'].notna()]

rows = []
for r in P0.itertuples():
    e0, ex = r.entry, r.exit
    if not np.isfinite(ex) or ex <= e0:
        continue
    base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
    if r.pattern == 'never':
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state='never'))
    elif r.pattern == 'prevalent':
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state='prevalent'))
    else:
        if r.onset > e0:
            rows.append(dict(base, start=e0, stop=min(r.onset, ex), event=0, state='never'))
            g = '<60' if r.onset < 60 else ('60-74' if r.onset < 75 else '>=75')
            rows.append(dict(base, start=max(r.onset, e0), stop=ex, event=r.event, state='inc_' + g))
        else:
            g = '<60' if r.onset < 60 else ('60-74' if r.onset < 75 else '>=75')
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state='inc_' + g))

T = pd.DataFrame(rows)
T = T[T['stop'] > T['start']].reset_index(drop=True)
T['male'] = (T['sex'] == 1).astype(float)
T['smoke_ever'] = (T['smoke'] >= 1).astype(float)
T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
T = T.dropna(subset=['edu', 'male', 'smoke_ever']).reset_index(drop=True)
for s_ in ['prevalent', 'inc_<60', 'inc_60-74', 'inc_>=75']:
    T['d_' + s_] = (T['state'] == s_).astype(float)
T['d_prev'] = (T['state'] == 'prevalent').astype(float)
T['d_lt60'] = (T['state'] == 'inc_<60').astype(float)
T['d_6074'] = (T['state'] == 'inc_60-74').astype(float)
T['d_75'] = (T['state'] == 'inc_>=75').astype(float)
T['d_ever'] = (T['state'] != 'never').astype(float)

P("="*78 + "\n# HRS 重做：暴露 5 态\n" + "="*78)
P(T.groupby('state').agg(seg=('pid','size'), persons=('pid','nunique'),
   py=('start', lambda x: 0), events=('event','sum')).to_string())
P("\n[人-年]")
for k, x in T.groupby('state'):
    P(f"  {k:12s} py={(x['stop']-x['start']).sum():9,.0f}  events={int(x['event'].sum()):5,}")

def fit(form, keys, lab):
    D = T[['pid','start','stop','event','male','edu','smoke_ever','d_prev','d_lt60','d_6074','d_75','d_ever']].copy()
    m = CoxTimeVaryingFitter()
    m.fit(D, id_col='pid', start_col='start', stop_col='stop', event_col='event', formula=form)
    P(f"\n--- {lab}: seg={len(D):,} persons={D['pid'].nunique():,} events={int(m.event_observed.sum()):,} ---")
    out = []
    for k in keys:
        q = m.summary.loc[k]
        est, lo, hi, pv = q['exp(coef)'], q['exp(coef) lower 95%'], q['exp(coef) upper 95%'], q['p']
        cod = {'d_prev':'Prevalent at entry (onset unknown)', 'd_lt60':'Onset <60 (incident)',
               'd_6074':'Onset 60-74 (incident)', 'd_75':'Onset >=75 (incident)',
               'd_ever':'Any arthritis'}[k]
        P(f"  {cod:36s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={pv:.2e}  E={e_val(hr_to_rr(est)):.2f}/{e_for_bound(lo,hi):.2f}")
        out.append(dict(contrast=cod, kind='HR', est=est, lo=lo, hi=hi, p=pv,
                        e_point=e_val(hr_to_rr(est)), e_bound=e_for_bound(lo, hi),
                        n=len(D), events=int(m.event_observed.sum())))
    return out

rows_out = []
rows_out += fit('d_ever + male + edu + smoke_ever', ['d_ever'], 'Any arthritis vs never（参照组=从未患病）')
rows_out += fit('d_prev + d_lt60 + d_6074 + d_75 + male + edu + smoke_ever',
                ['d_prev', 'd_lt60', 'd_6074', 'd_75'],
                'Prevalent + incident onset strata vs never')
pd.DataFrame(rows_out).to_csv(os.path.join(OUT, "table4_hrs_mortality_v2.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s20_hrs_redo.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 table4_hrs_mortality_v2.csv")
