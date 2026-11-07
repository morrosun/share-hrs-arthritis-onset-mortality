# -*- coding: utf-8 -*-
"""Step 85: HRS mortality cohort under the proposed final rules.

Changes relative to table4_hrs_mortality_v5.csv:
  (a) same-year deaths in never/prevalent are retained with 0.5 years instead of
      being dropped (removes the asymmetry with the incident strata);
  (b) respondents younger than 40 at entry (spouses/partners) are excluded.

Both are reported separately and together, so the cost of each decision is visible.
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

h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

persons = []
for pid, a in h.groupby('pid'):
    fp = a[a['arthre'] == 1]
    base = dict(pid=pid, entry=a['age'].iloc[0], last=a['age'].iloc[-1], dead=a['radyear'].max(),
                byear=a['rabyear'].iloc[0], sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0],
                smoke=a['smoken'].iloc[0])
    if fp.empty:
        persons.append(dict(base, pattern='never', onset=np.nan, pos=np.nan)); continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2 if pat == 'incident' else np.nan
    persons.append(dict(base, pattern=pat, onset=mid, pos=fp['age'].iloc[0]))
P0 = pd.DataFrame(persons)
P0['event'] = P0['dead'].notna().astype(int)
P0['exit'] = np.where(P0['event'] == 1, P0['dead'] - P0['byear'], P0['last'])
P0 = P0[P0['exit'].notna()].copy()

def grp_of(x): return '<60' if x < 60 else ('60-74' if x < 75 else '>=75')


def build(retain_same_year, drop_under40):
    rows = []
    for r in P0.itertuples():
        e0, ex = r.entry, r.exit
        if not np.isfinite(ex):
            continue
        if drop_under40 and e0 < 40:
            continue
        zero = ex <= e0
        if zero:
            if not (retain_same_year and r.event == 1):
                continue
            ex = e0 + 0.5
        base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
        if r.pattern == 'never':
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state='never'))
        elif r.pattern == 'prevalent':
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state='prevalent'))
        else:
            ex_inc = max(ex, r.pos + 0.5) if r.event == 1 else ex
            if r.pos > e0:
                rows.append(dict(base, start=e0, stop=min(r.pos, ex), event=0, state='never'))
            if ex_inc > max(r.pos, e0):
                rows.append(dict(base, start=max(r.pos, e0), stop=ex_inc, event=r.event,
                                 state='inc_' + grp_of(r.onset)))
    T = pd.DataFrame(rows)
    T = T[T['stop'] > T['start']].reset_index(drop=True)
    T['male'] = (T['sex'] == 1).astype(float)
    T['smoke_ever'] = (T['smoke'] >= 1).astype(float)
    T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
    T = T.dropna(subset=['edu', 'male', 'smoke_ever']).reset_index(drop=True)
    for st, col in [('prevalent', 'd_prev'), ('inc_<60', 'd_lt60'),
                    ('inc_60-74', 'd_6074'), ('inc_>=75', 'd_75')]:
        T[col] = (T['state'] == st).astype(float)
    T['d_ever'] = (T['state'] != 'never').astype(float)
    return T


def fit(T, label):
    m = CoxTimeVaryingFitter()
    m.fit(T, id_col='pid', event_col='event', start_col='start', stop_col='stop',
          formula='d_ever + male + edu + smoke_ever')
    s = m.summary.loc['d_ever']
    persons = T[['pid']].drop_duplicates().shape[0]
    P(f"  {label:44s} n={persons:,}  deaths={int(T['event'].sum()):,}  "
      f"any arthritis {s['exp(coef)']:.3f} ({s['exp(coef) lower 95%']:.3f}-{s['exp(coef) upper 95%']:.3f})")
    m2 = CoxTimeVaryingFitter()
    m2.fit(T, id_col='pid', event_col='event', start_col='start', stop_col='stop',
           formula='d_prev + d_lt60 + d_6074 + d_75 + male + edu + smoke_ever')
    out = {}
    for k in ['d_prev', 'd_lt60', 'd_6074', 'd_75']:
        q = m2.summary.loc[k]
        out[k] = (q['exp(coef)'], q['exp(coef) lower 95%'], q['exp(coef) upper 95%'])
    P("      five-state: " + " / ".join(f"{out[k][0]:.2f}" for k in ['d_prev', 'd_lt60', 'd_6074', 'd_75']))
    return persons, int(T['event'].sum()), s['exp(coef)'], out


P("=" * 78)
P("# HRS mortality cohort under four rule combinations")
P("=" * 78)
res = {}
for lab, ret, drop in [
    ("current (drop same-year, keep <40)", False, False),
    ("retain same-year only", True, False),
    ("drop <40 only", False, True),
    ("retain same-year + drop <40", True, True),
]:
    P("\n" + lab)
    res[lab] = fit(build(ret, drop), lab)

io.open(f"{OUT}\\s85_hrs_final.txt", 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE", f"{OUT}\\s85_hrs_final.txt")
