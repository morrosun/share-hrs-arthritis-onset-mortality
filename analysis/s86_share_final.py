# -*- coding: utf-8 -*-
"""Step 86: SHARE prevalent cohort under the proposed final rules.

  (a) same-year deaths retained with 0.5 years (currently dropped);
  (b) the 2020 entry wave (wave 8) excluded (its entrants have no survivor
      person-time, so only decedents contribute).

Four combinations are fitted with the Table 3 model 4 specification (age as the
time scale, delayed entry, adjusted for sex, education and smoking).
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S)
s['agew'] = s['yr'] - s['rabyear']
s = s.sort_values(['pid', 'wave'])

first_w = s.groupby('pid')['wave'].min().rename('w0')
s = s.merge(first_w, left_on='pid', right_index=True)
ent = s[s['wave'] == s['w0']].set_index('pid')
person = pd.DataFrame({
    'pid': ent.index,
    'w0': ent['w0'], 'entry_yr': ent['yr'], 'entry_age': ent['agew'],
    'byear': ent['rabyear'], 'sex': ent['ragender'],
    'edu': ent['raeducl'], 'smoke': ent['smoken'],
    'country': ent['country'],
}).reset_index(drop=True)
person['last_yr'] = s.groupby('pid')['yr'].max().reindex(person['pid']).values
person['deadyr'] = s.groupby('pid')['radyear'].max().reindex(person['pid']).values
person['onset_age'] = (s.dropna(subset=['radiagarthr'])
                        .groupby('pid')['radiagarthr'].first()
                        .reindex(person['pid']).values)

p = person[person['onset_age'].notna()].copy()
p['onset_age'] = p['onset_age'].where(p['onset_age'] >= 5)
p = p[p['onset_age'].notna() & p['byear'].notna()]
p['event'] = p['deadyr'].notna().astype(int)
p['exit_age'] = np.where(p['event'] == 1, p['deadyr'] - p['byear'], p['last_yr'] - p['byear'])
p = p[p['exit_age'].notna() & (p['entry_age'] >= 40) & (p['exit_age'] <= 110)]
p['smoke_ever'] = (p['smoke'] >= 1).astype(float)
p['male'] = (p['sex'] == 1).astype(float)
p['edu'] = pd.to_numeric(p['edu'], errors='coerce')
p['grp'] = pd.cut(p['onset_age'], [4, 59, 74, 120], labels=['<60', '60-74', '>=75'])
p['g_60_74'] = (p['grp'] == '60-74').astype(float)
p['g_ge75'] = (p['grp'] == '>=75').astype(float)

P("=" * 78)
P("# SHARE prevalent cohort under four rule combinations")
P("=" * 78)
P(f"eligible base (onset known, entry>=40): {len(p):,}  deaths={int(p.loc[p['exit_age'] > p['entry_age'], 'event'].sum()):,}")


def build(retain_same_year, drop_wave8):
    d = p.copy()
    if drop_wave8:
        d = d[d['w0'] != 8]
    zero = d['exit_age'] <= d['entry_age']
    if retain_same_year:
        d.loc[zero & (d['event'] == 1), 'exit_age'] = d.loc[zero & (d['event'] == 1), 'entry_age'] + 0.5
        d = d[d['exit_age'] > d['entry_age']]
    else:
        d = d[~zero]
    d = d.dropna(subset=['exit_age', 'event', 'male', 'edu', 'smoke_ever'])
    d['py'] = d['exit_age'] - d['entry_age']
    return d


def fit(d, label):
    m = CoxPHFitter()
    m.fit(d, duration_col='exit_age', event_col='event', entry_col='entry_age',
          formula='g_60_74 + g_ge75 + male + edu + smoke_ever')
    b = np.array([m.summary.loc[k, 'coef'] for k in ['g_60_74', 'g_ge75']])
    V = m.variance_matrix_.loc[['g_60_74', 'g_ge75'], ['g_60_74', 'g_ge75']].values
    chi2 = float(b @ np.linalg.inv(V) @ b); pv = float(stats.chi2.sf(chi2, 2))
    a = m.summary.loc['g_60_74']; c = m.summary.loc['g_ge75']
    P(f"  {label:44s} n={len(d):,} deaths={int(d['event'].sum()):,} py={d['py'].sum():,.0f}  "
      f"60-74 {a['exp(coef)']:.2f} ({a['exp(coef) lower 95%']:.2f}-{a['exp(coef) upper 95%']:.2f})  "
      f">=75 {c['exp(coef)']:.2f} ({c['exp(coef) lower 95%']:.2f}-{c['exp(coef) upper 95%']:.2f})  "
      f"P={pv:.3g}")
    return dict(n=len(d), deaths=int(d['event'].sum()), py=float(d['py'].sum()),
                hr=a['exp(coef)'], lo=c['exp(coef) lower 95%'], hi=c['exp(coef) upper 95%'],
                hr75=c['exp(coef)'], lo75=c['exp(coef) lower 95%'], hi75=c['exp(coef) upper 95%'])


for lab, ret, w8 in [
    ("current (drop same-year, keep wave 8)", False, False),
    ("retain same-year only", True, False),
    ("drop wave 8 only", False, True),
    ("retain same-year + drop wave 8", True, True),
]:
    P("\n" + lab)
    fit(build(ret, w8), lab)

io.open(os.path.join(OUT, "s86_share_final.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE", os.path.join(OUT, "s86_share_final.txt"))
