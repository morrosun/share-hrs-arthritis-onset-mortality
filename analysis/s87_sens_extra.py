# -*- coding: utf-8 -*-
"""Step 87: two sensitivity rows for Table S1 panel A that s10 does not emit.

  (a) delayed entry at the later of interview and reported diagnosis;
  (b) complete-case analysis for smoking.
Appends the rows to out/table4_sensitivity.csv so the table keeps the same shape.
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter

OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

p = pd.read_csv(os.path.join(OUT, "analysis_share.csv"), encoding='utf-8-sig')
p['grp'] = p['grp'].astype(str).str.strip()
p['event'] = p['event'].astype(int)
p['male'] = (p['sex'] == 1).astype(float)
p['smoke_ever'] = (p['smoke'] >= 1).astype(float)
p['g_60_74'] = (p['grp'] == '60-74').astype(float)
p['g_ge75'] = (p['grp'] == '>=75').astype(float)
p = p.dropna(subset=['exit_age', 'event', 'entry_age', 'male', 'edu', 'smoke_ever'])
P(f"base n={len(p):,} deaths={int(p['event'].sum()):,}")


def fit(d, entry_col='entry_age'):
    m = CoxPHFitter()
    m.fit(d, duration_col='exit_age', event_col='event', entry_col=entry_col,
          formula='g_60_74 + g_ge75 + male + edu + smoke_ever')
    a, b = m.summary.loc['g_60_74'], m.summary.loc['g_ge75']
    return (f"{a['exp(coef)']:.2f} ({a['exp(coef) lower 95%']:.2f}\u2013{a['exp(coef) upper 95%']:.2f})",
            f"{b['exp(coef)']:.2f} ({b['exp(coef) lower 95%']:.2f}\u2013{b['exp(coef) upper 95%']:.2f})")


# (a) delayed entry at the later of interview and diagnosis
q = p.copy()
q['entry_lt'] = np.maximum(q['entry_age'], q['onset_age'])
P(f"(a) onset after entry: {(q['onset_age'] > q['entry_age']).sum():,}")
a1, a2 = fit(q, entry_col='entry_lt')
P(f"    delayed entry at later of interview and diagnosis: {a1} / {a2}  n={len(q):,}")

# (b) complete case for smoking (smoke missing)
n_missing = int(pd.read_csv(os.path.join(OUT, "analysis_share.csv"), encoding='utf-8-sig')['smoke'].isna().sum())
cc = p[p['smoke'].notna()].copy()
b1, b2 = fit(cc)
P(f"(b) complete case for smoking: {b1} / {b2}  n={len(cc):,} deaths={int(cc['event'].sum()):,} (dropped {n_missing})")

df = pd.read_csv(os.path.join(OUT, "table4_sensitivity.csv"), encoding='utf-8-sig')
df = df[~df['scenario'].str.startswith(('Delayed entry', 'Complete case'))]
rows = pd.DataFrame([
    dict(scenario='Delayed entry at later of interview and diagnosis',
         hr_6074=a1, hr_75=a2, n=len(q), events=int(q['event'].sum()), note=''),
    dict(scenario=f'Complete case for smoking ({n_missing} excluded)',
         hr_6074=b1, hr_75=b2, n=len(cc), events=int(cc['event'].sum()), note=''),
])
df = pd.concat([df, rows], ignore_index=True)
df.to_csv(os.path.join(OUT, "table4_sensitivity.csv"), index=False, encoding='utf-8-sig')
P("\nWROTE table4_sensitivity.csv")
P(df.to_string())
io.open(os.path.join(OUT, "s87_sens_extra.txt"), 'w', encoding='utf-8').write("\n".join(lines))
