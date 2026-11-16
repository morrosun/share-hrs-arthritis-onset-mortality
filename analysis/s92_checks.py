# -*- coding: utf-8 -*-
"""Step 92: 两个补充计算。

(1) Table S6 面板 C 的精确系数（正文引用的是对数尺度系数，CSV 只存四舍五入的 HR）。
(2) 未记录死亡的 tipping point：在 ≥75 岁层中，把已删失者按退出年龄从大到小逐例
    视为「在末次访谈时死亡」，求使 ≥75 vs <60 的 HR 达到 1.00 所需的最少例数与比例。
    同时给出「全部删失者均死亡」的极端上界。

输出 out/s92_checks.txt
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


d = pd.read_csv(os.path.join(OUT, "analysis_share.csv"), encoding='utf-8-sig')
d['male'] = (d['sex'] == 1).astype(float)
d['smoke_ever'] = (d['smoke'] >= 1).astype(float)
base_cov = ['male', 'edu', 'smoke_ever']


def fit(df, formula):
    m = CoxPHFitter()
    m.fit(df, duration_col='exit_age', event_col='event', entry_col='entry_age', formula=formula)
    return m


def hr75(df):
    m = fit(df, 'g_60_74 + g_ge75 + male + edu + smoke_ever')
    q = m.summary.loc['g_ge75']
    return q['exp(coef)'], q['exp(coef) lower 95%'], q['exp(coef) upper 95%']


# ---------------- (1) S6 panel C ----------------
P("=" * 78)
P("# Table S6 panel C: affine proxy, exact log-scale coefficients")
P("=" * 78)
d['onset10'] = d['onset_age'] / 10.0
d['proxy'] = (75 - d['onset_age']) / 10.0
m1 = fit(d, 'onset10 + male + edu + smoke_ever')
m2 = fit(d, 'proxy + male + edu + smoke_ever')
b1 = float(m1.params_['onset10']); b2 = float(m2.params_['proxy'])
P(f"  onset per +10 y : beta={b1:+.6f}  HR={np.exp(b1):.4f} "
  f"({np.exp(m1.confidence_intervals_.loc['onset10'].iloc[0]):.4f}-"
  f"{np.exp(m1.confidence_intervals_.loc['onset10'].iloc[1]):.4f})")
P(f"  affine proxy    : beta={b2:+.6f}  HR={np.exp(b2):.4f} "
  f"({np.exp(m2.confidence_intervals_.loc['proxy'].iloc[0]):.4f}-"
  f"{np.exp(m2.confidence_intervals_.loc['proxy'].iloc[1]):.4f})")
P(f"  sum of betas    : {b1 + b2:+.9f}   product of HRs: {np.exp(b1 + b2):.8f}")
P(f"  log-likelihood  : {m1.log_likelihood_:.6f} vs {m2.log_likelihood_:.6f}  "
  f"(difference {abs(m1.log_likelihood_ - m2.log_likelihood_):.2e})")

# ---------------- (2) tipping point ----------------
P("")
P("=" * 78)
P("# Tipping point for unrecorded deaths in the >=75 stratum")
P("=" * 78)
d['g_60_74'] = (d['grp'] == '60-74').astype(float)
d['g_ge75'] = (d['grp'] == '>=75').astype(float)
h0, lo0, hi0 = hr75(d)
P(f"  main model: >=75 vs <60 HR {h0:.3f} ({lo0:.3f}-{hi0:.3f})")

cen = d[(d['grp'] == '>=75') & (d['event'] == 0)].sort_values('exit_age', ascending=False)
P(f"  censored in >=75 stratum: {len(cen):,} of {int((d['grp']=='>=75').sum()):,}")

# all censored die at their exit age (extreme bound)
dall = d.copy(); dall.loc[cen.index, 'event'] = 1
ha, loa, hia = hr75(dall)
P(f"  extreme: every censored respondent in the >=75 stratum dies at the last interview "
  f"-> HR {ha:.3f} ({loa:.3f}-{hia:.3f})")

# binary search on the number converted, oldest exit first
lo_m, hi_m = 0, len(cen)
while lo_m < hi_m:
    mid = (lo_m + hi_m) // 2
    dd = d.copy()
    dd.loc[cen.index[:mid], 'event'] = 1
    h, _, _ = hr75(dd)
    if h < 1.0:
        lo_m = mid + 1
    else:
        hi_m = mid

dd = d.copy(); dd.loc[cen.index[:lo_m], 'event'] = 1
h, lo, hi = hr75(dd)
P(f"  tipping point: converting the {lo_m:,} oldest censored records "
  f"({100*lo_m/len(cen):.1f}% of the censored in the >=75 stratum) gives HR {h:.3f} ({lo:.3f}-{hi:.3f})")
if lo_m > 0:
    dd2 = d.copy(); dd2.loc[cen.index[:lo_m - 1], 'event'] = 1
    h2, _, _ = hr75(dd2)
    P(f"  one fewer ({lo_m-1:,}) gives {h2:.3f}")
for frac in (0.25, 0.5, 0.75):
    m = int(len(cen) * frac)
    dd3 = d.copy(); dd3.loc[cen.index[:m], 'event'] = 1
    h3, _, _ = hr75(dd3)
    P(f"  monotonicity check: {m:,} converted ({100*frac:.0f}%) -> HR {h3:.3f}")
P("")
P("  Interpretation: extra deaths are assumed to occur at the last interview, so no extra person-time is")
P("  added and the conversion is the most favourable possible to the null. This is a lower bound on the")
P("  number of unrecorded deaths required.")


# ---------------- (3) 重写 Table S6 面板 C，使其与主模型同一套协变量编码 ----------------
src = pd.read_csv(os.path.join(OUT, "tableS6_duration_v5.csv"), encoding='utf-8-sig')
src = src[src['panel'] != 'C']
ci1l, ci1h = m1.confidence_intervals_.loc['onset10'].tolist()
ci2l, ci2h = m2.confidence_intervals_.loc['proxy'].tolist()
newC = pd.DataFrame([
    dict(panel='C', cohort='SHARE', term='Age at onset, per +10 y', n=len(d),
         est=round(float(np.exp(b1)), 4), lo=round(float(np.exp(ci1l)), 4),
         hi=round(float(np.exp(ci1h)), 4), p=round(float(m1.summary.loc['onset10', 'p']), 6),
         loglik=round(float(m1.log_likelihood_), 6)),
    dict(panel='C', cohort='SHARE', term='Duration at attained age 75, per +10 y', n=len(d),
         est=round(float(np.exp(b2)), 4), lo=round(float(np.exp(ci2l)), 4),
         hi=round(float(np.exp(ci2h)), 4), p=round(float(m2.summary.loc['proxy', 'p']), 6),
         loglik=round(float(m2.log_likelihood_), 6)),
    dict(panel='C', cohort='SHARE', term='HR product (duration x onset)', n=None,
         est=round(float(np.exp(b1 + b2)), 6), loglik=round(float(m1.log_likelihood_ - m2.log_likelihood_), 8)),
])
out6 = pd.concat([src, newC], ignore_index=True)
out6.to_csv(os.path.join(OUT, "tableS6_duration_v18.csv"), index=False, encoding='utf-8-sig')
P(f"\nrewrote panel C on the analysis cohort (n={len(d):,}) -> tableS6_duration_v18.csv")

io.open(os.path.join(OUT, "s92_checks.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE s92_checks.txt")
