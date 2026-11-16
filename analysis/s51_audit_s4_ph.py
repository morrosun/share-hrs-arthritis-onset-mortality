# -*- coding: utf-8 -*-
"""Step 51: fourth-round review -- Table S4 risk-set audit and PH period estimates.

1) Rebuild the SHARE prevalent cohort exactly as s43/s48 do, and check per person
   whether the start of arthritic person-time is at or after the reported diagnosis age.
2) Quantify the records with onset > entry: their person-years, deaths and the
   single years they contribute (this is the suspected source of "55-104"/"70-104").
3) Recompute single-year overlap and the 20/50/100 py band HRs after excluding them.
4) Check the high-age tail (>=85 y).
5) Recover period-specific HRs with 95% CI for the SHARE prevalent model and the
   exposure x follow-up-period interaction P values.

Outputs: out/s51_audit_s4_ph.txt, out/tableS4_support_fine_v7.csv,
         out/tableS1_ph_period_v7.csv
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S)
s['age'] = s['yr'] - s['rabyear']

agg = s.groupby('pid').agg(entry=('age', 'min'), last=('age', 'max'),
                           sex=('ragender', 'first'), byear=('rabyear', 'first'),
                           edu=('raeducl', 'first'), smoke=('smoken', 'max'),
                           w0=('wave', 'min'))
dg = s.dropna(subset=['radiagarthr']).groupby('pid')['radiagarthr'].first().rename('onset')
dd = s.groupby('pid')['radyear'].max().rename('dead')
D = agg.join(dg).join(dd).dropna(subset=['onset'])
D = D[D['onset'] >= 5]
D['event'] = D['dead'].notna().astype(int)
D['exit'] = np.where(D['event'] == 1, D['dead'] - D['byear'], D['last'])
# v18 cohort rules, aligned with s43 and s33: a death in the calendar year of entry leaves no
# interval on the integer age scale and is retained with 0.5 years, and the 2020 entry wave is
# removed because its entrants have at most one interview. Without these two lines this script
# rebuilds the pre-v18 cohort (18,445) and Table S4 disagrees with Table 5 by 30 deaths
# (51 same-year + 12 with onset > entry - 33 from the 2020 wave; see s95_dump_s4_deaths.py).
_zero = D['exit'] <= D['entry']
D.loc[_zero & (D['event'] == 1), 'exit'] = D.loc[_zero & (D['event'] == 1), 'entry'] + 0.5
D = D[(D['exit'] > D['entry']) & (D['entry'] >= 40) & (D['exit'] <= 110) & (D['w0'] != 8)].copy()
D['male'] = (D['sex'] == 1).astype(float)
D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
D['grp'] = grp(D['onset'])
D = D.dropna(subset=['grp', 'edu']).reset_index()
D['py'] = D['exit'] - D['entry']

P("=" * 88)
P("# 1. SHARE prevalent cohort: per-person check of 'risk start >= diagnosis age'")
P("=" * 88)
P(f"n = {len(D):,}   deaths = {int(D['event'].sum()):,}   person-years = {D['py'].sum():,.0f}")
bad = D[D['onset'] > D['entry'] + 1e-9].copy()
P(f"Records with onset > entry (diagnosis reported at an age older than the entry age): {len(bad):,}")
P(f"  their person-years = {bad['py'].sum():,.1f}  deaths = {int(bad['event'].sum()):,}")
P(f"  onset - entry: median {np.median(bad['onset']-bad['entry']):.1f} y, "
  f"range {np.min(bad['onset']-bad['entry']):.1f} to {np.max(bad['onset']-bad['entry']):.1f} y")
P("\n  by onset stratum:")
for g, d in bad.groupby('grp', observed=True):
    P(f"    {str(g):6s}: n={len(d):3d}  py={d['py'].sum():8.1f}  deaths={int(d['event'].sum()):3d}  "
      f"entry {d['entry'].min():.0f}-{d['entry'].max():.0f}  onset {d['onset'].min():.0f}-{d['onset'].max():.0f}")
# person-time credited as 'arthritic' before the reported diagnosis
bad['pre_dx_py'] = np.maximum(0, np.minimum(bad['exit'], bad['onset']) - bad['entry'])
P(f"  person-years falling BETWEEN entry and reported diagnosis (i.e. credited as "
  f"arthritic before diagnosis): {bad['pre_dx_py'].sum():,.1f}")

# Do these 85 records reflect a real logical error, or a wave-year vs interview-year mismatch?
P("\n  Data-origin check: does the harmonised interview age (agey) / interview year (iwy) "
  "differ from the wave-year derived age used in the analysis?")
first_rows = (s.sort_values(['pid', 'wave']).groupby('pid').first())
chk = bad.set_index('pid').join(first_rows[['age', 'agey', 'iwy', 'iwm', 'wave', 'yr']],
                                rsuffix='_first')
P(f"    wave-year age (age) vs harmonised agey: mean difference "
  f"{(chk['age']-chk['agey']).mean():+.2f} y, "
  f"|diff| >= 1 y in {int((np.abs(chk['age']-chk['agey'])>=1).sum()):,} of {len(chk):,} records")
P(f"    iwy - mapped wave year: median {np.nanmedian(chk['iwy']-chk['yr']):+.1f} y, "
  f"range {np.nanmin(chk['iwy']-chk['yr']):+.0f} to {np.nanmax(chk['iwy']-chk['yr']):+.0f} y")
P(f"    using agey as entry, onset > agey remains in "
  f"{int((bad.set_index('pid')['onset'] > chk['agey']).sum()):,} of {len(bad):,} records")

# ---------------- 2. single-year risk sets ----------------
def single_year(d):
    out = {}
    for _, r in d.iterrows():
        e, x, ev = r['entry'], r['exit'], r['event']
        a = max(40, np.floor(e))
        while a < x and a < 110:
            s0, e0 = max(e, a), min(x, a + 1)
            if e0 > s0:
                v = out.setdefault(int(a), [0.0, 0])
                v[0] += e0 - s0
            a += 1
        if ev and np.isfinite(x):
            da = int(min(np.floor(x - 1e-9), 109))
            out.setdefault(da, [0.0, 0])[1] += 1
    return out


def overlap(D_):
    SY = {g: single_year(d) for g, d in D_.groupby('grp', observed=True)}
    res = {}
    for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
        ca, cb = SY.get(a, {}), SY.get(b, {})
        both = sorted(k for k in set(ca) | set(cb)
                      if ca.get(k, [0, 0])[0] > 0 and cb.get(k, [0, 0])[0] > 0)
        res[f"{a} vs {b}"] = (both, ca, cb)
    return res

P("\n" + "=" * 88)
P(f"# 2. Single-year common support: all {len(D):,} vs the "
  f"{int((D['onset'] <= D['entry'] + 1e-9).sum()):,} with onset <= entry")
P("=" * 88)
good = D[D['onset'] <= D['entry'] + 1e-9].copy()
P(f"restricted cohort n = {len(good):,}  deaths = {int(good['event'].sum()):,}")
allov = overlap(D)
goodov = overlap(good)
rows = []
for key in allov:
    both, ca, cb = allov[key]
    gboth, gca, gcb = goodov[key]
    lo, hi = (both[0], both[-1] + 1) if both else (None, None)
    glo, ghi = (gboth[0], gboth[-1] + 1) if gboth else (None, None)
    da = sum(ca[k][1] for k in both); db = sum(cb[k][1] for k in both)
    gda = sum(gca[k][1] for k in gboth); gdb = sum(gcb[k][1] for k in gboth)
    P(f"\n  {key}")
    P(f"    all {len(D):,} : {lo}-{hi} ({len(both)} single years), deaths {da:,} / {db:,}")
    P(f"    restricted : {glo}-{ghi} ({len(gboth)} single years), deaths {gda:,} / {gdb:,}")
    removed = sorted(set(both) - set(gboth))
    P(f"    single years present ONLY because of onset>entry records: "
      f"{removed[:12]}{' ...' if len(removed) > 12 else ''}  (n={len(removed)})")
    rows.append(dict(design='SHARE prevalent', contrast=key, audit='single-year',
                     cohort=f'all {len(D):,}', threshold=None,
                     window=None if lo is None else f"{lo}-{hi}",
                     deaths_a=da, deaths_b=db, hr=None, lo_ci=None, hi_ci=None))
    rows.append(dict(design='SHARE prevalent', contrast=key, audit='single-year',
                     cohort=f'restricted {len(good):,}', threshold=None,
                     window=None if glo is None else f"{glo}-{ghi}",
                     deaths_a=gda, deaths_b=gdb, hr=None, lo_ci=None, hi_ci=None))

# high-age tail
P(f"\n  High-age tail of the >=75 stratum (all {len(D):,}):")
SY75 = single_year(D[D['grp'] == '>=75'])
for age in (75, 80, 85, 90, 95, 100, 104):
    py, d = SY75.get(age, [0, 0])
    P(f"    age {age:3d}: {py:8.1f} py, {d:4d} deaths")
P(f"    maximum exit age in cohort = {D['exit'].max():.0f}; "
  f"n with exit > 90 = {int((D['exit']>90).sum()):,}; > 100 = {int((D['exit']>100).sum()):,}")

# ---------------- 3. 5-year band HRs ----------------
def band(d):
    out = {}
    for g_, dd in d.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(dd['entry'], dd['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0, e0 = max(e, a), min(x, a + 5)
                if e0 > s0:
                    bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g_] = bands
    return out


def window(B, a, b, thr):
    if a not in B or b not in B:
        return []
    segs = []; cur = None
    for k in sorted(set(B[a]) | set(B[b])):
        ok = B[a].get(k, 0) >= thr and B[b].get(k, 0) >= thr
        if ok and cur is None: cur = [k, k + 5]
        elif ok: cur[1] = k + 5
        elif cur is not None: segs.append(tuple(cur)); cur = None
    if cur is not None: segs.append(tuple(cur))
    return segs


def pair(d, a, b, lo, hi):
    X = d.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'] == a].empty or X[X['grp'] == b].empty:
        return None
    X['cmp'] = (X['grp'] == a).astype(float)
    try:
        m = CoxPHFitter()
        m.fit(X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']],
              duration_col='x', event_col='ev', entry_col='e')
    except Exception:
        return None
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=q['exp(coef)'],
                lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'])

P("\n" + "=" * 88)
P("# 3. Band-restricted HRs: do the 85 records affect any estimate?")
P("=" * 88)
for label, d in [(f'all {len(D):,}', D), (f'restricted {len(good):,}', good)]:
    B = band(d)
    P(f"\n  [{label}]")
    for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
        key = f"{a} vs {b}"
        for thr in (20, 50, 100):
            segs = window(B, a, b, thr)
            if not segs:
                P(f"    {key:18s} thr {thr:3d}: not estimable")
                rows.append(dict(design='SHARE prevalent', contrast=key, audit='5-year band',
                                 cohort=label, threshold=thr, window='none',
                                 deaths_a=None, deaths_b=None, hr=None, lo_ci=None, hi_ci=None))
                continue
            lo5, hi5 = segs[0][0], segs[-1][1]
            r = pair(d, a, b, lo5, hi5)
            if r is None:
                P(f"    {key:18s} thr {thr:3d}: window {lo5:.0f}-{hi5:.0f}, fit failed")
                continue
            P(f"    {key:18s} thr {thr:3d}: {lo5:.0f}-{hi5:.0f} y, n={r['n']:,}, "
              f"deaths={r['deaths']:,}, HR {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})")
            rows.append(dict(design='SHARE prevalent', contrast=key, audit='5-year band',
                             cohort=label, threshold=thr, window=f"{lo5:.0f}-{hi5:.0f}",
                             deaths_a=r['deaths'], deaths_b=None,
                             hr=round(r['hr'], 2), lo_ci=round(r['lo'], 2), hi_ci=round(r['hi'], 2)))

# main three-group model (Table 3 model 4) under both cohorts
P("\n  Main all-ages three-group model (Table 3, model 4):")
for label, d in [(f'all {len(D):,}', D), (f'restricted {len(good):,}', good)]:
    X = d.copy()
    X['g_60_74'] = (X['grp'] == '60-74').astype(float)
    X['g_ge75'] = (X['grp'] == '>=75').astype(float)
    m = CoxPHFitter()
    m.fit(X[['entry', 'exit', 'event', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']],
          duration_col='exit', event_col='event', entry_col='entry')
    a, b = m.summary.loc['g_60_74'], m.summary.loc['g_ge75']
    P(f"    {label:20s} n={len(X):,}  60-74: {a['exp(coef)']:.3f} "
      f"({a['exp(coef) lower 95%']:.3f}-{a['exp(coef) upper 95%']:.3f})  "
      f">=75: {b['exp(coef)']:.3f} ({b['exp(coef) lower 95%']:.3f}-{b['exp(coef) upper 95%']:.3f})")

# delayed entry at max(entry, onset) -- the Table S1 sensitivity, on the full cohort
X = D.copy()
X['entry2'] = np.maximum(X['entry'], X['onset'])
X = X[X['exit'] > X['entry2']].copy()
X['g_60_74'] = (X['grp'] == '60-74').astype(float)
X['g_ge75'] = (X['grp'] == '>=75').astype(float)
m = CoxPHFitter()
m.fit(X[['entry2', 'exit', 'event', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']],
      duration_col='exit', event_col='event', entry_col='entry2')
a, b = m.summary.loc['g_60_74'], m.summary.loc['g_ge75']
P(f"    entry = max(entry, onset), n={len(X):,}  60-74: {a['exp(coef)']:.3f} "
  f"({a['exp(coef) lower 95%']:.3f}-{a['exp(coef) upper 95%']:.3f})  "
  f">=75: {b['exp(coef)']:.3f} ({b['exp(coef) lower 95%']:.3f}-{b['exp(coef) upper 95%']:.3f})")

# ---------------- 4. PH: period-specific HRs with CI ----------------
P("\n" + "=" * 88)
P("# 4. Proportional hazards: period-specific HRs with 95% CI (SHARE prevalent)")
P("=" * 88)
c = pd.read_csv(os.path.join(OUT, "analysis_share.csv"), encoding='utf-8-sig')
c = c.rename(columns={'entry_age': 'entry', 'exit_age': 'exit'})
c['grp'] = c['grp'].astype(str).str.strip()
c['g_60_74'] = (c['grp'] == '60-74').astype(float)
c['g_ge75'] = (c['grp'] == '>=75').astype(float)
c['male'] = (c['sex'] == 1).astype(float)
c['smoke_ever'] = (c['smoke'] >= 1).astype(float)
c = c.dropna(subset=['exit', 'entry', 'male', 'edu', 'smoke_ever'])
c = c[c['exit'] > c['entry']].copy()
P(f"cohort n={len(c):,} deaths={int(c['event'].sum()):,}")
recs = []
for _, r in c.iterrows():
    e, x, ev = r['entry'], r['exit'], int(r['event'])
    cut = e + 5
    base = dict(g_60_74=r['g_60_74'], g_ge75=r['g_ge75'], male=r['male'],
                edu=r['edu'], smoke_ever=r['smoke_ever'])
    recs.append(dict(**base, e=e, x=min(x, cut), ev=ev if x <= cut else 0, period=0))
    if x > cut:
        recs.append(dict(**base, e=cut, x=x, ev=ev, period=1))
S = pd.DataFrame(recs)
P(f"split records: {len(S):,} (period 0: {len(S[S['period']==0]):,}, "
  f"period 1: {len(S[S['period']==1]):,})")
phrows = []
for per in (0, 1):
    q = S[S['period'] == per]
    m = CoxPHFitter()
    m.fit(q[['e', 'x', 'ev', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']],
          duration_col='x', event_col='ev', entry_col='e')
    for key, lab in [('g_60_74', '60-74 vs <60'), ('g_ge75', '>=75 vs <60')]:
        z = m.summary.loc[key]
        P(f"  {'<5 y' if per==0 else '>=5 y':8s} {lab:16s} HR {z['exp(coef)']:.2f} "
          f"({z['exp(coef) lower 95%']:.2f}-{z['exp(coef) upper 95%']:.2f})  p={z['p']:.4g}")
        phrows.append(dict(period='<5 y' if per == 0 else '>=5 y', contrast=lab,
                           hr=round(z['exp(coef)'], 2), lo=round(z['exp(coef) lower 95%'], 2),
                           hi=round(z['exp(coef) upper 95%'], 2), p=round(float(z['p']), 5),
                           n=len(q), deaths=int(q['ev'].sum())))
m = CoxPHFitter()
m.fit(S[['e', 'x', 'ev', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever', 'period']],
      duration_col='x', event_col='ev', entry_col='e',
      formula="g_60_74 + g_ge75 + male + edu + smoke_ever + period + g_60_74:period + g_ge75:period")
P("  interaction P values:")
for key, lab in [('g_60_74:period', '60-74 x period'), ('g_ge75:period', '>=75 x period')]:
    if key in m.summary.index:
        z = m.summary.loc[key]
        P(f"    {lab:20s} P = {z['p']:.4g}")
        phrows.append(dict(period='interaction', contrast=lab, hr=None, lo=None, hi=None,
                           p=round(float(z['p']), 5), n=len(S), deaths=int(S['ev'].sum())))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "tableS4_support_fine_v7.csv"), index=False, encoding='utf-8-sig')
pd.DataFrame(phrows).to_csv(os.path.join(OUT, "tableS1_ph_period_v7.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s51_audit_s4_ph.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/tableS4_support_fine_v7.csv, out/tableS1_ph_period_v7.csv, out/s51_audit_s4_ph.txt")
print("DONE")
