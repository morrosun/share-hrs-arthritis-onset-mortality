# -*- coding: utf-8 -*-
"""Step 99: code-level audit of the two rule-B shared-window estimates (v30).

The manuscript reports 0.609 (60-74 vs <60, 60-80 y) and 0.611 (>=75 vs 60-74,
75-95 y) under rule B, and a Poisson check that returned 0.710 and 0.699. The
review asks for (i) verification of the window boundaries, the group indicator and
the death/person-time aggregation, and (ii) an explanation of the 0.10 gap between
the Poisson and Cox figures.

This script does four things:

  1. band audit  - person-years by group in every five-year band around each
                   boundary, the >=20 py rule applied band by band, and the
                   contiguous support window that results;
  2. window audit- respondents, deaths by group, person-years by group, deaths
                   dropped because they fall outside the window, and the same
                   quantities recomputed with an independent loop;
  3. model ladder- Cox (attained age) unadjusted, one covariate at a time, and
                   fully adjusted; the fully adjusted fit replicated with
                   statsmodels PHReg (a different implementation);
  4. Poisson ladder - five-year and one-year bands, without and with the same
                   three covariates, so the 0.71 -> 0.61 gap can be attributed to
                   age resolution and adjustment rather than left unexplained.

Outputs: out/s99_ruleB_ladder.txt, out/tableS9_ladder.csv
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


# ------------------------------------------------------------------ cohort
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H)
h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty:
        continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=x['age'].iloc[-1], dead=x['radyear'].max(),
                     byear=x['rabyear'].iloc[0], sex=x['ragender'].iloc[0],
                     edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0],
                     poswave=fp['wave'].iloc[0], lastwave=x['wave'].max()))
I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['event'] = I['dead'].notna().astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['male'] = (I['sex'] == 1).astype(float)
I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
I['grp'] = grp(I['onset'])
I = I[(I['onset'] >= 40)].dropna(subset=['grp', 'edu'])
I['entry'] = I['pos']
fix = (I['exit'] <= I['entry']) & (I['event'] == 1)
I.loc[fix, 'exit'] = I.loc[fix, 'entry'] + 0.5
I = I[I['exit'] > I['entry']].copy()
B = I[I['lastwave'] > I['poswave']].copy()

P("=" * 92)
P("# 0. Cohort")
P("=" * 92)
P(f"incident cohort (rule A): n={len(I):,}  deaths={int(I['event'].sum()):,}")
P(f"rule B (a later interview exists): n={len(B):,}  deaths={int(B['event'].sum()):,}")


# ------------------------------------------------------------- band audit
def band_py(D, g, step=5):
    """person-years in each age band for one group."""
    d = D[D['grp'].astype(str) == g]
    out = {}
    for e, x in zip(d['entry'], d['exit']):
        a = max(40, np.floor(e / step) * step)
        while a < x and a < 110:
            s0, e0 = max(e, a), min(x, a + step)
            if e0 > s0:
                out[a] = out.get(a, 0.0) + (e0 - s0)
            a += step
    return out


def support(D, a, b, step=5):
    pa, pb = band_py(D, a, step), band_py(D, b, step)
    bands = sorted(set(pa) | set(pb))
    segs, cur = [], None
    for band in bands:
        ok = pa.get(band, 0) >= 20 and pb.get(band, 0) >= 20
        if ok and cur is None:
            cur = [band, band + step]
        elif ok:
            cur[1] = band + step
        elif cur is not None:
            segs.append(tuple(cur))
            cur = None
    if cur is not None:
        segs.append(tuple(cur))
    return segs, pa, pb


PAIRS = [('60-74', '<60', 60.0, 80.0), ('>=75', '60-74', 75.0, 95.0)]

P("")
P("=" * 92)
P("# 1. Band audit: person-years by group, >=20 py rule, and the resulting window")
P("=" * 92)
for a, b, lo, hi in PAIRS:
    segs, pa, pb = support(B, a, b)
    P("")
    P(f"  {a} vs {b}: contiguous support segments {segs}")
    P(f"  {'band':>8} {a:>10} {b:>10}  rule")
    bands = sorted(set(pa) | set(pb))
    keep = [x for x in bands if lo <= x < hi]
    show = [x for x in bands if lo - 10 <= x <= hi + 5]
    for band in show:
        ok = pa.get(band, 0) >= 20 and pb.get(band, 0) >= 20
        mark = 'in window' if (lo <= band < hi and ok) else ('below 20' if lo <= band < hi else '')
        P(f"  {band:>6}-{band+5:<2} {pa.get(band,0):>10.1f} {pb.get(band,0):>10.1f}  {mark}")
    P(f"  window used: {lo:.0f}-{hi:.0f} y; bands inside the window {keep[0]}-{keep[-1]+5}; "
      f"excluded below: {[x for x in bands if x < lo][-2:]}; excluded above: {[x for x in bands if x >= hi][:2]}")

# ----------------------------------------------------------- window audit
P("")
P("=" * 92)
P("# 2. Window audit: respondents, deaths by group, person-years by group")
P("=" * 92)


def frame(d, a, b, lo, hi):
    X = d[d['grp'].astype(str).isin([a, b])].copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    return X[X['x'] > X['e']].copy()


def py_by_group(X, a, b):
    out = {}
    for g in (a, b):
        d = X[X['grp'].astype(str) == g]
        out[g] = float((d['x'] - d['e']).sum())
    return out


for a, b, lo, hi in PAIRS:
    X = frame(B, a, b, lo, hi)
    nd = X.groupby('grp', observed=True)['ev'].sum().to_dict()
    py = py_by_group(X, a, b)
    # independent recount of deaths in the window, without the ev column
    n_ind = 0
    for r in B[B['grp'].astype(str).isin([a, b])].itertuples():
        if r.event == 1 and lo <= r.exit <= hi and min(r.exit, hi) > max(r.entry, lo):
            n_ind += 1
    dropped = int(((B['grp'].astype(str).isin([a, b])) & (B['event'] == 1) &
                   (B['exit'] > hi)).sum())
    P("")
    P(f"  {a} vs {b}   window {lo:.0f}-{hi:.0f} y")
    P(f"    n = {len(X):,}   deaths = {int(X['ev'].sum()):,} "
      f"({a} {nd.get(a,0)}, {b} {nd.get(b,0)}); independent recount {n_ind}")
    P(f"    person-years = {py[a]:.1f} ({a}) + {py[b]:.1f} ({b}) = {sum(py.values()):.1f}")
    P(f"    deaths with age at death above the window, censored at the boundary: {dropped}")
    P(f"    group indicator: {(X['grp'].astype(str) == a).sum():,} in {a}, "
      f"{(X['grp'].astype(str) == b).sum():,} in {b}; cmp=1 is {a} vs {b}")

# --------------------------------------------------- composition of rule B
P("")
P("=" * 92)
P("# 2b. What rule B removes, by onset group")
P("=" * 92)
rem = I[I['lastwave'] == I['poswave']]
P(f"  removed: {len(rem):,} respondents, {int(rem['event'].sum()):,} deaths "
  f"(all removed died: {bool(int(rem['event'].sum()) == len(rem))})")
P(f"  {'group':<8}{'n kept':>9}{'deaths kept':>13}{'n removed':>11}{'deaths removed':>16}{'% of group deaths':>20}")
for g in ['<60', '60-74', '>=75']:
    keep = I[I['grp'].astype(str) == g]
    rm = rem[rem['grp'].astype(str) == g]
    pct = 100 * rm['event'].sum() / keep['event'].sum() if keep['event'].sum() else np.nan
    P(f"  {g:<8}{len(keep):>9,}{int(keep['event'].sum()):>13,}{len(rm):>11,}"
      f"{int(rm['event'].sum()):>16,}{pct:>19.1f}%")

# ------------------------------------------------------------ Cox ladder
P("")
P("=" * 92)
P("# 3. Cox ladder (attained age as the time scale, delayed entry)")
P("=" * 92)


def cox(X, a, covars):
    Z = X.copy()
    Z['cmp'] = (Z['grp'].astype(str) == a).astype(float)
    m = CoxPHFitter()
    m.fit(Z[['e', 'x', 'ev', 'cmp'] + covars], duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return float(q['exp(coef)']), float(q['exp(coef) lower 95%']), float(q['exp(coef) upper 95%'])


COVARS = ['male', 'edu', 'smoke_ever']
SPECS = [('none', []), ('sex', ['male']), ('education', ['edu']), ('smoking', ['smoke_ever']),
         ('sex+education+smoking', COVARS)]
cox_res = {}
for a, b, lo, hi in PAIRS:
    X = frame(B, a, b, lo, hi)
    P("")
    P(f"  {a} vs {b}   window {lo:.0f}-{hi:.0f}   n={len(X):,} deaths={int(X['ev'].sum()):,}")
    for lab, cv in SPECS:
        hr, l, u = cox(X, a, cv)
        cox_res[((a, b), lab)] = (hr, l, u)
        P(f"    Cox, {lab:<22}: {hr:.4f} ({l:.3f}-{u:.3f})")

# independent implementation: R survival::coxph on the same two frames
P("")
P("  independent implementation: R survival::coxph, same window and covariates")
frames = {}
for i, (a, b, lo, hi) in enumerate(PAIRS, 1):
    X = frame(B, a, b, lo, hi).copy()
    X['cmp'] = (X['grp'].astype(str) == a).astype(int)
    X[['pid', 'e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']].to_csv(
        os.path.join(OUT, f"s99_frame_c{i}.csv"), index=False)
    frames[i] = (a, b)
R_SCRIPT = r'''args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(survival))
for (f in args) {
  d <- read.csv(f)
  fit <- coxph(Surv(e, x, ev) ~ cmp + male + edu + smoke_ever, data=d, ties="efron")
  s <- summary(fit)
  cat(sprintf("%s  n=%d  deaths=%d  HR=%.4f (%.3f-%.3f)\n", basename(f), nrow(d),
              sum(d$ev), s$conf.int[1,1], s$conf.int[1,3], s$conf.int[1,4]))
  fit0 <- coxph(Surv(e, x, ev) ~ cmp, data=d, ties="efron")
  s0 <- summary(fit0)
  cat(sprintf("    unadjusted HR=%.4f (%.3f-%.3f)\n", s0$conf.int[1,1],
              s0$conf.int[1,3], s0$conf.int[1,4]))
}
'''
rpath = os.path.join(OUT, "_s99_coxph_check.R")
open(rpath, "w", encoding="utf-8").write(R_SCRIPT)
import shutil, subprocess
if shutil.which("Rscript") is None:
    P("    Rscript not found; optional R re-check skipped")
else:
    try:
        rr = subprocess.run(["Rscript", rpath,
                             os.path.join(OUT, "s99_frame_c1.csv"),
                             os.path.join(OUT, "s99_frame_c2.csv")],
                            capture_output=True, text=True, timeout=300)
        for ln in (rr.stdout or "").splitlines():
            P("    " + ln)
        if rr.returncode != 0:
            P("    R returned " + str(rr.returncode) + ": " + (rr.stderr or "")[:400])
    except Exception as exc:
        P(f"    optional R re-check skipped: {exc}")

# -------------------------------------------------------- Poisson ladder
P("")
P("=" * 92)
P("# 4. Poisson ladder: age resolution and adjustment")
P("=" * 92)


def person_period(d, a, b, lo, hi, step):
    """One row per respondent-band, carrying person-time and the death indicator
    of that band. A death is assigned to the band in which it occurred, so it
    shares the row with the exposure it accrued in that band."""
    rec = []
    for r in d[d['grp'].astype(str).isin([a, b])].itertuples():
        e, x = max(r.entry, lo), min(r.exit, hi)
        if x <= e:
            continue
        died = 1 if (r.event == 1 and e < r.exit <= hi) else 0
        dband = np.floor((r.exit - 1e-9) / step) * step if died else None
        acc = {}
        aa = np.floor(e / step) * step
        while aa < x:
            s0, e0 = max(e, aa), min(x, aa + step)
            if e0 > s0:
                acc[float(aa)] = acc.get(float(aa), 0.0) + (e0 - s0)
            aa += step
        for band, py in acc.items():
            rec.append((str(r.grp), band, py, 1 if (died and band == dband) else 0,
                        r.male, r.edu, r.smoke_ever))
    return pd.DataFrame(rec, columns=['grp', 'band', 'py', 'd', 'male', 'edu', 'smoke_ever'])


def poisson_fit(d, a, b, lo, hi, step, covars):
    R = person_period(d, a, b, lo, hi, step)
    if covars:
        f = 'd ~ C(grp, Treatment(reference="%s")) + C(band) + %s' % (b, ' + '.join(covars))
        m = smf.glm(f, data=R, family=sm.families.Poisson(),
                    offset=np.log(R['py'])).fit()
        key = [k for k in m.params.index if k.startswith('C(grp')][0]
    else:
        T = R.groupby(['grp', 'band'], as_index=False).agg(py=('py', 'sum'), d=('d', 'sum'))
        T['is_a'] = (T['grp'] == a).astype(int)
        m = smf.glm('d ~ is_a + C(band)', data=T, family=sm.families.Poisson(),
                    offset=np.log(T['py'])).fit()
        key = 'is_a'
    ci = m.conf_int().loc[key]
    return float(np.exp(m.params[key])), float(np.exp(ci[0])), float(np.exp(ci[1]))


ladder = []
for a, b, lo, hi in PAIRS:
    P("")
    P(f"  {a} vs {b}   window {lo:.0f}-{hi:.0f}")
    for step, cv, lab in [(5, [], '5-year bands, no covariates'),
                          (5, ['male', 'edu', 'smoke_ever'], '5-year bands + covariates'),
                          (1, [], '1-year bands, no covariates'),
                          (1, ['male', 'edu', 'smoke_ever'], '1-year bands + covariates')]:
        hr, l, u = poisson_fit(B, a, b, lo, hi, step, cv)
        ladder.append(dict(contrast=f"{a} vs {b}", spec=lab, hr=hr, lo=l, hi=u))
        P(f"    Poisson, {lab:<34}: {hr:.4f} ({l:.3f}-{u:.3f})")

# ------------------------------------------------------------ closeness
P("")
P("=" * 92)
P("# 5. How close are the two contrasts under each specification?")
P("=" * 92)
P(f"  {'specification':<44}{'60-74 vs <60':>16}{'>=75 vs 60-74':>18}{'|diff|':>10}")
for lab in [s[0] for s in SPECS]:
    h1 = cox_res[(('60-74', '<60'), lab)][0]
    h2 = cox_res[(('>=75', '60-74'), lab)][0]
    P(f"  Cox, {lab:<40}{h1:>16.4f}{h2:>18.4f}{abs(h1-h2):>10.4f}")
for spec in ['5-year bands, no covariates', '5-year bands + covariates',
             '1-year bands, no covariates', '1-year bands + covariates']:
    h1 = [r['hr'] for r in ladder if r['spec'] == spec and r['contrast'] == '60-74 vs <60'][0]
    h2 = [r['hr'] for r in ladder if r['spec'] == spec and r['contrast'] == '>=75 vs 60-74'][0]
    P(f"  Poisson, {spec:<36}{h1:>16.4f}{h2:>18.4f}{abs(h1-h2):>10.4f}")

# write the ladder for the table builder
L = pd.DataFrame(ladder)
L['comparison'] = np.where(L['contrast'] == '60-74 vs <60', 'c1', 'c2')
wide = L.pivot(index='spec', columns='comparison', values='hr')
wide['diff'] = (wide['c1'] - wide['c2']).abs()
wide['family'] = 'Poisson'
wide = wide.reset_index()

rows = []
for lab in [s[0] for s in SPECS]:
    h1, h2 = cox_res[(('60-74', '<60'), lab)][0], cox_res[(('>=75', '60-74'), lab)][0]
    rows.append(dict(spec=lab, c1=h1, c2=h2, diff=abs(h1 - h2), family='Cox'))
C = pd.DataFrame(rows)
C['spec'] = 'Cox, ' + C['spec']
wide['spec'] = 'Poisson, ' + wide['spec']
ALL = pd.concat([C, wide[['spec', 'c1', 'c2', 'diff', 'family']]], ignore_index=True)
ALL.to_csv(os.path.join(OUT, "tableS9_ladder.csv"), index=False, encoding='utf-8-sig')

P("")
P("Wrote out/tableS9_ladder.csv")
io.open(os.path.join(OUT, "s99_ruleB_ladder.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("Wrote out/s99_ruleB_ladder.txt")

# ------------------------------------------------ Table S12 exports (v31)
# Panel A: person-years and deaths in each five-year band of each window.
band_rows = []
for a, b, lo, hi in PAIRS:
    pa, pb = band_py(B, a), band_py(B, b)
    D = B[B['grp'].astype(str).isin([a, b])]
    da, db = {}, {}
    for r in D.itertuples():
        # a death exactly at the lower boundary has no time inside the window
        if r.event == 1 and lo < r.exit <= hi:
            band = float(np.floor((r.exit - 1e-9) / 5) * 5)
            tgt = da if str(r.grp) == a else db
            tgt[band] = tgt.get(band, 0) + 1
    for band in sorted(x for x in (set(pa) | set(pb)) if lo <= x < hi):
        band_rows.append(dict(contrast=f"{a} vs {b}", window=f"{lo:g}\u2013{hi:g}",
                              band=f"{band:g}\u2013{band+5:g}",
                              py_a=pa.get(band, 0.0), py_b=pb.get(band, 0.0),
                              deaths_a=da.get(band, 0), deaths_b=db.get(band, 0)))
    band_rows.append(dict(contrast=f"{a} vs {b}", window=f"{lo:g}\u2013{hi:g}", band='Total in bands',
                          py_a=sum(pa.get(float(x), 0.0) for x in range(int(lo), int(hi), 5)),
                          py_b=sum(pb.get(float(x), 0.0) for x in range(int(lo), int(hi), 5)),
                          deaths_a=sum(da.values()), deaths_b=sum(db.values())))
    # deaths exactly at the lower boundary: no time inside the window, excluded from the fit
    ba = sum(1 for r in D.itertuples() if r.event == 1 and r.exit == lo and str(r.grp) == a)
    bb = sum(1 for r in D.itertuples() if r.event == 1 and r.exit == lo and str(r.grp) == b)
    band_rows.append(dict(contrast=f"{a} vs {b}", window=f"{lo:g}\u2013{hi:g}",
                          band='At lower boundary (excluded)', py_a=0.0, py_b=0.0,
                          deaths_a=ba, deaths_b=bb))
    band_rows.append(dict(contrast=f"{a} vs {b}", window=f"{lo:g}\u2013{hi:g}",
                          band='All deaths in window', py_a=float('nan'), py_b=float('nan'),
                          deaths_a=sum(da.values()) + ba, deaths_b=sum(db.values()) + bb))
BA = pd.DataFrame(band_rows)
BA.to_csv(os.path.join(OUT, "tableS12_bands.csv"), index=False, encoding='utf-8-sig')

# Panel B: every specification, both ratios, their difference.
lab_rows = []
for fam, spec in [('Cox', 'none'), ('Cox', 'sex'), ('Cox', 'education'), ('Cox', 'smoking'),
                  ('Cox', 'sex+education+smoking')]:
    h1, l1, u1 = cox_res[(('60-74', '<60'), spec)]
    h2, l2, u2 = cox_res[(('>=75', '60-74'), spec)]
    lab_rows.append(dict(family=fam, spec=spec, c1=h1, lo1=l1, hi1=u1, c2=h2, lo2=l2, hi2=u2, diff=abs(h1 - h2)))
for spec in ['5-year bands, no covariates', '5-year bands + covariates',
             '1-year bands, no covariates', '1-year bands + covariates']:
    r1 = [r for r in ladder if r['spec'] == spec and r['contrast'] == '60-74 vs <60'][0]
    r2 = [r for r in ladder if r['spec'] == spec and r['contrast'] == '>=75 vs 60-74'][0]
    lab_rows.append(dict(family='Poisson', spec=spec, c1=r1['hr'], lo1=r1['lo'], hi1=r1['hi'],
                         c2=r2['hr'], lo2=r2['lo'], hi2=r2['hi'], diff=abs(r1['hr'] - r2['hr'])))
LB = pd.DataFrame(lab_rows)
LB.to_csv(os.path.join(OUT, "tableS12_ladder.csv"), index=False, encoding='utf-8-sig')
P("")
P(f"Table S12: {len(BA)-2} band rows + 2 totals; {len(LB)} specifications, "
  f"diff {LB['diff'].min():.4f}-{LB['diff'].max():.4f}")
P("Wrote out/tableS12_bands.csv and out/tableS12_ladder.csv")
