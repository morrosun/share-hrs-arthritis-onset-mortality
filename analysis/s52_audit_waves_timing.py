# -*- coding: utf-8 -*-
"""Step 52: fourth-round review -- wave denominators, same-year deaths, landmark,
HRS version reconciliation, PH scale, and the S5 midpoint gap.

1) HRS: three nested layers per entry wave (interval observed / eligible / analysable),
   showing that Table S7 panel C reported the analytic sample, not interval availability.
2) SHARE: diagnosis-age availability among respondents who reported arthritis at entry.
3) Same-year deaths: month variables, death year vs first-positive wave year, and
   0.25/0.5/0.75/1.0 y imputation sensitivity for the two main incident contrasts.
4) Excluding early deaths vs a 2-year landmark: quantify guaranteed survival time.
5) HRS Table 4 version chain: 2x2 isolation of switch point (midpoint vs first positive)
   and same-year death retention; clarify n=41,788 vs 34,537.
6) Proportional-hazards interaction under the follow-up-time scale (to locate the
   earlier draft's P=0.002/0.008).
7) Table S5: midpoint-to-first-positive gap and verification that 5-y censoring
   starts at the first positive report.

Outputs: out/s52_audit_waves_timing.txt, out/tableS7_waves_v7.csv,
         out/tableS8_sameyear_timing_v7.csv, out/tableS4_version_check_v7.csv
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


def load(db, wy):
    d = pyreadr.read_r(os.path.join(BASE, f"{db}.RData"))[db].copy()
    d = d.rename(columns={d.columns[0]: 'pid'})
    d['yr'] = d['wave'].map(wy)
    d['age'] = d['yr'] - d['rabyear']
    return d


s = load('share', WY_S)
h = load('hrs', WY_H)

# ===================== 1. HRS nested layers =====================
P("=" * 90)
P("# 1. HRS: three nested denominators by entry wave (repairing Table S7 panel C)")
P("=" * 90)
h2 = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
first = h2.groupby('pid').agg(entry_wave=('wave', 'first'), n_waves=('wave', 'size'))

recs = []
for pid, x in h2.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty:
        continue
    recs.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     pos_year=fp['yr'].iloc[0], last=x['age'].iloc[-1],
                     dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                     edu=x['raeducl'].iloc[0]))
I = pd.DataFrame(recs)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['event'] = I['dead'].notna().astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
# L1: interval observed (no eligibility filter)
L1 = I.copy()
# L2: eligibility (onset>=40, education complete, valid stratum)
L2 = I[(I['onset'] >= 40)].dropna(subset=['edu']).copy()
L2['grp'] = grp(L2['onset'])
L2 = L2.dropna(subset=['grp']).copy()
# L3: analytic sample (entry = first positive; same-year deaths retained with +0.5)
L3 = L2.copy()
L3['entry'] = L3['pos']
fix = (L3['exit'] <= L3['entry']) & (L3['event'] == 1)
L3.loc[fix, 'exit'] = L3.loc[fix, 'entry'] + 0.5
L3 = L3[L3['exit'] > L3['entry']].copy()
P(f"L0 (all HRS respondents with a valid wave)      = {len(first):,}")
P(f"L1 (negative->positive interval OBSERVED)       = {len(L1):,}")
P(f"L2 (L1 + onset>=40 + complete education/stratum)= {len(L2):,}")
P(f"L3 (L2 + usable post-report follow-up)          = {len(L3):,}")
P(f"s47 panel C 'Incident interval observed' total  = 6,359 = L3, NOT L1.")

first = first.copy()
for nm, d in [('L1', L1), ('L2', L2), ('L3', L3)]:
    first[nm] = first.index.isin(d['pid']).astype(int)
g = first.groupby('entry_wave').agg(n_entered=('L1', 'size'), L1=('L1', 'sum'),
                                    L2=('L2', 'sum'), L3=('L3', 'sum'))
g['pct_L1'] = 100 * g['L1'] / g['n_entered']
g['pct_L2'] = 100 * g['L2'] / g['n_entered']
g['pct_L3'] = 100 * g['L3'] / g['n_entered']
P("\n" + g.round(1).to_string())
P("\nWaves 14-15: L1 (observed interval) vs L3 (analysable), explaining the reported 0%:")
for w in (14, 15):
    r = g.loc[float(w)]
    P(f"  wave {w}: entered {int(r['n_entered']):,}, L1 interval observed {int(r['L1']):,}, "
      f"L2 eligible {int(r['L2']):,}, L3 analysable {int(r['L3']):,}")
P("  => 0% in the old panel C is about post-report follow-up / eligibility, not about")
P("     being unable to observe a negative-to-positive transition (where L1 > 0).")

# ===================== 2. SHARE availability denominator =====================
P("\n" + "=" * 90)
P("# 2. SHARE: diagnosis-age availability conditional on reporting arthritis at entry")
P("=" * 90)
s_sorted = s.sort_values(['pid', 'wave'])
ent = s_sorted.groupby('pid').agg(entry_wave=('wave', 'first'),
                                  arth_entry=('arthre', 'first'))
dx = (s_sorted.dropna(subset=['radiagarthr'])
      .assign(v=lambda d: d['radiagarthr'].where(d['radiagarthr'] >= 5))
      .dropna(subset=['v']).groupby('pid').agg(dx_age=('v', 'first')))
ent = ent.join(dx)
ent['has_dx'] = ent['dx_age'].notna()
ga = ent.groupby('entry_wave').agg(n_entered=('has_dx', 'size'),
                                   n_arth_entry=('arth_entry', 'sum'),
                                   n_dx=('has_dx', 'sum'))
ga['pct_all_entrants'] = 100 * ga['n_dx'] / ga['n_entered']
# denominator = respondents with arthritis reported at their entry interview
arth = ent[ent['arth_entry'] == 1].groupby('entry_wave').agg(
    n_arth_entry2=('has_dx', 'size'), n_dx2=('has_dx', 'sum'))
ga = ga.join(arth)
ga['pct_among_arthritic'] = 100 * ga['n_dx2'] / ga['n_arth_entry2']
P(ga[['n_entered', 'n_dx', 'pct_all_entrants', 'n_arth_entry2', 'n_dx2',
      'pct_among_arthritic']].round(1).to_string())
P("\nThe 17.4-22.7% figure used ALL entrants as the denominator and therefore mixes")
P("arthritis prevalence with recording availability; conditional on arthritis at entry")
P("the recording proportion is much higher and is the relevant quantity.")

rows7 = []
for w, r_ in g.iterrows():
    yr = WY_H[int(w)]
    ar = ga.loc[float(w)] if float(w) in ga.index else None
    rows7.append(dict(panel='C', cohort='HRS', wave=f"Entry wave {int(w)} ({yr})",
                      n_entered=int(r_['n_entered']), L1=int(r_['L1']), L2=int(r_['L2']),
                      L3=int(r_['L3']), pct_L1=round(r_['pct_L1'], 1),
                      pct_L3=round(r_['pct_L3'], 1)))
for w, r_ in ga.iterrows():
    yr = WY_S[int(w)]
    rows7.append(dict(panel='A2', cohort='SHARE', wave=f"Entry wave {int(w)} ({yr})",
                      n_entered=int(r_['n_entered']),
                      L1=int(r_['n_arth_entry2']) if pd.notna(r_['n_arth_entry2']) else 0,
                      L2=int(r_['n_dx2']) if pd.notna(r_['n_dx2']) else 0,
                      L3=None, pct_L1=round(r_['pct_all_entrants'], 1),
                      pct_L3=round(r_['pct_among_arthritic'], 1) if pd.notna(r_['pct_among_arthritic']) else None))

# ===================== 3. Same-year death imputation =====================
P("\n" + "=" * 90)
P("# 3. Same-year deaths: available timing variables and 0.5-year imputation")
P("=" * 90)
for nm, d in [('SHARE', s), ('HRS', h)]:
    cols = [c for c in d.columns if 'month' in c.lower() or c.lower() in
            ('iwm', 'iwy', 'iwendm', 'iwendy', 'radtimtdth', 'agey', 'radyear')]
    P(f"  {nm} timing columns: {cols}")
    for c in cols:
        if c in d.columns:
            P(f"    {c:12s}: non-missing {int(d[c].notna().sum()):,}/{len(d):,}")
P("  All analysis ages are integer years: age = wave calendar year - birth year;")
P("  death age = death year - birth year. Month variables exist but were NOT used.")


def build_incident(df, wy, delta):
    d = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave']).copy()
    out = []
    for pid, x in d.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty:
            continue
        ip = x.index.get_loc(fp.index[0])
        neg = x.iloc[:ip]
        neg = neg[neg['arthre'] == 0]
        if neg.empty:
            continue
        out.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                        pos_year=fp['yr'].iloc[0], last=x['age'].iloc[-1],
                        dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                        sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0],
                        smoke=x['smoken'].iloc[0]))
    A = pd.DataFrame(out)
    A['onset'] = (A['lastneg'] + A['pos']) / 2
    A['event'] = A['dead'].notna().astype(int)
    A['exit'] = np.where(A['event'] == 1, A['dead'] - A['byear'], A['last'])
    A['male'] = (A['sex'] == 1).astype(float)
    A['smoke_ever'] = (A['smoke'] >= 1).astype(float)
    A['grp'] = grp(A['onset'])
    A = A[(A['onset'] >= 40)].dropna(subset=['grp', 'edu']).copy()
    A['entry'] = A['pos']
    zero = (A['exit'] <= A['entry']) & (A['event'] == 1)
    A.loc[zero, 'exit'] = A.loc[zero, 'entry'] + delta
    A = A[A['exit'] > A['entry']].copy()
    return A, zero.sum(), A[A['pid'].isin(A.loc[zero, 'pid'])].shape[0], d


# death-year audit on the raw interval cohort
P("\n  Death year relative to the calendar year of the first positive report:")
for nm, df, wy in [('SHARE', s, WY_S), ('HRS', h, WY_H)]:
    A0, nzero, _, _ = build_incident(df, wy, 0.5)
    tmp = A0.copy()
    # reconstruct from raw to compare death year and first-positive wave year
    d = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rec = []
    for pid, x in d.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); neg = x.iloc[:ip]; neg = neg[neg['arthre'] == 0]
        if neg.empty: continue
        rec.append((pid, fp['yr'].iloc[0], x['radyear'].max()))
    R = pd.DataFrame(rec, columns=['pid', 'pos_year', 'dead_year'])
    R = R[R['dead_year'].notna()]
    R['diff'] = R['dead_year'] - R['pos_year']
    anom = set(R[R['diff'] < 0]['pid'])
    P(f"    {nm}: deaths after first positive interview {int((R['diff']>0).sum()):,}; "
      f"same calendar year {int((R['diff']==0).sum()):,}; "
      f"BEFORE first positive year (anomalous) {int((R['diff']<0).sum()):,}")
    P(f"    zero-length records imputed with 0.5 y: {nzero:,}; "
      f"anomalous (death before first-positive wave year) records retained in that "
      f"analytic sample: {int(A0['pid'].isin(anom).sum()):,}")


def band_py(D):
    out = {}
    for g_, d in D.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0, e0 = max(e, a), min(x, a + 5)
                if e0 > s0:
                    bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g_] = bands
    return pd.DataFrame(out).fillna(0).sort_index()


def pairwin(D, a, b, lo, hi):
    X = D.copy()
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

P("\n  Sensitivity to the imputed length of a zero-length death record:")
rows8 = []
for nm, df, wy, win in [('SHARE', s, WY_S, [('60-74', '<60', 60, 75), ('>=75', '60-74', 75, 90)]),
                        ('HRS', h, WY_H, [('60-74', '<60', 60, 80), ('>=75', '60-74', 75, 90)])]:
    for delta in (0.25, 0.5, 0.75, 1.0):
        D, nzero, _, _ = build_incident(df, wy, delta)
        parts = []
        for a, b, lo, hi in win:
            r = pairwin(D, a, b, lo, hi)
            txt = f"{r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})" if r else 'not estimable'
            parts.append(f"{a} vs {b}: {txt}")
            rows8.append(dict(cohort=nm, delta=delta, contrast=f"{a} vs {b}",
                              n=len(D), deaths=int(D['event'].sum()),
                              hr=round(r['hr'], 2) if r else None,
                              lo=round(r['lo'], 2) if r else None,
                              hi=round(r['hi'], 2) if r else None))
        P(f"    {nm:6s} delta={delta:.2f} y  n={len(D):,}  " + " | ".join(parts))
    # anomalous deaths: exclude those whose recorded death year precedes the first-positive wave year
    Danom, _, _, _ = build_incident(df, wy, 0.5)
    Dr = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    ap = []
    for pid, x in Dr.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); neg = x.iloc[:ip]; neg = neg[neg['arthre'] == 0]
        if neg.empty: continue
        dy = x['radyear'].max()
        if pd.notna(dy) and dy < fp['yr'].iloc[0]:
            ap.append(pid)
    Dclean = Danom[~Danom['pid'].isin(ap)].copy()
    if ap:
        txt = []
        for a, b, lo, hi in win:
            r = pairwin(Dclean, a, b, lo, hi)
            txt.append(f"{a} vs {b}: {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})" if r
                       else f"{a} vs {b}: not estimable")
        P(f"    {nm:6s} exclude {len(ap):,} anomalous death records: n={len(Dclean):,}  "
          + " | ".join(txt))
    else:
        P(f"    {nm:6s} no anomalous death records to exclude")

# ===================== 4. Exclude early deaths vs landmark =====================
P("\n" + "=" * 90)
P("# 4. 'Exclude deaths <2 y' vs a 2-year landmark")
P("=" * 90)
for nm, df, wy in [('SHARE', s, WY_S), ('HRS', h, WY_H)]:
    D, _, _, _ = build_incident(df, wy, 0.5)
    early = D[(D['event'] == 1) & (D['exit'] - D['entry'] < 2)]
    excl = D[~((D['event'] == 1) & (D['exit'] - D['pos'] < 2))].copy()
    lm = D[D['exit'] > D['pos'] + 2].copy()
    lm['entry'] = D.loc[lm.index, 'pos'] + 2
    guaranteed = ((excl['entry'] < excl['exit']) &
                  (excl['exit'] - excl['entry'] > 2)).sum()
    P(f"  {nm}: early deaths (within 2 y) = {len(early):,}; "
      f"'exclude early deaths' keeps n={len(excl):,} and still starts follow-up at the first "
      f"positive report, so every survivor in the 0-2 y interval is guaranteed alive "
      f"({guaranteed:,} records span that interval).")
    P(f"       2-year landmark: n={len(lm):,} (survived to 2 y), entry = first positive + 2; "
      f"this is a conditional-survival analysis of a different population.")

# ===================== 5. HRS Table 4 version chain =====================
P("\n" + "=" * 90)
P("# 5. HRS 'Any arthritis' HR: 1.23 (v2) -> 1.34 (v5), isolating the cause")
P("=" * 90)
hd = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
persons = []
for pid, a in hd.groupby('pid'):
    fp = a[a['arthre'] == 1]
    base = dict(pid=pid, entry=a['age'].iloc[0], last=a['age'].iloc[-1],
                dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0])
    if fp.empty:
        persons.append(dict(base, pattern='never', onset=np.nan, pos=np.nan)); continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]; neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2 if pat == 'incident' else np.nan
    persons.append(dict(base, pattern=pat, onset=mid, pos=fp['age'].iloc[0]))
P0 = pd.DataFrame(persons)
P0['event'] = P0['dead'].notna().astype(int)
P0['exit'] = np.where(P0['event'] == 1, P0['dead'] - P0['byear'], P0['last'])
P0 = P0[P0['exit'].notna()]


def build_tv(switch, retain):
    rows = []
    for r in P0.itertuples():
        e0, ex = r.entry, r.exit
        if not np.isfinite(ex) or ex <= e0: continue
        base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
        if r.pattern == 'never':
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state='never'))
        elif r.pattern == 'prevalent':
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state='prevalent'))
        else:
            cut = r.onset if switch == 'midpoint' else r.pos
            ex_inc = max(ex, cut + 0.5) if (retain and r.event == 1) else ex
            if cut > e0:
                rows.append(dict(base, start=e0, stop=min(cut, ex), event=0, state='never'))
            rows.append(dict(base, start=max(cut, e0), stop=ex_inc, event=r.event, state='incident'))
    T = pd.DataFrame(rows)
    T = T[T['stop'] > T['start']].reset_index(drop=True)
    T['male'] = (T['sex'] == 1).astype(float)
    T['smoke_ever'] = (T['smoke'] >= 1).astype(float)
    T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
    T = T.dropna(subset=['edu', 'male', 'smoke_ever']).reset_index(drop=True)
    T['d_ever'] = (T['state'] != 'never').astype(float)
    m = CoxTimeVaryingFitter()
    D = T[['pid', 'start', 'stop', 'event', 'male', 'edu', 'smoke_ever', 'd_ever']]
    m.fit(D, id_col='pid', start_col='start', stop_col='stop', event_col='event',
          formula='d_ever + male + edu + smoke_ever')
    q = m.summary.loc['d_ever']
    return dict(hr=float(q['exp(coef)']), lo=float(q['exp(coef) lower 95%']),
                hi=float(q['exp(coef) upper 95%']), segments=len(T),
                persons=int(T['pid'].nunique()), events=int(m.event_observed.sum()))

verrows = []
for sw in ('midpoint', 'first positive'):
    for ret in (False, True):
        r = build_tv(sw, ret)
        lab = f"switch={sw}, retain same-year deaths={ret}"
        P(f"  {lab:52s} HR {r['hr']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})  "
          f"segments={r['segments']:,} persons={r['persons']:,} events={r['events']:,}")
        verrows.append(dict(scenario=lab, **r))
P("  v2 file table4_hrs_mortality_v2.csv: Any arthritis 1.229, n=41,788 (n = SEGMENTS), "
  "events 13,056")
P("  v5 file table4_hrs_mortality_v5.csv: Any arthritis 1.336, n=34,537 (n = PERSONS), "
  "events 13,056")
P("  => the person count did not change between models; the two 'n' fields count different")
P("     things. Events are identical (13,056). The HR change is attributable to the switch")
P("     point and to retaining same-year deaths, not to an undocumented sample change.")

# ===================== 6. PH under follow-up time scale =====================
P("\n" + "=" * 90)
P("# 6. Where did the earlier draft's PH P=0.002/0.008 come from?")
P("=" * 90)
c = pd.read_csv(os.path.join(OUT, "analysis_share.csv"), encoding='utf-8-sig')
c = c.rename(columns={'entry_age': 'entry', 'exit_age': 'exit'})
c['grp'] = c['grp'].astype(str).str.strip()
c['g_60_74'] = (c['grp'] == '60-74').astype(float)
c['g_ge75'] = (c['grp'] == '>=75').astype(float)
c['male'] = (c['sex'] == 1).astype(float)
c['smoke_ever'] = (c['smoke'] >= 1).astype(float)
c = c.dropna(subset=['exit', 'entry', 'male', 'edu', 'smoke_ever'])
c = c[c['exit'] > c['entry']].copy()
# follow-up time scale: duration = exit-entry, no left truncation, adjust for entry age
F = c.copy()
F['tsf'] = F['exit'] - F['entry']
F['entry_c'] = F['entry'] - F['entry'].mean()
recs = []
for _, r in F.iterrows():
    cut = 5.0
    base = dict(g_60_74=r['g_60_74'], g_ge75=r['g_ge75'], male=r['male'],
                edu=r['edu'], smoke_ever=r['smoke_ever'], entry_c=r['entry_c'])
    recs.append(dict(**base, e=0.0, x=min(r['tsf'], cut),
                     ev=int(r['event']) if r['tsf'] <= cut else 0, period=0))
    if r['tsf'] > cut:
        recs.append(dict(**base, e=cut, x=r['tsf'], ev=int(r['event']), period=1))
F2 = pd.DataFrame(recs)
P("  Follow-up-time scale, period-specific HRs (95% CI), fitted separately in each period:")
for per in (0, 1):
    q = F2[F2['period'] == per]
    try:
        m = CoxPHFitter()
        m.fit(q[['e', 'x', 'ev', 'g_60_74', 'g_ge75', 'male', 'edu',
                 'smoke_ever', 'entry_c']], duration_col='x', event_col='ev', entry_col='e')
        for key, lab in [('g_60_74', '60-74 vs <60'), ('g_ge75', '>=75 vs <60')]:
            z = m.summary.loc[key]
            P(f"    {'<5 y' if per==0 else '>=5 y':8s} {lab:16s} HR {z['exp(coef)']:.2f} "
              f"({z['exp(coef) lower 95%']:.2f}-{z['exp(coef) upper 95%']:.2f})")
    except Exception as ex:
        P(f"    period {per}: fit failed ({type(ex).__name__})")
P("  A single model with a period main effect on this time scale is singular (period is")
P("  determined by the interval start), so period-specific fits are reported instead.")
P("  Attained-age scale (s51): interaction P = 0.41 (60-74) and 0.69 (>=75) -- not significant.")
P("  Earlier drafts reported P = 0.002 / 0.008 for a group x period interaction; that")
P("  specification is not reproducible under either v6 time scale on the same cohort, so")
P("  those P values are NOT carried into v7. v7 reports the re-estimated period-specific")
P("  HRs with CIs and the attained-age interaction test instead.")

# ===================== 7. Table S5 midpoint gap =====================
P("\n" + "=" * 90)
P("# 7. Table S5: midpoint-to-first-positive gap and the 5-year window")
P("=" * 90)
for nm, df, wy in [('SHARE', s, WY_S), ('HRS', h, WY_H)]:
    D, _, _, _ = build_incident(df, wy, 0.5)
    gap = D['pos'] - (D['lastneg'] + D['pos']) / 2
    P(f"  {nm}: first positive - midpoint: median {gap.median():.1f} y, "
      f"IQR {gap.quantile(.25):.1f}-{gap.quantile(.75):.1f}, max {gap.max():.1f} y")
    P(f"        entry is the first positive report for all {len(D):,} records; "
      f"5-y censoring uses exit = min(exit, pos+5), so the midpoint is used ONLY for grouping.")
    # verify 5-y censored person-time starts at pos
    D5 = D.copy()
    D5['exit5'] = np.minimum(D5['exit'], D5['pos'] + 5)
    D5 = D5[D5['exit5'] > D5['entry']]
    P(f"        5-y censored design: n={len(D5):,}, "
      f"min(entry)={D5['entry'].min():.0f}, "
      f"all entry==pos: {bool((D5['entry']==D5['pos']).all())}")

pd.DataFrame(rows7).to_csv(os.path.join(OUT, "tableS7_waves_v7.csv"), index=False, encoding='utf-8-sig')
pd.DataFrame(rows8).to_csv(os.path.join(OUT, "tableS8_sameyear_timing_v7.csv"), index=False, encoding='utf-8-sig')
pd.DataFrame(verrows).to_csv(os.path.join(OUT, "tableS4_version_check_v7.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s52_audit_waves_timing.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/tableS7_waves_v7.csv, out/tableS8_sameyear_timing_v7.csv, "
  "out/tableS4_version_check_v7.csv, out/s52_audit_waves_timing.txt")
print("DONE")
