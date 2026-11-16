# -*- coding: utf-8 -*-
"""Step 96: audit the five-year band window against the single-year overlap.

For every pairwise contrast in Table 5 this reports
  (a) the band window and the single-year overlap,
  (b) person-years and deaths that fall inside the band window but outside the
      single-year overlap, split by stratum -- these are the ages at which only
      one stratum is observed,
  (c) the unrounded HR from the windowed fit and from the unrestricted fit
      (the two Table 5 columns), so that a coincidental agreement at two
      decimals can be distinguished from an identical fit,
  (d) deaths in the reference stratum inside the overlap (used to justify the
      "not estimable" rows).

Outputs: out/s96_window_audit.txt, out/s96_window_audit.csv
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
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


def build_incident(df, wave_map):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map)
    df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rows = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty:
            continue
        ip = x.index.get_loc(fp.index[0])
        prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty:
            continue
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(),
                         byear=x['rabyear'].iloc[0], sex=x['ragender'].iloc[0],
                         edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
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
    return I[I['exit'] > I['entry']].copy()


def prevalent_share(s):
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
    zero = D['exit'] <= D['entry']
    D.loc[zero & (D['event'] == 1), 'exit'] = D.loc[zero & (D['event'] == 1), 'entry'] + 0.5
    D = D[(D['exit'] > D['entry']) & (D['entry'] >= 40) & (D['exit'] <= 110) & (D['w0'] != 8)].copy()
    D['male'] = (D['sex'] == 1).astype(float)
    D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
    D['grp'] = grp(D['onset'])
    return D.dropna(subset=['grp', 'edu']).reset_index()


def window5(I):
    W = I.copy()
    W['exit'] = np.minimum(W['exit'], W['pos'] + 5)
    W['event'] = np.where(W['exit'] < I['exit'] - 1e-9, 0, W['event'])
    return W[W['exit'] > W['entry']].copy()


def band_py(D):
    out = {}
    for g, d in D.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0, e0 = max(e, a), min(x, a + 5)
                if e0 > s0:
                    bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g] = bands
    return pd.DataFrame(out).fillna(0).sort_index()


def support(D, a, b, thr=20):
    T = band_py(D)
    segs, cur = [], None
    for band in T.index:
        ok = (T.loc[band, a] >= thr) and (T.loc[band, b] >= thr)
        if ok and cur is None:
            cur = [band, band + 5]
        elif ok:
            cur[1] = band + 5
        elif cur is not None:
            segs.append(tuple(cur))
            cur = None
    if cur is not None:
        segs.append(tuple(cur))
    return segs


def single_year_rows(D):
    """one row per (pid, single year) with person-time, to get the single-year overlap"""
    recs = []
    for r in D.itertuples():
        a = max(40, np.floor(r.entry))
        while a < r.exit and a < 110:
            s0, e0 = max(r.entry, a), min(r.exit, a + 1)
            if e0 > s0:
                recs.append((r.pid, str(r.grp), int(a), e0 - s0, 0))
            a += 1
        if r.event and np.isfinite(r.exit):
            da = int(min(np.floor(r.exit - 1e-9), 109))
            recs.append((r.pid, str(r.grp), da, 0.0, 1))
    return pd.DataFrame(recs, columns=['pid', 'grp', 'age', 'py', 'dead'])


def fit(d, a, b, lo, hi):
    X = d.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].astype(str).isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'].astype(str) == a].empty or X[X['grp'].astype(str) == b].empty:
        return None
    X['cmp'] = (X['grp'].astype(str) == a).astype(float)
    m = CoxPHFitter()
    m.fit(X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']],
          duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=float(q['exp(coef)']),
                lo=float(q['exp(coef) lower 95%']), hi=float(q['exp(coef) upper 95%']))


s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S)
s['age'] = s['yr'] - s['rabyear']
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})

I_s = build_incident(s, WY_S)
I_h = build_incident(h, WY_H)
designs = [('SHARE prevalent', prevalent_share(s)),
           ('SHARE incident', I_s),
           ('SHARE incident 5y', window5(I_s)),
           ('HRS incident', I_h),
           ('HRS incident 5y', window5(I_h))]

rows = []
for name, D in designs:
    P("=" * 96)
    P(f"# {name}   n={len(D):,}  deaths={int(D['event'].sum()):,}")
    P("=" * 96)
    SY = single_year_rows(D)
    piv = SY.pivot_table(index='age', columns='grp', values='py', aggfunc='sum').fillna(0)
    dth = SY[SY['dead'] == 1].groupby(['age', 'grp']).size().unstack(fill_value=0)
    for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
        segs = support(D, a, b)
        if not segs:
            P(f"  {a} vs {b}: no qualifying band")
            continue
        lo, hi = segs[0][0], segs[-1][1]
        # single-year overlap = ages at which both strata contribute person-time
        if a in piv.columns and b in piv.columns:
            both = sorted(k for k in piv.index if piv.loc[k, a] > 0 and piv.loc[k, b] > 0
                          and lo <= k < hi)
        else:
            both = []
        ov_lo, ov_hi = (both[0], both[-1] + 1) if both else (None, None)
        # person-time and deaths inside the band window but outside the single-year overlap
        inwin = SY[(SY['age'] >= lo) & (SY['age'] < hi)]
        out_ov = inwin[~inwin['age'].isin(both)]
        py_a_out = out_ov.loc[out_ov['grp'] == a, 'py'].sum()
        py_b_out = out_ov.loc[out_ov['grp'] == b, 'py'].sum()
        d_a_out = out_ov.loc[out_ov['grp'] == a, 'dead'].sum()
        d_b_out = out_ov.loc[out_ov['grp'] == b, 'dead'].sum()
        rw = fit(D, a, b, lo, hi)
        rf = fit(D, a, b, 0, 1000)
        rt = fit(D, a, b, ov_lo, ov_hi) if both else None
        # deaths in the reference stratum (b) inside the single-year overlap
        d_b_ov = int(dth.loc[[k for k in both if k in dth.index], b].sum()) if both else 0
        P(f"  {a} vs {b}")
        P(f"    band window {lo:.0f}-{hi:.0f}   single-year overlap {ov_lo}-{ov_hi} "
          f"({len(both)} years)")
        P(f"    inside window but outside overlap: {a} {py_a_out:6.1f} py / {int(d_a_out)} d "
          f"| {b} {py_b_out:6.1f} py / {int(d_b_out)} d")
        if rw and rf:
            P(f"    windowed   HR {rw['hr']:.4f} ({rw['lo']:.4f}-{rw['hi']:.4f}) n={rw['n']:,} d={rw['deaths']:,}")
            P(f"    all ages   HR {rf['hr']:.4f} ({rf['lo']:.4f}-{rf['hi']:.4f}) n={rf['n']:,} d={rf['deaths']:,}")
            P(f"    difference in log HR: {abs(np.log(rw['hr']) - np.log(rf['hr'])):.5f}")
        if rt:
            P(f"    trimmed to the overlap {ov_lo:.0f}-{ov_hi:.0f}: HR {rt['hr']:.4f} "
              f"({rt['lo']:.4f}-{rt['hi']:.4f}) n={rt['n']:,} d={rt['deaths']:,}")
            P(f"    |log HR| band-window vs trimmed: {abs(np.log(rw['hr']) - np.log(rt['hr'])):.5f}")
        P(f"    deaths in {b} inside the overlap: {d_b_ov}")
        P("")
        rows.append(dict(design=name, contrast=f"{a} vs {b}", band_window=f"{lo:.0f}-{hi:.0f}",
                         single_year=f"{ov_lo}-{ov_hi}" if both else 'none',
                         n_years=len(both),
                         py_out_a=round(py_a_out, 1), d_out_a=int(d_a_out),
                         py_out_b=round(py_b_out, 1), d_out_b=int(d_b_out),
                         hr_window=round(rw['hr'], 4) if rw else None,
                         hr_all=round(rf['hr'], 4) if rf else None,
                         hr_trim=round(rt['hr'], 4) if rt else None,
                         lo_trim=round(rt['lo'], 4) if rt else None,
                         hi_trim=round(rt['hi'], 4) if rt else None,
                         n_trim=rt['n'] if rt else None, d_trim=rt['deaths'] if rt else None,
                         n_window=rw['n'] if rw else None, d_window=rw['deaths'] if rw else None,
                         d_ref_in_overlap=d_b_ov))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "s96_window_audit.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s96_window_audit.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("Wrote out/s96_window_audit.txt / .csv")
