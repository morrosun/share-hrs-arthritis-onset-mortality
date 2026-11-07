# -*- coding: utf-8 -*-
"""Step 95: locate the SHARE prevalent death-count discrepancy between Table 5 and Table S4.

Three cohorts are rebuilt side by side:

  MAIN  out/analysis_share.csv          -- Table 5 (s33) cohort, 18,587
  NEW   s43 prevalent_share() logic     -- same-year deaths imputed at 0.5 y, wave 8 removed,
                                           then restricted to onset <= entry
  OLD   s51 audit_s4_ph.py logic        -- no imputation, wave 8 kept, no onset restriction,
                                           then restricted to onset <= entry  (feeds Table S4)

For the 75-105 window of the >=75 vs 60-74 contrast every death is dumped per person,
the pid sets are differenced, and each discordant death is attributed to one code path:
same-year death, 2020 entry wave, onset > entry restriction, or window edge.

Outputs: out/s95_death_dump.csv, out/s95_dump_s4_deaths.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
LO, HI = 75.0, 105.0
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S)
s['age'] = s['yr'] - s['rabyear']


def base_frame(impute_same_year, drop_wave8):
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
    D['same_year'] = (D['event'] == 1) & (D['exit'] <= D['entry'] + 1e-9)
    if impute_same_year:
        D.loc[D['same_year'], 'exit'] = D.loc[D['same_year'], 'entry'] + 0.5
    D['male'] = (D['sex'] == 1).astype(float)
    D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
    D['grp'] = grp(D['onset'])
    D['onset_gt_entry'] = D['onset'] > D['entry'] + 1e-9
    D['wave8'] = D['w0'] == 8
    keep = (D['exit'] > D['entry']) & (D['entry'] >= 40) & (D['exit'] <= 110)
    if drop_wave8:
        keep &= ~D['wave8']
    D = D[keep].dropna(subset=['grp', 'edu']).copy()
    D['py'] = D['exit'] - D['entry']
    return D.reset_index()


OLD = base_frame(impute_same_year=False, drop_wave8=False)
OLD = OLD[~OLD['onset_gt_entry']].copy()          # s51 "restricted 18,445"
NEW = base_frame(impute_same_year=True, drop_wave8=True)
NEW = NEW[~NEW['onset_gt_entry']].copy()           # s43 prevalent cohort

MAIN = pd.read_csv(f"{OUT}\\analysis_share.csv", encoding='utf-8-sig')
MAIN = MAIN.rename(columns={'entry_age': 'entry', 'exit_age': 'exit'})
MAIN['grp'] = MAIN['grp'].astype(str).str.strip()
MAIN = MAIN.dropna(subset=['exit', 'entry', 'edu']).copy()
MAIN = MAIN[MAIN['exit'] > MAIN['entry']].copy()
MAIN['same_year'] = (MAIN['event'] == 1) & (MAIN['py'] <= 0.5 + 1e-9)
MAIN['wave8'] = MAIN['w0'] == 8
MAIN['onset'] = MAIN['onset_age']
MAIN['onset_gt_entry'] = MAIN['onset'] > MAIN['entry'] + 1e-9

P("=" * 92)
P("# Cohort sizes")
P("=" * 92)
for lab, d in [('MAIN (Table 5, analysis_share.csv)', MAIN), ('NEW (s43, imputed + no wave 8)', NEW),
               ('OLD (s51, feeds Table S4)', OLD)]:
    P(f"  {lab:42s} n={len(d):,}  deaths={int(d['event'].sum()):,}  py={d['py'].sum():,.0f}")


def window_deaths(d, a, b, lo=LO, hi=HI):
    """Replicate pair(): returns the subset whose death is counted, plus the analysis frame."""
    X = d.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].astype(str).isin([a, b])) & (X['x'] > X['e'])].copy()
    return X


P("")
P("=" * 92)
P("# 75-105 window, >=75 vs 60-74: deaths counted by each cohort")
P("=" * 92)
frames = {}
for lab, d in [('MAIN', MAIN), ('NEW', NEW), ('OLD', OLD)]:
    X = window_deaths(d, '>=75', '60-74')
    frames[lab] = X
    P(f"  {lab:5s} n={len(X):,}  deaths={int(X['ev'].sum()):,}  "
      f"(>=75: n={int((X['grp'].astype(str)=='>=75').sum()):,} "
      f"d={int(X.loc[X['grp'].astype(str)=='>=75','ev'].sum()):,}; "
      f"60-74: n={int((X['grp'].astype(str)=='60-74').sum()):,} "
      f"d={int(X.loc[X['grp'].astype(str)=='60-74','ev'].sum()):,})")

# ---- per-person death dump ----
def dump(X, tag):
    z = X[X['ev'] == 1].copy()
    z['cohort'] = tag
    return z[['pid', 'cohort', 'grp', 'entry', 'exit', 'onset', 'event', 'same_year',
              'wave8', 'onset_gt_entry', 'edu', 'smoke_ever']]


ALL = pd.concat([dump(frames['MAIN'], 'MAIN'), dump(frames['NEW'], 'NEW'),
                 dump(frames['OLD'], 'OLD')], ignore_index=True)
ALL.to_csv(f"{OUT}\\s95_death_dump.csv", index=False, encoding='utf-8-sig')

sets = {k: set(v.loc[v['ev'] == 1, 'pid']) for k, v in frames.items()}
P("")
P("=" * 92)
P("# pid set differences among deaths counted in the window")
P("=" * 92)
for x, y in [('MAIN', 'OLD'), ('NEW', 'OLD'), ('MAIN', 'NEW')]:
    only_x = sets[x] - sets[y]
    only_y = sets[y] - sets[x]
    P(f"  {x} minus {y}: {len(only_x):,}   |   {y} minus {x}: {len(only_y):,}")

# ---- attribution of MAIN-only and OLD-only deaths ----
ref = NEW.set_index('pid')
def attribute(pids, frame_tag):
    rows = []
    src = {'MAIN': MAIN, 'NEW': NEW, 'OLD': OLD}[frame_tag].set_index('pid')
    for p in sorted(pids):
        if p not in src.index:
            reason = 'dropped before the window (see cohort filters)'
            rows.append(dict(pid=p, reason=reason))
            continue
        r = src.loc[p]
        if r['same_year'] if 'same_year' in src.columns else False:
            reason = 'same-year death (exit<=entry; imputed 0.5 y in MAIN/NEW, dropped in OLD)'
        elif r.get('wave8', False):
            reason = '2020 entry wave (removed in MAIN/NEW, kept in OLD)'
        elif r.get('onset_gt_entry', False):
            reason = 'onset > entry (excluded by the restricted cohort in NEW/OLD)'
        elif r['exit'] > HI + 1e-9:
            reason = 'death after the upper window edge (105 y)'
        elif r['exit'] <= LO + 1e-9:
            reason = 'death at or before the lower window edge (75 y)'
        else:
            reason = 'unexplained: present in both cohorts, counted differently'
        rows.append(dict(pid=p, grp=str(r['grp']), entry=float(r['entry']), exit=float(r['exit']),
                         onset=float(r['onset']), reason=reason))
    return pd.DataFrame(rows)


P("")
P("=" * 92)
P("# Attribution")
P("=" * 92)
for tag, other in [('MAIN', 'OLD'), ('OLD', 'MAIN')]:
    diff = sets[tag] - sets[other]
    if not diff:
        continue
    A = attribute(diff, tag)
    P(f"\n  deaths counted in {tag} but not in {other}: {len(A)}")
    if 'reason' in A.columns:
        for k, v in A['reason'].value_counts().items():
            P(f"    {v:3d}  {k}")
        A.to_csv(f"{OUT}\\s95_attr_{tag}_only.csv", index=False, encoding='utf-8-sig')
        shown = A[A['reason'].str.startswith('unexplained', na=False)]
        if len(shown):
            P("    --- unexplained rows ---")
            P(shown.to_string(index=False))

# ---- where do the cohorts themselves diverge? ----
P("")
P("=" * 92)
P("# Cohort-level reconciliation (all ages, before the window)")
P("=" * 92)
for lab, d in [('MAIN', MAIN), ('NEW', NEW), ('OLD', OLD)]:
    z = d[d['event'] == 1]
    P(f"  {lab:5s} n={len(d):,}  deaths={len(z):,}  "
      f"same-year={int(z['same_year'].sum()) if 'same_year' in z.columns else 'n/a'}  "
      f"wave8={int(z['wave8'].sum()) if 'wave8' in z.columns else 'n/a'}  "
      f"onset>entry={int(z['onset_gt_entry'].sum()) if 'onset_gt_entry' in z.columns else 'n/a'}")

io.open(f"{OUT}\\s95_dump_s4_deaths.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s95_death_dump.csv, out/s95_dump_s4_deaths.txt")
print("DONE")
