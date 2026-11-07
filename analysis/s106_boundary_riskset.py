# -*- coding: utf-8 -*-
"""Step 106: at the instant a boundary death occurs, is the older group in the risk set?

A death at exactly 60 (or 75) years falls on the lower boundary of its window. If no
member of the older group is at risk at that age, the death carries no information about
the contrast and dropping it is what the partial likelihood implies, not an arbitrary
convention. If some members are at risk, dropping it discards information and the
sensitivity analysis in Table S12 panel C is doing real work.

Output: out/s106_boundary_riskset.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

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


h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
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
                     last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                     sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0],
                     poswave=fp['wave'].iloc[0], lastwave=x['wave'].max()))
I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['event'] = (I['dead'].notna()).astype(int)
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

PAIRS = [('60-74', '<60', 60.0, 80.0), ('>=75', '60-74', 75.0, 95.0)]
for rname, D in [('rule A', I), ('rule B', B)]:
    P("=" * 78)
    P(f"# {rname}")
    P("=" * 78)
    for a, b, lo, hi in PAIRS:
        Dp = D[D['grp'].astype(str).isin([a, b])].copy()
        # window construction used by the fits
        Dp['e'] = Dp['entry'].clip(lower=lo)
        Dp['x'] = Dp['exit'].clip(upper=hi)
        bd = Dp[(Dp['event'] == 1) & (Dp['exit'] == lo)]
        P("")
        P(f"{a} vs {b}, window {lo:.0f}-{hi:.0f}: {len(bd)} deaths at exactly {lo:.0f} years "
          f"({int((bd['grp'].astype(str) == a).sum())} in {a}, {int((bd['grp'].astype(str) == b).sum())} in {b})")
        for t in [lo]:
            for who, g in (('older', a), ('reference', b)):
                risk = Dp[(Dp['grp'].astype(str) == g) & (Dp['e'] <= t) & (Dp['x'] > t) & ~(
                    (Dp['event'] == 1) & (Dp['exit'] == t))]
                P(f"  at age {t:.0f}: {who} group risk set = {len(risk)} respondents, "
                  f"{risk['x'].sub(risk['e']).sum():.1f} person-years remaining")
        # same question a hair inside the window, where the retained variants place the death
        for t in [lo + 1e-6, lo + 0.5]:
            risk = Dp[(Dp['grp'].astype(str) == a) & (Dp['e'] <= t) & (Dp['x'] > t)]
            P(f"  at age {t:.6f}: older group risk set = {len(risk)} respondents")

io.open(f"{OUT}\\s106_boundary_riskset.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s106_boundary_riskset.txt")
