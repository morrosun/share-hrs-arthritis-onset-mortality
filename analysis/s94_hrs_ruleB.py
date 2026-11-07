# -*- coding: utf-8 -*-
"""Step 94: HRS 发病队列在「至少两次访谈」规则下的窗口限制对比。

v21 正文把「主规则 1.04（Table 5，窗口限制）」与「规则 B 0.61（Table S9，全年龄三组模型）」
并列，两者估计量不同。本脚本按 Table 5 的同一算法（5 岁带共同支持、窗口内 Cox），
在规则 B 的子队列上重估 60–74 vs <60 与 ≥75 vs 60–74，使比较同一口径。

规则 B 定义（与 s69 一致）：首次阳性报告之后至少还有一次访谈，即 lastwave > poswave。

输出 out/s94_hrs_ruleB.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
out = []


def P(s=""):
    lines.append(str(s)); print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    if neg.empty:
        continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                     sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0],
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

P("=" * 78)
P("# HRS incident cohort")
P("=" * 78)
P(f"eligible: {len(I):,}  deaths: {int(I['event'].sum()):,}")
B = I[I['lastwave'] > I['poswave']].copy()
P(f"rule B (a later interview exists): {len(B):,}  deaths: {int(B['event'].sum()):,}")
P(f"removed: {len(I)-len(B):,}, all died: {bool(((I['lastwave'] == I['poswave']) & (I['event'] == 1)).sum() == len(I)-len(B))}")


def band_py(D):
    out = {}
    for g, d in D.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0 = max(e, a); e0 = min(x, a + 5)
                if e0 > s0:
                    bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g] = bands
    return pd.DataFrame(out).fillna(0).sort_index()


def support(D, a, b):
    T = band_py(D)
    segs = []; cur = None
    for band in T.index:
        ok = (T.loc[band, a] >= 20) and (T.loc[band, b] >= 20)
        if ok and cur is None:
            cur = [band, band + 5]
        elif ok:
            cur[1] = band + 5
        elif cur is not None:
            segs.append(tuple(cur)); cur = None
    if cur is not None:
        segs.append(tuple(cur))
    return segs


def pair(D, a, b, lo, hi):
    X = D.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'] == a].empty or X[X['grp'] == b].empty:
        return None
    X['cmp'] = (X['grp'] == a).astype(float)
    m = CoxPHFitter()
    m.fit(X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']],
          duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=q['exp(coef)'],
                lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'],
                nd_a=int(X.loc[X['grp'] == a, 'ev'].sum()), nd_b=int(X.loc[X['grp'] == b, 'ev'].sum()))


for lab, D in [("main (all incident)", I), ("rule B (>=2 interviews)", B)]:
    P("")
    P("=" * 78)
    P(f"# {lab}: n={len(D):,} deaths={int(D['event'].sum()):,}")
    P("=" * 78)
    for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
        segs = support(D, a, b)
        if not segs:
            P(f"  {a} vs {b}: no shared support -> not estimable")
            continue
        lo, hi = segs[0][0], segs[-1][1]
        r = pair(D, a, b, lo, hi)
        if r is None:
            P(f"  {a} vs {b}: support {segs} but one group empty -> not estimable")
            continue
        P(f"  {a} vs {b}: window {lo}-{hi} y, n={r['n']:,}, deaths={r['deaths']:,} "
          f"({r['nd_a']} vs {r['nd_b']}) -> HR {r['hr']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})")
        out.append(dict(rule=lab, contrast=f"{a} vs {b}", window=f"{lo:.0f}\u2013{hi:.0f}",
                        n=r['n'], deaths=r['deaths'], hr=r['hr'], lo=r['lo'], hi=r['hi']))

pd.DataFrame(out).to_csv(f"{OUT}\\tableS9_window_v18.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s94_hrs_ruleB.txt", 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE s94_hrs_ruleB.txt")
