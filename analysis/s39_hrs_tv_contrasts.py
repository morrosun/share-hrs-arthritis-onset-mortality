# -*- coding: utf-8 -*-
"""Step 39: HRS 五态时依 Cox 模型内，三个发病层之间的直接对比。

做法：数据、协变量、风险集与 s29 完全相同，只改变参照状态（重新参数化），
这样 HR 与 95% CI 直接由模型给出，无需手工组合协方差矩阵。
  - 参照 = inc_<60  -> inc_60-74 vs inc_<60、inc_>=75 vs inc_<60
  - 参照 = inc_60-74 -> inc_>=75 vs inc_60-74
另核对 d_never 的 HR 是否等于 1 / (inc_<60 vs never)，以确认参数化等价。

输出 out/table4_hrs_tv_contrasts_v5.csv、out/s39_hrs_tv_contrasts.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H)
h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

persons = []
for pid, a in h.groupby('pid'):
    fp = a[a['arthre'] == 1]
    base = dict(pid=pid, entry=a['age'].iloc[0], last=a['age'].iloc[-1], dead=a['radyear'].max(),
                byear=a['rabyear'].iloc[0], sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0],
                smoke=a['smoken'].iloc[0])
    if fp.empty:
        persons.append(dict(base, pattern='never', onset=np.nan, pos=np.nan))
        continue
    ip = a.index.get_loc(fp.index[0])
    prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2 if pat == 'incident' else np.nan
    persons.append(dict(base, pattern=pat, onset=mid, pos=fp['age'].iloc[0]))
P0 = pd.DataFrame(persons)
P0['event'] = P0['dead'].notna().astype(int)
P0['exit'] = np.where(P0['event'] == 1, P0['dead'] - P0['byear'], P0['last'])
P0 = P0[P0['exit'].notna()]


def grp_of(x):
    return '<60' if x < 60 else ('60-74' if x < 75 else '>=75')


rows = []
for r in P0.itertuples():
    e0, ex = r.entry, r.exit
    if not np.isfinite(ex) or ex <= e0:
        continue
    base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
    if r.pattern in ('never', 'prevalent'):
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state=r.pattern))
    else:
        ex_inc = max(ex, r.pos + 0.5) if r.event == 1 else ex
        if r.pos > e0:
            rows.append(dict(base, start=e0, stop=min(r.pos, ex), event=0, state='never'))
        rows.append(dict(base, start=max(r.pos, e0), stop=ex_inc, event=r.event,
                         state='inc_' + grp_of(r.onset)))

T = pd.DataFrame(rows)
T = T[T['stop'] > T['start']].reset_index(drop=True)
T['male'] = (T['sex'] == 1).astype(float)
T['smoke_ever'] = (T['smoke'] >= 1).astype(float)
T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
T = T.dropna(subset=['edu', 'male', 'smoke_ever']).reset_index(drop=True)
for c, s in [('d_never', 'never'), ('d_prev', 'prevalent'), ('d_lt60', 'inc_<60'),
             ('d_6074', 'inc_60-74'), ('d_75', 'inc_>=75')]:
    T[c] = (T['state'] == s).astype(float)

P("=" * 78)
P("# HRS 五态时依模型内，发病层之间的直接对比（v5）")
P("=" * 78)
for k, x in T.groupby('state'):
    P(f"  {k:12s} py={(x['stop']-x['start']).sum():9,.0f}  events={int(x['event'].sum()):5,}  persons={x['pid'].nunique():,}")

LAB = {'d_6074': 'Onset 60-74 vs <60 (incident, within five-state model)',
       'd_75': 'Onset >=75 vs <60 (incident, within five-state model)',
       'd_75b': 'Onset >=75 vs 60-74 (incident, within five-state model)',
       'd_never': 'Never vs <60 (consistency check)',
       'd_prev': 'Prevalent vs <60 (consistency check)',
       'd_lt60': 'Onset <60 vs 60-74 (consistency check)'}
out = []


def fit(ref, cols, keys, lab):
    D = T[['pid', 'start', 'stop', 'event', 'male', 'edu', 'smoke_ever'] + cols].copy()
    m = CoxTimeVaryingFitter()
    m.fit(D, id_col='pid', start_col='start', stop_col='stop', event_col='event',
          formula=' + '.join(cols + ['male', 'edu', 'smoke_ever']))
    P(f"\n--- {lab}（参照状态 = {ref}）: persons={D['pid'].nunique():,} events={int(m.event_observed.sum()):,} ---")
    for k, name in keys:
        q = m.summary.loc[k]
        est, lo, hi, pv = q['exp(coef)'], q['exp(coef) lower 95%'], q['exp(coef) upper 95%'], q['p']
        P(f"  {name:62s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={pv:.3g}")
        out.append(dict(reference=ref, contrast=name, hr=round(est, 3), lo=round(lo, 3),
                        hi=round(hi, 3), p=float(f"{pv:.3g}"),
                        n_persons=D['pid'].nunique(), events=int(m.event_observed.sum())))
    return m


m1 = fit('inc_<60', ['d_never', 'd_prev', 'd_6074', 'd_75'],
         [('d_6074', LAB['d_6074']), ('d_75', LAB['d_75']),
          ('d_never', LAB['d_never']), ('d_prev', LAB['d_prev'])],
         'Model 1')
m2 = fit('inc_60-74', ['d_never', 'd_prev', 'd_lt60', 'd_75'],
         [('d_75', LAB['d_75b']), ('d_lt60', LAB['d_lt60'])],
         'Model 2')

# 一致性核对：原模型 inc_<60 vs never = 1.203，故 never vs inc_<60 应约 0.831
q = m1.summary.loc['d_never']
P(f"\n[一致性] 1 / 1.203 = {1/1.203:.3f}；模型 1 的 never vs <60 = {q['exp(coef)']:.3f}")
P("[一致性] 模型 1 的 d_6074 × 模型 2 的 d_75 应等于模型 1 的 d_75（同一对比链）")
a = m1.summary.loc['d_6074']['exp(coef)']
b = m2.summary.loc['d_75']['exp(coef)']
c = m1.summary.loc['d_75']['exp(coef)']
P(f"  {a:.3f} × {b:.3f} = {a*b:.3f}  vs  {c:.3f}")

pd.DataFrame(out).to_csv(f"{OUT}\\table4_hrs_tv_contrasts_v5.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s39_hrs_tv_contrasts.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 out/table4_hrs_tv_contrasts_v5.csv")
P("写出 out/s39_hrs_tv_contrasts.txt")
