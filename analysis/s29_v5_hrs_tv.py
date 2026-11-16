# -*- coding: utf-8 -*-
"""Step 29 (v5): HRS 时依五态模型，把发病暴露切换点从「区间中点」改为「首次阳性报告年龄」。
中点切换会把 [中点, 首次阳性] 这段保证存活的时间计为发病暴露（保证生存时间偏倚）。
v5 口径：阳性报告前全部为 never；首次阳性报告起才计入 inc_ 层。
prevalent（基线已患病）保持不变。输出 out/table4_hrs_mortality_v5.csv
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def hr_to_rr(hr): return (1 - 0.5**np.sqrt(hr)) / (1 - 0.5**np.sqrt(1/hr))
def e_val(rr):
    if rr < 1: rr = 1/rr
    return rr + np.sqrt(rr*(rr-1))

h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

persons = []
for pid, a in h.groupby('pid'):
    fp = a[a['arthre'] == 1]
    base = dict(pid=pid, entry=a['age'].iloc[0], last=a['age'].iloc[-1], dead=a['radyear'].max(),
                byear=a['rabyear'].iloc[0], sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0],
                smoke=a['smoken'].iloc[0])
    if fp.empty:
        persons.append(dict(base, pattern='never', onset=np.nan, pos=np.nan)); continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    # v5：分层 = 区间中点（与 v4 相同）；切换/入组 = 首次阳性
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2 if pat == 'incident' else np.nan
    persons.append(dict(base, pattern=pat, onset=mid, pos=fp['age'].iloc[0]))
P0 = pd.DataFrame(persons)
P0['event'] = P0['dead'].notna().astype(int)
P0['exit'] = np.where(P0['event'] == 1, P0['dead'] - P0['byear'], P0['last'])
P0 = P0[P0['exit'].notna()]

def grp_of(x): return '<60' if x < 60 else ('60-74' if x < 75 else '>=75')

rows = []
for r in P0.itertuples():
    e0, ex = r.entry, r.exit
    # v18 口径（2026-11-07 作者确认）：
    #  (a) never/prevalent 的同年死亡（零长度记录）与 incident 一致，插补 0.5 年；
    #  (b) 剔除入组时 <40 岁者（配偶/伴侣，位于队列定义之外）。
    if not np.isfinite(ex): continue
    if e0 < 40: continue
    if ex <= e0:
        if r.event == 1:
            ex = e0 + 0.5
        else:
            continue
    base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
    if r.pattern == 'never':
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state='never'))
    elif r.pattern == 'prevalent':
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state='prevalent'))
    else:
        # v5：切换点 = 首次阳性报告年龄；阳性前全部为 never
        # 若死亡发生在首次阳性同一访谈区间（ex<=pos），保留事件：inc 段取 pos..pos+0.5（见 s31 方案 B）
        ex_inc = max(ex, r.pos + 0.5) if r.event == 1 else ex
        if r.pos > e0:
            rows.append(dict(base, start=e0, stop=min(r.pos, ex), event=0, state='never'))
        rows.append(dict(base, start=max(r.pos, e0), stop=ex_inc, event=r.event, state='inc_' + grp_of(r.onset)))

T = pd.DataFrame(rows)
T = T[T['stop'] > T['start']].reset_index(drop=True)
T['male'] = (T['sex'] == 1).astype(float)
T['smoke_ever'] = (T['smoke'] >= 1).astype(float)
T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
T = T.dropna(subset=['edu', 'male', 'smoke_ever']).reset_index(drop=True)
T['d_prev'] = (T['state'] == 'prevalent').astype(float)
T['d_lt60'] = (T['state'] == 'inc_<60').astype(float)
T['d_6074'] = (T['state'] == 'inc_60-74').astype(float)
T['d_75'] = (T['state'] == 'inc_>=75').astype(float)
T['d_ever'] = (T['state'] != 'never').astype(float)

P("="*78 + "\n# HRS 时依五态（v5：切换点 = 首次阳性报告年龄）\n" + "="*78)
for k, x in T.groupby('state'):
    P(f"  {k:12s} py={(x['stop']-x['start']).sum():9,.0f}  events={int(x['event'].sum()):5,}  persons={x['pid'].nunique():,}")

out = []
def fit(form, cols, keys, lab):
    D = T[['pid', 'start', 'stop', 'event', 'male', 'edu', 'smoke_ever'] + cols].copy()
    m = CoxTimeVaryingFitter()
    m.fit(D, id_col='pid', start_col='start', stop_col='stop', event_col='event', formula=form)
    P(f"\n--- {lab}: persons={D['pid'].nunique():,} events={int(m.event_observed.sum()):,} ---")
    for k in keys:
        q = m.summary.loc[k]
        est, lo, hi, pv = q['exp(coef)'], q['exp(coef) lower 95%'], q['exp(coef) upper 95%'], q['p']
        cod = {'d_ever': 'Any arthritis', 'd_prev': 'Prevalent at entry (onset unknown)',
               'd_lt60': 'Onset <60 (incident, from first positive)',
               'd_6074': 'Onset 60-74 (incident, from first positive)',
               'd_75': 'Onset >=75 (incident, from first positive)'}[k]
        P(f"  {cod:44s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={pv:.2e}")
        # Store full precision: the manuscript rounds once, at display time. Rounding
        # here as well would round twice and could move a value across a 0.01 boundary
        # (for example 1.2946 -> 1.295 -> 1.30 instead of 1.29).
        out.append(dict(model=lab, contrast=cod, hr=float(est), lo=float(lo), hi=float(hi),
                        p=float(f"{pv:.3g}"), n_persons=D['pid'].nunique(), events=int(m.event_observed.sum())))

fit('d_ever + male + edu + smoke_ever', ['d_ever'], ['d_ever'], 'Any vs never')
fit('d_prev + d_lt60 + d_6074 + d_75 + male + edu + smoke_ever', ['d_prev', 'd_lt60', 'd_6074', 'd_75'],
    ['d_prev', 'd_lt60', 'd_6074', 'd_75'], 'Five-state')

pd.DataFrame(out).to_csv(os.path.join(OUT, "table4_hrs_mortality_v5.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s29_v5_hrs_tv.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 table4_hrs_mortality_v5.csv")
