# -*- coding: utf-8 -*-
"""Step 90: Figure 3 的分年龄死亡率，按 v18 最终队列口径重算。

v5 版图 3 的两个口径缺陷（与 v18 主分析不一致）：
  (a) `if ex <= e0: continue` 直接丢掉同年死亡者（HRS 348 名 never/prevalent），
      而 v18 主分析统一插补 0.5 年；
  (b) 未剔除入组年龄 <40 岁者（318 名配偶/伴侣），而 v18 死亡率队列已剔除。

本脚本用 s85_hrs_final.py 的 build(retain_same_year=True, drop_under40=True)
生成同一套人-区间表（34,568 人 / 13,389 死亡），再按 5 岁带切分人年，
使图 3 与表 4 完全共用同一风险集。

输出：
  out/fig3_rates_v18.csv      5 岁带 × 五态的人年、死亡、率
  out/s90_fig3_v18.txt        口径核对 + 新旧率对照
"""
import os
import sys, io, math, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s)); print(s, flush=True)


h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

persons = []
for pid, a in h.groupby('pid'):
    fp = a[a['arthre'] == 1]
    base = dict(pid=pid, entry=a['age'].iloc[0], last=a['age'].iloc[-1],
                dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0],
                smoke=a['smoken'].iloc[0])
    if fp.empty:
        persons.append(dict(base, pattern='never', onset=np.nan, pos=np.nan)); continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2 if pat == 'incident' else np.nan
    persons.append(dict(base, pattern=pat, onset=mid, pos=fp['age'].iloc[0]))

P0 = pd.DataFrame(persons)
P0['event'] = P0['dead'].notna().astype(int)
P0['exit'] = np.where(P0['event'] == 1, P0['dead'] - P0['byear'], P0['last'])
P0 = P0[P0['exit'].notna()].copy()


def grp_of(x):
    return '<60' if x < 60 else ('60-74' if x < 75 else '>=75')


# ---- 与 s85 build(retain_same_year=True, drop_under40=True) 完全相同 ----
rows = []
for r in P0.itertuples():
    e0, ex = r.entry, r.exit
    if not np.isfinite(ex):
        continue
    if e0 < 40:
        continue
    if ex <= e0:
        if r.event != 1:
            continue
        ex = e0 + 0.5
    base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
    if r.pattern == 'never':
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state='never'))
    elif r.pattern == 'prevalent':
        rows.append(dict(base, start=e0, stop=ex, event=r.event, state='prevalent'))
    else:
        ex_inc = max(ex, r.pos + 0.5) if r.event == 1 else ex
        if r.pos > e0:
            rows.append(dict(base, start=e0, stop=min(r.pos, ex), event=0, state='never'))
        if ex_inc > max(r.pos, e0):
            rows.append(dict(base, start=max(r.pos, e0), stop=ex_inc, event=r.event,
                             state='on_' + grp_of(r.onset)))

T = pd.DataFrame(rows)
T = T[T['stop'] > T['start']].reset_index(drop=True)
# 与 s85 一致：male / smoke_ever 由 0/1 派生（缺失吸烟记为 0），只对教育做 dropna
T['male'] = (T['sex'] == 1).astype(float)
T['smoke_ever'] = (T['smoke'] >= 1).astype(float)
T['edu'] = pd.to_numeric(T['edu'], errors='coerce')
T = T.dropna(subset=['edu', 'male', 'smoke_ever']).reset_index(drop=True)
P(f"[missing smoking among modelled rows] {int(pd.isna(T['smoke']).sum()):,} of {len(T):,} rows; "
  f"{int(T.loc[pd.isna(T['smoke']), 'pid'].nunique()):,} respondents coded as never-smokers")

P("=" * 78)
P("# Figure 3 cohort check (v18 rules: same-year deaths kept, entry age >= 40)")
P("=" * 78)
P(f"persons={T['pid'].nunique():,}  deaths={int(T['event'].sum()):,}  "
  f"person-years={float((T['stop'] - T['start']).sum()):,.0f}")
P("  (Table 4 target: 34,568 respondents / 13,389 deaths)")
P("")
P(T.groupby('state').apply(lambda d: pd.Series({
    'persons': d['pid'].nunique(),
    'deaths': int(d['event'].sum()),
    'py': float((d['stop'] - d['start']).sum())}), include_groups=False).round(0).to_string())


# ---- 5 岁带切分（锚在 5 的倍数，45 岁起，事件记入含退出年龄的那一段）----
def split_band(row, lo=45, hi=110, w=5):
    out = []
    a = max(math.floor(row['start'] / w) * w, lo)
    while a < row['stop'] and a < hi:
        s0 = max(row['start'], a); e0 = min(row['stop'], a + w)
        if e0 > s0 + 1e-9:
            out.append((row['state'], a, e0 - s0, 0))
        a += w
    if out and row['event'] == 1:
        out[-1] = (out[-1][0], out[-1][1], out[-1][2], 1)
    return out


recs = []
for _, row in T.iterrows():
    recs += split_band(row)
F3 = pd.DataFrame(recs, columns=['grp', 'band', 'py', 'death'])
g3 = (F3.groupby(['grp', 'band'], observed=True)
        .agg(py=('py', 'sum'), death=('death', 'sum')).reset_index())
g3['rate'] = 1000 * g3['death'] / g3['py']
g3 = g3[g3['py'] >= 200].sort_values(['grp', 'band']).reset_index(drop=True)

ORDER = ['never', 'prevalent', 'on_<60', 'on_60-74', 'on_>=75']
g3['grp'] = pd.Categorical(g3['grp'], ORDER, ordered=True)
g3 = g3.sort_values(['grp', 'band'])
g3.to_csv(os.path.join(OUT, "fig3_rates_v18.csv"), index=False, encoding='utf-8-sig')

P("")
P("=" * 78)
P("# Rates per 1000 person-years (v18)")
P("=" * 78)
piv = g3.pivot_table(index='band', columns='grp', values='rate', observed=True)
P(piv.round(1).to_string())
P("")
P("band = lower bound of the 5-year band; plotted at band + 2.5")

# ---- 新旧对照 ----
old = pd.read_csv(os.path.join(OUT, "fig3_rates_v5.csv"), encoding='utf-8-sig')
old['grp'] = old['grp'].astype(str).str.strip()
new = g3.copy(); new['grp'] = new['grp'].astype(str)
m = old.merge(new, on=['grp', 'band'], suffixes=('_v5', '_v18'), how='outer')
P("")
P("=" * 78)
P("# v5 -> v18 差异（仅列有变化的单元格）")
P("=" * 78)
m['d_py'] = m['py_v18'] - m['py_v5']
m['d_dt'] = m['death_v18'] - m['death_v5']
m['d_rt'] = m['rate_v18'] - m['rate_v5']
chg = m[(m['d_py'].abs() > 0.5) | (m['d_dt'].abs() > 0.5)]
P(chg[['grp', 'band', 'py_v5', 'py_v18', 'death_v5', 'death_v18',
       'rate_v5', 'rate_v18', 'd_rt']].round(2).to_string(index=False))

io.open(os.path.join(OUT, "s90_fig3_v18.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE out/fig3_rates_v18.csv + out/s90_fig3_v18.txt")
