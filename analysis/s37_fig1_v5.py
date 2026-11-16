# -*- coding: utf-8 -*-
"""Step 37: 核对 Figure 1A/1B 与 Figure 2 的数字，并生成 Figure 1B 的 v5 起病年龄分布数据。

输出 out/s37_fig1_audit.txt、out/fig1_onset_dist_v5.csv
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


# ---- 旧 Figure 1B 数据 ----
old = pd.read_csv(os.path.join(OUT, "fig1_onset_dist.csv"), encoding='utf-8-sig')
P("=" * 78)
P("# 旧 out/fig1_onset_dist.csv（fig1_fig3.R 当前读取的文件）")
P("=" * 78)
for lib, d in old.groupby('lib'):
    P(f"  {lib}: n={int(d['n'].sum()):,}  age 范围 {d['age'].min()}-{d['age'].max()}  bin 数 {len(d)}")
P("  HRS 前 8 个 bin:")
P(old[old['lib'] == 'HRS'].head(8).to_string(index=False))

old2 = pd.read_csv(os.path.join(OUT, "fig1_onset_v2.csv"), encoding='utf-8-sig')
P("")
P(f"# out/fig1_onset_v2.csv（s25 生成，v2 时声称已改为仅发病病例）")
for lib, d in old2.groupby('lib'):
    P(f"  {lib}: n={int(d['n'].sum()):,}  age 范围 {d['age'].min()}-{d['age'].max()}  bin 数 {len(d)}")

# ---- Figure 1A / Figure 2 数据来源 ----
flow = pd.read_csv(os.path.join(OUT, "fig1_flow.csv"), header=None, encoding='utf-8-sig')
P("")
P("# Figure 1A 流程（out/fig1_flow.csv）")
P(flow.to_string(index=False, header=False))
hr = pd.read_csv(os.path.join(OUT, "fig1_hr.csv"), encoding='utf-8-sig')
P("")
P("# Figure 2C 森林图（out/fig1_hr.csv，SHARE prevalent 年龄尺度 Cox）")
P(hr.to_string(index=False))

# ---- 生成 v5 的 HRS 发病分布 ----
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H)
h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
ons = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty:
        continue
    ons.append((neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2)
ons = pd.Series(ons, name='onset')
P("")
P("=" * 78)
P("# v5 口径的 HRS 起病年龄分布（全体发病者，区间中点，无过滤）")
P("=" * 78)
P(f"  n={len(ons):,}  中位 {ons.median():.0f}  IQR {ons.quantile(.25):.0f}-{ons.quantile(.75):.0f}  "
  f"最小 {ons.min():.1f} 最大 {ons.max():.1f}")

bins = np.floor(ons / 5) * 5 + 2.5
hv = bins.value_counts().sort_index().rename_axis('age').reset_index(name='n')
hv['lib'] = 'HRS'
hv['pct'] = 100 * hv['n'] / hv['n'].sum()
hv = hv[['lib', 'age', 'n', 'pct']]

sv = old[old['lib'] == 'SHARE'][['lib', 'age', 'n']].copy()
sv['pct'] = 100 * sv['n'] / sv['n'].sum()
v5 = pd.concat([sv[['lib', 'age', 'n', 'pct']], hv], ignore_index=True)
v5.to_csv(os.path.join(OUT, "fig1_onset_dist_v5.csv"), index=False, encoding='utf-8-sig')
P(f"  写出 out/fig1_onset_dist_v5.csv（SHARE {int(sv['n'].sum()):,} + HRS {int(hv['n'].sum()):,}，{len(v5)} 行）")

io.open(os.path.join(OUT, "s37_fig1_audit.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("写出 out/s37_fig1_audit.txt")
