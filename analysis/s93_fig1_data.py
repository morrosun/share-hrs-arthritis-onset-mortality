# -*- coding: utf-8 -*-
"""Step 93: Figure 1 的数据（v19 口径）。

v5 版的 fig1_flow.csv 仍写着 18,530（4,875 死亡），且只画到「无随访区间」一步，
没有 2020 波次与入组年龄的处理。本脚本直接由分析数据集重算：
  SHARE 流程 136,843 → 47,900 → 27,264 → 18,587 (4,932 死亡)
  四步排除：无出生年 3；入组 <40 岁 9；2020 入组波 3,353；仅一次访谈且存活 5,312
  （3 + 9 + 3,353 + 5,312 = 8,677 = 27,264 − 18,587）

起病年龄分布：SHARE 用 analysis_share.csv 的 onset_age（18,587 人），
HRS 沿用 v5 已算出的 7,252 名发病者的区间中点。

输出 out/fig1_flow_v18.csv、out/fig1_onset_dist_v18.csv、out/s93_fig1.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd

OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []


def P(s=""):
    lines.append(str(s)); print(s, flush=True)


d = pd.read_csv(f"{OUT}\\analysis_share.csv", encoding='utf-8-sig')
P("=" * 78)
P("# Figure 1 flow check")
P("=" * 78)
P(f"analytic cohort n={len(d):,} deaths={int(d['event'].sum()):,}")
n0, n1, n2 = 136843, 47900, 27264
n3 = len(d)
ex = dict(never=n0 - n1, noage=n1 - n2)
P(f"  {n0:,} -> {n1:,} (-{ex['never']:,}) -> {n2:,} (-{ex['noage']:,}) -> {n3:,} (-{n2-n3:,})")
P(f"  last step must equal 3 + 9 + 3,353 + 5,312 = {3+9+3353+5312:,}  (actual {n2-n3:,})")

flow = pd.DataFrame([
    ("Respondents in SHARE, waves 1-8", n0),
    ("Ever reported a physician diagnosis of arthritis", n1),
    ("Valid age at first diagnosis (5-105 years)", n2),
    ("Analytic cohort", n3),
], columns=['lab', 'n'])
flow.to_csv(f"{OUT}\\fig1_flow_v18.csv", index=False, encoding='utf-8-sig')

# ---- onset distribution ----
bins = np.floor(d['onset_age'] / 5) * 5 + 2.5
sv = bins.value_counts().sort_index().rename_axis('age').reset_index(name='n')
sv['lib'] = 'SHARE'
old = pd.read_csv(f"{OUT}\\fig1_onset_dist_v5.csv", encoding='utf-8-sig')
hv = old[old['lib'] == 'HRS'][['lib', 'age', 'n']].copy()
dist = pd.concat([sv[['lib', 'age', 'n']], hv], ignore_index=True)
dist['pct'] = 100 * dist['n'] / dist.groupby('lib')['n'].transform('sum')
dist.to_csv(f"{OUT}\\fig1_onset_dist_v18.csv", index=False, encoding='utf-8-sig')
P("")
P(f"  SHARE onset distribution n={int(sv['n'].sum()):,} (median {d['onset_age'].median():.0f})")
P(f"  HRS   onset distribution n={int(hv['n'].sum()):,} (rows {len(hv)})")

io.open(f"{OUT}\\s93_fig1.txt", 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE fig1_flow_v18.csv + fig1_onset_dist_v18.csv")
