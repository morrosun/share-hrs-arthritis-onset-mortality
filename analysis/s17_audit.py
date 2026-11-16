# -*- coding: utf-8 -*-
"""Step 17: 时间结构审计（回应评审第 1 条）
1) 统计每个起病组里「诊断前的人-年」有多少、涉及多少人 —— 这就是不死时间偏倚的量。
2) 用 max(入组年龄, 诊断年龄) 作延迟进入重拟合。
3) 核验 radiagarthr 与「首次报告阳性波次」是否自洽（回忆/记录一致性）。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter

OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

d = pd.read_csv(os.path.join(OUT, "analysis_share.csv"))
d['grp'] = d['grp'].astype(str).str.strip()
d['male'] = (d['sex'] == 1).astype(float)
d['smoke_ever'] = (d['smoke'] >= 1).astype(float)

P("="*78 + "\n# A. 诊断前时间被计入「诊断后组」的规模\n" + "="*78)
d['pre_dx'] = (d['onset_age'] - d['entry_age']).clip(lower=0)
d['has_pre'] = (d['onset_age'] > d['entry_age']).astype(int)
g = d.groupby('grp').agg(n=('pid', 'size'), deaths=('event', 'sum'),
                         py=('py', 'sum'),
                         n_onset_after_entry=('has_pre', 'sum'),
                         py_before_dx=('pre_dx', 'sum'))
g['pct_py_before_dx'] = (100 * g['py_before_dx'] / g['py']).round(1)
g['pct_pers'] = (100 * g['n_onset_after_entry'] / g['n']).round(1)
P(g.round(3).to_string())
P(f"\n合计：{int(d['has_pre'].sum()):,} / {len(d):,} 人（{100*d['has_pre'].mean():.1f}%）的诊断发生在首次访谈之后，")
P(f"      合计 {d['pre_dx'].sum():,.0f} 人-年（占全部 {d['py'].sum():,.0f} 人-年的 {100*d['pre_dx'].sum()/d['py'].sum():.1f}%）落在诊断之前。")

P("\n[按组：这部分的规模]")
for k in ['<60', '60-74', '>=75']:
    x = d[d['grp'] == k]
    P(f"  {k:6s}: n={len(x):6,}  诊断晚于入组者 {int(x['has_pre'].sum()):5,} "
      f"({100*x['has_pre'].mean():4.1f}%)  诊断前人-年 {x['pre_dx'].sum():9,.0f} "
      f"({100*x['pre_dx'].sum()/x['py'].sum():4.1f}% of group py)")

P("\n[最极端的情形：诊断年龄 ≥75 但入组时才 60 出头]")
x = d[(d['grp'] == '>=75') & (d['entry_age'] < 75)]
P(f"  n={len(x)} 人  诊断前随访 {x['pre_dx'].sum():,.0f} 人-年  死亡 {int(x['event'].sum())}")
P(f"  这些人的平均入组年龄 {x['entry_age'].mean():.1f}，平均诊断年龄 {x['onset_age'].mean():.1f}")

P("\n" + "="*78 + "\n# B. 主模型重拟合：延迟进入改为 max(入组, 诊断)\n" + "="*78)
def fit(df, entry_col, label):
    D = df[['exit_age', entry_col, 'event', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']].dropna()
    D = D[D['exit_age'] > D[entry_col]]
    m = CoxPHFitter()
    m.fit(D, duration_col='exit_age', event_col='event', entry_col=entry_col)
    P(f"\n--- {label}  n={len(D):,}  deaths={int(D['event'].sum()):,} ---")
    s = m.summary
    out = {}
    for k, key in [('60-74', 'g_60_74'), ('>=75', 'g_ge75')]:
        out[k] = (s.loc[key, 'exp(coef)'], s.loc[key, 'exp(coef) lower 95%'],
                  s.loc[key, 'exp(coef) upper 95%'], s.loc[key, 'p'])
        P(f"  {k:6s} {out[k][0]:.2f} ({out[k][1]:.2f}-{out[k][2]:.2f})  p={out[k][3]:.3g}")
    return out

d['g_60_74'] = (d['grp'] == '60-74').astype(float)
d['g_ge75'] = (d['grp'] == '>=75').astype(float)

res_old = fit(d, 'entry_age', '旧：延迟进入 = 首次访谈年龄（诊断前时间计入该组）')
d['entry_dx'] = np.maximum(d['entry_age'], d['onset_age'])
res_new = fit(d, 'entry_dx', '新：延迟进入 = max(首次访谈, 诊断年龄)')

P("\n[对比]")
P(f"  60-74 旧 {res_old['60-74'][0]:.2f} ({res_old['60-74'][1]:.2f}-{res_old['60-74'][2]:.2f})"
  f"  → 新 {res_new['60-74'][0]:.2f} ({res_new['60-74'][1]:.2f}-{res_new['60-74'][2]:.2f})")
P(f"  >=75  旧 {res_old['>=75'][0]:.2f} ({res_old['>=75'][1]:.2f}-{res_old['>=75'][2]:.2f})"
  f"  → 新 {res_new['>=75'][0]:.2f} ({res_new['>=75'][1]:.2f}-{res_new['>=75'][2]:.2f})")

io.open(os.path.join(OUT, "s17_audit.txt"), 'w', encoding='utf-8').write("\n".join(lines))
