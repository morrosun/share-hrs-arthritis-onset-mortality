# -*- coding: utf-8 -*-
"""Step 91: 比例风险的分时段估计（Table S1 panel C），按 v18/v19 队列重算。

v7 版该表由一次性内联脚本生成，n at risk（7,995 / 10,592）无法与当前队列复用：
它们既不等于「人年 <5 年者」（6,696），也不等于「存活满 5 年者」（11,891），
且相加恰为 18,587，说明是某一版的另一种切分。本脚本改为明确定义：

  把每个人按「入组后 5 年」切成 2 段（年龄尺度）：
    第 1 段 [entry, min(exit, entry+5))
    第 2 段 [max(entry, entry+5), exit)   （exit > entry+5 时才有）
  报告每段的「贡献人年的人数」「人年」「死亡」「HR」，并给出交互作用的 2 自由度 Wald 检验。

输出 out/tableS1_ph_period_v18.csv、out/s91_ph_v18.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from lifelines import CoxTimeVaryingFitter
from scipy import stats

OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []


def P(s=""):
    lines.append(str(s)); print(s, flush=True)


d = pd.read_csv(os.path.join(OUT, "analysis_share.csv"), encoding='utf-8-sig')
d = d.dropna(subset=['entry_age', 'exit_age', 'event', 'male' if 'male' in d else 'sex', 'edu'])
d['male'] = (d['sex'] == 1).astype(float)
d['smoke_ever'] = (d['smoke'] >= 1).astype(float)
d['g_60_74'] = (d['grp'] == '60-74').astype(float)
d['g_ge75'] = (d['grp'] == '>=75').astype(float)
P(f"cohort n={len(d):,} deaths={int(d['event'].sum()):,} py={float((d['exit_age']-d['entry_age']).sum()):,.0f}")

# ---- 切成两段 ----
rows = []
for r in d.itertuples():
    t0, t1 = r.entry_age, r.exit_age
    cut = t0 + 5.0
    if t1 <= cut:
        rows.append(dict(pid=r.pid, start=t0, stop=t1, event=r.event, period='<5 y',
                         g_60_74=r.g_60_74, g_ge75=r.g_ge75, male=r.male,
                         edu=r.edu, smoke_ever=r.smoke_ever))
    else:
        rows.append(dict(pid=r.pid, start=t0, stop=cut, event=0, period='<5 y',
                         g_60_74=r.g_60_74, g_ge75=r.g_ge75, male=r.male,
                         edu=r.edu, smoke_ever=r.smoke_ever))
        rows.append(dict(pid=r.pid, start=cut, stop=t1, event=r.event, period='>=5 y',
                         g_60_74=r.g_60_74, g_ge75=r.g_ge75, male=r.male,
                         edu=r.edu, smoke_ever=r.smoke_ever))

T = pd.DataFrame(rows)
T = T[T['stop'] > T['start']].reset_index(drop=True)
T['late'] = (T['period'] == '>=5 y').astype(float)
T['x1_early'] = T['g_60_74'] * (1 - T['late'])
T['x1_late'] = T['g_60_74'] * T['late']
T['x2_early'] = T['g_ge75'] * (1 - T['late'])
T['x2_late'] = T['g_ge75'] * T['late']

P("")
P("=" * 78)
P("# Person-time by period")
P("=" * 78)
summ = T.groupby('period').agg(respondents=('pid', 'nunique'),
                               py=('stop', lambda s: np.nan),
                               deaths=('event', 'sum'))
summ['py'] = T.groupby('period').apply(lambda x: float((x['stop'] - x['start']).sum()), include_groups=False)
summ = summ[['respondents', 'py', 'deaths']]
P(summ.round(0).to_string())
P(f"check: respondents {int(summ['respondents'].sum()):,} (a person can appear in both periods); "
  f"py {summ['py'].sum():,.0f}; deaths {int(summ['deaths'].sum()):,}")

# ---- 交互模型 ----
m = CoxTimeVaryingFitter()
m.fit(T, id_col='pid', event_col='event', start_col='start', stop_col='stop',
      formula='x1_early + x1_late + x2_early + x2_late + male + edu + smoke_ever')
out = []
rows_int = []
for k, per, lab in [('x1_early', '<5 y', '60-74 vs <60'), ('x1_late', '>=5 y', '60-74 vs <60'),
                    ('x2_early', '<5 y', '>=75 vs <60'), ('x2_late', '>=5 y', '>=75 vs <60')]:
    q = m.summary.loc[k]
    out.append(dict(period=per, contrast=lab, hr=q['exp(coef)'],
                    lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'], p=q['p']))
O = pd.DataFrame(out)
_py = T.groupby('period').apply(lambda x: float((x['stop'] - x['start']).sum()), include_groups=False)
_dt = T.groupby('period')['event'].sum()
O['py'] = O['period'].map(_py)
O['deaths'] = O['period'].map(_dt).astype(int)
P("")
P("=" * 78)
P("# Period-specific hazard ratios (attained-age scale, delayed entry)")
P("=" * 78)
P(O.round(3).to_string(index=False))

# ---- 交互检验：晚期段与早期段的系数差（1 df Wald）----
P("")
P("=" * 78)
P("# Exposure x period interaction (difference between the two periods)")
P("=" * 78)
V = np.asarray(m.variance_matrix_)
pos = {n: i for i, n in enumerate(m.params_.index)}
for name, a, b_ in [('60-74 x period', 'x1_late', 'x1_early'), ('>=75 x period', 'x2_late', 'x2_early')]:
    i, j = pos[a], pos[b_]
    diff = float(m.params_.iloc[i] - m.params_.iloc[j])
    se = float(np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j]))
    chi2 = (diff / se) ** 2
    P(f"  {name}: beta_diff={diff:+.3f} SE={se:.3f} chi2={chi2:.2f} P={stats.chi2.sf(chi2, 1):.4f} (1 df)")
    rows_int.append(dict(period='interaction', contrast=name, hr=np.nan, lo=np.nan, hi=np.nan,
                         p=float(stats.chi2.sf(chi2, 1))))

O = pd.concat([O, pd.DataFrame(rows_int)], ignore_index=True)
O.to_csv(os.path.join(OUT, "tableS1_ph_period_v18.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s91_ph_v18.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE tableS1_ph_period_v18.csv + s91_ph_v18.txt")
