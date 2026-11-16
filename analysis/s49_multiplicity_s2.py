# -*- coding: utf-8 -*-
"""Step 49: Table S2 炎症层的多重检验校正。

检验族（预先设定，共 7 个推断）：
  ELSA, log(WBC) ~ 起病年龄：连续每 +10 岁 (1)；分类模型 3 个对比 (<45 / 60-74 / >=75 vs 45-59) (3)
  CHARLS 2011, 关节炎状态：log(CRP) / WBC / 尿酸 (3)
校正：Holm-Bonferroni (FWER) 与 Benjamini-Hochberg (FDR)，
      分别按「全族 7 个」与「按队列分族（ELSA 4 / CHARLS 3）」计算。
输出 out/tableS2_multiplicity_v5.csv、out/s49_multiplicity.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.formula.api as smf

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_E = {1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010, 6: 2012, 7: 2014,
        8: 2016, 9: 2018, 89: 2017}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def load(db):
    d = pyreadr.read_r(os.path.join(BASE, f"{db}.RData"))[db].copy()
    return d.rename(columns={d.columns[0]: 'pid'})

rows = []

# ==================== ELSA ====================
e = load('elsa')
e['pid'] = e['pid'].astype(str)
e['yr'] = e['wave'].map(WY_E)
core = load('elsa_raw_core')
w = core[core['concept'] == 'wbc'].copy()
w['pid'] = w['pid'].astype(str)
w = w.rename(columns={'value': 'wbc'})[['pid', 'wave', 'wbc']]
w['yr'] = w['wave'].map(WY_E)

dg = (e[e['arthre'] == 1].dropna(subset=['radiagarthr'])
      .assign(radiagarthr=lambda x: x['radiagarthr'].where(x['radiagarthr'] >= 5))
      .dropna(subset=['radiagarthr'])
      .groupby('pid').agg(onset_age=('radiagarthr', 'first'), sex=('ragender', 'first')))
m = w.merge(dg, left_on='pid', right_index=True, how='inner')
m = m[m['yr'].notna()]
m['age_at_wbc'] = m['yr'] - e.groupby('pid')['rabyear'].first().reindex(m['pid']).values
m = m[m['age_at_wbc'] >= m['onset_age']]
m = m.sort_values(['pid', 'yr']).groupby('pid').first().reset_index()
m['logwbc'] = np.log(m['wbc'])
m['male'] = (m['sex'] == 1).astype(float)
D = m.dropna(subset=['logwbc', 'age_at_wbc', 'male']).copy()
D['grp'] = pd.cut(D['onset_age'], [4, 44, 59, 74, 120], labels=['<45', '45-59', '60-74', '>=75'])
D['onset10'] = D['onset_age'] / 10
P("=" * 84 + "\n# ELSA: log(WBC) 与起病年龄\n" + "=" * 84)
P(f"进入模型 n={len(D):,}")

mod_c = smf.ols("logwbc ~ onset10 + age_at_wbc + male", data=D).fit(cov_type='HC1')
q = mod_c.params['onset10']
rows.append(dict(cohort='ELSA', family='ELSA', test='WBC, per 10 years later onset', n=len(D),
                 estimate=100 * (np.exp(q) - 1),
                 lo=100 * (np.exp(q - 1.96 * mod_c.bse['onset10']) - 1),
                 hi=100 * (np.exp(q + 1.96 * mod_c.bse['onset10']) - 1),
                 p_raw=float(mod_c.pvalues['onset10'])))
P(f"  连续每 +10 岁: {rows[-1]['estimate']:+.2f}%  p={rows[-1]['p_raw']:.6f}")

mod_g = smf.ols("logwbc ~ C(grp, Treatment('45-59')) + age_at_wbc + male", data=D).fit(cov_type='HC1')
for g in ['<45', '60-74', '>=75']:
    key = f"C(grp, Treatment('45-59'))[T.{g}]"
    if key not in mod_g.params.index:
        P(f"  {g}: 缺失"); continue
    b = mod_g.params[key]
    rows.append(dict(cohort='ELSA', family='ELSA', test=f'WBC, {g} vs 45-59', n=len(D),
                     estimate=100 * (np.exp(b) - 1),
                     lo=100 * (np.exp(b - 1.96 * mod_g.bse[key]) - 1),
                     hi=100 * (np.exp(b + 1.96 * mod_g.bse[key]) - 1),
                     p_raw=float(mod_g.pvalues[key])))
    P(f"  {g:6s} vs 45-59: {rows[-1]['estimate']:+.2f}%  p={rows[-1]['p_raw']:.6f}")

# 分类模型的总体 2-df/3-df Wald 检验（作为额外的族定义参考）
try:
    wt = mod_g.wald_test(" = 0, ".join(
        [f"C(grp, Treatment('45-59'))[T.{g}]" for g in ['<45', '60-74', '>=75']]) + " = 0",
        scalar=True)
    P(f"  分类模型总体 Wald: F/chi2={float(wt.statistic):.3f}, p={float(wt.pvalue):.6f}")
except Exception as ex:
    P(f"  总体 Wald 检验跳过: {str(ex)[:80]}")

# ==================== CHARLS ====================
c = load('charls')
c['yr'] = c['wave'].map({1: 2011, 2: 2013, 3: 2015, 4: 2018, 5: 2020})
c['age'] = c['yr'] - c['rabyear']
c11 = c[(c['wave'] == 1) & c['arthre'].notna()].copy()
c11['male'] = (c11['ragender'] == 1).astype(float)
c11['edu_c'] = pd.Categorical(c11['raeducl'], [1, 2, 3], ordered=True)
c11['logcrp'] = np.log(c11['bl_crp'].where(c11['bl_crp'] > 0))
P("\n" + "=" * 84 + "\n# CHARLS 2011: 关节炎与生物标志物\n" + "=" * 84)

mm = c11.dropna(subset=['logcrp', 'age', 'male', 'edu_c'])
mod = smf.ols("logcrp ~ arthre + age + male + C(edu_c)", data=mm).fit(cov_type='HC1')
b = mod.params['arthre']
rows.append(dict(cohort='CHARLS', family='CHARLS', test='CRP, arthritis vs none', n=len(mm),
                 estimate=100 * (np.exp(b) - 1),
                 lo=100 * (np.exp(b - 1.96 * mod.bse['arthre']) - 1),
                 hi=100 * (np.exp(b + 1.96 * mod.bse['arthre']) - 1),
                 p_raw=float(mod.pvalues['arthre'])))
P(f"  CRP: +{rows[-1]['estimate']:.2f}%  n={len(mm):,}  p={rows[-1]['p_raw']:.6f}")

for v, lab in [('bl_wbc', 'WBC'), ('bl_ua', 'Uric acid')]:
    mm2 = c11.dropna(subset=[v, 'age', 'male'])
    m2 = smf.ols(f"{v} ~ arthre + age + male", data=mm2).fit(cov_type='HC1')
    b2 = m2.params['arthre']
    rows.append(dict(cohort='CHARLS', family='CHARLS', test=f'{lab}, arthritis vs none', n=len(mm2),
                     estimate=float(b2), lo=float(b2 - 1.96 * m2.bse['arthre']),
                     hi=float(b2 + 1.96 * m2.bse['arthre']), p_raw=float(m2.pvalues['arthre'])))
    P(f"  {lab:10s}: {b2:+.3f}  n={len(mm2):,}  p={rows[-1]['p_raw']:.6f}")

# ==================== 校正 ====================
def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 0.0
    for i, j in enumerate(order):
        prev = max(prev, min(1.0, (n - i) * p[j]))
        adj[j] = prev
    return adj

def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        j = order[i]
        prev = min(prev, p[j] * n / (i + 1))
        q[j] = min(1.0, prev)
    return q

R = pd.DataFrame(rows)
R['p_holm_all'] = holm(R['p_raw'].values)
R['q_bh_all'] = bh(R['p_raw'].values)
R['p_holm_family'] = np.nan
R['q_bh_family'] = np.nan
for fam, d in R.groupby('family'):
    R.loc[d.index, 'p_holm_family'] = holm(d['p_raw'].values)
    R.loc[d.index, 'q_bh_family'] = bh(d['p_raw'].values)

P("\n" + "=" * 84 + "\n# 校正结果\n" + "=" * 84)
P(R[['cohort', 'test', 'n', 'estimate', 'p_raw', 'p_holm_all', 'q_bh_all',
     'p_holm_family', 'q_bh_family']].round(5).to_string(index=False))
P(f"\n全族检验数 = {len(R)}；ELSA {int((R.family=='ELSA').sum())} 个，CHARLS {int((R.family=='CHARLS').sum())} 个")
P(f"原始 P<0.05 的检验数 = {int((R.p_raw<0.05).sum())}；"
  f"Holm 全族校正后 P<0.05 = {int((R.p_holm_all<0.05).sum())}；"
  f"BH 全族 q<0.05 = {int((R.q_bh_all<0.05).sum())}")
P(f"按队列分族：Holm 后 P<0.05 = {int((R.p_holm_family<0.05).sum())}；"
  f"BH 后 q<0.05 = {int((R.q_bh_family<0.05).sum())}")

R.to_csv(os.path.join(OUT, "tableS2_multiplicity_v5.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s49_multiplicity.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 tableS2_multiplicity_v5.csv / s49_multiplicity.txt")
print("DONE")
