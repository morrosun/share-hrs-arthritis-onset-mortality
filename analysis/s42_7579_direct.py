# -*- coding: utf-8 -*-
"""Step 42: 第三轮评审要求——把功能分析统一限定在 75–79 岁，直接比较各起病层。

- 暴露与状态定义与 s34（v5）完全一致
- 年龄以「以年计的连续 attained age」进入模型（不是每个单岁一个哑变量）
- 同时拟合含二次年龄项的模型，检查非线性是否改变组间结论
- 报告每组的 person-waves、人数、单岁年龄分布

输出 out/table2_7579_v5.csv、out/s42_7579.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
COM = ['hibpe', 'diabe', 'hearte', 'stroke', 'cancre', 'lunge']
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def gof(o):
    return '<60' if o < 60 else ('60-74' if o < 75 else '>=75')


# ---------------- HRS v5 暴露 ----------------
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = h[COM].fillna(0).sum(axis=1)
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
pers = {}
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        pers[pid] = ('never', np.nan, np.nan); continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]; neg = neg[neg['arthre'] == 0]
    if neg.empty:
        pers[pid] = ('prevalent', np.nan, np.nan); continue
    mid = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2
    pers[pid] = ('on', mid, fp['age'].iloc[0])


def hrs_state(row):
    pat, mid, pos = pers[row['pid']]
    if pat == 'never': return 'never'
    if pat == 'prevalent': return 'prevalent'
    return 'never' if row['age'] < pos else 'on_' + gof(mid)


h['state'] = h.apply(hrs_state, axis=1)
hh = h[(h['age'] >= 45) & (h['age'] <= 105)].copy()
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
hh['state'] = pd.Categorical(hh['state'], ['never', 'prevalent', 'on_<60', 'on_60-74', 'on_>=75'])

# ---------------- SHARE（基线自报诊断年龄）----------------
s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = s[COM].fillna(0).sum(axis=1)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
sd = {}
for pid, x in s.groupby('pid'):
    dg = x['radiagarthr'].dropna(); dg = dg[dg >= 5]
    sd[pid] = ('on_' + gof(dg.iloc[0])) if not dg.empty else 'never'
ss = s.copy(); ss['state'] = ss['pid'].map(sd).fillna('never')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)].copy()
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss['state'] = pd.Categorical(ss['state'], ['never', 'on_<60', 'on_60-74', 'on_>=75'])


def difftest(res, k1, k2, kind):
    b = res.params[k1] - res.params[k2]
    V = res.cov_params()
    se = np.sqrt(V.loc[k1, k1] + V.loc[k2, k2] - 2 * V.loc[k1, k2])
    p = 2 * (1 - st.norm.cdf(abs(b / se)))
    if kind == 'beta':
        return b, b - 1.96 * se, b + 1.96 * se, p
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


STATES = {'HRS': ['prevalent', 'on_<60', 'on_60-74', 'on_>=75'],
          'SHARE': ['on_<60', 'on_60-74', 'on_>=75']}
out = []
P("=" * 84)
P("# 75–79 岁共同年龄范围内的直接比较（v5）")
P("=" * 84)
P("年龄项：以年计的连续 attained age（线性）；quadratic 模型另加 (age-77.5)^2。")


def run(data, outcome, lib, kind, label):
    d = data[(data['age'] >= 75) & (data['age'] < 80)].dropna(subset=[outcome]).copy()
    d['age2'] = (d['age'] - 77.5) ** 2
    P(f"\n--- {lib} | {label} | 75-79: waves={len(d):,} persons={d['pid'].nunique():,} ---")
    dist = d.assign(single=np.floor(d['age']).astype(int)).groupby(
        ['state', 'single'], observed=True).size().unstack(fill_value=0)
    P("  单岁 person-waves 分布（state × 单岁年龄）：")
    P("    " + dist.to_string().replace("\n", "\n    "))
    for stt in STATES[lib]:
        q = d[d['state'] == stt]
        P(f"    {stt:11s} waves={len(q):,} persons={q['pid'].nunique():,} "
          f"mean age={q['age'].mean():.2f}")
    for model_tag, rhs in [('linear', 'C(state) + C(edu_c) + male + age'),
                           ('quadratic', 'C(state) + C(edu_c) + male + age + age2')]:
        try:
            if kind == 'beta':
                m = smf.mixedlm(f"{outcome} ~ {rhs}", d, groups=d['pid'].values).fit(reml=False, method='lbfgs')
            else:
                m = smf.gee(f"{outcome} ~ {rhs}", groups='pid', data=d,
                            family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
        except Exception as ex:
            P(f"  [{model_tag}] 拟合失败: {str(ex)[:90]}")
            continue
        P(f"  [{model_tag}]")
        for stt in STATES[lib]:
            k = f"C(state)[T.{stt}]"
            c, se, p = m.params[k], m.bse[k], m.pvalues[k]
            est, lo, hi = ((c, c - 1.96 * se, c + 1.96 * se) if kind == 'beta'
                           else (np.exp(c), np.exp(c - 1.96 * se), np.exp(c + 1.96 * se)))
            P(f"    vs never  {stt:11s} {est:.3f} ({lo:.3f}-{hi:.3f})  p={p:.3g}")
            out.append(dict(lib=lib, outcome=label, kind=kind, model=model_tag, term=stt,
                            est=round(est, 4), lo=round(lo, 4), hi=round(hi, 4), p=float(f"{p:.3g}"),
                            waves=len(d), persons=d['pid'].nunique()))
        for a, b in [('on_60-74', 'on_<60'), ('on_>=75', 'on_<60'), ('on_>=75', 'on_60-74')]:
            try:
                est, lo, hi, p = difftest(m, f"C(state)[T.{a}]", f"C(state)[T.{b}]", kind)
                lab = f"{a.replace('on_', '')} vs {b.replace('on_', '')}"
                P(f"    diff      {lab:18s} {est:+.3f} ({lo:+.3f} to {hi:+.3f})  p={p:.3g}")
                out.append(dict(lib=lib, outcome=label, kind=kind, model=model_tag, term=lab,
                                est=round(est, 4), lo=round(lo, 4), hi=round(hi, 4), p=float(f"{p:.3g}"),
                                waves=len(d), persons=d['pid'].nunique()))
            except Exception as ex:
                P(f"    diff {a} vs {b} 失败: {str(ex)[:60]}")


run(hh, 'adl6a', 'HRS', 'beta', 'ADL limitations (0-6)')
run(hh, 'iadl5a', 'HRS', 'beta', 'IADL limitations (0-5)')
run(hh, 'comorb', 'HRS', 'ratio', 'Comorbidity count (0-6)')
run(ss, 'mob', 'SHARE', 'ratio', 'Mobility limitations (0-2)')
run(ss, 'comorb', 'SHARE', 'ratio', 'Comorbidity count (0-6)')

pd.DataFrame(out).to_csv(os.path.join(OUT, "table2_7579_v5.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s42_7579.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 out/table2_7579_v5.csv 与 out/s42_7579.txt")
