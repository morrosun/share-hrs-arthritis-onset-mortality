# -*- coding: utf-8 -*-
"""Step 16: 表 3 重做 —— 四列，使「只换时间尺度」这一对比无混淆。
1 随访年、无左截断、不校正
2 随访年、校正入组年龄
3 随访年、校正入组年龄 + 性别/教育/吸烟
4 年龄尺度、左截断、校正性别/教育/吸烟
3 与 4 之间只差时间尺度（与「左截断」），这是本文的核心演示。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

p = pd.read_csv(f"{OUT}\\analysis_share.csv")
p['grp'] = p['grp'].astype(str).str.strip()
p['event'] = p['event'].astype(int)
p['male'] = (p['sex'] == 1).astype(float)
p['smoke_ever'] = (p['smoke'] >= 1).astype(float)
p['g_60_74'] = (p['grp'] == '60-74').astype(float)
p['g_ge75'] = (p['grp'] == '>=75').astype(float)
p['fu'] = (p['exit_age'] - p['entry_age']).clip(lower=0.1)
D = p.dropna(subset=['fu','event','entry_age','male','edu','smoke_ever']).copy()
D['entry_c'] = D['entry_age'] - D['entry_age'].mean()
P(f"队列 n={len(D)}  deaths={int(D['event'].sum())}  随访均值 {D['fu'].mean():.1f} 年")

def fit(dur, entry, formula):
    m = CoxPHFitter()
    kw = dict(duration_col=dur, event_col='event', formula=formula)
    if entry:
        kw['entry_col'] = entry
    m.fit(D, **kw)
    return m

SPECS = [
    ("1. Follow-up time scale, unadjusted", 'fu', None, "g_60_74 + g_ge75"),
    ("2. Follow-up time scale, adjusted for age at entry", 'fu', None, "g_60_74 + g_ge75 + entry_c"),
    ("3. Follow-up time scale, fully adjusted", 'fu', None,
     "g_60_74 + g_ge75 + entry_c + male + edu + smoke_ever"),
    ("4. Age time scale, left-truncated, fully adjusted", 'exit_age', 'entry_age',
     "g_60_74 + g_ge75 + male + edu + smoke_ever"),
]

rows = []
for lab, dur, entry, form in SPECS:
    m = fit(dur, entry, form)
    b = np.array([m.summary.loc[k, 'coef'] for k in ['g_60_74', 'g_ge75']])
    V = m.variance_matrix_.loc[['g_60_74', 'g_ge75'], ['g_60_74', 'g_ge75']].values
    chi2 = float(b @ np.linalg.inv(V) @ b)
    pv = float(stats.chi2.sf(chi2, 2))
    P(f"\n--- {lab} ---  梯度 chi2={chi2:.1f}, df=2, P={pv:.3g}")
    for k, key in [('60-74', 'g_60_74'), ('>=75', 'g_ge75')]:
        s = m.summary.loc[key]
        rows.append(dict(model=lab, grp=k, hr=s['exp(coef)'],
                         lo=s['exp(coef) lower 95%'], hi=s['exp(coef) upper 95%'],
                         chi2=round(chi2, 1), p_grad=pv))
        P(f"   {k}: {s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f})")

R = pd.DataFrame(rows)
R['txt'] = R.apply(lambda r: f"{r['hr']:.2f} ({r['lo']:.2f}\u2013{r['hi']:.2f})", axis=1)
R.to_csv(f"{OUT}\\table3_naive.csv", index=False, encoding='utf-8-sig')
P("\n" + R.pivot(index='grp', columns='model', values='txt').to_string())
io.open(f"{OUT}\\s16_table3.txt", 'w', encoding='utf-8').write("\n".join(lines))
