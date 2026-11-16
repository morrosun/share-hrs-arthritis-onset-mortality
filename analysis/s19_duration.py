# -*- coding: utf-8 -*-
"""Step 19: 病程 vs 起病年龄的可分性检验 + HRS 暴露重做
(A) SHARE 发病队列：把所有人限制在「诊断后 3 年内」，此时病程近似相同 → 再看起病年龄是否还有差异。
(B) HRS：区分基线已患病（起病年龄未知）与随访新报告（可定区间），各为一类；
    新报告者用区间中点定起病年龄，并同样做 3 年窗检验。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

# ==================== A. SHARE 发病队列 + 3 年窗 ====================
P("="*78 + "\n# A. SHARE 发病队列：全程 vs 诊断后 3 年窗\n" + "="*78)
s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s = s.sort_values(['pid', 'wave'])
rows = []
for pid, x in s.groupby('pid'):
    a = x.dropna(subset=['arthre'])
    if a.empty: continue
    fp = a[a['arthre'] == 1]
    if fp.empty: continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    if neg.empty: continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=a['age'].iloc[-1], dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                     sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['width'] = I['pos'] - I['lastneg']
I['event'] = I['dead'].notna().astype(int)
I['exit_full'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['male'] = (I['sex'] == 1).astype(float)
I['smoke_ever'] = (I['smoke'] >= 1).astype(float)

def fit_window(df, exit_col, lab):
    df = df.copy()
    df['grp'] = pd.cut(df['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
    df = df.dropna(subset=['grp'])
    df['g74'] = (df['grp'] == '60-74').astype(float)
    df['g75'] = (df['grp'] == '>=75').astype(float)
    D = df[['exit', 'onset', 'event', 'g74', 'g75', 'male', 'edu', 'smoke_ever']].dropna()
    D = D[D['exit'] > D['onset']]
    m = CoxPHFitter()
    m.fit(D, duration_col='exit', event_col='event', entry_col='onset')
    P(f"\n--- {lab}: n={len(D):,} deaths={int(D['event'].sum()):,} ---")
    for k, key in [('60-74','g74'), ('>=75','g75')]:
        q = m.summary.loc[key]
        P(f"  {k:6s} {q['exp(coef)']:.2f} ({q['exp(coef) lower 95%']:.2f}"
          f"-{q['exp(coef) upper 95%']:.2f})  p={q['p']:.3g}")
    return m

I['exit'] = I['exit_full']
fit_window(I, 'exit', 'SHARE incident, full follow-up')

# 3 年窗：所有人右删失于 onset+3
for w in [3, 5]:
    J = I.copy()
    J['exit'] = np.minimum(J['exit'], J['onset'] + w)
    J['event'] = np.where(J['exit'] < J['exit_full'] - 1e-9, 0, J['event'])
    fit_window(J, 'exit', f'SHARE incident, censored at onset+{w} y')

# 描述：诊断后前 3 年的粗死亡率
P("\n[诊断后 3 年内粗死亡率 /1000 人年]")
J = I.copy()
J['exit'] = np.minimum(J['exit'], J['onset'] + 3)
J['event'] = np.where(J['exit'] < J['exit_full'] - 1e-9, 0, J['event'])
J = J[(J['exit'] > J['onset']) & (J['onset'] >= 40)]
J['grp'] = pd.cut(J['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
for k, x in J.dropna(subset=['grp']).groupby('grp', observed=True):
    py = (x['exit'] - x['onset']).sum()
    P(f"  {k:6s} n={len(x):5,} deaths={int(x['event'].sum()):4,} py={py:8.0f} rate={1000*x['event'].sum()/py:6.1f}")

# ==================== B. HRS ====================
P("\n\n" + "="*78 + "\n# B. HRS：基线已患病 vs 随访新报告\n" + "="*78)
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    a = x
    fp = a[a['arthre'] == 1]
    if fp.empty:
        rows.append(dict(pid=pid, pattern='never', onset=np.nan, pos=np.nan, lastneg=np.nan,
                         entry=a['age'].iloc[0], last=a['age'].iloc[-1],
                         dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                         sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
        continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    pat = 'prevalent' if neg.empty else 'incident'
    rows.append(dict(pid=pid, pattern=pat,
                     onset=(neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2 if pat == 'incident' else np.nan,
                     pos=fp['age'].iloc[0],
                     lastneg=neg['age'].iloc[-1] if pat == 'incident' else np.nan,
                     entry=a['age'].iloc[0], last=a['age'].iloc[-1],
                     dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                     sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
H = pd.DataFrame(rows)
H['event'] = H['dead'].notna().astype(int)
H['exit'] = np.where(H['event'] == 1, H['dead'] - H['byear'], H['last'])
P(H['pattern'].value_counts().to_string())
P("\n[基线已患病者的入组年龄分布 —— 他们的起病年龄其实未知]")
pv = H[H['pattern'] == 'prevalent']
P(f"  n={len(pv):,}，入组年龄中位 {pv['entry'].median():.0f}（IQR {pv['entry'].quantile(.25):.0f}–{pv['entry'].quantile(.75):.0f}）")
P(f"  其中入组时已 {int((pv['entry']>=60).sum()):,} 人 ≥60 岁、{int((pv['entry']>=75).sum()):,} 人 ≥75 岁")
P("  → 旧做法把他们按「首次观察到报告的年龄」分入 60-74 / ≥75，属实质误分类")
inc = H[H['pattern'] == 'incident'].copy()
P(f"\n随访新报告者 n={len(inc):,}，区间宽度中位 {(inc['pos']-inc['lastneg']).median():.0f} 年")
P(f"  起病年龄（区间中点）中位 {inc['onset'].median():.0f}（IQR {inc['onset'].quantile(.25):.0f}–{inc['onset'].quantile(.75):.0f}）")
inc['male'] = (inc['sex'] == 1).astype(float)
inc['smoke_ever'] = (inc['smoke'] >= 1).astype(float)

def fit_h(df, exit_col, lab):
    df = df.copy()
    df['grp'] = pd.cut(df['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
    df = df.dropna(subset=['grp'])
    df['g74'] = (df['grp'] == '60-74').astype(float)
    df['g75'] = (df['grp'] == '>=75').astype(float)
    D = df[['exit', 'onset', 'event', 'g74', 'g75', 'male', 'edu', 'smoke_ever']].dropna()
    D = D[D['exit'] > D['onset']]
    m = CoxPHFitter()
    m.fit(D, duration_col='exit', event_col='event', entry_col='onset')
    P(f"\n--- {lab}: n={len(D):,} deaths={int(D['event'].sum()):,} ---")
    P(m.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(3).to_string())

P("\n[HRS 发病队列（延迟进入=区间中点）]")
fit_h(inc, 'exit', 'HRS incident, full follow-up')
for w in [3, 5]:
    J = inc.copy()
    J['exit'] = np.minimum(J['exit'], J['onset'] + w)
    J['event'] = np.where(J['exit'] < inc['exit'] - 1e-9, 0, inc['event'])
    fit_h(J, 'exit', f'HRS incident, censored at onset+{w} y')

io.open(os.path.join(OUT, "s19_duration.txt"), 'w', encoding='utf-8').write("\n".join(lines))
