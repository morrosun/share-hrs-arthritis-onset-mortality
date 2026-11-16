# -*- coding: utf-8 -*-
"""Step 18 (v2): SHARE 发病队列重做
关键发现：radiagarthr 基本只在首次访谈时被问到（27,504 中有自报诊断年龄者几乎都是基线已患病者），
17,874 名随访新报告者里仅 194 人有该字段。
故发病队列改用「首次阴性报告与首次阳性报告之间的区间中点」定义起病年龄（区间 2–4 年），
入组时点=区间中点（两端访谈均存活，故中点必然存活，可作延迟进入）。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S)
s['age'] = s['yr'] - s['rabyear']
s = s.sort_values(['pid', 'wave'])

rows = []
for pid, x in s.groupby('pid'):
    a = x.dropna(subset=['arthre'])
    if a.empty:
        continue
    fp = a[a['arthre'] == 1]
    if fp.empty:
        continue
    ip = a.index.get_loc(fp.index[0])
    prior = a.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    if neg.empty:
        continue                                  # 基线已患病 → 不属于发病队列
    rows.append(dict(pid=pid,
                     lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     dg=fp['radiagarthr'].iloc[0],
                     last=a['age'].iloc[-1], dead=a['radyear'].max(),
                     byear=a['rabyear'].iloc[0], sex=a['ragender'].iloc[0],
                     edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))

I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['event'] = I['dead'].notna().astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['width'] = I['pos'] - I['lastneg']

P("="*78 + "\n# A. 发病队列构造\n" + "="*78)
P(f"首访阴性、其后报告关节炎者: {len(I):,} 人")
P(f"  报告区间宽度（两次访谈间隔）: 中位 {I['width'].median():.0f} 年（IQR {I['width'].quantile(.25):.0f}–{I['width'].quantile(.75):.0f}）")
P(f"  其中自报诊断年龄可得: {int(I['dg'].notna().sum()):,} 人 —— 佐证 radiagarthr 主要只在基线问询")
P(f"  区间中点定义的起病年龄: 中位 {I['onset'].median():.0f}（IQR {I['onset'].quantile(.25):.0f}–{I['onset'].quantile(.75):.0f}）")

W = I[(I['exit'] > I['onset']) & (I['onset'] >= 40)].copy()
W['grp'] = pd.cut(W['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
W = W.dropna(subset=['grp'])
W['male'] = (W['sex'] == 1).astype(float)
W['smoke_ever'] = (W['smoke'] >= 1).astype(float)
W['py'] = W['exit'] - W['onset']
P(f"\n分析样本: {len(W):,} 人，死亡 {int(W['event'].sum()):,}，人-年 {W['py'].sum():,.0f}")
P(W.groupby('grp', observed=True).agg(n=('pid','size'), deaths=('event','sum'),
   py=('py','sum'), entry=('onset','mean'), exit=('exit','mean')).round(1).to_string())
P("\n[各粗死亡率 /1000 人年]")
for k, x in W.groupby('grp', observed=True):
    r = 1000 * x['event'].sum() / x['py'].sum()
    P(f"  {k:6s} n={len(x):5,} deaths={int(x['event'].sum()):4,} rate={r:6.1f}")

res = []
def run(df, onset_col, label):
    df = df.copy()
    df['g74'] = (df['grp'] == '60-74').astype(float)
    df['g75'] = (df['grp'] == '>=75').astype(float)
    D = df[['exit', onset_col, 'event', 'g74', 'g75', 'male', 'edu', 'smoke_ever']].dropna()
    D = D[D['exit'] > D[onset_col]]
    m = CoxPHFitter()
    m.fit(D, duration_col='exit', event_col='event', entry_col=onset_col)
    P(f"\n--- {label}  n={len(D):,}  deaths={int(D['event'].sum()):,} ---")
    for k, key in [('60-74', 'g74'), ('>=75', 'g75')]:
        q = m.summary.loc[key]
        P(f"  {k:6s} {q['exp(coef)']:.2f} ({q['exp(coef) lower 95%']:.2f}-{q['exp(coef) upper 95%']:.2f})  p={q['p']:.3g}")
        res.append(dict(design=label, grp=k, hr=q['exp(coef)'], lo=q['exp(coef) lower 95%'],
                        hi=q['exp(coef) upper 95%'], p=q['p'], n=len(D), events=int(D['event'].sum())))

run(W, 'onset', 'Incident cohort: onset = interval midpoint')

for lab, shift in [('lower bound (last negative report)', -0.5),
                   ('upper bound (first positive report)', 0.5)]:
    W2 = W.copy()
    W2['onset_s'] = W2['onset'] + shift * W2['width']
    W2['grp'] = pd.cut(W2['onset_s'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
    W2 = W2.dropna(subset=['grp'])
    run(W2, 'onset_s', f'Incident cohort: onset = {lab}')

pd.DataFrame(res).to_csv(os.path.join(OUT, "s18_incident.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s18_incident.txt"), 'w', encoding='utf-8').write("\n".join(lines))
