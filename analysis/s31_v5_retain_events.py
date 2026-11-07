# -*- coding: utf-8 -*-
"""Step 31: v5 发病队列（入组=首次阳性）中，exit<=entry 的人如何处理的敏感性分析。
方案 A（v5 原口径）：剔除 exit<=entry。
方案 B（保留事件）：exit<=entry 且已死亡者保留，exit 设为 entry+0.5 年（同年内死亡的事件不丢失）；
      exit<=entry 且存活者仍剔除（零时间，对偏似然无贡献）。
另报告：剔除者的年龄组构成与死亡数，以及方案 B 的 Cox 结果。
输出 out/s31_v5_retain_events.txt
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
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def build(df, wave_map, sex_col='ragender'):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map)
    df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rows = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty: continue
        rows.append(dict(pid=pid, pos=fp['age'].iloc[0], last=x['age'].iloc[-1],
                         dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                         sex=x[sex_col].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0],
                         lastneg=neg['age'].iloc[-1]))
    I = pd.DataFrame(rows)
    I['event'] = I['dead'].notna().astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['male'] = (I['sex'] == 1).astype(float)
    I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
    I['grp'] = pd.cut(I['lastneg'] / 2 + I['pos'] / 2, [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
    return I.dropna(subset=['grp', 'edu'])

def cox(D, label):
    D = D.copy()
    D['g74'] = (D['grp'] == '60-74').astype(float)
    D['g75'] = (D['grp'] == '>=75').astype(float)
    m = CoxPHFitter()
    m.fit(D[['exit', 'entry', 'event', 'g74', 'g75', 'male', 'edu', 'smoke_ever']],
          duration_col='exit', event_col='event', entry_col='entry')
    P(f"\n  [{label}] n={len(D):,} deaths={int(D['event'].sum()):,}")
    for k, key in [('60-74', 'g74'), ('>=75', 'g75')]:
        q = m.summary.loc[key]
        P(f"    {k} vs <60: HR {q['exp(coef)']:.2f} ({q['exp(coef) lower 95%']:.2f}-{q['exp(coef) upper 95%']:.2f}) p={q['p']:.3g}")

def run(I, name):
    P("=" * 78 + f"\n# {name}\n" + "=" * 78)
    I = I.copy()
    I['entry'] = I['pos']
    excl = I[I['exit'] <= I['entry']]
    P(f"总人数（首次阳性入组前）: {len(I):,}；exit<=entry: {len(excl):,}（其中死亡 {int(excl['event'].sum())}）")
    P("exit<=entry 按起病层分布:")
    P(excl.groupby('grp', observed=True).agg(n=('pid', 'size'), deaths=('event', 'sum')).to_string())
    # 方案 A
    A = I[I['exit'] > I['entry']]
    cox(A, 'A: 剔除 exit<=entry（v5 原口径）')
    # 方案 B：保留同年死亡事件
    B = I.copy()
    fix = (B['exit'] <= B['entry']) & (B['event'] == 1)
    B.loc[fix, 'exit'] = B.loc[fix, 'entry'] + 0.5
    B = B[B['exit'] > B['entry']]
    cox(B, 'B: 保留 exit<=entry 的死亡事件（exit=entry+0.5）')
    P("")

s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
run(build(s, WY_S), 'SHARE 发病队列')

h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
run(build(h, WY_H), 'HRS 发病队列')

io.open(f"{OUT}\\s31_v5_retain_events.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("写出 s31_v5_retain_events.txt")
