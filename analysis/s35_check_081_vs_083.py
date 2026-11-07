# -*- coding: utf-8 -*-
"""Step 35: 核对 SHARE prevalent 全年龄 >=75 vs <60 的 0.81（Table 3 模型4，三组同时入模）
与 0.83（s33 pair()，仅两组入模）的差异来源。
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

s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
agg = s.groupby('pid').agg(entry=('age', 'min'), last=('age', 'max'), sex=('ragender', 'first'),
                           byear=('rabyear', 'first'), edu=('raeducl', 'first'), smoke=('smoken', 'max'))
dg = s.dropna(subset=['radiagarthr']).groupby('pid')['radiagarthr'].first().rename('onset')
dd = s.groupby('pid')['radyear'].max().rename('dead')
D = agg.join(dg).join(dd).dropna(subset=['onset'])
D = D[D['onset'] >= 5]
D['event'] = D['dead'].notna().astype(int)
D['exit'] = np.where(D['event'] == 1, D['dead'] - D['byear'], D['last'])
D = D[(D['exit'] > D['entry']) & (D['entry'] >= 40) & (D['exit'] <= 110)].copy()
D['male'] = (D['sex'] == 1).astype(float)
D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
D['grp'] = pd.cut(D['onset'], [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])
D = D.dropna(subset=['grp', 'edu']).reset_index()
P(f"n={len(D):,} deaths={int(D['event'].sum()):,}")

# (1) 三组同时入模，参照 <60（与 Table 3 模型4 一致）
X = D.copy()
X['g6074'] = (X['grp'] == '60-74').astype(float)
X['g75'] = (X['grp'] == '>=75').astype(float)
X = X[['entry', 'exit', 'event', 'g6074', 'g75', 'male', 'edu', 'smoke_ever']]
m = CoxPHFitter()
m.fit(X, duration_col='exit', event_col='event', entry_col='entry')
for k in ['g6074', 'g75']:
    q = m.summary.loc[k]
    P(f"三组入模 {k}: HR {q['exp(coef)']:.3f} ({q['exp(coef) lower 95%']:.3f}-{q['exp(coef) upper 95%']:.3f})")

# (2) 仅两组（<60 vs >=75）入模，即 s33 pair() 的口径
Y = D[D['grp'].isin(['<60', '>=75'])].copy()
Y['cmp'] = (Y['grp'] == '>=75').astype(float)
Y = Y[['entry', 'exit', 'event', 'cmp', 'male', 'edu', 'smoke_ever']]
m2 = CoxPHFitter()
m2.fit(Y, duration_col='exit', event_col='event', entry_col='entry')
q = m2.summary.loc['cmp']
P(f"两组入模 >=75 vs <60: HR {q['exp(coef)']:.3f} ({q['exp(coef) lower 95%']:.3f}-{q['exp(coef) upper 95%']:.3f})")

io.open(f"{OUT}\\s35_check_081_vs_083.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("写出 s35_check_081_vs_083.txt")
