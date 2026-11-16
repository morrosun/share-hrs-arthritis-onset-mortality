# -*- coding: utf-8 -*-
"""Step 27: SHARE 中选择偏倚核查
47,900 名曾报告关节炎者中，27,264 有可用的自报诊断年龄，20,636 没有。
比较两组在入组年龄、随访长度、死亡比例上的差异，判断"有诊断年龄者"是否为一个选择性样本。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s = s.sort_values(['pid', 'wave'])

rows = []
for pid, x in s.groupby('pid'):
    ever = (x['arthre'] == 1).any()
    if not ever:
        continue
    dg = x['radiagarthr'].dropna()
    dg = dg[dg >= 5]
    rows.append(dict(pid=pid,
                     has_dg=int(not dg.empty),
                     onset=float(dg.iloc[0]) if not dg.empty else np.nan,
                     entry=x['age'].iloc[0], last=x['age'].iloc[-1],
                     dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                     sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0],
                     nwave=len(x), first_wave=x['wave'].iloc[0]))
D = pd.DataFrame(rows)
D['event'] = D['dead'].notna().astype(int)
D['exit'] = np.where(D['event'] == 1, D['dead'] - D['byear'], D['last'])
D['py'] = (D['exit'] - D['entry']).clip(lower=0)
D = D[(D['exit'] > D['entry']) & (D['entry'] >= 40)]

P("=" * 76 + "\n# SHARE 曾报告关节炎者：有 vs 无可用诊断年龄\n" + "=" * 76)
P(f"总人数 {len(D):,}")
g = D.groupby('has_dg').agg(n=('pid', 'size'), female=('sex', lambda x: 100 * (x == 0).mean()),
                            entry_age=('entry', 'median'), follow=('py', 'median'),
                            waves=('nwave', 'median'), first_wave=('first_wave', 'median'),
                            deaths=('event', 'sum'), py=('py', 'sum'))
g['death_pct'] = 100 * g['deaths'] / g['n']
g['rate'] = 1000 * g['deaths'] / g['py']
g.index = ['no usable onset age', 'usable onset age']
P(g.round(1).to_string())

P("\n[按入组波次分层 —— 判断缺失是否与入组时期相关]")
P(D.groupby(['first_wave', 'has_dg']).size().unstack(fill_value=0).to_string())
P("\n[按入组年龄分层 —— 是否有可用诊断年龄]")
D['eb'] = pd.cut(D['entry'], [40, 55, 65, 75, 85, 110])
P(D.groupby('eb', observed=True)['has_dg'].agg(['size', 'mean']).round(3).to_string())

P("\n[结论判据] 若两组死亡比例与入组年龄接近，则『有诊断年龄』主要反映问卷可得性而非健康选择。")
io.open(os.path.join(OUT, "s27_selection.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("DONE")
