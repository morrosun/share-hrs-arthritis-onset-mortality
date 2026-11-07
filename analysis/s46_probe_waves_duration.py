# -*- coding: utf-8 -*-
"""Step 46 (probe): 检查波次、诊断年龄可得率、病程变量的可计算性。

不产出正式结果，只为 s47/s48 的分析设计提供依据。
输出 out/s46_probe.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def load(db):
    d = pyreadr.read_r(f"{BASE}\\{db}.RData")[db].copy()
    return d.rename(columns={d.columns[0]: 'pid'})

s = load('share')
h = load('hrs')
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']

P("=" * 78 + "\n# SHARE 列名（与波次/诊断年龄/死亡相关）\n" + "=" * 78)
P([c for c in s.columns if any(k in c.lower() for k in
   ['wave', 'radiag', 'radyear', 'rabyear', 'arthre', 'age', 'country', 'ragender', 'raeducl', 'smoken'])])
P("\n[SHARE radiagarthr 非缺失按 wave]")
P(s.groupby('wave')['radiagarthr'].agg(['size', 'count']).assign(
    pct=lambda x: (100 * x['count'] / x['size']).round(1)).to_string())
P("\n[SHARE radiagarthr 有效值(>=5)按 wave]")
P(s.assign(ok=lambda d: d['radiagarthr'].where(d['radiagarthr'] >= 5)).groupby('wave')['ok']
   .agg(['size', 'count']).assign(pct=lambda x: (100 * x['count'] / x['size']).round(1)).to_string())

P("\n" + "=" * 78 + "\n# SHARE：按入组波次（首个 wave）的诊断年龄可得率\n" + "=" * 78)
first = (s.sort_values(['pid', 'wave']).groupby('pid')
         .agg(entry_wave=('wave', 'first'), entry_age=('age', 'first') if 'age' in s else ('rabyear', 'first'),
              n_waves=('wave', 'size')))
dg = (s.dropna(subset=['radiagarthr']).assign(v=lambda d: d['radiagarthr'].where(d['radiagarthr'] >= 5))
      .dropna(subset=['v']).groupby('pid')['v'].first().rename('onset_age'))
dg_wave = (s.dropna(subset=['radiagarthr']).assign(v=lambda d: d['radiagarthr'].where(d['radiagarthr'] >= 5))
           .dropna(subset=['v']).groupby('pid')['wave'].first().rename('dg_wave'))
first = first.join(dg).join(dg_wave)
first['has_dx'] = first['onset_age'].notna()
g = first.groupby('entry_wave').agg(n=('has_dx', 'size'), has_dx=('has_dx', 'sum'),
                                    onset_med=('onset_age', 'median'), n_waves=('n_waves', 'mean'))
g['pct'] = (100 * g['has_dx'] / g['n']).round(1)
P(g.round(2).to_string())
P(f"\n合计: {len(first):,} 人，有有效诊断年龄 {int(first['has_dx'].sum()):,} "
  f"({100*first['has_dx'].mean():.1f}%)")
P("\n[诊断年龄出现的波次 × 入组波次（交叉表，前 12 格）]")
P(pd.crosstab(first['entry_wave'], first['dg_wave'], dropna=False).to_string())

P("\n" + "=" * 78 + "\n# HRS：按入组波次，可定义发病区间的人数\n" + "=" * 78)
h2 = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave']).copy()
first_h = h2.groupby('pid').agg(entry_wave=('wave', 'first'), n_waves=('wave', 'size'))
pat = {}
for pid, x in h2.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        pat[pid] = 'never'
    else:
        ip = x.index.get_loc(fp.index[0])
        neg = x.iloc[:ip]
        pat[pid] = 'prevalent' if (neg[neg['arthre'] == 0].empty) else 'incident'
first_h['pattern'] = first_h.index.map(pat)
gh = pd.crosstab(first_h['entry_wave'], first_h['pattern'])
gh['total'] = gh.sum(axis=1)
for c in ['incident', 'prevalent', 'never']:
    if c in gh: gh['pct_' + c] = (100 * gh[c] / gh['total']).round(1)
P(gh.to_string())

P("\n" + "=" * 78 + "\n# 病程可计算性\n" + "=" * 78)
P("SHARE prevalent: duration_at_entry = entry_age - radiagarthr；")
P("HRS incident: duration since first positive = age_at_wave - age_at_first_positive；")
P("SHARE incident: 同上（用首次阳性 wave）。")
P("\n[SHARE radiagarthr 是否在后续波次重复出现（同一 pid 的非缺失波次数）]")
cnt = s.dropna(subset=['radiagarthr']).groupby('pid')['wave'].nunique()
P(cnt.value_counts().sort_index().head(10).to_string())

io.open(f"{OUT}\\s46_probe.txt", 'w', encoding='utf-8').write("\n".join(lines))
print("DONE")
