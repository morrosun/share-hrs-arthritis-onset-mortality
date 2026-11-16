# -*- coding: utf-8 -*-
"""Step 32: exit<=entry 剔除者的首次阳性波次分布（是否集中在末波）。
输出 out/s32_exit_wave_dist.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def dist(df, wave_map, name, last_wave):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map); df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rec = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
        if prior[prior['arthre'] == 0].empty: continue
        rec.append(dict(pid=pid, pos_wave=int(fp['wave'].iloc[0]), last_wave=int(x['wave'].iloc[-1]),
                        n_after=int((x['wave'] > fp['wave'].iloc[0]).sum())))
    R = pd.DataFrame(rec)
    R['no_followup'] = R['n_after'] == 0
    P("=" * 78 + f"\n# {name}: 首次阳性波次分布（全部发病队列）\n" + "=" * 78)
    t = R.groupby('pos_wave').agg(n=('pid', 'size'), no_followup=('no_followup', 'sum'))
    t['pct_no_followup'] = (100 * t['no_followup'] / t['n']).round(1)
    P(t.to_string())
    P(f"合计 {len(R):,}；无后续访谈 {int(R['no_followup'].sum()):,}（{100*R['no_followup'].mean():.1f}%）")
    P(f"无后续访谈中位于末波 {last_wave} 者: {int((R[R['no_followup']]['pos_wave']==last_wave).sum()):,}")
    P("")

s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
dist(s, WY_S, 'SHARE', 8)

h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
dist(h, WY_H, 'HRS', 15)

io.open(os.path.join(OUT, "s32_exit_wave_dist.txt"), 'w', encoding='utf-8').write("\n".join(lines))
