# -*- coding: utf-8 -*-
"""Step 30: 核对 v5 发病队列中 exit<=entry（首次阳性即末次访谈/死亡）被剔除的人数与原因。
entry = 首次阳性报告年龄；exit = 死亡年龄（若死亡）否则末次访谈年龄。
输出 out/s30_check_exit_eq_entry.txt
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

def incident_table(df, wave_map, label):
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
        pos = fp['age'].iloc[0]
        last = x['age'].iloc[-1]
        dead = x['radyear'].max()
        byear = x['rabyear'].iloc[0]
        n_after = int((x['age'] > pos).sum())          # 首次阳性之后的访谈次数
        rows.append(dict(pid=pid, pos=pos, last=last, dead=dead, byear=byear,
                         died=pd.notna(dead), n_waves=len(x), n_after=n_after,
                         last_wave_is_pos=(x['wave'].iloc[-1] == fp['wave'].iloc[0])))
    I = pd.DataFrame(rows)
    I['event'] = I['died'].astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['exit_le_entry'] = I['exit'] <= I['pos']
    P("=" * 78 + f"\n# {label}\n" + "=" * 78)
    P(f"首访阴性后报告关节炎者（发病队列全体）: {len(I):,}")
    P(f"exit<=entry（首次阳性即末次记录）: {int(I['exit_le_entry'].sum()):,}")
    ex = I[I['exit_le_entry']]
    P(f"  其中已死亡: {int(ex['event'].sum()):,}；仍存活（末次即首次阳性）: {int((ex['event']==0).sum()):,}")
    P(f"  首次阳性即最后一波（无后续访谈）: {int(ex['last_wave_is_pos'].sum()):,}")
    P(f"exit>entry（纳入 v5 分析）: {int((~I['exit_le_entry']).sum()):,}")
    # 对死亡但 exit<=entry 的人，检查死亡年份与阳性年份的先后
    d = ex[ex['event'] == 1]
    if len(d):
        P(f"  死亡且 exit<=entry 的死亡年龄-阳性年龄差（年）: 中位 {(d['exit']-d['pos']).median():.2f}，范围 {(d['exit']-d['pos']).min():.2f} ~ {(d['exit']-d['pos']).max():.2f}")
    # 无后续访谈的存活者：其 exit==pos 说明是末次访谈即阳性，而非数据缺失
    alive_ex = ex[ex['event'] == 0]
    if len(alive_ex):
        P(f"  存活且 exit<=entry 的 last-pos 差（年）: 中位 {(alive_ex['last']-alive_ex['pos']).median():.2f}，最大 {(alive_ex['last']-alive_ex['pos']).max():.2f}")
    P("")
    return I

s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
I_s = incident_table(s, WY_S, "SHARE 发病队列（v5 入组=首次阳性）")

h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
I_h = incident_table(h, WY_H, "HRS 发病队列（v5 入组=首次阳性）")

io.open(os.path.join(OUT, "s30_check_exit_eq_entry.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("写出 s30_check_exit_eq_entry.txt")
