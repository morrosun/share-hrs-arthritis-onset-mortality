# -*- coding: utf-8 -*-
"""Step 28 (v5): 时间结构核验（回应第三轮模拟评审）
1. 新发病例：起病年龄分层仍取区间中点，但延迟进入起点改为「首次阳性报告年龄」
   （中点入组会保证中点至首次阳性之间存活，不能消除保证生存时间偏倚）。
2. 同步对 HRS 新发病例做同一设计（时依 Cox 的 prevalent/incident 五态已在 s20 中以首次访谈入组）。
3. 核对 HRS 新发病例人数（7,252 / 7,241 / 7,251）的来源。
输出：out/s28_v5_entry_audit.txt, out/table_v5_entry_audit.csv
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

def build_incident(df, wave_map, id_col='pid'):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map)
    df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values([id_col, 'wave'])
    rows = []
    for pid, x in df.groupby(id_col):
        fp = x[x['arthre'] == 1]
        if fp.empty:
            continue
        ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty:
            continue                      # 基线已患病，不属于发病队列
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(),
                         byear=x['rabyear'].iloc[0], sex=x['ragender'].iloc[0],
                         edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
    I = pd.DataFrame(rows)
    I['onset'] = (I['lastneg'] + I['pos']) / 2
    I['width'] = I['pos'] - I['lastneg']
    I['event'] = I['dead'].notna().astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['male'] = (I['sex'] == 1).astype(float)
    I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
    I['grp'] = pd.cut(I['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
    return I.dropna(subset=['grp', 'edu'])

def fit_design(D, entry_col, label, rows_out):
    D = D[D['exit'] > D[entry_col]].copy()
    D['g74'] = (D['grp'] == '60-74').astype(float)
    D['g75'] = (D['grp'] == '>=75').astype(float)
    m = CoxPHFitter()
    m.fit(D[['exit', entry_col, 'event', 'g74', 'g75', 'male', 'edu', 'smoke_ever']],
          duration_col='exit', event_col='event', entry_col=entry_col)
    P(f"\n--- {label}  n={len(D):,}  deaths={int(D['event'].sum()):,}  py={(D['exit']-D[entry_col]).sum():,.0f} ---")
    for k, key in [('60-74', 'g74'), ('>=75', 'g75')]:
        q = m.summary.loc[key]
        P(f"  {k:6s} HR {q['exp(coef)']:.2f} ({q['exp(coef) lower 95%']:.2f}-{q['exp(coef) upper 95%']:.2f})  p={q['p']:.3g}")
        rows_out.append(dict(design=label, contrast=f"{k} vs <60", n=len(D), deaths=int(D['event'].sum()),
                             hr=round(q['exp(coef)'], 2), lo=round(q['exp(coef) lower 95%'], 2),
                             hi=round(q['exp(coef) upper 95%'], 2), p=float(f"{q['p']:.3g}")))

rows_out = []

# ---------- SHARE ----------
s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s = s.rename(columns={'wave': 'wave'})
rows = []
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s = s.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
rows = []
for pid, x in s.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty: continue
    ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
    neg = prior[prior['arthre'] == 0]
    if neg.empty: continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=x['age'].iloc[-1], dead=x['radyear'].max(),
                     byear=x['rabyear'].iloc[0], sex=x['ragender'].iloc[0],
                     edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['event'] = I['dead'].notna().astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['male'] = (I['sex'] == 1).astype(float)
I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
I['grp'] = pd.cut(I['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
I = I[(I['onset'] >= 40)].dropna(subset=['grp', 'edu'])
P("="*78 + "\n# SHARE 发病队列：分层=区间中点，入组 = 中点 vs 首次阳性\n" + "="*78)
P(f"分析样本（区间中点分层）: {len(I):,}")

I_mid = I.copy(); I_mid['entry_mid'] = I_mid['onset']
fit_design(I_mid, 'entry_mid', 'SHARE incident | strata midpoint, entry midpoint (v4)', rows_out)
I_pos = I.copy(); I_pos['entry_pos'] = I_pos['pos']
fit_design(I_pos, 'entry_pos', 'SHARE incident | strata midpoint, entry first positive (v5)', rows_out)

# ---------- HRS ----------
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
H = build_incident(h, WY_H)
H = H[H['onset'] >= 40]
P("\n" + "="*78 + "\n# HRS 发病队列\n" + "="*78)
P(f"HRS 首访阴性后报告关节炎者（分层前）: {len(H):,}")
P(f"分层后（edu/性别/吸烟可用，onset>=40）: {len(H):,}")
P(H.groupby('grp', observed=True).size().to_string())
H_mid = H.copy(); H_mid['entry_mid'] = H_mid['onset']
fit_design(H_mid, 'entry_mid', 'HRS incident | strata midpoint, entry midpoint (v4)', rows_out)
H_pos = H.copy(); H_pos['entry_pos'] = H_pos['pos']
fit_design(H_pos, 'entry_pos', 'HRS incident | strata midpoint, entry first positive (v5)', rows_out)

# ---------- 病例数核对 ----------
P("\n" + "="*78 + "\n# HRS 新发病例人数核对\n" + "="*78)
P(f"build_incident 所得（含 edu 缺失剔除前）: {len(build_incident(h, WY_H)):,}")
P(f"onset>=40 且 grp 可分配（当前 v5 口径）: {len(H):,}")
P("注：正文 7,252 来自旧脚本；表5A 分组合计 7,241 来自 s23；s20 时依 Cox 的 incident 三层合计 7,251（2483+3186+1582）。")

pd.DataFrame(rows_out).to_csv(os.path.join(OUT, "table_v5_entry_audit.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s28_v5_entry_audit.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 table_v5_entry_audit.csv / s28_v5_entry_audit.txt")
