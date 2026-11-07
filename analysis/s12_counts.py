# -*- coding: utf-8 -*-
"""Step 12: 只统计各结局模型的分析样本量（不拟合），供正文与表 2 的 N 列使用。"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, pyreadr

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)
def load(db): return pyreadr.read_r(f"{BASE}\\{db}.RData")[db]

WY_H = {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020}
WY_S = {1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020}
COMORB = ['hibpe','diabe','hearte','stroke','lunge','cancre']
BINS  = [45,60,70,80,90,106]
BLAB  = ['45-59','60-69','70-79','80-89','90+']

def comorb_count(df):
    sub = df[COMORB]
    return sub.fillna(0).sum(axis=1).where(sub.notna().sum(axis=1) >= 4)

rows = []

# ---------------- HRS ----------------
h = load('hrs').copy()
h = h.rename(columns={h.columns[0]:'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h['comorb'] = comorb_count(h)
h = h.sort_values(['pid','wave'])
pos  = h[h['arthre']==1].groupby('pid').agg(onset_yr=('yr','min'), onset_age=('age','min'))
ever = h.groupby('pid')['arthre'].max().rename('ever')
hp = pd.DataFrame({'pid': h['pid'].unique()}).set_index('pid').join(pos).join(ever).reset_index()
def gh(r):
    if r['ever'] != 1 or pd.isna(r['onset_age']): return 'never'
    a = r['onset_age']
    return '<60' if a < 60 else ('60-74' if a < 75 else '>=75')
hp['og'] = hp.apply(gh, axis=1)

hh = h.merge(hp[['pid','og','onset_yr']], on='pid', how='left')
hh = hh[((hh['og']=='never') | (hh['yr'] >= hh['onset_yr'])) & hh['adl6a'].notna()].copy()
hh = hh[(hh['age']>=45) & (hh['age']<=105)].reset_index(drop=True)
hh['og'] = pd.Categorical(hh['og'], ['<60','60-74','>=75','never'], ordered=False)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1,2,3], ordered=True)
hh['male'] = (hh['ragender']==1).astype(float)
hh = hh.dropna(subset=['age','adl6a','edu_c','male']).reset_index(drop=True)
hh['band'] = pd.cut(hh['age'], BINS, labels=BLAB, right=False)
hh = hh.dropna(subset=['band']).reset_index(drop=True)
rows.append(dict(lib='HRS', outcome='ADL count (0-6)', rows=len(hh), persons=hh['pid'].nunique()))
hh_i = hh.dropna(subset=['iadl5a']).reset_index(drop=True)
rows.append(dict(lib='HRS', outcome='IADL count (0-5)', rows=len(hh_i), persons=hh_i['pid'].nunique()))
hhc = hh.dropna(subset=['comorb']).reset_index(drop=True)
rows.append(dict(lib='HRS', outcome='Comorbidity count (0-6)', rows=len(hhc), persons=hhc['pid'].nunique()))

# ---------------- SHARE ----------------
s = load('share').copy()
s = s.rename(columns={s.columns[0]:'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['comorb'] = comorb_count(s)
s['mob'] = (s['walk100a'] + s['climsa']).where(s['walk100a'].notna() & s['climsa'].notna())
s = s.sort_values(['pid','wave'])
# SHARE 用 radiagarthr（诊断年龄）分组，与 s8 完全一致；不可用 arthre 首次阳性波代替
dg = s.dropna(subset=['radiagarthr']).assign(
        radiagarthr=lambda x: x['radiagarthr'].where(x['radiagarthr']>=5)).dropna(subset=['radiagarthr'])
dg = dg.groupby('pid').agg(onset_age=('radiagarthr','first'))
ever_s = s.groupby('pid')['arthre'].max().rename('ever')
sp = pd.DataFrame({'pid': s['pid'].unique()}).set_index('pid').join(dg).join(ever_s).reset_index()
def gs(r):
    if pd.isna(r['onset_age']) or r['ever'] != 1: return 'never'
    a = r['onset_age']
    return '<60' if a < 60 else ('60-74' if a < 75 else '>=75')
sp['og'] = sp.apply(gs, axis=1)
ss = s.merge(sp[['pid','og','onset_age']], on='pid', how='left')
ss['onset_yr'] = ss['rabyear'] + ss['onset_age']
ss = ss[((ss['og']=='never') | (ss['yr'] >= ss['onset_yr'])) & ss['mob'].notna()].copy()
ss = ss[(ss['age']>=45) & (ss['age']<=105)].reset_index(drop=True)
ss['og'] = pd.Categorical(ss['og'], ['<60','60-74','>=75','never'], ordered=False)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1,2,3], ordered=True)
ss['male'] = (ss['ragender']==1).astype(float)
ss = ss.dropna(subset=['age','mob','edu_c','male']).reset_index(drop=True)
ss['band'] = pd.cut(ss['age'], BINS, labels=BLAB, right=False)
ss = ss.dropna(subset=['band']).reset_index(drop=True)
rows.append(dict(lib='SHARE', outcome='Mobility limitation count (0-2)', rows=len(ss), persons=ss['pid'].nunique()))
ssc = ss.dropna(subset=['comorb']).reset_index(drop=True)
rows.append(dict(lib='SHARE', outcome='Comorbidity count (0-6)', rows=len(ssc), persons=ssc['pid'].nunique()))

t = pd.DataFrame(rows)
P("="*74 + "\n# 各结局模型分析样本\n" + "="*74)
P(t.to_string(index=False))
P("\n[按库汇总]")
P(t.groupby('lib')[['rows','persons']].sum().to_string())
t.to_csv(f"{OUT}\\table2_n.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s12_counts.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 table2_n.csv / s12_counts.txt")
