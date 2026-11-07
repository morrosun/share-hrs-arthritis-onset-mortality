# -*- coding: utf-8 -*-
"""Step 3
A) CHARLS 血生化覆盖（bl_crp / bl_wbc / bl_ua / bl_hgb）与单位
B) SHARE 诊断年龄队列：左截断生存设计能落多少事件
C) ELSA radiagarthr 非法值诊断（是否可用）
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, pandas as pd, numpy as np

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines=[]
def P(s=""):
    lines.append(str(s)); print(s)
def load(db): return pyreadr.read_r(os.path.join(BASE, db+".RData"))[db]

WY = {'charls':{1:2011,2:2013,3:2015,4:2018,5:2020},
      'share':{1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020},
      'elsa':{1:2002,2:2004,3:2006,4:2008,5:2010,6:2012,7:2014,8:2016,9:2018}}

# ---------- A. CHARLS 血生化 ----------
P("="*72+"\n# A. CHARLS 血生化\n"+"="*72)
c = load('charls')
for v in ['bl_crp','bl_wbc','bl_ua','bl_hgb']:
    if v not in c.columns: continue
    P(f"\n--- {v} ---")
    P(c.groupby('wave')[v].agg(['count','mean','median','min','max']).round(2).to_string())
    P(f"  整体: n={c[v].notna().sum()}  p1={c[v].quantile(.01):.1f} p99={c[v].quantile(.99):.1f}")
    # 与 arthre 的交叉：是否只在一个波次
    P(f"  非缺失值按波次占比: "
      + ", ".join(f"w{int(w)}={100*x:.0f}%" for w,x in (c[v].notna().groupby(c['wave']).mean()*100).items()))

# ---------- B. SHARE 诊断年龄队列 ----------
P("\n\n"+"="*72+"\n# B. SHARE 左截断生存队列\n"+"="*72)
s = load('share').copy()
s = s.rename(columns={s.columns[0]:'pid'})
s['yr'] = s['wave'].map(WY['share'])
s['age_at_w'] = s['yr'] - s['rabyear']

# 每人：入组波次（最早）、入组年龄、诊断年龄（取首个非缺失）
first = s.groupby('pid').agg(entry_w=('wave','min'), entry_yr=('yr','min'),
                             entry_age=('age_at_w','min'), sex=('ragender','first'),
                             byear=('rabyear','first'))
dg = s.dropna(subset=['radiagarthr']).groupby('pid')['radiagarthr'].first().rename('onset_age')
ev = s.groupby('pid')['arthre'].max().rename('ever')
dd = s.groupby('pid')['radyear'].max().rename('deadyr')
coh = first.join(dg).join(ev).join(dd)
P(f"SHARE 总人数: {len(coh)}")
P(f"  有诊断年龄: {coh['onset_age'].notna().sum()}")
P(f"  ever arthre=1: {(coh['ever']==1).sum()}")
P(f"  两者都有: {(coh['onset_age'].notna() & (coh['ever']==1)).sum()}")

# 清洗
bad0 = coh['onset_age']==0
bad5 = (coh['onset_age']>0)&(coh['onset_age']<5)
P(f"  非法值: ==0 {bad0.sum()}, 0<x<5 {bad5.sum()}")
coh['onset_age_c'] = coh['onset_age'].where(coh['onset_age']>=5)
# 诊断年龄 > 入组年龄 的：属"入组后新发"，起病年龄可保留
P(f"  起病早于入组(左截断): {(coh['onset_age_c'] < coh['entry_age']).sum()}")
P(f"  起病晚于/等于入组    : {(coh['onset_age_c'] >= coh['entry_age']).sum()}")

a = coh.dropna(subset=['onset_age_c']).copy()
a['grp'] = pd.cut(a['onset_age_c'], [4,44,59,74,120],
                  labels=['<45','45-59','60-74','>=75'])
a['died'] = a['deadyr'].notna()
a['fu'] = np.where(a['died'], a['deadyr']-a['entry_yr'], np.nan)
# 右删失：末次访视年
last_yr = s.groupby('pid')['yr'].max().rename('last_yr')
a = a.join(last_yr)
a['t_end'] = np.where(a['died'], a['deadyr'], a['last_yr'])
a['py'] = (a['t_end'] - a['entry_yr']).clip(lower=0)

P("\n--- 按起病年龄分档（入组即左截断）---")
tab = a.groupby('grp', observed=True).agg(
    n=('onset_age_c','size'), onset_med=('onset_age_c','median'),
    entry_age=('entry_age','mean'), deaths=('died','sum'),
    py=('py','sum'))
tab['death_pct'] = (100*tab['deaths']/tab['n']).round(1)
tab['rate_per1000py'] = (1000*tab['deaths']/tab['py']).round(1)
tab['entry_age']=tab['entry_age'].round(1); tab['py']=tab['py'].round(0)
P(tab.to_string())
P(f"\n  合计 n={len(a)}  deaths={a['died'].sum()}  PY={a['py'].sum():.0f}")

# 年龄标准化死亡率的粗算：按入组年龄分层内比较
P("\n--- 入组年龄分层内的死亡数（看混杂）---")
a['entry_grp'] = pd.cut(a['entry_age'], [49,59,69,79,120], labels=['50-59','60-69','70-79','80+'])
P(pd.crosstab(a['grp'], a['entry_grp'], values=a['died'], aggfunc='sum').to_string())
P("\n  各单元格 n:")
P(pd.crosstab(a['grp'], a['entry_grp']).to_string())

# ---------- C. ELSA ----------
P("\n\n"+"="*72+"\n# C. ELSA radiagarthr 非法值诊断\n"+"="*72)
e = load('elsa').copy()
e = e.rename(columns={e.columns[0]:'pid'})
e['yr'] = e['wave'].map(WY['elsa'])
e['age_at_w'] = e['yr'] - e['rabyear']
e['gap'] = e['radiagarthr'] - e['age_at_w']
P(f"records with both: {e['gap'].notna().sum()}")
P(e['gap'].describe(percentiles=[.001,.01,.05,.5,.95,.99,.999]).round(2).to_string())
P(f"  gap>1 的按波次计数:")
P(e[e['gap']>1].groupby('wave').size().to_string())
P(f"  gap>1 且 arthre==1 的占比: "
  f"{100*(e[e['gap']>1]['arthre']==1).mean():.1f}%")
P("  抽样看 gap 最大的 10 条 (pid, wave, age_at_w, radiagarthr):")
P(e.nlargest(10,'gap')[['pid','wave','age_at_w','radiagarthr','arthre']].to_string(index=False))

# ELSA 死亡覆盖
P(f"\n  ELSA radyear 非缺失: {e.groupby('pid')['radyear'].max().notna().sum()} / {e['pid'].nunique()}")

with open(os.path.join(OUT,'s3_feasibility.txt'),'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s3_feasibility.txt")
