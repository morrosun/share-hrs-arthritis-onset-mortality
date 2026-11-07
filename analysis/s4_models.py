# -*- coding: utf-8 -*-
"""Step 4: 决定性检验
A) SHARE：左截断 Cox（时间尺度=年龄），起病年龄分档 → 全因死亡
B) HRS：时依 Cox（关节炎状态随时间变化），起病年龄分档 → 全因死亡
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, pandas as pd, numpy as np
from lifelines import CoxPHFitter
from lifelines import CoxTimeVaryingFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines=[]
def P(s=""):
    lines.append(str(s)); print(s)
def load(db): return pyreadr.read_r(os.path.join(BASE, db+".RData"))[db]

WY = {'hrs':{5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020},
      'share':{1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020}}

# ---------------- A. SHARE ----------------
P("="*76+"\n# A. SHARE 左截断 Cox | 时间尺度 = 年龄 | 结局 = 全因死亡\n"+"="*76)
s = load('share').copy()
s = s.rename(columns={s.columns[0]:'pid'})
s['yr'] = s['wave'].map(WY['share'])
s['agew'] = s['yr'] - s['rabyear']

agg = s.groupby('pid').agg(entry_age=('agew','min'), entry_yr=('yr','min'),
                           last_yr=('yr','max'), sex=('ragender','first'),
                           byear=('rabyear','first'))
dg  = s.dropna(subset=['radiagarthr']).groupby('pid')['radiagarthr'].first().rename('onset_age')
dd  = s.groupby('pid')['radyear'].max().rename('deadyr')
cv  = s.groupby('pid').agg(edu=('raeducl','first'), smoke=('smoken','max'),
                           bmi=('bmi','first'))
coh = agg.join(dg).join(dd).join(cv)
coh['onset_age'] = coh['onset_age'].where(coh['onset_age']>=5)
coh = coh.dropna(subset=['onset_age'])
coh['exit_age'] = np.where(coh['deadyr'].notna(), coh['deadyr']-coh['byear'],
                           coh['last_yr']-coh['byear'])
coh['event'] = coh['deadyr'].notna().astype(int)
coh = coh[(coh['exit_age'] > coh['entry_age']) & (coh['entry_age']>=40) & (coh['exit_age']<=110)]
P(f"建模队列 n={len(coh)}  deaths={coh['event'].sum()}")

coh['grp'] = pd.cut(coh['onset_age'], [4,44,59,74,120], labels=['<45','45-59','60-74','>=75'])
tab = coh.groupby('grp', observed=True).agg(n=('event','size'), deaths=('event','sum'),
      entry_age=('entry_age','mean'), exit_age=('exit_age','mean'))
P("\n[描述]\n"+tab.round(1).to_string())

D = coh.rename(columns={'entry_age':'entry','exit_age':'stop'}).copy()
D['male'] = (D['sex']==1).astype(float)
D['smoke_ever'] = (D['smoke']>=1).astype(float)
D = D.dropna(subset=['stop','entry','event'])
for g in ['<45','60-74','>=75']:
    D['g_'+g.replace('-','_').replace('<','lt').replace('>=','ge')] = (D['grp']==g).astype(float)

P("\n--- Model 1: 仅起病年龄分档（ref=45-59）---")
m1 = CoxPHFitter()
D1 = D[['stop','entry','event','g_lt45','g_60_74','g_ge75','grp','male','edu','smoke_ever']].dropna()
m1.fit(D1, duration_col='stop', event_col='event', entry_col='entry',
       formula="g_lt45 + g_60_74 + g_ge75")
P(m1.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
P(f"  n={m1._n_examples}  events={int(D1['event'].sum())}  concordance={m1.concordance_index_:.3f}")

P("\n--- Model 2: + 性别 + 教育 + 吸烟 ---")
D2 = D[['stop','entry','event','g_lt45','g_60_74','g_ge75','male','edu','smoke_ever']].dropna()
m2 = CoxPHFitter()
m2.fit(D2, duration_col='stop', event_col='event', entry_col='entry',
       formula="g_lt45 + g_60_74 + g_ge75 + male + edu + smoke_ever")
P(m2.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
P(f"  n={m2._n_examples}  events={int(D2['event'].sum())}")

# 连续起病年龄（每 10 岁）
D['onset10'] = D['onset_age']/10
m3 = CoxPHFitter()
D3 = D[['stop','entry','event','onset10','male','edu','smoke_ever']].dropna()
m3.fit(D3, duration_col='stop', event_col='event', entry_col='entry',
       formula="onset10 + male + edu + smoke_ever")
P("\n--- Model 3: 起病年龄连续（每 +10 岁）---")
P(m3.summary.loc[['onset10'],['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())

# ---------------- B. HRS 时依 ----------------
P("\n\n"+"="*76+"\n# B. HRS 时依 Cox | 时间尺度 = 年龄 | 关节炎状态随时间更新\n"+"="*76)
h = load('hrs').copy()
h = h.rename(columns={h.columns[0]:'pid'})
h['yr'] = h['wave'].map(WY['hrs'])
h['agew'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['agew','arthre']).sort_values(['pid','wave'])

# 人级
hi = h.groupby('pid').agg(byear=('rabyear','first'), sex=('ragender','first'),
                          entry_age=('agew','min'), last_yr=('yr','max'))
ons = (h[h['arthre']==1].groupby('pid')
        .apply(lambda x: pd.Series({'onset_yr': x['yr'].iloc[0],
                                    'onset_age': x['agew'].iloc[0]}), include_groups=False))
ddh = h.groupby('pid')['radyear'].max().rename('deadyr')
hi = hi.join(ons).join(ddh)
hi['exit_year'] = np.where(hi['deadyr'].notna(), hi['deadyr'], hi['last_yr'])
hi['event'] = hi['deadyr'].notna().astype(int)
hi = hi[hi['exit_year'] > hi['yr'].min() if False else hi['exit_year'].notna()]
P(f"队列 n={len(hi)}  deaths={hi['event'].sum()}")

# 拆成 2 段（未患病期 / 患病后）
rows=[]
for pid, r in hi.iterrows():
    e0 = r['entry_age']; ex = r['exit_year']-r['byear']
    if ex <= e0: continue
    if pd.isna(r['onset_age']) or r['onset_age'] >= ex:
        rows.append((pid, e0, ex, r['event'], 0.0, 0.0, 0.0, r['sex'], 0))
    else:
        a1 = r['onset_age']
        if a1 > e0:
            rows.append((pid, e0, a1, 0, 0.0,0.0,0.0, r['sex'], 0))
        g = '<45' if a1<45 else ('45-59' if a1<60 else ('60-74' if a1<75 else '>=75'))
        rows.append((pid, max(a1,e0), ex, r['event'],
                     1.0 if g=='<45' else 0.0,
                     1.0 if g=='60-74' else 0.0,
                     1.0 if g=='>=75' else 0.0, r['sex'], 1))
T = pd.DataFrame(rows, columns=['pid','start','stop','event','g_lt45','g_60_74','g_ge75','sex','art_now'])
T = T[T['stop']>T['start']]
P(f"时依数据: {len(T)} 段, {T['pid'].nunique()} 人, events={int(T.groupby('pid')['event'].max().sum())}")
P(f"  曾患病者: {T.groupby('pid')['art_now'].max().sum()}")
# 未患病参照组 = 三段哑变量全 0；45-59 为参照
ctv = CoxTimeVaryingFitter()
T['male']=(T['sex']==1).astype(float)
ctv.fit(T[['pid','start','stop','event','g_lt45','g_60_74','g_ge75','male']],
        id_col='pid', start_col='start', stop_col='stop', event_col='event',
        formula="g_lt45 + g_60_74 + g_ge75 + male")
P("\n--- 时依 Cox（ref: 未患病；45-59 为患病组内参照，未单列）---")
P(ctv.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
P(f"  n_rows={len(T)}  events={int(T['event'].sum())}")

# 仅患病者内部对比（45-59 为 ref），用 onset 后左截断
P("\n--- 仅患病者：起病年龄内部对比（ref=45-59）---")
T2 = T[T['art_now']==1].copy()
ctv2 = CoxTimeVaryingFitter()
ctv2.fit(T2[['pid','start','stop','event','g_lt45','g_60_74','g_ge75','male']],
        id_col='pid', start_col='start', stop_col='stop', event_col='event',
        formula="g_lt45 + g_60_74 + g_ge75 + male")
P(ctv2.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
P(f"  n={T2['pid'].nunique()}  events={int(T2['event'].sum())}")

with open(os.path.join(OUT,'s4_models.txt'),'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s4_models.txt")
