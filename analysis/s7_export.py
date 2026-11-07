# -*- coding: utf-8 -*-
"""Step 7: 导出分析数据集 + 表1 + 图1 数据
SHARE 左截断队列（主），HRS 时依队列（对照）
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, pandas as pd, numpy as np
from scipy import stats
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines=[]
def P(s=""):
    lines.append(str(s)); print(s)
def load(db): return pyreadr.read_r(f"{BASE}\\{db}.RData")[db]

WY_S = {1:2004,2:2007,4:2011,5:2013,6:2015,7:2017,8:2020}

s = load('share').copy()
s = s.rename(columns={s.columns[0]:'pid'})
s['yr'] = s['wave'].map(WY_S)
s['agew'] = s['yr'] - s['rabyear']
s = s.sort_values(['pid','wave'])

# ---- 人级：入组波 = 首个有效波
first_w = s.groupby('pid')['wave'].min().rename('w0')
s = s.merge(first_w, left_on='pid', right_index=True)
ent = s[s['wave']==s['w0']].set_index('pid')
person = pd.DataFrame({
    'pid': ent.index,
    'w0': ent['w0'], 'entry_yr': ent['yr'], 'entry_age': ent['agew'],
    'byear': ent['rabyear'], 'sex': ent['ragender'],
    'edu': ent['raeducl'] if 'raeducl' in ent else np.nan,
    'smoke': ent['smoken'] if 'smoken' in ent else np.nan,
    'bmi': ent['bmi'] if 'bmi' in ent else np.nan,
    'hibpe': ent['hibpe'], 'diabe': ent['diabe'], 'hearte': ent['hearte'],
    'stroke': ent['stroke'], 'cancre': ent['cancre'], 'lunge': ent['lunge'],
    'country': ent['country'] if 'country' in ent else np.nan,
}).reset_index(drop=True)

person['last_yr'] = s.groupby('pid')['yr'].max().reindex(person['pid']).values
person['deadyr']  = s.groupby('pid')['radyear'].max().reindex(person['pid']).values
person['onset_age'] = (s.dropna(subset=['radiagarthr'])
                        .groupby('pid')['radiagarthr'].first()
                        .reindex(person['pid']).values)
person['ever'] = s.groupby('pid')['arthre'].max().reindex(person['pid']).values

P("="*72+"\n# SHARE 队列构建过程（逐步对账）\n"+"="*72)
P(f"总人数                        : {len(person)}")
n0=len(person)
p = person.copy()
m1 = p['onset_age'].notna()
P(f"有 radiagarthr                : {m1.sum()}")
p = p[m1]; 
p['onset_age'] = p['onset_age'].where(p['onset_age']>=5)
m2 = p['onset_age'].notna()
P(f"起病年龄≥5（剔除 0 与 <5）    : {m2.sum()}  （剔除 {m1.sum()-m2.sum()}）")
p = p[m2]
p['event'] = p['deadyr'].notna().astype(int)
p['exit_age'] = np.where(p['event']==1, p['deadyr']-p['byear'], p['last_yr']-p['byear'])
P(f"byear 缺失                    : {p['byear'].isna().sum()}")
P(f"exit_age ≤ entry_age          : {(p['exit_age']<=p['entry_age']).sum()}")
P(f"entry_age <40                 : {(p['entry_age']<40).sum()}")
P(f"exit_age >110                 : {(p['exit_age']>110).sum()}")
# v18 口径（2026-11-07 作者确认）：
#  (a) 同年死亡（零长度记录）统一插补 0.5 年，不再只保留 incident 的一侧；
#  (b) 剔除 2020 年入组波（wave 8）：该波入组者最多一次访谈，存活者无随访人年。
zero = p['exit_age'] <= p['entry_age']
n_zero_dead = int((zero & (p['event'] == 1)).sum())
p.loc[zero & (p['event'] == 1), 'exit_age'] = p.loc[zero & (p['event'] == 1), 'entry_age'] + 0.5
P(f"同年死亡插补 0.5 年           : {n_zero_dead}")
p = p[(p['exit_age']>p['entry_age']) & (p['entry_age']>=40) & (p['exit_age']<=110) & p['exit_age'].notna() & (p['w0']!=8)]
P(f"剔除 2020 入组波(w0=8)        : 队列 = {len(p)}  deaths={int(p['event'].sum())}")
P(f"最终建模队列                  : {len(p)}  deaths={p['event'].sum()}")

p['py'] = (p['exit_age']-p['entry_age']).clip(lower=0)
p['grp'] = pd.cut(p['onset_age'], [4,59,74,120], labels=['<60','60-74','>=75'])
p['smoke_ever'] = (p['smoke']>=1).astype(float)
p.to_csv(f"{OUT}\\analysis_share.csv", index=False, encoding='utf-8-sig')

# ---- 组内描述
def rate_ci(d, py):
    r = d/py*1000
    lo = r*np.exp(-1.96/np.sqrt(max(d,1)))
    hi = r*np.exp( 1.96/np.sqrt(max(d,1)))
    return r, lo, hi

P("\n"+"="*72+"\n# 分组描述 + 粗率\n"+"="*72)
rows=[]
for g, x in p.groupby('grp', observed=True):
    d = x['event'].sum(); py = x['py'].sum()
    r, lo, hi = rate_ci(d, py)
    rows.append(dict(grp=g, n=len(x), onset_med=x['onset_age'].median(),
        onset_q1=x['onset_age'].quantile(.25), onset_q3=x['onset_age'].quantile(.75),
        entry_mean=x['entry_age'].mean(), entry_sd=x['entry_age'].std(),
        deaths=int(d), py=py, rate=r, lo=lo, hi=hi))
G = pd.DataFrame(rows)
P(G.round(2).to_string(index=False))
G.to_csv(f"{OUT}\\fig1_groups.csv", index=False, encoding='utf-8-sig')

# ---- 校正 HR（左截断 Cox）
P("\n"+"="*72+"\n# 左截断 Cox（时间尺度=年龄）\n"+"="*72)
D = p.rename(columns={'entry_age':'entry','exit_age':'stop'}).copy()
D['male'] = (D['sex']==1).astype(float)
D['smoke_ever'] = (D['smoke']>=1).astype(float)
D['g_60_74'] = (D['grp']=='60-74').astype(float)
D['g_ge75'] = (D['grp']=='>=75').astype(float)
D2 = D[['stop','entry','event','g_60_74','g_ge75','male','edu','smoke_ever']].dropna()
cph = CoxPHFitter()
cph.fit(D2, duration_col='stop', event_col='event', entry_col='entry',
        formula="g_60_74 + g_ge75 + male + edu + smoke_ever")
P(cph.summary[['exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(3).to_string())
P(f"n={cph._n_examples} events={int(D2['event'].sum())}")

hr = {'<60':1.0}
ci = {'<60':(1.0,1.0)}
for k, key in [('60-74','g_60_74'),('>=75','g_ge75')]:
    hr[k] = cph.summary.loc[key,'exp(coef)']
    ci[k] = (cph.summary.loc[key,'exp(coef) lower 95%'], cph.summary.loc[key,'exp(coef) upper 95%'])
H = pd.DataFrame([{'grp':k,'hr':v,'lo':ci[k][0],'hi':ci[k][1]} for k,v in hr.items()])
H.to_csv(f"{OUT}\\fig1_hr.csv", index=False, encoding='utf-8-sig')
P("\n[H] 已写出 fig1_hr.csv")

# ---- 表1
P("\n"+"="*72+"\n# 表1 素材\n"+"="*72)
def ms(x): return f"{x.mean():.1f} ({x.std():.1f})"
def pct(x): return f"{100*x.mean():.1f}"
t1=[]
order = ['<60','60-74','>=75']
for g in order:
    x = p[p['grp']==g]
    t1.append({'起病年龄组':g,
        'N':len(x),
        '起病年龄 中位(IQR)': f"{x['onset_age'].median():.0f} ({x['onset_age'].quantile(.25):.0f}–{x['onset_age'].quantile(.75):.0f})",
        '入组年龄 均值(SD)': ms(x['entry_age']),
        '女性 %': pct(x['sex']==0),
        '教育 低/中/高 %': ' / '.join([pct(x['edu']==k) for k in [1,2,3]]) if x['edu'].notna().any() else 'NA',
        '吸烟 %': pct(x['smoke_ever']),
        'BMI 均值(SD)': ms(x['bmi']) if x['bmi'].notna().any() else 'NA',
        '高血压 %': pct(x['hibpe']==1), '糖尿病 %': pct(x['diabe']==1),
        '心脏病 %': pct(x['hearte']==1), '卒中 %': pct(x['stroke']==1),
        '死亡 n': int(x['event'].sum()),
        '人年': f"{x['py'].sum():.0f}",
        '粗死亡率 /1000人年 (95%CI)': f"{rate_ci(x['event'].sum(), x['py'].sum())[0]:.1f} ({rate_ci(x['event'].sum(), x['py'].sum())[1]:.1f}–{rate_ci(x['event'].sum(), x['py'].sum())[2]:.1f})",
    })
T1 = pd.DataFrame(t1).set_index('起病年龄组').T
P(T1.to_string())
T1.to_csv(f"{OUT}\\table1_share.csv", encoding='utf-8-sig')

# 教育/吸烟/性别 的取值分布核对
P("\n[核对] edu 取值: " + str(dict(p['edu'].value_counts().head(6))))
P("[核对] smoke 取值: " + str(dict(p['smoke'].value_counts().head(6))))
P("[核对] sex 取值: " + str(dict(p['sex'].value_counts().head(4))))

with open(f"{OUT}\\s7_export.txt",'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE")
