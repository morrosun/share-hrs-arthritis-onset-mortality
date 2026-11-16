# -*- coding: utf-8 -*-
"""Step 6: 机制层唯一可做的一条 —— ELSA 起病年龄 × WBC
限制：WBC 必须在诊断之后测量（排除反向因果）；年龄校正。
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, pandas as pd, numpy as np
import statsmodels.formula.api as smf

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines=[]
def P(s=""):
    lines.append(str(s)); print(s)
def load(db): return pyreadr.read_r(os.path.join(BASE, f"{db}.RData"))[db]

WY = {1:2002,2:2004,3:2006,4:2008,5:2010,6:2012,7:2014,8:2016,9:2018,89:2017}

e = load('elsa').copy()
e = e.rename(columns={e.columns[0]:'id'})
e['id'] = e['id'].astype(str)
e['yr'] = e['wave'].map(WY)
e['agew'] = e['yr'] - e['rabyear']

core = load('elsa_raw_core')
w = core[core['concept']=='wbc'].copy()
w['id'] = w['pid'].astype(str)
w = w.rename(columns={'value':'wbc'})[['id','wave','wbc']]
w['yr'] = w['wave'].map(WY)

# 人级：诊断年龄（仅 arthre=1 且合法）
dg = (e[e['arthre']==1].dropna(subset=['radiagarthr'])
        .assign(radiagarthr=lambda x: x['radiagarthr'].where(x['radiagarthr']>=5))
        .dropna(subset=['radiagarthr'])
        .groupby('id').agg(onset_age=('radiagarthr','first'),
                           sex=('ragender','first'),
                           edu=('raeducl','first'),
                           bmi=('mbmi','first') if 'mbmi' in e.columns else ('raeducl','first')))

# 取"诊断之后"最近一次 WBC
m = w.merge(dg, left_on='id', right_index=True, how='inner')
m = m[m['yr'] >= m['rabyear'] if False else m['yr'].notna()]
m['age_at_wbc'] = m['yr'] - e.groupby('id')['rabyear'].first().reindex(m['id']).values
m = m[m['age_at_wbc'] >= m['onset_age']]          # 仅诊断后测量
m = m.sort_values(['id','yr']).groupby('id').first().reset_index()

P("="*70+"\n# ELSA：起病年龄 × WBC（诊断后测量，年龄校正）\n"+"="*70)
P(f"可用样本: {len(m)} 人")
m['grp'] = pd.cut(m['onset_age'], [4,44,59,74,120], labels=['<45','45-59','60-74','>=75'])
P("\n[描述]")
t = m.groupby('grp', observed=True).agg(n=('wbc','size'), onset=('onset_age','median'),
        age_wbc=('age_at_wbc','mean'), wbc_mean=('wbc','mean'), wbc_med=('wbc','median'))
P(t.round(2).to_string())

m['logwbc'] = np.log(m['wbc'])
m['male'] = (m['sex']==1).astype(float)
D = m.dropna(subset=['logwbc','age_at_wbc','male','grp'])
P(f"\n进入模型 n={len(D)}")

P("\n--- OLS: log(WBC) ~ 起病年龄组 + 测量时年龄 + 性别（ref=45-59）---")
mod = smf.ols("logwbc ~ C(grp, Treatment('45-59')) + age_at_wbc + male", data=D).fit(cov_type='HC1')
res = pd.DataFrame({'coef':mod.params,'se':mod.bse,'p':mod.pvalues})
res['pct_change'] = (np.exp(res['coef'])-1)*100
P(res.round(3).to_string())

P("\n--- 连续：起病年龄每 +10 岁 ---")
D['onset10'] = D['onset_age']/10
mod2 = smf.ols("logwbc ~ onset10 + age_at_wbc + male", data=D).fit(cov_type='HC1')
r2 = pd.DataFrame({'coef':mod2.params,'se':mod2.bse,'p':mod2.pvalues})
r2['pct_change'] = (np.exp(r2['coef'])-1)*100
P(r2.round(3).to_string())

with open(os.path.join(OUT, "s6_mechanism.txt"),'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s6_mechanism.txt")
