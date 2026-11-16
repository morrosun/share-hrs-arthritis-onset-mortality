# -*- coding: utf-8 -*-
"""Step 24: 评审第二轮的四项补充
(A) HRS 新口径（仅 incident）的起病年龄中位数 —— 核对正文 64 岁是否沿用旧定义
(B) 共病「六项全非缺失」完整病例敏感性（现按 >=4 项计数，缺失按无病计）
(C) 区间上下界：分离「只改分组」与「同时改分组和进入时间」
(D) 功能分析在 70-79 内加连续年龄（单岁）后的组间对比
"""
import os
import sys, io, math, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010,
        11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def to_grp(v):
    return pd.cut(v, [40, 59.999, 74.999, 200], labels=['<60', '60-74', '>=75'])

# ==================== A. HRS 新旧口径的起病年龄分布 ====================
P("=" * 84 + "\n# A. HRS 起病年龄：旧口径（首次阳性报告）vs 新口径（仅 incident，区间中点）\n" + "=" * 84)
h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        rows.append(dict(pid=pid, pattern='never', onset_new=np.nan, onset_old=np.nan,
                         entry=x['age'].iloc[0]))
        continue
    ip = x.index.get_loc(fp.index[0]); neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    old = fp['age'].iloc[0]
    if neg.empty:
        rows.append(dict(pid=pid, pattern='prevalent', onset_new=np.nan, onset_old=old,
                         entry=x['age'].iloc[0]))
    else:
        rows.append(dict(pid=pid, pattern='incident',
                         onset_new=(neg['age'].iloc[-1] + old) / 2, onset_old=old,
                         entry=x['age'].iloc[0]))
Hd = pd.DataFrame(rows)
P(Hd['pattern'].value_counts().to_string())
for lab, col, sub in [('旧口径：全部病例，首次阳性报告年龄', 'onset_old', Hd[Hd['pattern'] != 'never']),
                      ('新口径：仅 incident，区间中点', 'onset_new', Hd[Hd['pattern'] == 'incident'])]:
    v = sub[col].dropna()
    P(f"\n{lab}\n  n={len(v):,}  中位 {v.median():.0f}（IQR {v.quantile(.25):.0f}–{v.quantile(.75):.0f}）"
      f"  <60: {100*(v<60).mean():.1f}%")
P("\n[旧口径下 prevalent 病例被赋予的\"起病年龄\" = 首次访谈年龄，其中 ≥60 岁者 "
  f"{int((Hd[(Hd['pattern']=='prevalent')]['onset_old']>=60).sum()):,} 人]")

# ==================== B. 共病完整病例敏感性 ====================
P("\n\n" + "=" * 84 + "\n# B. 共病计数：>=4 项非缺失（现用）vs 六项全非缺失（严格）\n" + "=" * 84)
COM = ['hibpe', 'diabe', 'hearte', 'stroke', 'cancre', 'lunge']

def comorb(df, strict=False):
    x = df[COM]
    n_ok = x.notna().sum(axis=1)
    s = x.fillna(0).sum(axis=1)
    keep = (n_ok == len(COM)) if strict else (n_ok >= 4)
    return s.where(keep), n_ok

# HRS
h2 = h.copy()
h2['cm'], h2['n_ok'] = comorb(h2)
rows = []
for pid, x in h2.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        rows.append(dict(pid=pid, state='never', onset=np.nan)); continue
    ip = x.index.get_loc(fp.index[0]); neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty:
        rows.append(dict(pid=pid, state='prevalent', onset=np.nan))
    else:
        o = (neg['age'].iloc[-1] + fp['age'].iloc[0]) / 2
        g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
        rows.append(dict(pid=pid, state='on_' + g, onset=o))
S = pd.DataFrame(rows)
hh = h2.merge(S, on='pid', how='left')
hh = hh[(hh['age'] >= 45) & (hh['age'] <= 105)]
hh = hh[(~hh['state'].str.startswith('on_')) | (hh['age'] >= hh['onset'])]
hh['male'] = (hh['ragender'] == 1).astype(float)
hh['edu_c'] = pd.Categorical(hh['raeducl'], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=['age', 'male', 'edu_c'])
hh['band'] = pd.cut(hh['age'], [45, 60, 70, 80, 90, 106],
                    labels=['45-59', '60-69', '70-79', '80-89', '90+'], right=False)
hh = hh.dropna(subset=['band'])
hh['state'] = pd.Categorical(hh['state'], ['never', 'prevalent', 'on_<60', 'on_60-74', 'on_>=75'])

P("\n[HRS 共病缺失结构]")
P(f"  人-波 {len(hh):,}")
P(f"  六项全有 {(hh['n_ok']==6).mean()*100:.1f}%  |  >=4 项 {(hh['n_ok']>=4).mean()*100:.1f}%")
P("  各层缺失率（六项中平均缺失项数）:")
P(hh.groupby('state', observed=True)['n_ok'].agg(n='size', mean_ok='mean').round(2).to_string())

for lab, strict in [('>=4 present (as reported)', False), ('all 6 present (complete case)', True)]:
    d = hh.copy()
    if strict:
        d = d[d['n_ok'] == 6]
    m = smf.gee("cm ~ C(state) + C(band) + C(edu_c) + male", groups='pid', data=d,
                family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
    P(f"\n--- HRS comorbidity, {lab}: waves={len(d):,} ---")
    out = []
    for k in m.params.index:
        if k.startswith('C(state)'):
            est = np.exp(m.params[k]); lo = np.exp(m.conf_int().loc[k, 0]); hi = np.exp(m.conf_int().loc[k, 1])
            P(f"  {k[10:-1]:10s} {est:.2f} ({lo:.2f}-{hi:.2f})  p={m.pvalues[k]:.3g}")
            out.append(dict(model=lab, state=k[10:-1], ratio=est, lo=lo, hi=hi, p=m.pvalues[k],
                            waves=len(d)))

# SHARE 同样
s = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
s['cm'], s['n_ok'] = comorb(s)
rows = []
for pid, x in s.groupby('pid'):
    dg = x['radiagarthr'].dropna(); dg = dg[dg >= 5]
    if dg.empty:
        rows.append(dict(pid=pid, state='never', onset=np.nan)); continue
    o = dg.iloc[0]
    g = '<60' if o < 60 else ('60-74' if o < 75 else '>=75')
    rows.append(dict(pid=pid, state='on_' + g, onset=o))
S2 = pd.DataFrame(rows)
ss = s.merge(S2, on='pid', how='left')
ss = ss[(ss['age'] >= 45) & (ss['age'] <= 105)]
ss = ss[(~ss['state'].str.startswith('on_')) | (ss['age'] >= ss['onset'])]
ss['male'] = (ss['ragender'] == 1).astype(float)
ss['edu_c'] = pd.Categorical(ss['raeducl'], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=['age', 'male', 'edu_c'])
ss['band'] = pd.cut(ss['age'], [45, 60, 70, 80, 90, 106],
                    labels=['45-59', '60-69', '70-79', '80-89', '90+'], right=False)
ss = ss.dropna(subset=['band'])
ss['state'] = pd.Categorical(ss['state'], ['never', 'on_<60', 'on_60-74', 'on_>=75'])

P("\n[SHARE 共病缺失结构]")
P(f"  人-波 {len(ss):,}  六项全有 {(ss['n_ok']==6).mean()*100:.1f}%  >=4 项 {(ss['n_ok']>=4).mean()*100:.1f}%")
P(ss.groupby('state', observed=True)['n_ok'].agg(n='size', mean_ok='mean').round(2).to_string())
for lab, strict in [('>=4 present (as reported)', False), ('all 6 present (complete case)', True)]:
    d = ss.copy()
    if strict:
        d = d[d['n_ok'] == 6]
    m = smf.gee("cm ~ C(state) + C(band) + C(edu_c) + male", groups='pid', data=d,
                family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
    P(f"\n--- SHARE comorbidity, {lab}: waves={len(d):,} ---")
    for k in m.params.index:
        if k.startswith('C(state)'):
            est = np.exp(m.params[k]); lo = np.exp(m.conf_int().loc[k, 0]); hi = np.exp(m.conf_int().loc[k, 1])
            P(f"  {k[10:-1]:10s} {est:.2f} ({lo:.2f}-{hi:.2f})  p={m.pvalues[k]:.3g}")
            out.append(dict(model=lab, state=k[10:-1], ratio=est, lo=lo, hi=hi, p=m.pvalues[k],
                            waves=len(d)))

pd.DataFrame(out).to_csv(os.path.join(OUT, "tableS3_comorb_completecase.csv"), index=False, encoding='utf-8-sig')

# ==================== C. 上下界：只改分组 vs 同时改进入时间 ====================
P("\n\n" + "=" * 84 + "\n# C. 区间上下界：分离「只改分组」与「同时改分组和进入时间」\n" + "=" * 84)
rows = []
for pid, x in s.groupby('pid'):
    a = x.dropna(subset=['arthre'])
    if a.empty: continue
    fp = a[a['arthre'] == 1]
    if fp.empty: continue
    ip = a.index.get_loc(fp.index[0]); neg = a.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty: continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=a['age'].iloc[-1], dead=a['radyear'].max(), byear=a['rabyear'].iloc[0],
                     sex=a['ragender'].iloc[0], edu=a['raeducl'].iloc[0], smoke=a['smoken'].iloc[0]))
I = pd.DataFrame(rows)
I['mid'] = (I['lastneg'] + I['pos']) / 2
I['width'] = I['pos'] - I['lastneg']
I['event'] = I['dead'].notna().astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['male'] = (I['sex'] == 1).astype(float)
I['smoke_ever'] = (I['smoke'] >= 1).astype(float)

def fit(df, entry_col, grp_col, lab):
    d = df.copy()
    d['g74'] = (d[grp_col] == '60-74').astype(float)
    d['g75'] = (d[grp_col] == '>=75').astype(float)
    D = d[['exit', entry_col, 'event', 'g74', 'g75', 'male', 'edu', 'smoke_ever']].dropna()
    D = D[D['exit'] > D[entry_col]]
    if len(D) < 500:
        P(f"\n--- {lab}: n={len(D)} 太少 ---"); return
    m = CoxPHFitter(); m.fit(D, duration_col='exit', event_col='event', entry_col=entry_col)
    P(f"\n--- {lab}: n={len(D):,} deaths={int(D['event'].sum()):,} ---")
    for k, key in [('60-74', 'g74'), ('>=75', 'g75')]:
        q = m.summary.loc[key]
        P(f"  {k:6s} {q['exp(coef)']:.2f} ({q['exp(coef) lower 95%']:.2f}-{q['exp(coef) upper 95%']:.2f})")
        globals().setdefault('res_c', []).append(
            dict(scenario=lab, grp=k, hr=q['exp(coef)'], lo=q['exp(coef) lower 95%'],
                 hi=q['exp(coef) upper 95%'], n=len(D), deaths=int(D['event'].sum())))

I['grp'] = to_grp(I['mid'])
B = I[(I['exit'] > I['mid']) & (I['mid'] >= 40)].dropna(subset=['grp'])
fit(B, 'mid', 'grp', 'midpoint, entry = midpoint (main)')

for lab, sh in [('lower bound', -0.5), ('upper bound', +0.5)]:
    W = B.copy()
    W['alt'] = W['mid'] + sh * W['width']
    W['grp_alt'] = to_grp(W['alt'])
    W = W.dropna(subset=['grp_alt'])
    # (a) 只改分组，进入时间仍为中点
    fit(W, 'mid', 'grp_alt', f'{lab}: group re-assigned, entry kept at midpoint')
    # (b) 分组与进入时间同步移动
    V = W[W['exit'] > W['alt']]
    fit(V, 'alt', 'grp_alt', f'{lab}: group AND entry moved')

pd.DataFrame(res_c).to_csv(os.path.join(OUT, "tableS4_bounds.csv"), index=False, encoding='utf-8-sig')

# ==================== D. 功能分析：70-79 内加连续年龄 ====================
P("\n\n" + "=" * 84 + "\n# D. 70-79 岁内：年龄段 vs 单岁连续年龄\n" + "=" * 84)
ADL = ['walkra', 'dressa', 'batha', 'eata', 'beda', 'toilta']
for lib, d, oc, fam, arg in [('HRS ADL', hh, 'adl6a', 'gaussian', None),
                             ('HRS IADL', hh, 'iadl5a', 'gaussian', None),
                             ('SHARE mobility', ss, 'mob', 'poisson', None)]:
    if oc not in d.columns:
        P(f"\n{lib}: 无 {oc}，跳过"); continue
    sub = d[(d['age'] >= 70) & (d['age'] < 80)].dropna(subset=[oc])
    if len(sub) < 200:
        P(f"\n{lib}: 70-79 内 n={len(sub)} 太少，跳过"); continue
    P(f"\n--- {lib}, ages 70-79: person-waves={len(sub):,} ---")
    P(sub.groupby('state', observed=True)[oc].agg(['size', 'mean']).round(2).to_string())
    for vlab, rhs in [('band only (never as band ref)', "C(state) + C(band) + C(edu_c) + male"),
                      ('band + continuous age', "C(state) + C(band) + age + C(edu_c) + male"),
                      ('continuous age only', "C(state) + age + C(edu_c) + male")]:
        if vlab.startswith('band only') and lib.startswith('S'):
            rhs = rhs.replace(' + C(band)', '')
        try:
            if fam == 'gaussian':
                m = smf.mixedlm(f"{oc} ~ {rhs}", sub, groups=sub['pid'].values).fit(reml=False, method='lbfgs')
                kind = 'beta'
            else:
                m = smf.gee(f"{oc} ~ {rhs}", groups='pid', data=sub,
                            family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
                kind = 'ratio'
            P(f"  [{vlab}]")
            for k in m.params.index:
                if k.startswith('C(state)'):
                    c = m.params[k]; se = m.bse[k]; p = m.pvalues[k]
                    if kind == 'beta':
                        P(f"    {k[10:-1]:10s} {c:+.2f} ({c-1.96*se:+.2f} to {c+1.96*se:+.2f})  p={p:.3g}")
                    else:
                        P(f"    {k[10:-1]:10s} {np.exp(c):.2f} ({np.exp(c-1.96*se):.2f}-{np.exp(c+1.96*se):.2f})  p={p:.3g}")
        except Exception as ex:
            P(f"  [{vlab}] 失败: {str(ex)[:70]}")

io.open(os.path.join(OUT, "s24_extras.txt"), 'w', encoding='utf-8').write("\n".join(lines))
print("\nWROTE s24_extras.txt")
