# -*- coding: utf-8 -*-
"""Step 47: 按入组波次核查诊断年龄可得率 + 波次/招募时期敏感性分析。

A) SHARE 全体受访者按入组波次的诊断年龄可得率（分母=该波次首次入组人数）
B) SHARE prevalent 主模型（Table 3 model 4）在波次限制 / 分层 / 调整下的稳健性
C) HRS 按入组波次可定义发病区间的比例，以及发病队列按招募时期的敏感性
输出：
  out/tableS7_waves_v5.csv       (panel A/B/C 合并，供表格渲染)
  out/tableS7_wave_availability.csv (panel A 单独)
  out/s47_wave_sensitivity.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy import stats as st

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
    d = d.rename(columns={d.columns[0]: 'pid'})
    d['yr'] = d['wave'].map(WY_S if db == 'share' else WY_H)
    d['age'] = d['yr'] - d['rabyear']
    return d

s = load('share')
h = load('hrs')

# ==================== A. SHARE 诊断年龄可得率 ====================
P("=" * 84 + "\n# A. SHARE：按入组波次的诊断年龄可得率\n" + "=" * 84)
first = (s.sort_values(['pid', 'wave']).groupby('pid')
         .agg(entry_wave=('wave', 'first'), entry_age=('age', 'first'), n_waves=('wave', 'size')))
dx = (s.dropna(subset=['radiagarthr'])
      .assign(v=lambda d: d['radiagarthr'].where(d['radiagarthr'] >= 5))
      .dropna(subset=['v']).groupby('pid')
      .agg(onset_age=('v', 'first'), dx_wave=('wave', 'first')))
first = first.join(dx)
first['has_dx'] = first['onset_age'].notna()
# 诊断年龄记录波次是否等于个人入组波次
both = first.dropna(subset=['dx_wave'])
same_wave = float((both['entry_wave'] == both['dx_wave']).mean())
P(f"有有效诊断年龄者中，记录波次 = 个人入组波次的比例: {same_wave*100:.2f}% "
  f"(n={len(both):,}/{len(both):,})")

g = first.groupby('entry_wave').agg(n_entered=('has_dx', 'size'), n_dx_age=('has_dx', 'sum'),
                                    onset_med=('onset_age', 'median'),
                                    n_waves=('n_waves', 'mean'))
g['availability_pct'] = 100 * g['n_dx_age'] / g['n_entered']
coh = pd.read_csv(f"{OUT}\\analysis_share.csv", encoding='utf-8-sig')
gc = coh.groupby('w0').agg(n_analytic=('event', 'size'), deaths=('event', 'sum'),
                           entry_age=('entry_age', 'mean'), py=('py', 'sum'))
g = g.join(gc)
g['crude_rate'] = 1000 * g['deaths'] / g['py']
P(g.round(2).to_string())

# 趋势检验：可得率是否与招募年份相关（排除 wave 7 的 10 人）
q = first[first['entry_wave'] != 7].copy()
tab = pd.crosstab(q['entry_wave'], q['has_dx'])
chi2, pv, dof, _ = st.chi2_contingency(tab)
P(f"\n各入组波次可得率差异: chi2={chi2:.1f}, df={dof}, P={pv:.3g} "
  f"(排除 wave 7 的 10 名新入组者)")
yr = q.groupby('entry_wave')['has_dx'].mean()
yr.index = [WY_S[int(w)] for w in yr.index]
r, pr = st.pearsonr(yr.index.values.astype(float), yr.values)
P(f"可得率与日历年（波次水平）相关: r={r:+.2f}, P={pr:.3g}")
P(f"可得率范围（wave 7 除外）: {yr.min()*100:.1f}%–{yr.max()*100:.1f}%")

rows = []
for w, r_ in g.iterrows():
    rows.append(dict(panel='A', cohort='SHARE', scenario=f"Entry wave {int(w)} ({WY_S[int(w)]})",
                     n=int(r_['n_entered']), n_dx=int(r_['n_dx_age']),
                     availability=round(r_['availability_pct'], 1),
                     onset_med=r_['onset_med'], n_analytic=(int(r_['n_analytic']) if pd.notna(r_['n_analytic']) else 0),
                     deaths=(int(r_['deaths']) if pd.notna(r_['deaths']) else 0),
                     crude_rate=(round(r_['crude_rate'], 1) if pd.notna(r_['crude_rate']) else None),
                     hr=None, lo=None, hi=None, hr2=None, lo2=None, hi2=None))
rows.append(dict(panel='A', cohort='SHARE', scenario='Chi-square across entry waves (excl. wave 7)',
                 n=None, n_dx=None, availability=round(float(pv), 6), onset_med=None,
                 n_analytic=None, deaths=None, crude_rate=None,
                 hr=None, lo=None, hi=None, hr2=None, lo2=None, hi2=None))

# ==================== B. SHARE prevalent 波次敏感性 ====================
P("\n" + "=" * 84 + "\n# B. SHARE prevalent：波次/招募时期敏感性（Table 3 model 4 口径）\n" + "=" * 84)
c = coh.copy()
c = c.rename(columns={'entry_age': 'entry', 'exit_age': 'exit'})
c['grp'] = c['grp'].astype(str).str.strip()
c['g_60_74'] = (c['grp'] == '60-74').astype(float)
c['g_ge75'] = (c['grp'] == '>=75').astype(float)
c['male'] = (c['sex'] == 1).astype(float)
c['smoke_ever'] = (c['smoke'] >= 1).astype(float)
c['entry_year'] = c['w0'].map(WY_S)
c = c.dropna(subset=['exit', 'entry', 'male', 'edu', 'smoke_ever'])
c = c[c['exit'] > c['entry']].copy()
P(f"主分析队列 n={len(c):,} deaths={int(c['event'].sum()):,}")

FORM = "g_60_74 + g_ge75 + male + edu + smoke_ever"

def fit(d, label, strata=None, extra=""):
    dd = d.copy()
    form = FORM + ((" + " + extra) if extra else "")
    cols = ['exit', 'entry', 'event', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']
    if extra: cols += [extra]
    if strata: cols += strata
    dd = dd[cols].dropna()
    m = CoxPHFitter()
    kw = dict(duration_col='exit', event_col='event', entry_col='entry', formula=form)
    if strata: kw['strata'] = strata
    m.fit(dd, **kw)
    a = m.summary.loc['g_60_74']; b = m.summary.loc['g_ge75']
    P(f"  {label:44s} n={len(dd):6,} deaths={int(dd['event'].sum()):5,}  "
      f"60-74: {a['exp(coef)']:.2f} ({a['exp(coef) lower 95%']:.2f}-{a['exp(coef) upper 95%']:.2f})  "
      f">=75: {b['exp(coef)']:.2f} ({b['exp(coef) lower 95%']:.2f}-{b['exp(coef) upper 95%']:.2f})")
    return dict(n=len(dd), deaths=int(dd['event'].sum()),
                hr=a['exp(coef)'], lo=a['exp(coef) lower 95%'], hi=a['exp(coef) upper 95%'],
                hr2=b['exp(coef)'], lo2=b['exp(coef) lower 95%'], hi2=b['exp(coef) upper 95%'])

SCEN = [
    ('Main: all entry waves', c, None, ""),
    ('Stratified by entry wave (baseline hazard)', c[c['w0'] != 7], ['w0'], ""),
    ('Adjusted for recruitment year (continuous)', c, None, "entry_year"),
    ('Early recruits only (waves 1-2; 2004-2007)', c[c['w0'].isin([1, 2])], None, ""),
    ('Wave 1 recruits only (2004)', c[c['w0'] == 1], None, ""),
    ('Later recruits only (waves 4-8; 2011-2020)', c[c['w0'].isin([4, 5, 6, 7, 8])], None, ""),
    ('Excluding final-wave entrants (wave 8)', c[c['w0'] != 8], None, ""),
]
P("")
for lab, d, strata, extra in SCEN:
    try:
        r = fit(d, lab, strata=strata, extra=extra)
        rows.append(dict(panel='B', cohort='SHARE', scenario=lab, n=r['n'], deaths=r['deaths'],
                         n_dx=None, availability=None, onset_med=None, n_analytic=None,
                         crude_rate=None,
                         hr=round(r['hr'], 2), lo=round(r['lo'], 2), hi=round(r['hi'], 2),
                         hr2=round(r['hr2'], 2), lo2=round(r['lo2'], 2), hi2=round(r['hi2'], 2)))
    except Exception as ex:
        P(f"  {lab}: FAILED {str(ex)[:100]}")

# ==================== C. HRS ====================
P("\n" + "=" * 84 + "\n# C. HRS：按入组波次的发病区间可得率与招募时期敏感性\n" + "=" * 84)

def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])

def build_incident(df, wave_map):
    df = df.copy().dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rows = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty: continue
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                         entry_wave=x['wave'].iloc[0],
                         sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
    I = pd.DataFrame(rows)
    I['onset'] = (I['lastneg'] + I['pos']) / 2
    I['event'] = I['dead'].notna().astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['male'] = (I['sex'] == 1).astype(float)
    I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
    I['grp'] = grp(I['onset'])
    I = I[(I['onset'] >= 40)].dropna(subset=['grp', 'edu'])
    I['entry'] = I['pos']
    fix = (I['exit'] <= I['entry']) & (I['event'] == 1)
    I.loc[fix, 'exit'] = I.loc[fix, 'entry'] + 0.5
    return I[I['exit'] > I['entry']].copy()

h2 = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
fh = h2.groupby('pid').agg(entry_wave=('wave', 'first'), n_waves=('wave', 'size'))
inc_all = build_incident(h, WY_H)
fh['incident'] = fh.index.isin(inc_all['pid'])
gh = fh.groupby('entry_wave').agg(n_entered=('incident', 'size'), n_incident=('incident', 'sum'),
                                  n_waves=('n_waves', 'mean'))
gh['interval_pct'] = 100 * gh['n_incident'] / gh['n_entered']
P(gh.round(2).to_string())
P(f"\n合计 {len(fh):,} 人中 {int(fh['incident'].sum()):,} 人可定义发病区间 "
  f"({100*fh['incident'].mean():.1f}%)；可得率由 wave 5 的 "
  f"{gh.loc[5.0,'interval_pct']:.1f}% 降至 wave 15 的 {gh.loc[15.0,'interval_pct']:.1f}%")
for w, r_ in gh.iterrows():
    rows.append(dict(panel='C-avail', cohort='HRS', scenario=f"Entry wave {int(w)} ({WY_H[int(w)]})",
                     n=int(r_['n_entered']), n_dx=int(r_['n_incident']),
                     availability=round(r_['interval_pct'], 1), onset_med=None,
                     n_analytic=None, deaths=None, crude_rate=None,
                     hr=None, lo=None, hi=None, hr2=None, lo2=None, hi2=None))

# 招募时期敏感性：同一 Table 5 窗口内对比，按入组波次分组重估
def band_py(D):
    out = {}
    for g_, d in D.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0 = max(e, a); e0 = min(x, a + 5)
                if e0 > s0: bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g_] = bands
    return pd.DataFrame(out).fillna(0).sort_index()

def pair(D, a, b, lo, hi):
    X = D.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'] == a].empty or X[X['grp'] == b].empty: return None
    X['cmp'] = (X['grp'] == a).astype(float)
    m = CoxPHFitter()
    m.fit(X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']],
          duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=q['exp(coef)'],
                lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'])

I = inc_all.merge(fh[['entry_wave']].reset_index(), on='pid', how='left', suffixes=('', '_first'))
I['entry_wave'] = I['entry_wave_first'] if 'entry_wave_first' in I else I['entry_wave']
P("\n[招募时期敏感性：窗口固定为 60-75 岁（60-74 vs <60）与 75-90 岁（>=75 vs 60-74）]")
PERIODS = [('Early (waves 5-8; 2000-2006)', [5, 6, 7, 8]),
           ('Middle (waves 9-11; 2008-2012)', [9, 10, 11]),
           ('Late (waves 12-15; 2014-2020)', [12, 13, 14, 15])]
for lab, ws in [('All entry waves', None)] + PERIODS:
    d = I if ws is None else I[I['entry_wave'].isin(ws)]
    res = {}
    for a, b, lo, hi, key in [('60-74', '<60', 60, 75, 'c1'), ('>=75', '60-74', 75, 90, 'c2')]:
        try:
            r = pair(d, a, b, lo, hi)
            res[key] = (round(r['hr'], 2), round(r['lo'], 2), round(r['hi'], 2), int(r['deaths'])) if r else None
        except Exception as ex:
            res[key] = None
    P(f"  {lab:38s} n={len(d):6,}  60-74 vs <60: {res['c1']}   >=75 vs 60-74: {res['c2']}")
    rows.append(dict(panel='C-sens', cohort='HRS', scenario=lab, n=len(d),
                     deaths=(res['c1'][3] if res['c1'] else None),
                     n_dx=None, availability=None, onset_med=None, n_analytic=None, crude_rate=None,
                     hr=(res['c1'][0] if res['c1'] else None), lo=(res['c1'][1] if res['c1'] else None),
                     hi=(res['c1'][2] if res['c1'] else None),
                     hr2=(res['c2'][0] if res['c2'] else None), lo2=(res['c2'][1] if res['c2'] else None),
                     hi2=(res['c2'][2] if res['c2'] else None)))

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}\\tableS7_waves_v5.csv", index=False, encoding='utf-8-sig')
R[R['panel'] == 'A'].to_csv(f"{OUT}\\tableS7_wave_availability.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s47_wave_sensitivity.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 tableS7_waves_v5.csv / s47_wave_sensitivity.txt")
print("DONE")
