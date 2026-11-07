# -*- coding: utf-8 -*-
"""Step 48: 病程分析（预先设定估计目标）。

估计目标（唯一可识别者）：
  diagnosis-anchored mortality trajectory —— 以「首次阳性报告」为时间原点，
  估计 (i) 诊断后各病程区间的死亡率，(ii) 在相同诊断后时长下，不同起病年龄层
  的死亡风险比（调整首报年龄、性别、教育、吸烟）。
  这是诊断后生存的描述性对比，不是病程的因果效应。

不可识别性：attained age = onset age + time since onset。任何以 attained age
为时间尺度的模型，其 onset-age 系数与病程系数是同一线性对比的重参数化。
本脚本用 SHARE prevalent 队列做数值验证（panel C）。

  A. HRS incident：诊断后各病程区间 × 起病层的死亡率
  B. HRS incident：诊断锚定 Cox（时间轴 = 自首报时长）
  C. SHARE prevalent：onset10 与 duration-at-reference-age 的重参数化等价性
  D. SHARE prevalent：入组时病程（entry_age - onset_age）的分布与关联

输出 out/tableS6_duration_v5.csv、out/s48_duration.txt
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
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []
def P(s=""):
    lines.append(str(s)); print(s, flush=True)

def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])

rows = []
def add(panel, **kw):
    base = dict(panel=panel, cohort=None, stratum=None, term=None, n=None, deaths=None,
                py=None, rate=None, est=None, lo=None, hi=None, p=None, loglik=None)
    base.update(kw)
    rows.append(base)

# ==================== 数据 ====================
s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H); h['age'] = h['yr'] - h['rabyear']

def build_incident(df):
    df = df.copy().dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    out = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0]); prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty: continue
        out.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                        last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                        sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
    I = pd.DataFrame(out)
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

def prevalent_share():
    agg = s.groupby('pid').agg(entry=('age', 'min'), last=('age', 'max'), sex=('ragender', 'first'),
                               byear=('rabyear', 'first'), edu=('raeducl', 'first'), smoke=('smoken', 'max'),
                               w0=('wave', 'min'))
    dg = s.dropna(subset=['radiagarthr']).groupby('pid')['radiagarthr'].first().rename('onset')
    dd = s.groupby('pid')['radyear'].max().rename('dead')
    D = agg.join(dg).join(dd).dropna(subset=['onset'])
    D = D[D['onset'] >= 5]
    D['event'] = D['dead'].notna().astype(int)
    D['exit'] = np.where(D['event'] == 1, D['dead'] - D['byear'], D['last'])
    _zero = D['exit'] <= D['entry']
    D.loc[_zero & (D['event'] == 1), 'exit'] = D.loc[_zero & (D['event'] == 1), 'entry'] + 0.5
    D = D[(D['exit'] > D['entry']) & (D['entry'] >= 40) & (D['exit'] <= 110) & (D['w0'] != 8)].copy()
    D['male'] = (D['sex'] == 1).astype(float)
    D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
    D['grp'] = grp(D['onset'])
    return D.dropna(subset=['grp', 'edu']).reset_index()

I = build_incident(h)
D = prevalent_share()
P("=" * 84 + "\n# 样本\n" + "=" * 84)
P(f"HRS incident (v5) n={len(I):,} deaths={int(I['event'].sum()):,}")
P(f"SHARE prevalent  n={len(D):,} deaths={int(D['event'].sum()):,}")

# ==================== A. 诊断后病程区间 × 起病层的粗死亡率 ====================
P("\n" + "=" * 84 + "\n# A. HRS：诊断后病程区间的死亡率（时间原点 = 首次阳性报告）\n" + "=" * 84)
BOUNDS = [0, 2, 5, 10, np.inf]
LAB = ['0-2', '2-5', '5-10', '>=10']
seg = []
for _, r in I.iterrows():
    stop = r['exit'] - r['entry']
    for j in range(4):
        a, b = BOUNDS[j], BOUNDS[j + 1]
        s0 = max(0.0, a); e0 = min(stop, b)
        if e0 > s0 + 1e-9:
            ev = 0
            if j == 3 or (stop <= b + 1e-9 and stop > a):
                ev = int(r['event']) if (e0 >= stop - 1e-9) else 0
            seg.append((r['grp'], LAB[j], e0 - s0, ev))
        if stop <= b + 1e-9: break
S = pd.DataFrame(seg, columns=['grp', 'dur_band', 'py', 'death'])
G = S.groupby(['grp', 'dur_band'], observed=True).agg(py=('py', 'sum'), deaths=('death', 'sum')).reset_index()
G['rate'] = 1000 * G['deaths'] / G['py']
P(G.pivot(index='dur_band', columns='grp', values='rate').round(1).to_string())
P("\n[人年与死亡数]")
P(G.pivot(index='dur_band', columns='grp', values='deaths').to_string())
for _, r in G.iterrows():
    add('A', cohort='HRS', stratum=str(r['grp']), term=r['dur_band'],
        n=int(I[I['grp'] == r['grp']].shape[0]), deaths=int(r['deaths']),
        py=round(r['py'], 0), rate=round(r['rate'], 1))
P("\n[全队列合计（不分起病层）]")
GT = S.groupby('dur_band', observed=True).agg(py=('py', 'sum'), deaths=('death', 'sum'))
GT['rate'] = 1000 * GT['deaths'] / GT['py']
P(GT.round(1).to_string())
for band, r in GT.iterrows():
    add('A', cohort='HRS', stratum='All', term=band,
        n=len(I), deaths=int(r['deaths']), py=round(r['py'], 0), rate=round(r['rate'], 1))

# ==================== B. 诊断锚定 Cox ====================
P("\n" + "=" * 84 + "\n# B. HRS：诊断锚定 Cox（时间轴 = 自首报时长）\n" + "=" * 84)
X = I.copy()
X['tsr'] = X['exit'] - X['entry']
X['entry0'] = 0.0
X['g_60_74'] = (X['grp'] == '60-74').astype(float)
X['g_ge75'] = (X['grp'] == '>=75').astype(float)
X['age_report_c'] = X['entry'] - X['entry'].mean()
X = X[X['tsr'] > 0].copy()

def fit_tsr(cols, form, label):
    m = CoxPHFitter()
    m.fit(X[cols + ['tsr', 'entry0', 'event']].dropna(),
          duration_col='tsr', event_col='event', entry_col='entry0', formula=form)
    P(f"\n--- {label} --- n={m._n_examples:,} loglik={m.log_likelihood_:.3f}")
    for key, lab in [('g_60_74', '60-74 vs <60'), ('g_ge75', '>=75 vs <60')]:
        q = m.summary.loc[key]
        P(f"  {lab:16s} HR {q['exp(coef)']:.3f} ({q['exp(coef) lower 95%']:.3f}-{q['exp(coef) upper 95%']:.3f})  p={q['p']:.3g}")
        add('B', cohort='HRS', term=lab, n=int(m._n_examples),
            est=round(q['exp(coef)'], 3), lo=round(q['exp(coef) lower 95%'], 3),
            hi=round(q['exp(coef) upper 95%'], 3), p=round(float(q['p']), 5),
            loglik=round(float(m.log_likelihood_), 3), stratum=label)
    return m

COV = ['g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']
fit_tsr(COV, "g_60_74 + g_ge75 + male + edu + smoke_ever", 'Unadjusted for age at first report')
fit_tsr(COV + ['age_report_c'],
        "g_60_74 + g_ge75 + age_report_c + male + edu + smoke_ever",
        'Adjusted for age at first report (primary duration estimand)')

# 对照：attained-age 尺度（v5 主口径，全年龄）
Y = I.copy()
Y['g_60_74'] = (Y['grp'] == '60-74').astype(float)
Y['g_ge75'] = (Y['grp'] == '>=75').astype(float)
m_age = CoxPHFitter()
m_age.fit(Y[['entry', 'exit', 'event', 'g_60_74', 'g_ge75', 'male', 'edu', 'smoke_ever']].dropna(),
          duration_col='exit', event_col='event', entry_col='entry',
          formula="g_60_74 + g_ge75 + male + edu + smoke_ever")
P(f"\n--- 对照：attained-age 尺度（v5 Table 5 全年龄口径）---")
for key, lab in [('g_60_74', '60-74 vs <60'), ('g_ge75', '>=75 vs <60')]:
    q = m_age.summary.loc[key]
    P(f"  {lab:16s} HR {q['exp(coef)']:.3f} ({q['exp(coef) lower 95%']:.3f}-{q['exp(coef) upper 95%']:.3f})")
    add('B', cohort='HRS', term=lab, n=int(m_age._n_examples),
        est=round(q['exp(coef)'], 3), lo=round(q['exp(coef) lower 95%'], 3),
        hi=round(q['exp(coef) upper 95%'], 3), stratum='Attained-age time scale (comparison)',
        loglik=round(float(m_age.log_likelihood_), 3))

# ==================== C. 不可识别性数值验证 ====================
P("\n" + "=" * 84 + "\n# C. SHARE：病程系数 = 起病年龄系数的重参数化\n" + "=" * 84)
Z = D.copy()
Z['onset10'] = Z['onset'] / 10
Z['dur_ref10'] = (75 - Z['onset']) / 10        # 在 75 岁时的病程，每 10 年
BASECOV = ['entry', 'exit', 'event', 'male', 'edu', 'smoke_ever']
res = {}
for key, var, label in [('onset', 'onset10', 'Age at onset, per +10 y'),
                        ('duration', 'dur_ref10', 'Duration at attained age 75, per +10 y')]:
    m = CoxPHFitter()
    m.fit(Z[BASECOV + [var]].dropna(), duration_col='exit', event_col='event',
          entry_col='entry', formula=f"{var} + male + edu + smoke_ever")
    q = m.summary.loc[var]
    res[key] = dict(coef=float(q['coef']), se=float(q['se(coef)']), hr=float(q['exp(coef)']),
                    lo=float(q['exp(coef) lower 95%']), hi=float(q['exp(coef) upper 95%']),
                    loglik=float(m.log_likelihood_), p=float(q['p']))
    P(f"  {label:42s} beta={res[key]['coef']:+.6f}  HR={res[key]['hr']:.4f} "
      f"({res[key]['lo']:.4f}-{res[key]['hi']:.4f})  loglik={res[key]['loglik']:.6f}")
    add('C', cohort='SHARE', term=label, n=len(Z), est=round(res[key]['hr'], 4),
        lo=round(res[key]['lo'], 4), hi=round(res[key]['hi'], 4),
        p=round(res[key]['p'], 6), loglik=round(res[key]['loglik'], 6))
P(f"  beta(duration) = -beta(onset)? {np.isclose(res['duration']['coef'], -res['onset']['coef'], atol=1e-8)}")
P(f"  HR(duration) x HR(onset) = {res['duration']['hr']*res['onset']['hr']:.8f} (应=1)")
P(f"  loglik 差 = {res['duration']['loglik']-res['onset']['loglik']:.2e}")
P("  => 在以 attained age 为时间尺度的模型中，'病程效应'与'起病年龄效应'是同一系数，无法单独识别。")
add('C', cohort='SHARE', term='HR product (duration x onset)', est=round(res['duration']['hr']*res['onset']['hr'], 6),
    loglik=round(res['duration']['loglik']-res['onset']['loglik'], 8))

# ==================== D. SHARE 入组时病程 ====================
P("\n" + "=" * 84 + "\n# D. SHARE：诊断至入组的病程（duration at entry）\n" + "=" * 84)
Z['dur_entry'] = Z['entry'] - Z['onset']
P(f"入组时病程: 中位 {Z['dur_entry'].median():.1f} 年 (IQR "
  f"{Z['dur_entry'].quantile(.25):.1f}-{Z['dur_entry'].quantile(.75):.1f})，"
  f"均值 {Z['dur_entry'].mean():.1f} 年；范围 {Z['dur_entry'].min():.1f}-{Z['dur_entry'].max():.1f}")
P(f"负值（入组前诊断年龄大于入组年龄）: {int((Z['dur_entry']<0).sum()):,} 人")
Z['dur_band'] = pd.cut(Z['dur_entry'], [-100, 5, 15, 200], labels=['<5', '5-15', '>=15'])
tmp = Z.groupby('dur_band', observed=True).agg(n=('event', 'size'), deaths=('event', 'sum'))
Z['_py'] = Z['exit'] - Z['entry']
tmp['py'] = Z.groupby('dur_band', observed=True)['_py'].sum()
tmp['rate'] = 1000 * tmp['deaths'] / tmp['py']
P(tmp.round(1).to_string())
for band, r in tmp.iterrows():
    add('D', cohort='SHARE', stratum=str(band) + ' y', term='Duration at entry',
        n=int(r['n']), deaths=int(r['deaths']), py=round(r['py'], 0), rate=round(r['rate'], 1))

Z['d_5_15'] = (Z['dur_band'] == '5-15').astype(float)
Z['d_ge15'] = (Z['dur_band'] == '>=15').astype(float)
m_d = CoxPHFitter()
m_d.fit(Z[BASECOV + ['d_5_15', 'd_ge15']].dropna(), duration_col='exit', event_col='event',
        entry_col='entry', formula="d_5_15 + d_ge15 + male + edu + smoke_ever")
P("\n[age-scale Cox，参考 <5 年]")
for key, lab in [('d_5_15', '5-15 vs <5 y'), ('d_ge15', '>=15 vs <5 y')]:
    q = m_d.summary.loc[key]
    P(f"  {lab:16s} HR {q['exp(coef)']:.3f} ({q['exp(coef) lower 95%']:.3f}-{q['exp(coef) upper 95%']:.3f})")
    add('D', cohort='SHARE', term=lab, n=int(m_d._n_examples),
        est=round(q['exp(coef)'], 3), lo=round(q['exp(coef) lower 95%'], 3),
        hi=round(q['exp(coef) upper 95%'], 3), p=round(float(q['p']), 6))
P("\n注：duration at entry = entry age - onset age，同时携带入组年龄信息，")
P("    因此上述 HR 不是病程效应；它与 Table 3 的起病年龄系数并不互为倒数。")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}\\tableS6_duration_v5.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s48_duration.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\n写出 tableS6_duration_v5.csv / s48_duration.txt")
print("DONE")
