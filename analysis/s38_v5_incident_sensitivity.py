# -*- coding: utf-8 -*-
"""Step 38: v5 口径下发病队列的敏感性分析（供 Table S1 Panel B）。

统一规则：
  - 入组 = 首次阳性报告年龄 pos（landmark 场景为 pos+2）
  - 分层 = 区间中点（上下界场景只改分层，不改入组，避免重新引入保证时间）
  - 主口径：exit<=entry 且已死亡者保留（exit=entry+0.5）；存活者零时间剔除
  - 每个对比在共同年龄支持窗口内估计；无支持标 not estimable

输出 out/tableS1_incident_v5.csv、out/s38_v5_incident_sensitivity.txt
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
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


def build_base(df, wave_map):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map)
    df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rows = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty:
            continue
        ip = x.index.get_loc(fp.index[0])
        neg = x.iloc[:ip]
        neg = neg[neg['arthre'] == 0]
        if neg.empty:
            continue
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                         sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0]))
    I = pd.DataFrame(rows)
    I['event'] = I['dead'].notna().astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['male'] = (I['sex'] == 1).astype(float)
    I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
    return I


SCENARIOS = [
    ('main', 'Main v5 (retain same-year deaths)'),
    ('schemeA', 'Exclude all exit<=entry (scheme A)'),
    ('lower', 'Onset = last negative (grouping only)'),
    ('upper', 'Onset = first positive (grouping only)'),
    ('exclude2y', 'Exclude deaths <2 y after first positive'),
    ('landmark2', '2-year landmark (entry = first positive +2)'),
]


def make(I, key):
    D = I.copy()
    if key == 'lower':
        D['onset'] = D['lastneg']
    elif key == 'upper':
        D['onset'] = D['pos']
    else:
        D['onset'] = (D['lastneg'] + D['pos']) / 2
    D['grp'] = grp(D['onset'])
    D = D[(D['onset'] >= 40)].dropna(subset=['grp', 'edu']).copy()
    D['entry'] = D['pos'] + (2 if key == 'landmark2' else 0)
    if key == 'exclude2y':
        D = D[~((D['event'] == 1) & (D['exit'] - D['pos'] < 2))].copy()
    # 只有主口径与其上下界变体保留同年死亡事件；其余场景一律不保留
    if key in ('main', 'lower', 'upper'):
        fix = (D['exit'] <= D['entry']) & (D['event'] == 1)
        D.loc[fix, 'exit'] = D.loc[fix, 'entry'] + 0.5
    return D[D['exit'] > D['entry']].copy()


def band_py(D):
    out = {}
    for g, d in D.groupby('grp', observed=True):
        bands = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / 5) * 5)
            while a < x and a < 110:
                s0 = max(e, a); e0 = min(x, a + 5)
                if e0 > s0:
                    bands[a] = bands.get(a, 0) + (e0 - s0)
                a += 5
        out[g] = bands
    return pd.DataFrame(out).fillna(0).sort_index()


def support(D, a, b):
    T = band_py(D)
    if a not in T.columns or b not in T.columns:
        return []
    segs = []; cur = None
    for band in T.index:
        ok = (T.loc[band, a] >= 20) and (T.loc[band, b] >= 20)
        if ok and cur is None:
            cur = [band, band + 5]
        elif ok:
            cur[1] = band + 5
        elif cur is not None:
            segs.append(tuple(cur)); cur = None
    if cur is not None:
        segs.append(tuple(cur))
    return segs


def pair(D, a, b, lo, hi):
    X = D.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'] == a].empty or X[X['grp'] == b].empty:
        return None
    X['cmp'] = (X['grp'] == a).astype(float)
    X = X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']]
    try:
        m = CoxPHFitter()
        m.fit(X, duration_col='x', event_col='ev', entry_col='e')
    except Exception as e:
        P(f"    Cox 拟合失败: {type(e).__name__}: {e}")
        return None
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=q['exp(coef)'],
                lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'])


rows = []
SAMPLE_N = {}
for lib, path, wmap in [('SHARE', 'share.RData', WY_S), ('HRS', 'hrs.RData', WY_H)]:
    df = pyreadr.read_r(f"{BASE}\\{path}")['share' if lib == 'SHARE' else 'hrs'].copy()
    df = df.rename(columns={df.columns[0]: 'pid'})
    I = build_base(df, wmap)
    P("=" * 78)
    P(f"# {lib} 发病队列敏感性（v5）")
    P("=" * 78)
    for key, label in SCENARIOS:
        D = make(I, key)
        SAMPLE_N[(lib, label)] = len(D)
        P(f"\n[{label}] n={len(D):,} deaths={int(D['event'].sum()):,}")
        for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
            segs = support(D, a, b)
            if not segs:
                P(f"  {a} vs {b}: 无共同支持 -> not estimable")
                rows.append(dict(cohort=lib, scenario=label, contrast=f"{a} vs {b}",
                                 support='none', n=None, deaths=None, hr=None, lo=None, hi=None))
                continue
            lo, hi = segs[0][0], segs[-1][1]
            r = pair(D, a, b, lo, hi)
            if r is None:
                P(f"  {a} vs {b}: 窗口内无足够人数 -> not estimable")
                rows.append(dict(cohort=lib, scenario=label, contrast=f"{a} vs {b}",
                                 support=f"{lo}-{hi}", n=None, deaths=None, hr=None, lo=None, hi=None))
                continue
            P(f"  {a} vs {b}: [{lo},{hi}) n={r['n']:,} deaths={r['deaths']:,} "
              f"HR {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})")
            rows.append(dict(cohort=lib, scenario=label, contrast=f"{a} vs {b}",
                             support=f"{lo}-{hi}", n=r['n'], deaths=r['deaths'],
                             hr=round(r['hr'], 2), lo=round(r['lo'], 2), hi=round(r['hi'], 2)))
    P("")

R = pd.DataFrame(rows)
R['sample_n'] = R.apply(lambda r: SAMPLE_N[(r['cohort'], r['scenario'])], axis=1)
R.to_csv(f"{OUT}\\tableS1_incident_v5.csv", index=False, encoding='utf-8-sig')
P(f"写出 out/tableS1_incident_v5.csv（{len(R)} 行）")
io.open(f"{OUT}\\s38_v5_incident_sensitivity.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("写出 out/s38_v5_incident_sensitivity.txt")
