# -*- coding: utf-8 -*-
"""Step 43: 第三轮评审要求——比 5 岁段人年更细的共同支持审计。

1) 按单岁计算每个设计、每个起病层的在险人年与死亡数；
2) 对每个两两对比，给出双方同时有在险人年的单岁范围，以及该范围内两组的死亡数；
3) 5 岁段门槛敏感性：>=20 / >=50 / >=100 人年，分别给出窗口与窗口内 HR；
4) 窗口内分析同时限制入组下限与退出上限（与 s33 一致）。

输出 out/s43_support_fine.txt、out/tableS4_support_fine_v5.csv
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


def prevalent_share(s):
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
    # 审计队列：仅保留报告诊断年龄不晚于入组年龄者（v13 起的口径，否则会出现诊断前的风险集）
    D = D[D['onset'] <= D['entry'] + 1e-9].copy()
    D['male'] = (D['sex'] == 1).astype(float)
    D['smoke_ever'] = (D['smoke'] >= 1).astype(float)
    D['grp'] = grp(D['onset'])
    return D.dropna(subset=['grp', 'edu']).reset_index()


def build_incident(df, wave_map):
    df = df.copy()
    df['yr'] = df['wave'].map(wave_map); df['age'] = df['yr'] - df['rabyear']
    df = df.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])
    rows = []
    for pid, x in df.groupby('pid'):
        fp = x[x['arthre'] == 1]
        if fp.empty: continue
        ip = x.index.get_loc(fp.index[0])
        neg = x.iloc[:ip]; neg = neg[neg['arthre'] == 0]
        if neg.empty: continue
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
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


def single_year(D):
    """返回 {grp: {单岁: [人年, 死亡数]}}"""
    out = {}
    for g, d in D.groupby('grp', observed=True):
        cell = {}
        for e, x, ev in zip(d['entry'], d['exit'], d['event']):
            a = max(40, np.floor(e))
            while a < x and a < 110:
                s0 = max(e, a); e0 = min(x, a + 1)
                if e0 > s0:
                    v = cell.setdefault(int(a), [0.0, 0])
                    v[0] += e0 - s0
                a += 1
            if ev and np.isfinite(x):
                da = int(min(np.floor(x - 1e-9), 109))
                cell.setdefault(da, [0.0, 0])[1] += 1
        out[g] = cell
    return out


def band(D, w=5):
    out = {}
    for g, d in D.groupby('grp', observed=True):
        cell = {}
        for e, x in zip(d['entry'], d['exit']):
            a = max(40, np.floor(e / w) * w)
            while a < x and a < 110:
                s0 = max(e, a); e0 = min(x, a + w)
                if e0 > s0: cell[a] = cell.get(a, 0) + (e0 - s0)
                a += w
        out[g] = cell
    return out


def window(D, a, b, thr):
    B = band(D)
    if a not in B or b not in B: return []
    segs = []; cur = None
    for k in sorted(set(B[a]) | set(B[b])):
        ok = B[a].get(k, 0) >= thr and B[b].get(k, 0) >= thr
        if ok and cur is None: cur = [k, k + 5]
        elif ok: cur[1] = k + 5
        elif cur is not None: segs.append(tuple(cur)); cur = None
    if cur is not None: segs.append(tuple(cur))
    return segs


def pair(D, a, b, lo, hi):
    X = D.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].isin([a, b])) & (X['x'] > X['e'])].copy()
    if X.empty or X[X['grp'] == a].empty or X[X['grp'] == b].empty: return None
    X['cmp'] = (X['grp'] == a).astype(float)
    X = X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']]
    try:
        m = CoxPHFitter(); m.fit(X, duration_col='x', event_col='ev', entry_col='e')
    except Exception:
        return None
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=q['exp(coef)'],
                lo=q['exp(coef) lower 95%'], hi=q['exp(coef) upper 95%'])


s = pyreadr.read_r(f"{BASE}\\share.RData")['share'].copy()
s = s.rename(columns={s.columns[0]: 'pid'})
s['yr'] = s['wave'].map(WY_S); s['age'] = s['yr'] - s['rabyear']
h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})

designs = [
    ('SHARE prevalent', prevalent_share(s)),
    ('SHARE incident v5', build_incident(s, WY_S)),
    ('HRS incident v5', build_incident(h, WY_H)),
]
rows = []
for name, D in designs:
    P("=" * 84)
    P(f"# {name}  n={len(D):,} deaths={int(D['event'].sum()):,}")
    P("=" * 84)
    SY = single_year(D)
    for a, b in [('60-74', '<60'), ('>=75', '60-74'), ('>=75', '<60')]:
        ca, cb = SY.get(a, {}), SY.get(b, {})
        both = sorted(k for k in set(ca) | set(cb) if ca.get(k, [0, 0])[0] > 0 and cb.get(k, [0, 0])[0] > 0)
        if both:
            lo, hi = both[0], both[-1] + 1
            da = sum(ca[k][1] for k in both); db = sum(cb[k][1] for k in both)
            pa = sum(ca[k][0] for k in both); pb = sum(cb[k][0] for k in both)
            P(f"  {a} vs {b}: 单岁共同在险 {lo}-{hi}（{len(both)} 个单岁）")
            P(f"    {a}: {pa:,.0f} 人年 / {da:,} 死亡；{b}: {pb:,.0f} 人年 / {db:,} 死亡")
        else:
            lo = hi = None; da = db = 0; pa = pb = 0
            P(f"  {a} vs {b}: 无任何单岁共同在险")
        rows.append(dict(design=name, contrast=f"{a} vs {b}", audit='single-year',
                         threshold=None, window=None if lo is None else f"{lo}-{hi}",
                         n_years=len(both), py_a=round(pa), py_b=round(pb),
                         deaths_a=da, deaths_b=db, hr=None, lo_ci=None, hi_ci=None))
        for thr in (20, 50, 100):
            segs = window(D, a, b, thr)
            if not segs:
                P(f"    [门槛 {thr} 人年/5岁段] 无合格窗口 -> not estimable")
                rows.append(dict(design=name, contrast=f"{a} vs {b}", audit='5-year band',
                                 threshold=thr, window='none', n_years=0, py_a=None, py_b=None,
                                 deaths_a=None, deaths_b=None, hr=None, lo_ci=None, hi_ci=None))
                continue
            lo5, hi5 = segs[0][0], segs[-1][1]
            r = pair(D, a, b, lo5, hi5)
            if r is None:
                P(f"    [门槛 {thr}] 窗口 {lo5:.0f}-{hi5:.0f} 内无法拟合")
                rows.append(dict(design=name, contrast=f"{a} vs {b}", audit='5-year band',
                                 threshold=thr, window=f"{lo5:.0f}-{hi5:.0f}", n_years=None,
                                 py_a=None, py_b=None, deaths_a=None, deaths_b=None,
                                 hr=None, lo_ci=None, hi_ci=None))
                continue
            P(f"    [门槛 {thr}] {lo5:.0f}-{hi5:.0f}: n={r['n']:,} deaths={r['deaths']:,} "
              f"HR {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})")
            rows.append(dict(design=name, contrast=f"{a} vs {b}", audit='5-year band',
                             threshold=thr, window=f"{lo5:.0f}-{hi5:.0f}", n_years=None,
                             py_a=None, py_b=None, deaths_a=r['deaths'], deaths_b=None,
                             hr=round(r['hr'], 2), lo_ci=round(r['lo'], 2), hi_ci=round(r['hi'], 2)))
    P("")

pd.DataFrame(rows).to_csv(f"{OUT}\\tableS4_support_fine_v5.csv", index=False, encoding='utf-8-sig')
io.open(f"{OUT}\\s43_support_fine.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("写出 out/tableS4_support_fine_v5.csv 与 out/s43_support_fine.txt")
