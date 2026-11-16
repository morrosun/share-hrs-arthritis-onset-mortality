# -*- coding: utf-8 -*-
"""Step 36: 逐人核对 v5 发病队列样本数差异（s30 / s31 / s33 三套构建口径）。

输出 out/s36_reconcile_n.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def base_table(df, wave_map):
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
        prior = x.iloc[:ip]
        neg = prior[prior['arthre'] == 0]
        if neg.empty:
            continue
        rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                         last=x['age'].iloc[-1], dead=x['radyear'].max(),
                         byear=x['rabyear'].iloc[0], edu=x['raeducl'].iloc[0],
                         sex=x['ragender'].iloc[0], smoke=x['smoken'].iloc[0],
                         pos_wave=fp['wave'].iloc[0], last_wave=x['wave'].iloc[-1]))
    I = pd.DataFrame(rows)
    I['onset'] = (I['lastneg'] + I['pos']) / 2
    I['event'] = I['dead'].notna().astype(int)
    I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
    I['entry'] = I['pos']
    I['exit_le_entry'] = I['exit'] <= I['entry']
    # s31 的分组边界：pd.cut(onset, [40, 59, 74, 120])
    I['grp_s31'] = pd.cut(I['onset'], [40, 59, 74, 120], labels=['<60', '60-74', '>=75'])
    # s33 的分组边界：pd.cut(onset, [4, 59.999, 74.999, 120]) 且 onset >= 40
    I['grp_s33'] = pd.cut(I['onset'], [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])
    return I


def retain_events(B):
    """方案 B：exit<=entry 且已死亡者把 exit 设为 entry+0.5，随后剔除零时间者。"""
    B = B.copy()
    fix = (B['exit'] <= B['entry']) & (B['event'] == 1)
    B.loc[fix, 'exit'] = B.loc[fix, 'entry'] + 0.5
    return B[B['exit'] > B['entry']].copy()


def show_diff(name, a, b, I, limit=15):
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    P(f"  {name}: 仅在前者 {len(only_a)} 人，仅在后者 {len(only_b)} 人")
    for tag, ids in [('仅前者', only_a), ('仅后者', only_b)]:
        if not ids:
            continue
        d = I[I['pid'].isin(ids)]
        P(f"    {tag}原因：edu 缺失 {int(d['edu'].isna().sum())}，"
          f"onset<=40 {int((d['onset'] <= 40).sum())}，onset>120 {int((d['onset'] > 120).sum())}，"
          f"exit<=entry {int(d['exit_le_entry'].sum())}，其中死亡 {int(d.loc[d['exit_le_entry'], 'event'].sum())}")
        show = d[['pid', 'onset', 'grp_s31', 'grp_s33', 'edu', 'pos', 'exit', 'event', 'exit_le_entry']].head(limit)
        P("    " + show.to_string(index=False).replace("\n", "\n    "))


def run(I, label):
    P("=" * 78)
    P(f"# {label}")
    P("=" * 78)
    P(f"[s30 口径] 发病队列全体（无 grp/edu 过滤）= {len(I):,}")
    P(f"  exit<=entry = {int(I['exit_le_entry'].sum()):,}，其中死亡 {int(I.loc[I['exit_le_entry'], 'event'].sum()):,}")
    P(f"  exit>entry  = {int((~I['exit_le_entry']).sum()):,}")

    s31 = I.dropna(subset=['grp_s31', 'edu']).copy()
    s33 = I[(I['onset'] >= 40)].dropna(subset=['grp_s33', 'edu']).copy()
    P(f"[s31 口径] 过滤 grp_s31 / edu 后 = {len(s31):,}；exit<=entry = {int(s31['exit_le_entry'].sum()):,}（死亡 {int(s31.loc[s31['exit_le_entry'], 'event'].sum()):,}）")
    P(f"[s33 口径] onset>=40 + 过滤 grp_s33 / edu 后 = {len(s33):,}；exit<=entry = {int(s33['exit_le_entry'].sum()):,}（死亡 {int(s33.loc[s33['exit_le_entry'], 'event'].sum()):,}）")

    A31 = s31[~s31['exit_le_entry']].copy()
    B31 = retain_events(s31)
    B33 = retain_events(s33)
    P(f"[方案 A / s31] 直接剔除 exit<=entry = {len(A31):,}")
    P(f"[方案 B / s31] 保留同年死亡事件 = {len(B31):,}，死亡 {int(B31['event'].sum()):,}")
    P(f"[方案 B / s33] 保留同年死亡事件 = {len(B33):,}，死亡 {int(B33['event'].sum()):,}")

    P("")
    P("逐人差集：")
    show_diff("s30全体 vs s31基础集", I['pid'], s31['pid'], I)
    show_diff("s31基础集 vs s33基础集", s31['pid'], s33['pid'], I)
    show_diff("方案B(s31) vs 方案B(s33)", B31['pid'], B33['pid'], I)

    # 分组边界差异（不影响人数，但影响 HR）
    both = I[I['pid'].isin(set(s31['pid']) & set(s33['pid']))].copy()
    mism = both[both['grp_s31'].astype(str) != both['grp_s33'].astype(str)]
    P(f"  分组边界不一致（s31 用 59/74，s33 用 59.999/74.999）: {len(mism):,} 人")
    if len(mism):
        P("    " + mism[['pid', 'onset', 'grp_s31', 'grp_s33']].head(10).to_string(index=False).replace("\n", "\n    "))
    P("")


share = pyreadr.read_r(os.path.join(BASE, "share.RData"))['share'].copy()
share = share.rename(columns={share.columns[0]: 'pid'})
hrs = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
hrs = hrs.rename(columns={hrs.columns[0]: 'pid'})

run(base_table(share, WY_S), 'SHARE 发病队列')
run(base_table(hrs, WY_H), 'HRS 发病队列')

io.open(os.path.join(OUT, "s36_reconcile_n.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("写出 out/s36_reconcile_n.txt")
