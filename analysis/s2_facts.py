# -*- coding: utf-8 -*-
"""Step 2: 补齐四项关键事实
1) 炎症指标到底在哪个库（CHARLS 有没有 CRP/WBC/尿酸）
2) radiagarthr 的合法性与与 arthre 的一致性
3) ADL/IADL 各库可用性
4) 死亡信息（radyear）覆盖
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, pandas as pd, numpy as np

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines=[]
def P(s=""):
    lines.append(str(s)); print(s)

def load(db):
    f = os.path.join(BASE, db + ".RData")
    if not os.path.exists(f):
        P(f"[MISS] {f}")
        return None
    return pyreadr.read_r(f)[db]

P("="*72+"\n# 1. 炎症/生化变量是否存在\n"+"="*72)
LAB = ['crp','hscrp','wbc','uric','ua','fibrinogen','esr','il6','tnf','albumin','hgb','neutro']
for db in ['charls','hrs','share','elsa','mhas','klosa']:
    d = load(db)
    if d is None:
        continue
    cols=[c.lower() for c in d.columns]
    hit = [c for c in d.columns if any(k in c.lower() for k in LAB)]
    P(f"{db:8s} lab-like cols: {hit if hit else '❌ 无'}")

P("\n"+"="*72+"\n# 2. radiagarthr 合法性（SHARE / ELSA / KLoSA）\n"+"="*72)
for db in ['share','elsa','klosa']:
    d = load(db)
    if d is None:
        continue
    v = d[['arthre','radiagarthr','rabyear','wave']].copy()
    d['_a'] = d['wave']
    P(f"\n--- {db} ---")
    P("  按 arthre 分组 radiagarthr 非缺失率:")
    P("  " + d.groupby('arthre')['radiagarthr'].agg(
        n='size', nonmiss=lambda x: x.notna().sum(),
        pct=lambda x: round(100*x.notna().mean(),1)).to_string().replace("\n","\n  "))
    P(f"  radiagarthr==0: {(v['radiagarthr']==0).sum()}   <5: {((v['radiagarthr']>0)&(v['radiagarthr']<5)).sum()}")
    # 诊断年龄 > 当前年龄 的不可能值
    d['_age'] = d['agey'] - (2018 - d['wave']*2) if 'agey' in d and db=='elsa' else d.get('agey')
    bad = d[(d['radiagarthr'].notna()) & (d['agey'].notna()) &
            (d['radiagarthr'] > d['agey']+1)]
    P(f"  诊断年龄 > 当前年龄+1 的疑似非法值: {len(bad)}  "
      f"({100*len(bad)/max(1,d['radiagarthr'].notna().sum()):.2f}%)")
    # 每人内部一致性：radiagarthr 是否跨波稳定
    per = d.dropna(subset=['radiagarthr']).groupby(d.columns[0])['radiagarthr'].agg(['nunique','count'])
    per = per[per['count']>=2]
    P(f"  跨波重复报告者 n={len(per)}  其中诊断年龄不变的: {(per['nunique']==1).sum()} "
      f"({100*(per['nunique']==1).mean():.1f}%)")

P("\n"+"="*72+"\n# 3. ADL / IADL 可用性\n"+"="*72)
for db in ['hrs','charls','share','elsa','mhas','klosa']:
    d = load(db)
    if d is None:
        continue
    cand = [c for c in d.columns if any(k in c.lower() for k in ('adl','iadl','walkra','dadliv','mobility','adltot'))]
    P(f"{db:8s}: {cand}")
    for c in cand:
        nm = d[c].notna().sum()
        P(f"          {c:14s} nonmiss={nm:7d}  mean={d[c].mean():.2f}")

P("\n"+"="*72+"\n# 4. 死亡信息\n"+"="*72)
for db in ['hrs','charls','share','elsa','mhas','klosa']:
    d = load(db)
    if d is None:
        continue
    if 'radyear' not in d.columns: P(f"{db}: 无 radyear"); continue
    yr = d.groupby(d.columns[0])['radyear'].max()
    P(f"{db:8s} 有 rabyear: {'rabyear' in d.columns}  "
      f"死亡人数={yr.notna().sum():6d} / {len(yr):6d} ({100*yr.notna().mean():4.1f}%)  "
      f"死亡年 median={yr.median():.0f}  iwstat取值={sorted(d['iwstat'].dropna().unique())[:6] if 'iwstat' in d else 'NA'}")

with open(os.path.join(OUT,'s2_facts.txt'),'w',encoding='utf-8') as f: f.write("\n".join(lines))
print("\nWROTE s2_facts.txt")
