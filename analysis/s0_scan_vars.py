# -*- coding: utf-8 -*-
"""Step 0: 变量盘点 —— 关节炎起病年龄设计可行性
只做只读扫描，不改数据。
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr
import pandas as pd

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
os.makedirs(OUT, exist_ok=True)

DBS = ["hrs", "charls", "share", "elsa", "mhas", "klosa", "creles", "lasi"]
lines = []
def P(s=""):
    lines.append(str(s))
    print(s)

for db in DBS:
    f = os.path.join(BASE, db + ".RData")
    if not os.path.exists(f):
        P(f"[MISS] {f}"); continue
    d = pyreadr.read_r(f)[db]
    cols = list(d.columns)
    P(f"\n{'='*70}\n### {db}  shape={d.shape}\n{'='*70}")
    # 1) 关节炎相关
    arth = [c for c in cols if 'arthr' in c.lower()]
    P(f"[arthr*] n={len(arth)}: {arth}")
    # 2) 死亡 / 生存
    death = [c for c in cols if c.lower() in
             ('radyear','radmonth','died','alive','raddate','radyear_b','rwstat','iwstat',
              'radyear_e','deadyr','dth','death','time_death')]
    if not death:
        death = [c for c in cols if any(k in c.lower() for k in ('died','radyear','iwstat','dth'))][:20]
    P(f"[death*] {death}")
    # 3) ADL / IADL
    adl = [c for c in cols if any(k in c.lower() for k in ('adl','iadl','walkra','mobility'))]
    P(f"[adl*] {adl[:20]}")
    # 4) 共病
    comorb = [c for c in cols if any(k in c.lower() for k in
              ('hibpe','diabe','heart','strok','lunge','cancre','demene','hchole','psyche','comorb'))]
    P(f"[comorb*] {comorb[:30]}")
    # 5) 起病/病程线索：age at diagnosis
    onset = [c for c in cols if any(k in c.lower() for k in ('age','year','since','dur','onset'))]
    P(f"[onset-hint] {[c for c in onset if not c.islower() or 'age' in c.lower()][:40]}")
    # 6) 年龄/出生年/体重
    P(f"[anchor] agey={[c for c in cols if 'agey' in c]}, rabyear={'rabyear' in cols}, "
      f"ragender={'ragender' in cols}, wave={'wave' in cols}, inw={[c for c in cols if c.startswith('inw')][:5]}")
    # 7) arthre 实测：按波次非缺失值与阳性率
    for c in arth:
        if c.startswith('arthre') or c == 'arthre':
            g = d.groupby('wave')[c].agg(['count', 'sum'])
            g['pct'] = (g['sum'] / g['count'] * 100).round(1)
            P(f"  -- {c} by wave:\n{g.to_string()}")
    P(f"[wave] {sorted(d['wave'].dropna().unique().tolist())}  n_person={d.iloc[:,0].nunique()}")

with open(os.path.join(OUT, "s0_var_scan.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\nWROTE", os.path.join(OUT, "s0_var_scan.txt"))
