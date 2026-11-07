# -*- coding: utf-8 -*-
"""Step 1: 关键诊断
1) 各库波次覆盖与 arthre / radiagarthr 的实测覆盖
2) HRS/CHARLS 发病队列能落到多少人、起病年龄分布能否分出 YORA
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, pandas as pd, numpy as np

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
lines = []
def P(s=""):
    lines.append(str(s)); print(s)

WAVE_YEAR = {
 'hrs':   {5:2000,6:2002,7:2004,8:2006,9:2008,10:2010,11:2012,12:2014,13:2016,14:2018,15:2020,16:2022},
 'charls':{1:2011,2:2013,3:2015,4:2018,5:2020},
 'share': {1:2004,2:2007,3:2008,4:2011,5:2013,6:2015,7:2017,8:2020},
 'elsa':  {1:2002,2:2004,3:2006,4:2008,5:2010,6:2012,7:2014,8:2016,9:2018},
 'mhas':  {1:2001,2:2003,3:2012,4:2015,5:2018,6:2021},
 'klosa': {1:2006,2:2008,3:2010,4:2012,5:2014,6:2016,7:2018,8:2020},
}

def load(db):
    return pyreadr.read_r(os.path.join(BASE, db + ".RData"))[db]

# ---------- A. 覆盖诊断 ----------
for db in ['hrs','charls','share','elsa','mhas','klosa']:
    d = load(db)
    P(f"\n{'='*72}\n### {db}   rows={len(d)}  persons={d.iloc[:,0].nunique()}\n{'='*72}")
    cols = list(d.columns)
    art  = [c for c in cols if c in ('arthre',)][0]
    dg   = 'radiagarthr' if 'radiagarthr' in cols else None
    d['_wy'] = d['wave'].map(WAVE_YEAR[db])
    rows = []
    for w, x in d.groupby('wave'):
        rows.append({
            'wave': w,
            'yr': WAVE_YEAR[db].get(int(w)),
            'n_rows': len(x),
            'n_pers': x.iloc[:,0].nunique(),
            'age_mean': round(x['agey'].mean(),1) if 'agey' in x else np.nan,
            'arthre_n': int(x[art].notna().sum()),
            'arthre_pct': round(x[art].mean()*100,1),
            'radiag_n': int(x[dg].notna().sum()) if dg else 0,
            'radiag_med': round(x[dg].median(),1) if dg and x[dg].notna().any() else np.nan,
            'iwstat': dict(x['iwstat'].value_counts().head(4)) if 'iwstat' in x else {},
        })
    g = pd.DataFrame(rows).set_index('wave')
    P(g.to_string())
    if dg:
        v = d[dg].dropna()
        P(f"\n  radiagarthr n={len(v)}  mean={v.mean():.1f} min={v.min():.0f} max={v.max():.0f}")
        P(f"  分布: <45={100*(v<45).mean():.1f}%  45-59={100*((v>=45)&(v<60)).mean():.1f}%  "
          f"60-74={100*((v>=60)&(v<75)).mean():.1f}%  >=75={100*(v>=75).mean():.1f}%")
    P(f"  [radiagarthr 存在?] {bool(dg)}   [rxarthr 存在?] {'rxarthr' in cols}")

# ---------- B. 发病队列实算 HRS ----------
def incidence_cohort(db, label):
    d = load(db).copy()
    d = d.rename(columns={d.columns[0]: 'pid'})
    d['_yr'] = d['wave'].map(WAVE_YEAR[db])
    P(f"\n\n{'#'*72}\n# 发病队列实算 — {label}\n{'#'*72}")
    # 每人：首次有效波、基线 arthre
    d = d.sort_values(['pid','wave'])
    has = d[d['arthre'].notna()]
    base = has.groupby('pid').first()[['wave','_yr','arthre','rabyear','ragender']]
    base = base.rename(columns={'wave':'w0','_yr':'yr0','arthre':'art0'})
    P(f"有任意有效 arthre 者: {len(base)}")
    P(f"  基线(首次有效波) arthre=0 (无关节炎): {(base['art0']==0).sum()}")
    P(f"  基线 arthre=1 (已患病，剔除)       : {(base['art0']==1).sum()}")
    # 发病：首个 arthre==1 的波次
    neww = has[has['arthre']==1].groupby('pid').first()
    base['w1st'] = neww['wave'].reindex(base.index)
    base['yr1st'] = neww['_yr'].reindex(base.index)
    coh = base[base['art0']==0].copy()
    coh['onset_age'] = coh['yr1st'] - coh['rabyear']
    coh['base_age'] = coh['yr0'] - coh['rabyear']
    P(f"\n发病队列 n={len(coh)}")
    P(f"  随访中转为 arthre=1 : {coh['yr1st'].notna().sum()}  ({100*coh['yr1st'].notna().mean():.1f}%)")
    P(f"  始终阴性(右删失)    : {coh['yr1st'].isna().sum()}")
    P(f"  基线年龄 mean={coh['base_age'].mean():.1f}  "
      f"p10={coh['base_age'].quantile(.1):.0f} p50={coh['base_age'].median():.0f} p90={coh['base_age'].quantile(.9):.0f}")
    P(f"  基线年龄 <45 岁者: {(coh['base_age']<45).sum()}  "
      f"45-49: {((coh['base_age']>=45)&(coh['base_age']<50)).sum()}")
    o = coh['onset_age'].dropna()
    if len(o):
        P(f"\n  起病年龄 (报告时点) n={len(o)} mean={o.mean():.1f} p50={o.median():.0f} "
          f"min={o.min():.0f} max={o.max():.0f}")
        P(f"    分档 <45={100*(o<45).mean():.1f}%  45-59={100*((o>=45)&(o<60)).mean():.1f}%  "
          f"60-74={100*((o>=60)&(o<75)).mean():.1f}%  >=75={100*(o>=75).mean():.1f}%")
        P(f"    绝对数: <45={(o<45).sum()}  45-59={((o>=45)&(o<60)).sum()}  "
          f"60-74={((o>=60)&(o<75)).sum()}  >=75={(o>=75).sum()}")
    # 死亡
    if 'radyear' in d.columns:
        dd = d.groupby('pid')['radyear'].max()
        coh['deadyr'] = dd.reindex(coh.index)
        coh['died'] = coh['deadyr'].notna()
        P(f"\n  死亡: {coh['died'].sum()} ({100*coh['died'].mean():.1f}%)  "
          f"其中起病者中死亡 {coh.loc[coh['yr1st'].notna(),'died'].sum()}")
        coh['fu'] = np.where(coh['died'], coh['deadyr']-coh['yr0'], np.nan)
        P(f"  随访年数(死亡者) mean={coh['fu'].mean():.1f}")
    return coh

incidence_cohort('hrs', 'HRS')
incidence_cohort('charls', 'CHARLS')

with open(os.path.join(OUT,'s1_diagnosis.txt'),'w',encoding='utf-8') as f:
    f.write("\n".join(lines))
print("\nWROTE s1_diagnosis.txt")
