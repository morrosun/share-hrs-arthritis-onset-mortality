# -*- coding: utf-8 -*-
"""Step 53: assemble the corrected Table S4 for v7.

The SHARE prevalent rows come from s51, restricted to the respondents whose reported
diagnosis age is not later than their entry age (18,502 on the v18 cohort); the incident
rows are unchanged from s43. Output out/tableS4_support_v7.csv with the column layout
expected by tables4_html().
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd

OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
old = pd.read_csv(os.path.join(OUT, "tableS4_support_fine_v5.csv"), encoding='utf-8-sig')
new = pd.read_csv(os.path.join(OUT, "tableS4_support_fine_v7.csv"), encoding='utf-8-sig')

keep = ['design', 'contrast', 'audit', 'threshold', 'window',
        'deaths_a', 'deaths_b', 'hr', 'lo_ci', 'hi_ci']
prev = new[new['cohort'].astype(str).str.startswith('restricted')][keep].copy()
print('SHARE prevalent source cohort:', prev['cohort'].unique().tolist() if 'cohort' in prev else 'n/a')
prev['design'] = 'SHARE prevalent'
inc = old[old['design'] != 'SHARE prevalent'][keep].copy()
inc['design'] = inc['design'].replace({
    'SHARE incident v5': 'SHARE incident',
    'HRS incident v5': 'HRS incident'})
R = pd.concat([prev, inc], ignore_index=True)
R = R.sort_values(['design', 'contrast', 'audit', 'threshold'],
                  key=lambda c: c.astype(str) if c.name == 'audit' else c)
R.to_csv(os.path.join(OUT, "tableS4_support_v7.csv"), index=False, encoding='utf-8-sig')
print(R.groupby(['design', 'audit']).size().to_string())
print("\nSHARE prevalent single-year overlap after restriction:")
print(R[(R['design'] == 'SHARE prevalent') & (R['audit'] == 'single-year')]
      [['contrast', 'window', 'deaths_a', 'deaths_b']].to_string(index=False))
print("\nWrote out/tableS4_support_v7.csv")
