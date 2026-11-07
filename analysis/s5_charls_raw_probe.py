# -*- coding: utf-8 -*-
"""查 CHARLS 原始表是否含"确诊年份/年龄"字段（RData 面板未收录）"""
import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import psycopg2
CONN = "host=%s port=%s user=%s password=%s dbname=%s" % (
    os.environ.get("CHARLS_DB_HOST", "localhost"),
    os.environ.get("CHARLS_DB_PORT", "5432"),
    os.environ.get("CHARLS_DB_USER", "postgres"),
    os.environ.get("CHARLS_DB_PASSWORD", ""),
    os.environ.get("CHARLS_DB_NAME", "charls"))
conn = psycopg2.connect(CONN); cur = conn.cursor()

print("="*70); print("# 1. 含 diag / year / age 的列（按表）"); print("="*70)
cur.execute("""
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema='public'
  AND (column_name ILIKE '%diag%' OR column_name ILIKE '%year%')
ORDER BY table_name, column_name LIMIT 200
""")
rows = cur.fetchall()
print(f"命中 {len(rows)} 列")
for t, c in rows[:200]:
    print(f"  {t:28s} {c}")

print("\n" + "="*70); print("# 2. CHARLS 表清单（原始波次表）"); print("="*70)
cur.execute("""
SELECT table_name FROM information_schema.tables
WHERE table_schema='public' AND table_type='BASE TABLE'
ORDER BY table_name
""")
tabs = [r[0] for r in cur.fetchall()]
print(f"共 {len(tabs)} 张表")
print(", ".join(tabs))

print("\n" + "="*70); print("# 3. 变量标签表里搜 arthritis / 诊断年"); print("="*70)
cur.execute("""SELECT table_name FROM information_schema.tables
WHERE table_schema='public' AND (table_name ILIKE '%label%' OR table_name ILIKE '%var%')""")
print("标签表:", [r[0] for r in cur.fetchall()])
conn.close()
