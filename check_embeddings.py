import sys, os, json
sys.path.append('Agent1-RAG')
from backend.database_fixed import get_db_cursor
with get_db_cursor() as cur:
    cur.execute('SELECT COUNT(*) AS c FROM documents')
    print('total_rows=' + str(cur.fetchone()['c']))
    cur.execute('SELECT COUNT(*) AS c FROM documents WHERE embedding IS NOT NULL')
    print('with_embedding=' + str(cur.fetchone()['c']))
    cur.execute('SELECT id, title, LENGTH(content) AS len, chunk_index FROM documents WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT 20')
    print('sample_with_embedding=' + json.dumps([dict(r) for r in cur.fetchall()], ensure_ascii=False))
