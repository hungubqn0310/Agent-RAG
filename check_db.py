import sys, os, json
sys.path.append('Agent1-RAG')
from backend.database_fixed import get_db_cursor

try:
    with get_db_cursor() as cur:
        cur.execute('SELECT COUNT(*) AS c FROM documents')
        row = cur.fetchone()
        print('documents_count=' + str(row['c'] if row else 0))
        cur.execute('SELECT id, title, file_path, chunk_index, total_chunks FROM documents ORDER BY id DESC LIMIT 3')
        rows = cur.fetchall()
        print('sample_rows=' + json.dumps([dict(r) for r in rows], ensure_ascii=False))
except Exception as e:
    print('error=' + str(e))
