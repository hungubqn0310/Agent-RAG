import os
import asyncio
import logging
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agent1_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD must be set in .env file")

# Cấu hình connection pool
MIN_CONN = 5
MAX_CONN = 20
connection_pool = None

@contextmanager
def get_db_connection():
    """Context manager để lấy và trả connection từ pool"""
    global connection_pool
    if connection_pool is None:
        connection_pool = SimpleConnectionPool(
            MIN_CONN,
            MAX_CONN,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)

@contextmanager
def get_db_cursor(commit=False):
    """Context manager để lấy cursor với connection từ pool"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        finally:
            cursor.close()

def initialize_db():
    """Khởi tạo database và schema"""
    try:
        # Kết nối đến database mặc định postgres để tạo DB mới nếu cần
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Created database '{DB_NAME}'")
        
        cursor.close()
        conn.close()

        # Khởi tạo schema trong database đích
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Đọc schema từ file
            schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema.sql")
            with open(schema_path, "r", encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Split và execute từng statement
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                if statement:
                    cursor.execute(statement)
            
            conn.commit()
            cursor.close()
            
        logger.info("Database schema initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

async def search_similar_documents(query_embedding, query_text=None, limit=5, threshold=0.7):
    """Search for similar documents using vector similarity"""
    try:
        with get_db_cursor() as cursor:
            # Convert numpy array to list for PostgreSQL
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            if query_text:
                # Sử dụng RRF như trong mcp-rag với hỗ trợ unaccent để match tiếng Việt tốt hơn
                sql_query = """
                CREATE EXTENSION IF NOT EXISTS unaccent;
                WITH vector_results AS (
                    SELECT 
                        id,
                        ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT 100
                ),
                keyword_results AS (
                    SELECT 
                        id,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(to_tsvector('simple', unaccent(content)), plainto_tsquery('simple', unaccent(%s))) DESC
                        ) AS rank
                    FROM documents
                    WHERE to_tsvector('simple', unaccent(content)) @@ plainto_tsquery('simple', unaccent(%s))
                    ORDER BY ts_rank_cd(to_tsvector('simple', unaccent(content)), plainto_tsquery('simple', unaccent(%s))) DESC
                    LIMIT 200
                ),
                fused_results AS (
                    SELECT
                        id,
                        COALESCE(1.0 / (60 + vr.rank), 0.0) + COALESCE(1.0 / (60 + kr.rank), 0.0) AS rrf_score
                    FROM vector_results vr
                    FULL OUTER JOIN keyword_results kr USING (id)
                )
                SELECT 
                    d.id, d.title, d.content, d.file_path, d.chunk_index, d.total_chunks, d.metadata,
                    fr.rrf_score as similarity
                FROM fused_results fr
                JOIN documents d ON fr.id = d.id
                WHERE fr.rrf_score >= %s
                ORDER BY fr.rrf_score DESC
                LIMIT %s
                """
                cursor.execute(sql_query, (embedding_list, embedding_list, query_text, query_text, query_text, threshold, limit))
            else:
                # Chỉ dùng vector similarity
                sql_query = """
                SELECT id, title, content, file_path, chunk_index, total_chunks, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents 
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                cursor.execute(sql_query, (embedding_list, embedding_list, threshold, embedding_list, limit))
            
            results = cursor.fetchall()
            
            return results
            
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

async def insert_document(title, content, file_path, embedding, chunk_index, total_chunks, metadata):
    """Insert a new document with embedding"""
    try:
        with get_db_cursor(commit=True) as cursor:
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            cursor.execute("""
                INSERT INTO documents (title, content, file_path, embedding, embedding_status, chunk_index, total_chunks, metadata)
                VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s::jsonb)
                RETURNING id
            """, (title, content, file_path, embedding_list, 'completed', chunk_index, total_chunks, json.dumps(metadata)))
            
            result = cursor.fetchone()
            return result['id'] if result else None
            
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        return None

async def delete_document_by_path(file_path):
    """Delete document by file path"""
    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute("DELETE FROM documents WHERE file_path = %s", (file_path,))
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return False

async def get_document_by_path(file_path):
    """Get document by file path"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM documents WHERE file_path = %s", (file_path,))
            return cursor.fetchone()
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        return None 

async def insert_documents_batch(chunks: List[Dict[str, Any]]):
    """Insert multiple documents in batch"""
    try:
        with get_db_cursor(commit=True) as cursor:
            # Prepare data for batch insert
            data = []
            for chunk in chunks:
                embedding_list = chunk['embedding'].tolist() if isinstance(chunk['embedding'], np.ndarray) else chunk['embedding']
                data.append((
                    chunk['title'],
                    chunk['content'],
                    chunk['file_path'],
                    embedding_list,
                    'completed',
                    chunk['chunk_index'],
                    chunk['total_chunks'],
                    json.dumps(chunk['doc_metadata'])
                ))
            
            # Batch insert using execute_values
            execute_values(
                cursor,
                """
                INSERT INTO documents (title, content, file_path, embedding, embedding_status, chunk_index, total_chunks, metadata)
                VALUES %s
                """,
                data,
                template="(%s, %s, %s, %s::vector, %s, %s, %s, %s::jsonb)"
            )
            
    except Exception as e:
        logger.error(f"Error batch inserting documents: {e}")
        raise 