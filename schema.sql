-- Kích hoạt extension nếu chưa có
CREATE EXTENSION IF NOT EXISTS vector;

-- Tạo bảng documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    file_path TEXT UNIQUE, -- Đảm bảo mỗi chunk có file_path duy nhất
    embedding VECTOR(1536), -- Kích thước cho text-embedding-3-small
    embedding_status TEXT,
    chunk_index INTEGER,
    total_chunks INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tạo indexes để tăng tốc độ truy vấn
-- Index HNSW cho tìm kiếm vector nhanh (sử dụng cosine similarity)
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);

-- Index GIN cho tìm kiếm từ khóa (full-text search) nhanh
CREATE INDEX IF NOT EXISTS documents_content_fts_idx ON documents USING gin(to_tsvector('simple', content));

-- Index trên metadata path để xóa nhanh hơn
CREATE INDEX IF NOT EXISTS documents_metadata_path_idx ON documents USING gin ((metadata -> 'original_path'));