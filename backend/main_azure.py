from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import aiofiles
from pathlib import Path
import asyncio
import sys
import hashlib
import time
from PIL import Image
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_fixed import get_db_connection,initialize_db,insert_document,insert_documents_batch,delete_document_by_path,get_document_by_path
from services.document_processor_azure import DocumentProcessor
from services.ai_search_azure import AISearchService
from services.voice_service import VoiceService

app = FastAPI(title="AI Document Search API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - point to parent directory
static_dir = "../static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Warning: Static directory '{static_dir}' not found. Static files will not be served.")

# Initialize services
document_processor = DocumentProcessor()
ai_search_service = AISearchService()
voice_service = VoiceService()

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    include_external: bool = True

class SearchResponse(BaseModel):
    response: str
    citations: List[dict]
    sources_count: int
    external_sources_count: int

class UploadResponse(BaseModel):
    message: str
    file_id: str
    chunks_processed: int

class VoiceRequest(BaseModel):
    text: str
    language: str = "vi"

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_image_hash(file_path: str) -> str:
    """Calculate perceptual hash for images to detect similar images"""
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 8x8 for perceptual hash
            img = img.resize((8, 8), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Calculate average pixel value
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            
            # Create hash based on pixels above/below average
            hash_bits = []
            for pixel in pixels:
                hash_bits.append('1' if pixel > avg else '0')
            
            # Convert to hex
            hash_string = ''.join(hash_bits)
            return hex(int(hash_string, 2))[2:].zfill(16)
    except Exception as e:
        print(f"Error calculating image hash: {e}")
        return calculate_file_hash(file_path)  # Fallback to file hash

def get_image_metadata(file_path: str) -> dict:
    """Get image metadata for comparison"""
    try:
        with Image.open(file_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size_bytes': os.path.getsize(file_path)
            }
    except Exception as e:
        print(f"Error getting image metadata: {e}")
        return {'size_bytes': os.path.getsize(file_path)}

def generate_unique_filename(original_filename: str, upload_dir: str) -> str:
    """Generate unique filename if file already exists"""
    base_name, ext = os.path.splitext(original_filename)
    file_path = os.path.join(upload_dir, original_filename)
    
    if not os.path.exists(file_path):
        return original_filename
    
    counter = 1
    while True:
        new_filename = f"{base_name}_{counter}{ext}"
        new_file_path = os.path.join(upload_dir, new_filename)
        if not os.path.exists(new_file_path):
            return new_filename
        counter += 1

async def check_file_exists_in_db(file_path: str) -> bool:
    """Check if file already exists in database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE file_path = %s", (file_path,))
            count = cursor.fetchone()[0]
            return count > 0
    except Exception as e:
        print(f"Error checking file in database: {e}")
        return False

async def check_duplicate_file(file_path: str, file_hash: str, is_image: bool = False) -> Optional[str]:
    """Check if file with same content already exists in database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if is_image:
                # For images, check both file hash and image hash
                cursor.execute("""
                    SELECT file_path, metadata FROM documents 
                    WHERE metadata->>'file_hash' = %s 
                    OR metadata->>'image_hash' = %s
                    LIMIT 1
                """, (file_hash, file_hash))
            else:
                # For other files, check file hash
                cursor.execute("""
                    SELECT file_path, metadata FROM documents 
                    WHERE metadata->>'file_hash' = %s 
                    LIMIT 1
                """, (file_hash,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
        return None
    except Exception as e:
        print(f"Error checking duplicate file: {e}")
        return None

async def check_similar_image(file_path: str, image_hash: str) -> Optional[str]:
    """Check for similar images using perceptual hash"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path, metadata FROM documents 
                WHERE metadata->>'image_hash' IS NOT NULL
                AND metadata->>'file_type' = 'image'
            """)
            
            results = cursor.fetchall()
            for result in results:
                stored_hash = result[1].get('image_hash', '')
                if stored_hash and _compare_image_hashes(image_hash, stored_hash):
                    return result[0]
        return None
    except Exception as e:
        print(f"Error checking similar images: {e}")
        return None

def _compare_image_hashes(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """Compare two image hashes with a threshold for similarity"""
    try:
        # Convert hex to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(64)
        bin2 = bin(int(hash2, 16))[2:].zfill(64)
        
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
        return distance <= threshold
    except Exception as e:
        print(f"Error comparing image hashes: {e}")
        return False

async def delete_file_from_db(file_path: str) -> bool:
    """Delete all chunks of a file from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE file_path = %s", (file_path,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting file from database: {e}")
        return False

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    initialize_db()
    print("Database initialized successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Document Search API is running"}

# Upload file endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.png', '.jpg', '.jpeg'}
        
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File type not supported")
        
        # Generate unique filename to avoid conflicts
        unique_filename = generate_unique_filename(file.filename, UPLOAD_DIR)
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Check if file with same content already exists in database
        file_hash = calculate_file_hash(file_path)
        is_image = file_ext in ['.png', '.jpg', '.jpeg']
        
        # Check for exact duplicates
        existing_file = await check_duplicate_file(file_path, file_hash, is_image)
        
        if existing_file:
            # File with same content already exists
            os.remove(file_path)  # Remove the duplicate file
            raise HTTPException(
                status_code=409, 
                detail=f"File with same content already exists: {os.path.basename(existing_file)}"
            )
        
        # For images, check for similar images
        if is_image:
            image_hash = calculate_image_hash(file_path)
            similar_file = await check_similar_image(file_path, image_hash)
            
            if similar_file:
                os.remove(file_path)  # Remove the similar file
                raise HTTPException(
                    status_code=409, 
                    detail=f"Similar image already exists: {os.path.basename(similar_file)}"
                )
        
        # Check if file already exists in database (by path) and delete old version
        if await check_file_exists_in_db(file_path):
            await delete_file_from_db(file_path)
        
        # Process document
        chunks = document_processor.process_document(file_path)

        if not chunks:
            os.remove(file_path)  # Clean up file if processing failed
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        # Add enhanced metadata for future duplicate detection
        for chunk in chunks:
            chunk['doc_metadata']['file_hash'] = file_hash
            chunk['doc_metadata']['upload_time'] = time.time()
            chunk['doc_metadata']['file_type'] = 'image' if is_image else 'document'
            chunk['doc_metadata']['file_size'] = os.path.getsize(file_path)
            
            if is_image:
                chunk['doc_metadata']['image_hash'] = calculate_image_hash(file_path)
                chunk['doc_metadata']['image_metadata'] = get_image_metadata(file_path)

        # Insert chunks into database using batch processing
        await insert_documents_batch(chunks)
        
        return UploadResponse(
            message=f"File uploaded and processed successfully",
            file_id=file_path,
            chunks_processed=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if there was an error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        result = await ai_search_service.search_and_generate(
            query=request.query,
            include_external=request.include_external
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Voice to text endpoint
@app.post("/voice/speech-to-text")
async def speech_to_text(language: str = Form("vi-VN")):
    try:
        text = voice_service.speech_to_text(language=language)
        if text:
            return {"text": text, "success": True}
        else:
            return {"text": "", "success": False, "message": "Could not recognize speech"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text to speech endpoint
@app.post("/voice/text-to-speech")
async def text_to_speech(request: VoiceRequest):
    try:
        audio_data = voice_service.text_to_speech(request.text, request.language)
        if audio_data:
            return {"audio_data": audio_data.hex(), "success": True}
        else:
            return {"audio_data": "", "success": False, "message": "Could not generate speech"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get uploaded files
@app.get("/files")
async def get_uploaded_files():
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "uploaded_at": os.path.getctime(file_path)
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete file
@app.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            # Delete from database
            await delete_file_from_db(file_path)
            # Delete file
            os.remove(file_path)
            return {"message": f"File {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve uploaded files
@app.get("/files/{filename}")
async def serve_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Voice status
@app.get("/voice/status")
async def voice_status():
    return {
        "voice_enabled": voice_service.is_voice_enabled(),
        "available_languages": voice_service.get_available_languages(),
        "available_voices": voice_service.get_available_voices()
    }

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        # Get the absolute path to the templates directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        template_path = os.path.join(parent_dir, "templates", "index.html")
        
        print(f"Looking for template at: {template_path}")
        print(f"Template exists: {os.path.exists(template_path)}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError as e:
        print(f"Template not found: {e}")
        return HTMLResponse(content="<h1>Frontend not found. Please check templates/index.html</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
