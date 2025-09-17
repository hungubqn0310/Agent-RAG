import os
from typing import List, Dict, Any
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from functools import lru_cache

# Import custom prompts
try:
    from prompts.system_prompts import SYSTEM_PROMPT, GREETING_PROMPT, ERROR_PROMPT
except ImportError:
    # Fallback prompts if file doesn't exist
    SYSTEM_PROMPT = "Bạn là một trợ lý AI chuyên về tra cứu tài liệu. Hãy trả lời câu hỏi dựa trên các tài liệu được cung cấp một cách chi tiết và chính xác."
    GREETING_PROMPT = "Bạn là một trợ lý AI thân thiện. Hãy chào hỏi người dùng một cách thân thiện và mời họ đặt câu hỏi."
    ERROR_PROMPT = "Khi gặp lỗi, hãy thừa nhận vấn đề và đề xuất giải pháp thay thế."

class AzureOpenAIService:
    def __init__(self):
        # Cấu hình từ environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-3-small")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
        
        # Kiểm tra cấu hình
        self.is_configured = all([self.api_key, self.endpoint, self.embedding_deployment])
        
        if self.is_configured and not self._is_demo_key():
            try:
                # Khởi tạo embeddings client
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=self.embedding_deployment,
                    openai_api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key
                )
                
                # Khởi tạo chat client
                self.chat = AzureChatOpenAI(
                    azure_deployment=self.chat_deployment,
                    openai_api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    temperature=0.7
                )
                
                print("Azure OpenAI clients initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Azure OpenAI clients: {e}")
                self.is_configured = False
        else:
            print("Azure OpenAI configuration is incomplete or using demo keys. Running in demo mode.")
            self.is_configured = False
    
    def _is_demo_key(self) -> bool:
        """Check if using demo keys"""
        demo_keys = ["demo_key", "your_azure_openai_api_key_here", ""]
        return self.api_key in demo_keys or not self.api_key
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for a single text"""
        try:
            if self.is_configured:
                vector = self.embeddings.embed_query(text)
                return np.array(vector, dtype=np.float32)
            else:
                return self._create_mock_embedding(text)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return self._create_mock_embedding(text)
    
    def _create_mock_embedding(self, text: str) -> np.ndarray:
        """Create a mock embedding for demo purposes"""
        # Simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert to float array
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                value = int.from_bytes(chunk, byteorder='big')
                # Normalize to [-1, 1] range
                normalized = (value / (2**32 - 1)) * 2 - 1
                embedding.append(normalized)
        
        # Pad or truncate to 1536 dimensions
        while len(embedding) < 1536:
            embedding.append(0.0)
        return np.array(embedding[:1536], dtype=np.float32)
    
    def generate_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat completion using Azure OpenAI"""
        if self.is_configured:
            try:
                response = self.chat.invoke(messages)
                return response.content
            except Exception as e:
                print(f"Error generating chat completion: {e}")
                return self._create_mock_response(messages)
        else:
            return self._create_mock_response(messages)
    
    def _create_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Create a mock response for demo purposes"""
        user_message = ""
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        # Simple keyword-based responses
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ["xin chào", "hello", "hi", "chào"]):
            return "Xin chào! Tôi là trợ lý AI của bạn. Hiện tại tôi đang chạy ở chế độ demo. Để sử dụng đầy đủ tính năng, vui lòng cấu hình Azure OpenAI API keys trong file .env"
        
        elif any(word in user_lower for word in ["tài liệu", "document", "file", "tìm kiếm", "search"]):
            return "Tôi có thể giúp bạn tìm kiếm thông tin trong tài liệu. Hiện tại tôi đang chạy ở chế độ demo. Để tìm kiếm thực tế, vui lòng cấu hình database và API keys."
        
        elif any(word in user_lower for word in ["ai", "trí tuệ nhân tạo", "machine learning"]):
            return "Trí tuệ nhân tạo (AI) là công nghệ mô phỏng trí thông minh của con người trong máy móc. Hiện tại tôi đang chạy ở chế độ demo."
        
        elif any(word in user_lower for word in ["python", "lập trình", "programming"]):
            return "Python là một ngôn ngữ lập trình phổ biến, dễ học và mạnh mẽ. Hiện tại tôi đang chạy ở chế độ demo."
        
        else:
            return f"Tôi đã nhận được câu hỏi của bạn: '{user_message[:100]}...'. Hiện tại tôi đang chạy ở chế độ demo. Để có phản hồi chính xác, vui lòng cấu hình Azure OpenAI API keys trong file .env"
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response based on query and context"""
        messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": f"""
    Câu hỏi: {query}

    Tài liệu tham khảo (có kèm nguồn: tên file + số trang/sheet):
    {context}

    Yêu cầu:
    - Chỉ trả lời dựa trên tài liệu ở trên.
    - Khi trích dẫn, phải ghi rõ nguồn theo định dạng:
    "Nguồn: {{file_name}} — Trang {{page_number}}"
    - Nếu không tìm thấy thông tin, hãy nói rõ là không có.
    """
            }
        ]
        
        return self.generate_chat_completion(messages)

    def generate_greeting_response(self, query: str) -> str:
        """Generate greeting response"""
        messages = [
            {
                "role": "system", 
                "content": GREETING_PROMPT
            },
            {
                "role": "user", 
                "content": query
            }
        ]
        
        return self.generate_chat_completion(messages)
    
    def generate_error_response(self, error_message: str) -> str:
        """Generate error response"""
        messages = [
            {
                "role": "system", 
                "content": ERROR_PROMPT
            },
            {
                "role": "user", 
                "content": f"Lỗi: {error_message}"
            }
        ]
        
        return self.generate_chat_completion(messages)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents at once"""
        if self.is_configured:
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as e:
                print(f"Error embedding documents: {e}")
                return [self._create_mock_embedding(text).tolist() for text in texts]
        else:
            return [self._create_mock_embedding(text).tolist() for text in texts]

    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for a batch of texts"""
        try:
            if not texts:
                return []
            if self.is_configured:
                try:
                    vectors = self.embed_documents(texts)
                except Exception as e:
                    print(f"Error in embed_documents: {e}")
                    vectors = [self._create_mock_embedding(t).tolist() for t in texts]
            else:
                vectors = [self._create_mock_embedding(t).tolist() for t in texts]
            # Convert to numpy arrays
            return [np.array(v, dtype=np.float32) for v in vectors]
        except Exception as e:
            print(f"Error creating batch embeddings: {e}")
            return [self._create_mock_embedding(t) for t in texts]
