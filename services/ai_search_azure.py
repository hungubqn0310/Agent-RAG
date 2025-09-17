import os
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import re
from backend.database_fixed import search_similar_documents
from services.azure_openai_service import AzureOpenAIService

class AISearchService:
    def __init__(self):
        self.azure_openai = AzureOpenAIService()
        self.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    def is_simple_greeting(self, query: str) -> bool:
        """Check if query is a simple greeting that doesn't need document search"""
        query_lower = query.lower().strip()
        
        greeting_patterns = [
            r'^(xin )?chào( bạn)?( ạ)?[.!?]*$',
            r'^hello[.!?]*$',
            r'^hi[.!?]*$',
            r'^chào hỏi[.!?]*$',
            r'^(chúc )?buổi (sáng|chiều|tối)( tốt lành)?[.!?]*$',
            r'^good (morning|afternoon|evening)[.!?]*$',
            r'^cảm ơn( bạn)?( ạ)?[.!?]*$',
            r'^thank you[.!?]*$',
            r'^thanks[.!?]*$',
            r'^tạm biệt[.!?]*$',
            r'^bye[.!?]*$',
            r'^goodbye[.!?]*$',
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, query_lower):
                return True
        return False
    
    def get_simple_greeting_response(self, query: str) -> str:
        """Get appropriate response for simple greetings"""
        query_lower = query.lower().strip()
        
        if re.match(r'^(xin )?chào', query_lower) or query_lower in ['hello', 'hi']:
            return "Xin chào! Tôi có thể giúp bạn tìm hiểu thông tin từ các tài liệu đã tải lên. Bạn có câu hỏi gì không?"
        elif re.match(r'^(chúc )?buổi', query_lower) or re.match(r'^good (morning|afternoon|evening)', query_lower):
            return "Chúc bạn một ngày tốt lành! Tôi sẵn sàng hỗ trợ bạn tìm kiếm thông tin trong tài liệu."
        elif re.match(r'^(cảm ơn|thank)', query_lower):
            return "Không có gì! Tôi luôn sẵn sàng giúp đỡ bạn."
        elif re.match(r'^(tạm biệt|bye|goodbye)', query_lower):
            return "Tạm biệt! Chúc bạn một ngày tốt lành!"
        else:
            return "Xin chào! Tôi có thể giúp bạn tìm hiểu thông tin từ các tài liệu. Hãy đặt câu hỏi nhé!"
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for search query using Azure OpenAI"""
        try:
            embedding = self.azure_openai.create_embedding(query)
            return embedding
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return np.zeros(1536)
    
    async def search_documents(self, query: str, limit: int = 5, threshold: float = 0) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity"""
        try:
            # Create query embedding
            query_embedding = self.create_query_embedding(query)
            
            # Search in database với RRF
            results = await search_similar_documents(
                query_embedding=query_embedding,
                query_text=query,  # Thêm query_text để dùng RRF
                limit=limit,
                threshold=threshold
            )
            
            print(f"DEBUG: Found {len(results)} results for query: {query}")
            
            # Format results
            formatted_results = []
            for result in results:
                print(f"DEBUG: Result: {result}")
                formatted_results.append({
                    'id': result['id'],
                    'title': result['title'],
                    'content': result['content'],
                    'file_path': result['file_path'],
                    'chunk_index': result['chunk_index'],
                    'total_chunks': result['total_chunks'],
                    'doc_metadata': result['metadata'],
                    'similarity': float(result['similarity'])
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def search_external_sources(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search external sources using Google Search API"""
        if not self.google_api_key or not self.google_engine_id:
            print("Google Search API not configured")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_engine_id,
                'q': query,
                'num': num_results,
                'lr': 'lang_vi',  # Prefer Vietnamese results
                'gl': 'vn',       # Country: Vietnam
                'safe': 'medium'  # Safe search
            }
            
            print(f"Searching Google for: {query}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', ''),
                    'source': 'external'
                })
            
            print(f"Found {len(results)} external results")
            return results
        except Exception as e:
            print(f"Error searching external sources: {e}")
            return []
    
    def get_original_file_path(self, result: Dict[str, Any]) -> str:
        """Get original file path from metadata, removing chunk information"""
        try:
            # Try to get original_path from metadata first
            original_path = result.get('doc_metadata', {}).get('original_path', '')
            if original_path:
                return original_path
            
            # Fallback: remove chunk information from file_path
            file_path = result.get('file_path', '')
            if file_path and '#chunk' in file_path:
                return file_path.split('#chunk')[0]
            
            return file_path or ''
        except Exception as e:
            print(f"Error getting original file path: {e}")
            return ""
    
    def get_document_title(self, result: Dict[str, Any]) -> str:
        """Get proper document title, preferring original filename"""
        try:
            # Try to get original filename from metadata
            original_file = result.get('doc_metadata', {}).get('original_file', '')
            if original_file:
                # Remove extension for cleaner display
                return os.path.splitext(original_file)[0]
            
            # Fallback to title field
            return result.get('title', 'Unknown Document')
        except Exception as e:
            print(f"Error getting document title: {e}")
            return result.get('title', 'Unknown Document')
    
    def generate_response(self, query: str, document_results: List[Dict[str, Any]], 
                         external_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate AI response based on search results using Azure OpenAI"""
        try:
            # Check if this is a simple greeting
            if self.is_simple_greeting(query):
                simple_response = self.get_simple_greeting_response(query)
                return {
                    'response': simple_response,
                    'citations': [],
                    'sources_count': 0,
                    'external_sources_count': 0,
                    'raw_contents': [r['content'] for r in document_results]
                }
            
            # Prepare context from document results
            context = ""
            citations = []
            
            # Add internal document results
            if document_results:
                context += "Thông tin từ tài liệu nội bộ:\n"
                for i, result in enumerate(document_results, 1):
                    
                    original_file_path = self.get_original_file_path(result)
                    document_title = self.get_document_title(result)
                    
                    page_number = result['doc_metadata'].get('page_number', 1)

                    file_name = os.path.basename(original_file_path) if original_file_path else "unknown"
                    context += (
                        f"[Tài liệu {i}: {document_title} — {file_name}"
                        f" — Trang {page_number}]\n{result['content']}\n\n"
                    )
                    citations.append({
                        'document': document_title,
                        'file_path': original_file_path,
                        'page_number': page_number,
                        'source': 'internal'
                    })
            
            # Add external results if available
            if external_results:
                if document_results:
                    context += "\nThông tin từ tìm kiếm bên ngoài:\n"
                else:
                    context += "Thông tin từ tìm kiếm bên ngoài:\n"
                
                for i, result in enumerate(external_results, 1):
                    context += f"Nguồn {i}: {result['snippet']}\n"
                    citations.append({
                        'document': result['title'],
                        'link': result['link'],
                        'source': 'external'
                    })
            
            # Generate AI response
            if document_results or external_results:
                ai_response = self.azure_openai.generate_response(query, context)
                
                # Only add reference section if there are citations
                if citations:
                    formatted_answer = [ai_response.strip()]
                    formatted_answer.append("\n### Nguồn tham khảo")
                    
                    # Group citations by document to avoid duplicates
                    seen_documents = set()
                    for c in citations[:10]:  # cap list to keep UI clean
                        if c.get('link'):
                            # External source
                            formatted_answer.append(f"- {c['document']} — {c['link']}")
                        else:
                            # Internal document
                            file_name = os.path.basename(c.get('file_path', ''))
                            doc_key = f"{c['document']}_{file_name}"
                            
                            if doc_key not in seen_documents:
                                seen_documents.add(doc_key)
                                
                                if c.get('page_number', 1) > 1:
                                    formatted_answer.append(f"- {c['document']} — {file_name} (trang {c['page_number']})")
                                else:
                                    formatted_answer.append(f"- {c['document']} — {file_name}")
                    
                    final_text = "\n\n".join(formatted_answer)
                else:
                    final_text = ai_response.strip()
            else:
                # No results found - this should not happen with the new logic
                final_text = ("Không tìm thấy thông tin phù hợp trong tài liệu đã cung cấp và không thể tìm kiếm bên ngoài. "
                             "Bạn có thể thử: đặt câu hỏi cụ thể hơn, kiểm tra lại từ khóa có dấu/không dấu, "
                             "hoặc tải thêm tài liệu liên quan.")
            
            return {
                'response': final_text,
                'citations': citations,
                'sources_count': len(document_results),
                'external_sources_count': len(external_results) if external_results else 0,
                'raw_contents': [r['content'] for r in document_results]
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                'response': f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
                'citations': [],
                'sources_count': 0,
                'external_sources_count': 0
            }
    
    async def search_and_generate(self, query: str, include_external: bool = True) -> Dict[str, Any]:
        """Complete search and response generation - only search Google if NO internal results"""
        # Check for simple greetings first
        if self.is_simple_greeting(query):
            return self.generate_response(query, [], [])
        
        # Search internal documents first
        document_results = await self.search_documents(query)
        
        # Search Google if no relevant internal results (similarity < 0.3)
        external_results = []
        has_relevant_results = any(result.get('similarity', 0) >= 0 for result in document_results)
        
        if include_external and self.google_api_key and not has_relevant_results:
            print("No relevant internal results found (similarity < 0.3), searching Google...")
            external_results = self.search_external_sources(query)
        elif has_relevant_results:
            print(f"Found {len(document_results)} relevant internal results, using only internal data")
        else:
            print(f"Found {len(document_results)} internal results but none are relevant (similarity < 0.3)")
        
        # Generate response with available results
        response = self.generate_response(query, document_results, external_results)
        
        return response