# 📘 AI Document Search - Trợ lý tra cứu tài liệu thông minh

Hệ thống được xây dựng nhằm **hỗ trợ tra cứu và khai thác tri thức** từ các tài liệu nội bộ đa định dạng:

- 📄 **PDF**
- 📝 **Word**
- 📊 **Excel**
- 🖼️ **Hình ảnh quét**

---

## 🚀 Tính năng chính

- 🔎 **Tìm kiếm ngữ nghĩa**  
  Trả lời câu hỏi với thông tin trích dẫn cụ thể (**tên file, số trang, đường dẫn mở tài liệu**).

- 💬 **Giao tiếp tự nhiên**  
  - Nhập liệu bằng giọng nói (Speech-to-Text).  
  - Phản hồi bằng giọng nói (Text-to-Speech).  

- 🔄 **Tự động cập nhật dữ liệu**  
  Hệ thống sẽ tự đồng bộ khi có sự thay đổi trong kho tài liệu.

- 🌐 **Mở rộng tra cứu ngoài hệ thống**  
  Tích hợp tìm kiếm thông tin từ các nguồn bên ngoài (Google Search).

---

## 🛠️ Công nghệ sử dụng
- Python (FastAPI, LangChain, OpenAI/Azure OpenAI)
- Vector Database (PostgreSQL + pgvector)
- Frontend: HTML5, Bootstrap 5, JavaScript
- Speech: Web Speech API, TTS API

---

## 📦 Cách chạy dự án

```bash
# Clone repository
git clone https://github.com/hungubqn0310/Agent-RAG.git
cd Agent-RAG

# Cài đặt môi trường
pip install -r requirements.txt

# Chạy server
python backend/main_azure.py
