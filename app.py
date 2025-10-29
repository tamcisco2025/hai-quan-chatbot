# Chatbot Tra cứu Nghiệp vụ Hải quan (RAG)

Bộ khung mã nguồn mở, miễn phí, chạy trên **Hugging Face Spaces** với **Gradio + LlamaIndex + Chroma**. Trả lời **tiếng Việt**, **ngắn gọn có trích dẫn điều/khoản**, chỉ dựa trên **văn bản pháp luật VN (Luật/Nghị định/Thông tư)** do bạn tải lên. Hỗ trợ **tải file trực tiếp trên UI** và **lọc theo loại văn bản / năm / điều-khoản**.

---

## 1) Hướng dẫn triển khai nhanh (5–10 phút)

1. **Tạo GitHub repo** (Public) và thả 3 file sau vào root:

   * `app.py`
   * `requirements.txt`
   * `README.md` (bạn có thể dùng chính file này)
   * Tạo thư mục `data/` (để các văn bản đầu tiên vào đây, <50 file PDF/DOCX)

2. **Tạo Hugging Face Space**

   * Đăng nhập 👉 New Space → chọn **Gradio** → Public.
   * Kết nối với repo GitHub hoặc **Upload** trực tiếp `app.py`, `requirements.txt`, `README.md`, thư mục `data/`.
   * Chờ build xong là có URL công khai.

3. **Dùng thử**

   * Gõ câu hỏi (VD: *"Điều kiện miễn kiểm tra thực tế hàng hóa theo Thông tư 39/2018?"*).
   * Thử **Upload** thêm file ngay trên giao diện, chọn bộ lọc "Loại văn bản/Năm/Điều" rồi hỏi lại.

> Miễn phí hoàn toàn: dùng CPU của HF Spaces. Tốc độ vừa đủ cho demo/cộng đồng (<50 file).

---

## 2) Nguyên tắc trả lời & thông điệp bắt buộc

* **Chỉ** dựa trên tài liệu đã tải (RAG). Nếu không có căn cứ phù hợp:
  **"Chủ đề này tôi chưa được cập nhật. Xin hãy chọn chủ đề khác."**
* Trả lời **ngắn gọn**, có **trích dẫn**: `[Tên văn bản – Điều/Khoản (nếu có) – Trang]`.

---

## 3) `requirements.txt`

```txt
# UI
gradio==4.44.0

# RAG
llama-index==0.11.0
llama-index-embeddings-huggingface
llama-index-vector-stores-chroma
chromadb==0.5.5
sentence-transformers>=2.7.0

# Models (CPU)
transformers>=4.44.0
# Dòng dưới giúp cài torch CPU nhanh hơn trên nhiều môi trường, có thể bỏ nếu build lỗi
torch --extra-index-url https://download.pytorch.org/whl/cpu

# Parsers
pypdf
python-docx
unidecode
```

> **Ghi chú**: Nếu `torch` lỗi trên HF Spaces, xóa dòng `torch ...` để HF tự chọn phiên bản phù hợp.

---

## 4) `app.py`

```python
import os
import re
import io
import time
from typing import List, Dict, Any

import gradio as gr
from unidecode import unidecode

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Vector store: Chroma
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Lightweight open-source LLM for CPU (tiếng Việt khá ổn với prompt ngắn)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# =========================
# 0) Cấu hình
# =========================
DATA_DIR = os.getenv("DATA_DIR", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_NAME = "hai-quan-vn"

# Model ngắn gọn, chạy CPU nhanh
LLM_NAME = os.getenv("LLM_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

FALLBACK_MSG = "Chủ đề này tôi chưa được cập nhật. Xin hãy chọn chủ đề khác."

# Ngưỡng tối thiểu coi là có căn cứ
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.30"))
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# 1) LLM & Embeddings
# =========================
print("\n[Init] Loading LLM ...")
tok = AutoTokenizer.from_pretrained(LLM_NAME)
llm_pipe = pipeline(
    task="text-generation",
    model=AutoModelForCausalLM.from_pretrained(LLM_NAME),
    tokenizer=tok,
    max_new_tokens=512,
    do_sample=False,
)

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# =========================
# 2) Tiện ích metadata & parser
# =========================
TYPE_PATTERNS = [
    (r"\blu[aâ]t\b", "Luật"),
    (r"ngh[iị]\s*đ[iị]nh|nd\b", "Nghị định"),
    (r"th[oô]ng\s*t[uư]|tt\b", "Thông tư"),
]

YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")
DIEU_RE = re.compile(r"\bĐiều\s+(\d+[A-Z]?)", re.IGNORECASE)
KHOAN_RE = re.compile(r"\bKhoản\s+(\d+[A-Z]?)", re.IGNORECASE)


def guess_type_and_year(name: str) -> Dict[str, Any]:
    lower = unidecode(name.lower())
    vtype = None
    for pat, label in TYPE_PATTERNS:
        if re.search(pat, lower):
            vtype = label
            break
    year_match = YEAR_RE.search(name)
    year = int(year_match.group(1)) if year_match else None
    return {"loai": vtype, "nam": year}


def extract_refs(text: str) -> Dict[str, str]:
    d = DIEU_RE.search(text)
    k = KHOAN_RE.search(text)
    return {"dieu": d.group(1) if d else None, "khoan": k.group(1) if k else None}

# =========================
# 3) Dựng / nạp Index
# =========================
print("[Init] Building Chroma store ...")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(INDEX_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
index: VectorStoreIndex = None


def has_supported_files(path: str) -> bool:
    """Kiểm tra thư mục có chứa ít nhất 1 file .pdf/.docx/.txt hay không."""
    if not os.path.isdir(path):
        return False
    for root, _, files in os.walk(path):
        if any(f.lower().endswith((".pdf", ".docx", ".txt")) for f in files):
            return True
    return False


def load_documents_from_dir(path: str) -> List[Document]:
    # Bỏ qua nếu thư mục trống/không có file hỗ trợ để tránh ValueError
    if not has_supported_files(path):
        return []
    reader = SimpleDirectoryReader(
        input_dir=path,
        required_exts=[".pdf", ".docx", ".txt"],
        recursive=True,
    )
    docs = reader.load_data()
    enriched = []
    for d in docs:
        meta = d.metadata or {}
        name = meta.get("file_name") or os.path.basename(meta.get("filepath", ""))
        ty = guess_type_and_year(name)
        meta.update(ty)
        # Giữ nguyên các metadata có sẵn như page_label nếu có
        enriched.append(Document(text=d.text, metadata=meta))
    return enriched


def build_or_update_index(base_dirs: List[str]):
    global index
    docs: List[Document] = []
    for p in base_dirs:
        docs.extend(load_documents_from_dir(p))
    if not docs:
        # tạo index rỗng để không lỗi
        index = VectorStoreIndex.from_documents([], vector_store=vector_store)
        return
    index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)


print("[Init] Loading documents & indexing ...")
dirs = [p for p in [DATA_DIR, UPLOAD_DIR] if has_supported_files(p)]
if not dirs:
    # tạo index rỗng để app vẫn chạy khi chưa có tài liệu
    index = VectorStoreIndex.from_documents([], vector_store=vector_store)
else:
    build_or_update_index(dirs)
query_engine = index.as_query_engine(similarity_top_k=TOP_K, response_mode="compact")

# =========================
# 4) Truy vấn có bộ lọc
# =========================
SYSTEM_PROMPT = (
    "Bạn là trợ lý nghiệp vụ hải quan Việt Nam. Chỉ trả lời dựa trên tài liệu đã cung cấp. "
    "Nếu không có căn cứ phù hợp, hãy trả lời đúng thông điệp: 'Chủ đề này tôi chưa được cập nhật. Xin hãy chọn chủ đề khác.' "
    "Trả lời ngắn gọn, rõ ràng, kèm trích dẫn nguồn theo dạng [Tên văn bản – Điều/Khoản – Trang]."
)


def filter_nodes_by_metadata(nodes, loai: str = None, nam: int = None, dieu: str = None, khoan: str = None):
    # Lọc thô sau khi retrieve: giữ những node có metadata phù hợp
    def ok(node):
        m = node.metadata or {}
        if loai and m.get("loai") != loai:
            return False
        if nam and m.get("nam") != nam:
            return False
        # Nếu người dùng yêu cầu Điều/Khoản, ưu tiên đoạn có chứa
        t = node.get_content(metadata_mode="none")
        if dieu and (f"Điều {dieu}" not in t and f"ĐiỀu {dieu}" not in t):
            return False
        if khoan and (f"Khoản {khoan}" not in t and f"KhoẢn {khoan}" not in t):
            return False
        return True

    kept = [n for n in nodes if ok(n)]
    return kept if kept else nodes  # nếu lọc rỗng, trả về danh sách gốc để không mất thông tin


def format_citations(source_nodes) -> str:
    cites = []
    for sn in source_nodes:
        meta = sn.metadata or {}
        name = meta.get("file_name") or os.path.basename(meta.get("filepath", "Nguon"))
        page = meta.get("page_label") or meta.get("page")
        # cố gắng đoán điều/khoản trong chunk
        refs = extract_refs(sn.get_content(metadata_mode="none"))
        seg = name
        parts = []
        if refs.get("dieu"):
            parts.append(f"Điều {refs['dieu']}")
        if refs.get("khoan"):
            parts.append(f"Khoản {refs['khoan']}")
        if page is not None:
            parts.append(f"Trang {page}")
        if parts:
            seg += " – " + ", ".join(parts)
        cites.append(f"[{seg}]")
    # loại trùng
    uniq = []
    for c in cites:
        if c not in uniq:
            uniq.append(c)
    return " ".join(uniq)


def llm_answer(prompt: str) -> str:
    out = llm_pipe(SYSTEM_PROMPT + "\n\nCâu hỏi: " + prompt + "\n\nTrả lời:")
    return out[0]["generated_text"].split("Trả lời:")[-1].strip()


def answer(question: str, loai: str, nam: str, dieu: str, khoan: str):
    if not question or not question.strip():
        return "Hãy nhập câu hỏi.", ""

    # truy vấn trước
    res = query_engine.query(question)

    # kiểm tra độ phủ & lọc metadata
    nodes = getattr(res, "source_nodes", [])
    nodes = filter_nodes_by_metadata(nodes, loai or None, int(nam) if nam else None, dieu or None, khoan or None)

    # heuristic: nếu không có nguồn hoặc điểm thấp → fallback
    avg_sim = 0.0
    if nodes:
        sims = [getattr(n, "score", 0.0) or 0.0 for n in nodes]
        avg_sim = sum(sims) / max(1, len(sims))

    if not nodes or avg_sim < SIM_THRESHOLD:
        return FALLBACK_MSG, ""

    # Tạo câu trả lời ngắn gọn dựa vào context
    # ghép ngữ cảnh
    context = "\n\n".join([n.get_content(metadata_mode="none") for n in nodes[:3]])
    prompt = (
        SYSTEM_PROMPT
        + "\n\nTài liệu liên quan:\n" + context
        + "\n\nCâu hỏi: " + question
        + "\n\nYêu cầu: Trả lời ngắn gọn (3–6 câu), chỉ dựa vào tài liệu trên, dùng tiếng Việt, và chèn trích dẫn ở cuối."
    )
    text = llm_answer(prompt)

    cites = format_citations(nodes[:TOP_K])
    if cites:
        text = text.rstrip() + "\n\n" + cites
    return text, cites

# =========================
# 5) Upload file & cập nhật index
# =========================

def handle_upload(files: List[gr.File]):
    saved = []
    if not files:
        return "Chưa chọn file."
    for f in files:
        dest = os.path.join(UPLOAD_DIR, os.path.basename(f.name))
        with open(dest, "wb") as w:
            w.write(f.read())
        saved.append(os.path.basename(dest))
    # cập nhật index nhanh
    build_or_update_index([DATA_DIR, UPLOAD_DIR])
    global query_engine
    query_engine = index.as_query_engine(similarity_top_k=TOP_K, response_mode="compact")
    return f"Đã tải lên: {', '.join(saved)} (index đã được cập nhật)."

# =========================
# 6) Gradio UI
# =========================

def ui_answer(q, loai, nam, dieu, khoan):
    text, _ = answer(q, loai, nam, dieu, khoan)
    return text

with gr.Blocks(title="Chatbot Hải quan VN (RAG)") as demo:
    gr.Markdown("""
    # Chatbot Tra cứu Nghiệp vụ Hải quan (VN)
    - **Nội dung**: Luật / Nghị định / Thông tư do bạn tải lên.
    - **Nguyên tắc**: chỉ trả lời từ tài liệu. Nếu thiếu căn cứ: *Chủ đề này tôi chưa được cập nhật. Xin hãy chọn chủ đề khác.*
    """)

    with gr.Row():
        q = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: Điều kiện miễn kiểm tra thực tế hàng hóa theo Thông tư 39/2018?", scale=3)
    with gr.Row():
        loai = gr.Dropdown(label="Loại văn bản", choices=["", "Luật", "Nghị định", "Thông tư"], value="")
        nam = gr.Dropdown(label="Năm", choices=[""] + [str(y) for y in range(2005, 2031)], value="")
        dieu = gr.Textbox(label="Điều (ví dụ 10)", placeholder="", scale=1)
        khoan = gr.Textbox(label="Khoản (ví dụ 2)", placeholder="", scale=1)

    btn = gr.Button("Hỏi")
    out = gr.Markdown()

    btn.click(fn=ui_answer, inputs=[q, loai, nam, dieu, khoan], outputs=[out])

    gr.Markdown("""
    ### Tải thêm văn bản (PDF/DOCX/TXT)
    Chọn file để thêm vào cơ sở tri thức. Hệ thống sẽ **re-index** tự động.
    """)
    uploader = gr.File(label="Tải tệp (nhiều tệp)", file_types=[".pdf", ".docx", ".txt"], file_count="multiple")
    upmsg = gr.Markdown()
    uploader.upload(fn=handle_upload, inputs=[uploader], outputs=[upmsg])

    gr.Markdown(f"Cập nhật lần cuối: {time.strftime('%Y-%m-%d %H:%M:%S')} (giờ máy chủ)")

if __name__ == "__main__":
    demo.launch()
```

---

## 5) Kiểm thử nhanh (UAT)

* **Bộ 10 câu** (điều chỉnh theo bộ tài liệu của bạn):

  1. *Điều kiện miễn kiểm tra thực tế hàng hóa theo Thông tư 39/2018 là gì?*
  2. *Thành phần hồ sơ khi đăng ký tờ khai hải quan điện tử theo Nghị định X?*
  3. *Cách xác định trị giá tính thuế đối với hàng nhập khẩu theo …?*
  4. *Thời hạn nộp thuế xuất khẩu đối với …?*
  5. *Quy định về miễn, giảm, hoàn thuế trong trường hợp …?*
  6. *Trách nhiệm của người khai hải quan theo Luật …?*
  7. *Quy định về kiểm tra sau thông quan …?*
  8. *Quy định xử phạt vi phạm hành chính với hành vi …?*
  9. *Căn cứ pháp lý về xuất xứ hàng hóa …?*
  10. *Thủ tục với hàng quá cảnh theo …?*

**Tiêu chí đạt**

* ≥90% câu có ít nhất **1 trích dẫn** đúng văn bản và (nếu có) đúng **Điều/Khoản**.
* Khi thiếu căn cứ → trả về **thông điệp bắt buộc**.
* Thời gian phản hồi trên CPU miễn phí: < 20s với câu hỏi ngắn.

---

## 6) Gợi ý vận hành & cập nhật

* Đặt tên file rõ ràng, ví dụ: `Thong-tu-39-2018-BTC.pdf`, `Nghi-dinh-xx-2024.pdf`.
* Khi có văn bản mới/sửa đổi: chỉ cần **tải lên** qua UI hoặc thêm vào `data/` rồi push/rebuild.
* Với dữ liệu tăng dần: cân nhắc chuyển embeddings sang `bge-m3` hoặc `intfloat/multilingual-e5-large`.

---

## 7) Câu hỏi thường gặp (FAQ)

**Hỏi:** Sao câu trả lời đôi khi chung chung?
**Đáp:** Do dùng model nhỏ để chạy miễn phí CPU. Vì là RAG, chất lượng phụ thuộc độ sát của đoạn trích. Hãy tăng chất lượng tài liệu, đặt câu hỏi cụ thể, hoặc nâng cấp model.

**Hỏi:** Có thể bắt buộc trả lời chỉ từ tài liệu không?
**Đáp:** Có. Prompt hệ thống + ngưỡng `SIM_THRESHOLD` + fallback ở trên đã giới hạn. Có thể hạ `TOP_K`/tăng `SIM_THRESHOLD` để nghiêm hơn.

**Hỏi:** Có thể xuất câu trả lời ra PDF?
**Đáp:** Có thể bổ sung một nút tạo PDF (reportlab) — phần mở rộng nhỏ, không bắt buộc.

---

> **Bản quyền & trách nhiệm**: Mẫu này phục vụ mục đích thông tin. Vui lòng đối chiếu văn bản pháp luật chính thức khi sử dụng trong tình huống pháp lý cụ thể.
