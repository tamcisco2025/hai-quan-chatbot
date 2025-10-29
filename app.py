import os
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
