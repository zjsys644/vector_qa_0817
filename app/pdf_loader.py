import fitz  # PyMuPDF

def load_and_split_pdf(pdf_path, max_chars=500):
    doc = fitz.open(pdf_path)
    all_chunks = []
    for page in doc:
        text = page.get_text()
        # 按 max_chars 分片
        start = 0
        while start < len(text):
            chunk = text[start: start + max_chars].strip()
            if chunk:
                all_chunks.append(chunk)
            start += max_chars
    return all_chunks

