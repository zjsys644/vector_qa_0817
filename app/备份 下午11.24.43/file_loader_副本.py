import os
import json
import docx
import pandas as pd
import fitz
from PIL import Image

def split_file(file_path: str, max_chars: int = 500, overlap_ratio: float = 0.2, by: str = "chars_500"):
    ext = file_path.split('.')[-1].lower()

    def overlap_chunks(text, max_chars, overlap_ratio):
        if not max_chars or max_chars < 1:
            return [{"text": text}]
        stride = max(1, int(max_chars * (1 - overlap_ratio)))
        res = []
        i = 0
        n = len(text)
        while i < n:
            chunk = text[i:i + max_chars]
            if chunk.strip():
                res.append({"text": chunk})
            i += stride
        return res

    # PDF
    if ext == "pdf":
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        return overlap_chunks(text, max_chars, overlap_ratio)

    # 纯文本或 Markdown
    if ext in ("txt", "md"):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return overlap_chunks(content, max_chars, overlap_ratio)

    # Word (docx)
    if ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return overlap_chunks(text, max_chars, overlap_ratio)

    # Excel
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        return overlap_chunks(text, max_chars, overlap_ratio)

    # CSV/TSV
    if ext in ("csv", "tsv"):
        df = pd.read_csv(file_path, sep=None, engine="python")
        text = df.to_string(index=False)
        return overlap_chunks(text, max_chars, overlap_ratio)

    # jsonl 按行
    if ext == "jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 每行按 chunk 长度和 overlap 合并分割
        text = "".join([line for line in lines if line.strip()])
        return overlap_chunks(text, max_chars, overlap_ratio)

    # JSON
    if ext == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
            except Exception:
                f.seek(0)
                raw = f.read()
                lines = raw.splitlines()
                text = "\n".join([line for line in lines if line.strip()])
                return overlap_chunks(text, max_chars, overlap_ratio)
        if isinstance(content, list):
            # 多为聊天/知识QA/数组类，一条一块
            outs = []
            for item in content:
                s = json.dumps(item, ensure_ascii=False)
                outs.extend(overlap_chunks(s, max_chars, overlap_ratio))
            return outs
        elif isinstance(content, dict):
            outs = []
            for k, v in content.items():
                s = f"{k}: {json.dumps(v, ensure_ascii=False)}"
                outs.extend(overlap_chunks(s, max_chars, overlap_ratio))
            return outs
        else:
            txt = json.dumps(content, ensure_ascii=False)
            return overlap_chunks(txt, max_chars, overlap_ratio)

    # 图片
    if ext in ("jpg", "jpeg", "png", "bmp"):
        return [{"text": f"图片文件: {os.path.basename(file_path)}"}]

    raise ValueError("不支持的文件类型: %s" % ext)

