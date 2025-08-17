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

    # 纯文本/Markdown
    if ext in ("txt", "md"):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return overlap_chunks(content, max_chars, overlap_ratio)

    # Word
    if ext == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return overlap_chunks(text, max_chars, overlap_ratio)

    # Excel，每行一个文本块
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(file_path)
        rows = [",".join(map(str, df.columns))]
        rows += [",".join(map(str, row)) for row in df.values]
        return [{"text": row} for row in rows]

    # CSV/TSV，每行一个文本块
    if ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
        rows = [sep.join(map(str, df.columns))]
        rows += [sep.join(map(str, row)) for row in df.values]
        return [{"text": row} for row in rows]

        # jsonl 按行
    if ext == "jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        outs = []
        for i, line in enumerate(lines):
            # 不强制解析，避免坏行报错；解析成功就规范化一下
            txt = line
            try:
                obj = json.loads(line)
                txt = json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass
            for ch in overlap_chunks(txt, max_chars, overlap_ratio):
                ch["meta"] = {"json_path": f"$[{i}]", "content_type": "jsonl"}
                outs.append(ch)
        return outs

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

        outs = []

        # 生成可读的 JSON 路径，如 items[3].sku
        def jptr(keys):
            parts = []
            for k in keys:
                if isinstance(k, int):
                    parts.append(f"[{k}]")
                else:
                    parts.append(f".{k}" if parts else k)
            return "".join(parts) or "$"

        # 递归展开，遇到原子值就生成一条记录
        def walk(node, path_keys):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, path_keys + [k])
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, path_keys + [i])
            else:
                s = json.dumps(node, ensure_ascii=False)
                meta = {"json_path": jptr(path_keys), "content_type": "json"}
                # 对很长的值再切块，并继承 meta
                for ch in overlap_chunks(s, max_chars, overlap_ratio):
                    ch["meta"] = meta.copy()
                    outs.append(ch)

        if isinstance(content, (dict, list)):
            walk(content, [])
        else:
            # 顶层标量
            s = json.dumps(content, ensure_ascii=False)
            for ch in overlap_chunks(s, max_chars, overlap_ratio):
                ch["meta"] = {"json_path": "$", "content_type": "json"}
                outs.append(ch)

        return outs

    # 图片
    if ext in ("jpg", "jpeg", "png", "bmp"):
        return [{"text": f"图片文件: {os.path.basename(file_path)}"}]

    raise ValueError("不支持的文件类型: %s" % ext)

def process_file(file_path, chunk_size=500, overlap=0.2):
    return split_file(file_path, max_chars=chunk_size, overlap_ratio=overlap)
