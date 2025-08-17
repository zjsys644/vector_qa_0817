# File: app/file_loader.py

import os
import json
import re

import docx
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image  # 预留：如需读取图片尺寸等可用

# 去掉零宽字符 / BOM
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


def normalize_text(s: str) -> str:
    """轻量规范化：去零宽/BOM、压缩连续空格（保留换行）、去首尾空白。"""
    if not isinstance(s, str):
        s = str(s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"[ \t]+", " ", line) for line in s.split("\n"))
    return s.strip()


def make_base_meta(file_path: str, content_type: str) -> dict:
    return {
        "filename": os.path.basename(file_path),
        "content_type": content_type,
    }


def overlap_chunks(text: str, max_chars: int, overlap_ratio: float, base_meta: dict):
    """按字符数切片，带重叠。"""
    if not max_chars or max_chars < 1:
        return [{"text": normalize_text(text), "meta": dict(base_meta)}]
    stride = max(1, int(max_chars * (1 - overlap_ratio)))
    res = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : i + max_chars]
        if chunk.strip():
            res.append({"text": normalize_text(chunk), "meta": dict(base_meta)})
        i += stride
    return res


def split_file(
    file_path: str, max_chars: int = 500, overlap_ratio: float = 0.2, by: str = "chars_500"
):
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")

    # --- PDF ---
    if ext == "pdf":
        base_meta = make_base_meta(file_path, "pdf")
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        return overlap_chunks(text, max_chars, overlap_ratio, base_meta)

    # --- 纯文本 / Markdown ---
    if ext in ("txt", "md"):
        base_meta = make_base_meta(file_path, ext)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return overlap_chunks(content, max_chars, overlap_ratio, base_meta)

    # --- Word ---
    if ext == "docx":
        base_meta = make_base_meta(file_path, "docx")
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return overlap_chunks(text, max_chars, overlap_ratio, base_meta)

    # --- Excel ---
    if ext in ("xlsx", "xls"):
        base_meta = make_base_meta(file_path, "excel")
        df = pd.read_excel(file_path)
        outs = []
        header = ",".join(map(str, df.columns))
        outs.append({"text": normalize_text(header), "meta": {**base_meta, "row": 0}})
        for i, row in enumerate(df.values, start=1):
            line = ",".join(map(str, row))
            outs.append({"text": normalize_text(line), "meta": {**base_meta, "row": i}})
        return outs

    # --- CSV / TSV ---
    if ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        base_meta = make_base_meta(file_path, "tsv" if ext == "tsv" else "csv")
        df = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
        outs = []
        header = sep.join(map(str, df.columns))
        outs.append({"text": normalize_text(header), "meta": {**base_meta, "row": 0}})
        for i, row in enumerate(df.values, start=1):
            line = sep.join(map(str, row))
            outs.append({"text": normalize_text(line), "meta": {**base_meta, "row": i}})
        return outs

    # --- JSON Lines ---
    if ext == "jsonl":
        base_meta = make_base_meta(file_path, "jsonl")
        outs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                outs.append(
                    {
                        "text": normalize_text(line),
                        "meta": {**base_meta, "json_path": f"line:{idx}"},
                    }
                )
        return outs

    # --- JSON ---
    if ext == "json":
        base_meta = make_base_meta(file_path, "json")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
            except Exception:
                # 解析失败：按行拼回文本分块
                f.seek(0)
                raw = f.read()
                lines = raw.splitlines()
                text = "\n".join(line for line in lines if line.strip())
                return overlap_chunks(text, max_chars, overlap_ratio, base_meta)

        if isinstance(content, list):
            outs = []
            for i, item in enumerate(content):
                s = json.dumps(item, ensure_ascii=False)
                for ch in overlap_chunks(s, max_chars, overlap_ratio, base_meta):
                    ch["meta"]["json_path"] = f"[{i}]"
                    outs.append(ch)
            return outs

        elif isinstance(content, dict):
            outs = []
            for k, v in content.items():
                s = f"{k}: {json.dumps(v, ensure_ascii=False)}"
                for ch in overlap_chunks(s, max_chars, overlap_ratio, base_meta):
                    ch["meta"]["json_path"] = f".{k}"
                    outs.append(ch)
            return outs

        else:
            txt = json.dumps(content, ensure_ascii=False)
            return overlap_chunks(txt, max_chars, overlap_ratio, base_meta)

    # --- 图片 ---
    if ext in ("jpg", "jpeg", "png", "bmp", "gif", "webp"):
        base_meta = make_base_meta(file_path, "image")
        return [
            {
                "text": normalize_text(f"图片文件: {os.path.basename(file_path)}"),
                "meta": dict(base_meta),
            }
        ]

    # --- 其他不支持 ---
    raise ValueError(f"不支持的文件类型: {ext}")


def process_file(file_path: str, chunk_size: int = 500, overlap: float = 0.2):
    return split_file(file_path, max_chars=chunk_size, overlap_ratio=overlap)
