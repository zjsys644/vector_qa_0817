from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

import os

from app.config import QDRANT_COLLECTION
from app.file_loader import split_file
from app.embedder import get_embedder
from app.qdrant_client import QdrantDB
from app.models import (
    FileChunk,
    SearchRequest,
    SearchResult,
    UploadResult,
    FileListResult,
    SegmentsResult,
    DeleteResult
)

app = FastAPI(
    title="PDF Vector QA",
    description="支持多类型文件上传、分块、embedding 入库与向量检索，兼容 OpenWebUI MCP 工具。",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

embedder = get_embedder()
db = QdrantDB(QDRANT_COLLECTION)

@app.get("/", response_class=HTMLResponse, summary="上传页面")
def index():
    return FileResponse("app/static/upload.html")

@app.get("/admin", response_class=HTMLResponse, summary="后台管理页面")
def admin():
    return FileResponse("app/static/admin.html")

class UploadParams(BaseModel):
    custom_chars: int = 500
    overlap_ratio: float = 0.2

@app.post("/upload", response_model=UploadResult, summary="上传并分块文件")
async def upload(
    files: List[UploadFile] = File(..., description="支持多类型批量上传"),
    custom_chars: int = 500,
    overlap_ratio: float = 0.2
):
    """
    支持多文件上传，自动按参数分块、embedding、写入知识库。
    """
    results = []
    for file in files:
        contents = await file.read()
        path = f"tmp_{file.filename}"
        with open(path, "wb") as f:
            f.write(contents)
        chunks = split_file(path, max_chars=custom_chars, overlap_ratio=overlap_ratio)
        os.remove(path)
        for chunk in chunks:
            vector = embedder(chunk['text'])
            print("embed dim =", len(vector))
            meta = chunk.get('meta', {}).copy()    # 没有meta就用空dict
            meta["filename"] = file.filename
            db.insert(vector, chunk['text'], meta)
        results.append({"filename": file.filename, "segments": len(chunks)})
    return UploadResult(detail=results)

# main.py 里的 /search
# main.py 里 /search（仅演示关键部分）
@app.post("/search")
async def search(req: SearchRequest):
    query = (req.query or "").strip()
    if not query:
        return {
            "final_markdown": "暂时没有查到，请咨询人工客服或者期待更新。",
            "results": [],
            "meta": {"verbatim": True}
        }

    q_vec = embedder(query)
    hits = db.search(q_vec, top_k=30)  # 你原来多少就用多少

    if not hits:
        return {
            "final_markdown": "暂时没有查到，请咨询人工客服或者期待更新。",
            "results": [],
            "meta": {"verbatim": True}
        }

    def block(h):
        # 防止反引号破坏代码块
        t = (h.get("text") or "").replace("```", "`` `")
        return f"```\n{t}\n```"

    md = "【原文片段（逐字展示）】\n" + "\n\n".join(block(h) for h in hits[:3])
    # 如要附带来源/分数可继续拼 md

    return {
        "final_markdown": md,
        "results": hits,
        "meta": {"verbatim": True}
    }

@app.get("/list_files")
def list_files():
    results = db.list_files()
    return results

@app.delete("/delete_by_filename", response_model=DeleteResult, summary="删除指定文件及分块")
def delete_by_filename(filename: str):
    """
    删除指定文件相关所有知识库分片。
    """
    db.delete_by_filename(filename)
    return DeleteResult(detail=f"Deleted {filename}")

@app.get("/file_segments")
def file_segments(filename: str):
    segments = db.file_segments(filename)
    return segments
