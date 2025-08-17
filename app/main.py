# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import os
import logging

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pdf-vector-qa")

# -------------------- 依赖 --------------------
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
    DeleteResult,
    # 新增：只选不写需要
    IdsRequest,
    Chunk,
)

# -------------------- App --------------------
app = FastAPI(
    title="PDF Vector QA",
    description="支持多类型文件上传、分块、embedding 入库与向量检索，兼容 OpenWebUI MCP 工具。",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

embedder = get_embedder()
db = QdrantDB(QDRANT_COLLECTION)

# -------------------- Pages --------------------
@app.get("/", response_class=HTMLResponse, summary="上传页面")
def index():
    return FileResponse("app/static/upload.html")

@app.get("/admin", response_class=HTMLResponse, summary="后台管理页面")
def admin():
    return FileResponse("app/static/admin.html")

# -------------------- Models --------------------
class UploadParams(BaseModel):
    custom_chars: int = 500
    overlap_ratio: float = 0.2

# -------------------- Upload --------------------
@app.post("/upload", response_model=UploadResult, summary="上传并分块文件")
async def upload(
    files: List[UploadFile] = File(..., description="支持多类型批量上传"),
    custom_chars: int = 500,
    overlap_ratio: float = 0.2
):
    results: List[Dict[str, Any]] = []
    for file in files:
        contents = await file.read()
        path = f"tmp_{file.filename}"
        with open(path, "wb") as f:
            f.write(contents)

        try:
            chunks = split_file(path, max_chars=custom_chars, overlap_ratio=overlap_ratio)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

        # 入库
        for chunk in chunks:
            text = chunk.get("text", "") or ""
            meta = (chunk.get("meta") or {}).copy()
            meta["filename"] = file.filename
            try:
                vec = embedder(text)
                db.insert(vec, text, meta)
            except Exception:
                logger.exception("Insert chunk failed: %s", meta)

        results.append({"filename": file.filename, "segments": len(chunks)})

    return UploadResult(detail=results)

# --- /search 路由 ---
@app.post("/search", response_model=List[SearchResult], summary="向量检索")
async def search(req: SearchRequest):
    # 1) 读取入参（兼容旧的 max_results）
    query = (getattr(req, "query", "") or "").strip()
    if not query:
        logger.warning("[SEARCH] empty query")
        return []

    # top_k：优先用 req.top_k；没有则回退到旧字段 max_results；再没有用 15
    top_k_val = getattr(req, "top_k", None)
    legacy_max_results = getattr(req, "max_results", None)
    top_k = int(top_k_val if top_k_val is not None
                else (legacy_max_results if legacy_max_results is not None else 15))

    # max_return：优先用 req.max_return；没有就等于 top_k
    max_return_val = getattr(req, "max_return", None)
    max_return = int(max_return_val if max_return_val is not None else top_k)

    # min_score：默认 0.0（如果你想让 Qdrant 侧阈值生效，可以传这个给 score_threshold）
    min_score_val = getattr(req, "min_score", None)
    min_score = float(min_score_val if min_score_val is not None else 0.0)

    # 2) 向量化
    try:
        qvec = embedder(query)
        qdim = len(qvec) if hasattr(qvec, "__len__") else None
        logger.info("[SEARCH] query=%r dim=%s top_k=%d max_return=%d min_score=%.3f",
                    query[:80], qdim, top_k, max_return, min_score)
    except Exception:
        logger.exception("Embedder failed to encode query")
        return []

    # 3) Qdrant 检索（把 min_score 作为 score_threshold 传入）
    try:
        hits = db.search(qvec, top_k=top_k, query_text=query, score_threshold=min_score) or []
    except Exception:
        logger.exception("Qdrant search failed")
        return []

    # 4) 过滤与截断
    out = []
    for h in hits:
        try:
            score = float(h.get("score", 0.0))
            if score < min_score:
                continue
            out.append(
                SearchResult(
                    text=str(h.get("text") or ""),
                    meta=dict(h.get("meta") or {}),
                    score=score,
                )
            )
        except Exception:
            logger.exception("Bad hit row encountered")
            continue

    out = out[:max_return]
    logger.info("[SEARCH] hits_total=%d, hits_return=%d", len(hits), len(out))
    return out

# -------------------- Files / Segments --------------------
@app.get("/list_files", summary="列出已入库的文件")
def list_files():
    return db.list_files()

@app.delete("/delete_by_filename", response_model=DeleteResult, summary="删除指定文件及分块")
def delete_by_filename(filename: str):
    db.delete_by_filename(filename)
    return DeleteResult(detail=f"Deleted {filename}")

@app.get("/file_segments", summary="查看某个文件的所有分片")
def file_segments(filename: str):
    # 直接返回列表，前端可直接渲染
    return db.file_segments(filename)

# -------------------- Get by IDs（只选不写） --------------------
@app.post("/points_by_ids", response_model=List[Chunk], summary="按ID批量取原文")
async def points_by_ids(req: IdsRequest):
    ids = getattr(req, "ids", []) or []
    if not ids:
        return []
    rows = db.get_by_ids(ids)
    return [Chunk(id=r["id"], text=r["text"], meta=r["meta"]) for r in rows]

# -------------------- Health --------------------
@app.get("/healthz")
def health():
    return {"status": "ok"}
