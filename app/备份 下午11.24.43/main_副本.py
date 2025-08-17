from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from .embedder import Embedder
from .file_loader import split_file
from .qdrant_client import (
    add_texts, search, delete_by_filename, list_files, init_collection, get_points_by_filename
)
from .models import SearchRequest, SearchResponse, SearchResult

app = FastAPI(title="文件多类型知识库")
embedder = Embedder()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("app/static/upload.html")

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return FileResponse("app/static/admin.html")

@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    split_mode: str = Form("chars_500"),
    custom_chars: int = Form(None)
):
    init_collection()
    all_results = []
    allowed_exts = [
        ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xls",
        ".jsonl", ".json", ".png", ".jpg", ".jpeg"
    ]
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.filename}")
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        try:
            # 支持自定义分割方式
            if split_mode == "chars_500":
                segments = split_file(tmp_path, max_chars=500)
            elif split_mode == "chars_300":
                segments = split_file(tmp_path, max_chars=300)
            elif split_mode == "page":
                segments = split_file(tmp_path, by="page")
            elif split_mode == "custom" and custom_chars and custom_chars > 0:
                segments = split_file(tmp_path, max_chars=custom_chars)
            else:
                segments = split_file(tmp_path, max_chars=500)
            texts = [seg["text"] for seg in segments]
            payloads = []
            for seg in segments:
                meta = {k: v for k, v in seg.items() if k != "text"}
                meta["filename"] = file.filename
                payloads.append(meta)
            embeddings = embedder.embed(texts)
            add_texts(texts, embeddings, payloads=payloads)
            all_results.append({"file": file.filename, "segments": len(texts)})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{file.filename}: {str(e)}")
        finally:
            os.remove(tmp_path)
    return {"msg": f"上传并入库成功", "detail": all_results}

@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    init_collection()
    try:
        query_emb = embedder.embed([request.query])[0]
        hits = search(query_emb, request.top_k)
        results = [SearchResult(text=hit["text"], score=hit["score"], meta=hit.get("meta", {})) for hit in hits]
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_by_filename")
def api_delete_by_filename(filename: str = Query(...)):
    init_collection()
    delete_by_filename(filename)
    return {"msg": f"已删除文件 {filename} 的全部段落"}

@app.get("/list_files")
def api_list_files():
    init_collection()
    return list_files()

@app.get("/file_segments")
def file_segments(filename: str = Query(...)):
    return get_points_by_filename(filename)

@app.get("/tools_config.json")
def tools_config():
    ip = os.environ.get("MCP_OPENAPI_IP") or "localhost"
    port = os.environ.get("MCP_OPENAPI_PORT") or "8000"
    return JSONResponse({
        "name_for_human": "PDF Vector QA",
        "name_for_model": "pdf_vector_qa",
        "description_for_model": "上传文件后，支持多格式知识库问答和分割内容管理。/search 检索接口可直接用。",
        "description_for_human": "多类型文件知识库管理和问答，支持文件上传、检索和分割详情查看。",
        "api": {
            "type": "openapi",
            "url": f"http://{ip}:{port}/openapi.json",
            "is_user_authenticated": False
        },
        "auth": { "type": "none" }
    })

