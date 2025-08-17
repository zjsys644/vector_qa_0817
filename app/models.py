# app/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class FileChunk(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    min_score: Optional[float] = None
    max_results: Optional[int] = None   # None 表示不额外截断（由 top_k 决定）
    max_return: Optional[int] = None    # 兼容旧字段名
    path: Optional[str] = ""

class SearchResult(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Dict[str, Any]
    score: float

class UploadResult(BaseModel):
    detail: List[Dict[str, Any]]

class FileListResult(BaseModel):
    files: List[Dict[str, Any]]

class SegmentsResult(BaseModel):
    segments: List[FileChunk]

class DeleteResult(BaseModel):
    detail: str

# 只选不写：按 id 批量取原文
class IdsRequest(BaseModel):
    ids: List[str]

class Chunk(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Dict[str, Any]
