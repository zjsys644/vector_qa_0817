# app/qdrant_client.py
import os
import uuid
import logging
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from .config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM

logger = logging.getLogger("qdrant-db")

# 可通过环境变量控制阈值与回退行为
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0"))          # <=0 表示不设阈值
ENABLE_TEXT_FALLBACK = os.getenv("ENABLE_TEXT_FALLBACK", "1") != "0"
FALLBACK_SCAN_LIMIT = int(os.getenv("FALLBACK_SCAN_LIMIT", "2000"))  # 回退时最多扫描这么多点

client = QdrantClient(QDRANT_URL)

def init_collection():
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        logger.info("Created collection %s (dim=%d)", QDRANT_COLLECTION, EMBEDDING_DIM)

def add_texts(texts: List[str], embeddings: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None):
    init_collection()
    points = []
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        payload = (payloads[i] if payloads else {}) or {}
        payload["text"] = text
        points.append(PointStruct(id=str(uuid.uuid4()), vector=emb, payload=payload))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

def _normalize_hits(scored_points):
    out = []
    for r in scored_points:
        payload = r.payload or {}
        out.append({
            "id": str(getattr(r, "id", "")),   # <== 带上 id
            "text": payload.get("text", "") or "",
            "score": r.score,
            "meta": {k: v for k, v in payload.items() if k != "text"},
        })
    return out

def text_fallback(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    纯文本兜底：scroll 扫描部分点，返回包含关键字的前 top_k 条。
    """
    if not query_text:
        return []
    pts, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        with_payload=True,
        limit=FALLBACK_SCAN_LIMIT
    )
    rows: List[Dict[str, Any]] = []
    for p in pts:
        payload = p.payload or {}
        t = (payload.get("text") or "")
        if t and (query_text in t):
            rows.append({
                "text": t,
                "meta": {k: v for k, v in payload.items() if k != "text"},
                "score": 0.0
            })
            if len(rows) >= top_k:
                break
    return rows

def search(query_emb: List[float], top_k: int = 15, query_text: Optional[str] = None,
           score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    向量检索 + 关键字优先排序 + 纯文本兜底
    """
    init_collection()
    th = SCORE_THRESHOLD if score_threshold is None else score_threshold
    th = None if (th is None or th <= 0) else th

    logger.info("Qdrant.search top_k=%d, threshold=%s, query_text=%r", top_k, th, (query_text or "")[:50])

    result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_emb,
        limit=top_k,
        with_payload=True,
        score_threshold=th  # None => 不设阈值
    )
    hits = _normalize_hits(result)

    # 关键字优先（包含 query_text 的先排前）
    if query_text:
        contain = [h for h in hits if query_text in (h["text"] or "")]
        others = [h for h in hits if h not in contain]
        hits = contain + others

    # 纯文本回退，避免返回 []
    if not hits and ENABLE_TEXT_FALLBACK and query_text:
        logger.info("Qdrant.search got 0 hits; fallback to substring scan...")
        hits = text_fallback(query_text, top_k=top_k)

    return hits

def delete_by_filename(filename: str):
    init_collection()
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        )
    )

def list_files():
    init_collection()
    pts, _ = client.scroll(collection_name=QDRANT_COLLECTION, with_payload=True, limit=10000)
    from collections import Counter
    files = [ (p.payload or {}).get("filename", "") for p in pts ]
    stats = Counter(files)
    return [{"filename": k, "segments": v} for k, v in stats.items() if k]

def get_points_by_filename(filename: str, limit: int = 200):
    """
    ✅ 这就是你现在缺的函数：按文件名拿到分片
    """
    init_collection()
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    pts, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        with_payload=True,
        limit=limit,
        scroll_filter=Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))])
    )
    return [
        {
            "id": getattr(p, "id", ""),
            "text": (p.payload or {}).get("text", "") or "",
            "meta": {k: v for k, v in (p.payload or {}).items() if k != "text"}
        }
        for p in pts
    ]

def get_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    pts = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=ids,
        with_payload=True
    )
    rows = []
    for p in pts:
        pay = p.payload or {}
        rows.append({
            "id": str(getattr(p, "id", "")),
            "text": pay.get("text", "") or "",
            "meta": {k: v for k, v in pay.items() if k != "text"},
        })
    return rows

class QdrantDB:
    def __init__(self, collection_name):
        self.collection_name = collection_name

    def insert(self, vector, text, meta):
        add_texts([text], [vector], [meta])

    def search(self, query_emb, top_k: int = 15, query_text: Optional[str] = None,
               score_threshold: Optional[float] = None):
        return search(query_emb, top_k=top_k, query_text=query_text, score_threshold=score_threshold)

    def delete_by_filename(self, filename):
        return delete_by_filename(filename)

    def list_files(self):
        return list_files()

    def file_segments(self, filename, limit=200):
        return get_points_by_filename(filename, limit=limit)

    def get_by_ids(self, ids: List[str]):
        return get_by_ids(ids)
