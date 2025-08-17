from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from .config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM
import uuid

client = QdrantClient(QDRANT_URL)

def init_collection():
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )


def add_texts(texts, embeddings, payloads=None):
    init_collection()
    points = []
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        payload = payloads[i] if payloads else {}
        payload["text"] = text
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=emb,
            payload=payload
        ))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

def search(query_emb, top_k=15):
    init_collection()
    result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_emb,
        limit=top_k,
        with_payload=True
    )
    return [
        {
            "text": r.payload.get("text", ""),
            "score": r.score,
            "meta": {k: v for k, v in r.payload.items() if k != "text"}
        } for r in result
    ]

def delete_by_filename(filename):
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
    points, _ = client.scroll(collection_name=QDRANT_COLLECTION, with_payload=True, limit=10000)
    from collections import Counter
    files = [p.payload.get("filename", "") for p in points]
    stats = Counter(files)
    return [{"filename": k, "segments": v} for k, v in stats.items() if k]

def get_points_by_filename(filename, limit=200):
    init_collection()
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        with_payload=True,
        limit=limit,
        scroll_filter=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        )
    )
    return [
        {
            "id": getattr(p, "id", ""),
            "text": p.payload.get("text", ""),
            "meta": {k: v for k, v in p.payload.items() if k != "text"}
        }
        for p in points
    ]

class QdrantDB:
    def __init__(self, collection_name):
        self.collection_name = collection_name

    def insert(self, vector, text, meta):
        # 适配 insert 单个
        add_texts([text], [vector], [meta])

    def search(self, query_emb, top_k=15):
        return search(query_emb, top_k)

    def delete_by_filename(self, filename):
        return delete_by_filename(filename)

    def list_files(self):
        return list_files()

    def file_segments(self, filename, limit=200):
        return get_points_by_filename(filename, limit=limit)
