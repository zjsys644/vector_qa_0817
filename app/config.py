import os
from dotenv import load_dotenv

load_dotenv()

import os

EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "ollama")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM", "1024"))

OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

QDRANT_URL        = os.getenv("QDRANT_URL", "http://host.docker.internal:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_knowledge_bge_m3")
