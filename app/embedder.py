# app/embedder.py
import os, httpx

class Embedder:
    def __init__(self):
        self.method = os.getenv("EMBEDDING_METHOD", "ollama")
        self.model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
        if self.method not in {"ollama"}:
            raise ValueError(f"不支持的 EMBEDDING_METHOD: {self.method}")

        # Ollama 运行在宿主机 11434，容器里访问用 host.docker.internal
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    def __call__(self, text: str):
        if self.method == "ollama":
            with httpx.Client(timeout=20) as cli:
                r = cli.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.model, "prompt": text}
                )
            r.raise_for_status()
            data = r.json()
            return data["embedding"]

def get_embedder():
    return Embedder()
