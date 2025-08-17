import os, glob, time, json
from app.file_loader import split_file
from app.embedder import Embedder
from app.qdrant_client import add_texts, init_collection

WATCH_PATH = "/data/sync_folder"
HISTORY_FILE = "/data/ingested_files.json"
EMBEDDER = Embedder()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {}

def save_history(hist):
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f)

def main_loop():
    history = load_history()
    while True:
        for fname in glob.glob(WATCH_PATH + "/*"):
            mtime = int(os.path.getmtime(fname))
            if fname not in history or history[fname] < mtime:
                try:
                    segments = split_file(fname)
                    texts = [seg["text"] for seg in segments]
                    payloads = [{k: v for k, v in seg.items() if k != "text"} for seg in segments]
                    embeddings = EMBEDDER.embed(texts)
                    init_collection()
                    add_texts(texts, embeddings, payloads=payloads)
                    history[fname] = mtime
                    save_history(history)
                    print(f"已同步: {fname}")
                except Exception as e:
                    print(f"{fname} 同步失败: {e}")
        time.sleep(300)  # 5分钟一轮

if __name__ == "__main__":
    main_loop()

