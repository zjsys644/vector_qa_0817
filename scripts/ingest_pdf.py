import sys
import os
from app.pdf_loader import load_and_split_pdf
from app.embedder import DeepSeekEmbedder
from app.qdrant_client import add_texts, init_collection

def ingest(pdf_path):
    print(f"Loading PDF: {pdf_path}")
    chunks = load_and_split_pdf(pdf_path)
    print(f"Chunked into {len(chunks)} segments.")
    embedder = DeepSeekEmbedder()
    embeddings = embedder.embed(chunks)
    print(f"Embeddings obtained. Uploading to Qdrant...")
    init_collection()
    add_texts(chunks, embeddings)
    print("Upload done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_pdf.py path/to/your.pdf")
        sys.exit(1)
    ingest(sys.argv[1])

