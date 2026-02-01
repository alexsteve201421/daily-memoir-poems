import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
import tiktoken
from openai import OpenAI

BOOK_PATH = Path("book.txt")
DB_DIR = "chroma_db"
COLLECTION_NAME = "manu_shah_book"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_tokens: int = 450, overlap_tokens: int = 80):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_tokens
        if start < 0:
            start = 0
        if end == len(tokens):
            break
    return chunks

def main():
    if not BOOK_PATH.exists():
        raise FileNotFoundError("book.txt not found in project folder.")

    print("Loading book text...")
    raw = BOOK_PATH.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw)

    print("Chunking book...")
    chunks = chunk_text(text)
    print(f"Book loaded. Total chunks: {len(chunks)}")

    chroma = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i:05d}" for i in range(len(chunks))]

    print("Creating embeddings and storing them...")
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        embeddings = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        ).data

        collection.add(
            ids=batch_ids,
            documents=batch,
            embeddings=[e.embedding for e in embeddings],
        )

        print(f"Added {len(batch)} chunks...")

    print("Ingestion complete. Vector DB saved in chroma_db/")

if __name__ == "__main__":
    main()
