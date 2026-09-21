"""Re-embed all chunks in chunks_collection.jsonl with a local Ollama model and load into chunks_nomic."""

import json

import psycopg
import requests

JSONL_PATH = "data/chunks/chunks_collection.jsonl"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
DSN = "host=localhost port=5432 dbname=rag user=rag password=rag"
BATCH_SIZE = 50

INSERT_SQL = """
INSERT INTO chunks_nomic (
    chunk_id, doc_id, stock_symbol, part, section, chunk_idx,
    fiscal_year_end, filing_date, emb_model, text, embedding
) VALUES (
    %(chunk_id)s, %(doc_id)s, %(stock_symbol)s, %(part)s, %(section)s, %(chunk_idx)s,
    %(fiscal_year_end)s, %(filing_date)s, %(emb_model)s, %(text)s, %(embedding)s::vector
)
ON CONFLICT (chunk_id) DO NOTHING
"""


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "input": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]


def row_to_params(row: dict, embedding: list[float]) -> dict:
    meta = row["metadata"]
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": meta.get("doc_id"),
        "stock_symbol": meta.get("stock symbol") or meta.get("stock_symbol"),
        "part": meta.get("part"),
        "section": meta.get("section"),
        "chunk_idx": meta.get("chunk_idx"),
        "fiscal_year_end": meta.get("fiscal_year_end"),
        "filing_date": meta.get("filing_date"),
        "emb_model": MODEL,
        "text": row["text"],
        "embedding": "[" + ",".join(str(x) for x in embedding) + "]",
    }


def main():
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur, open(JSONL_PATH) as f:
            batch_rows = []
            total = 0
            for line in f:
                batch_rows.append(json.loads(line))
                if len(batch_rows) >= BATCH_SIZE:
                    embeddings = embed_batch([r["text"] for r in batch_rows])
                    params = [row_to_params(r, e) for r, e in zip(batch_rows, embeddings)]
                    cur.executemany(INSERT_SQL, params)
                    total += len(batch_rows)
                    print(f"embedded {total}", end="\r")
                    batch_rows = []
            if batch_rows:
                embeddings = embed_batch([r["text"] for r in batch_rows])
                params = [row_to_params(r, e) for r, e in zip(batch_rows, embeddings)]
                cur.executemany(INSERT_SQL, params)
                total += len(batch_rows)
        conn.commit()
    print(f"\ndone: {total} rows")


if __name__ == "__main__":
    main()
