"""Ingest data/chunks/chunks_collection.jsonl into the local Postgres `chunks` table."""

import json

import psycopg

DSN = "host=localhost port=5432 dbname=rag user=rag password=rag"
CHUNKS_PATH = "data/chunks/chunks_collection.jsonl"
BATCH_SIZE = 200

INSERT_SQL = """
INSERT INTO chunks (
    chunk_id, doc_id, stock_symbol, part, section, chunk_idx,
    fiscal_year_end, filing_date, emb_model, text, embedding
) VALUES (
    %(chunk_id)s, %(doc_id)s, %(stock_symbol)s, %(part)s, %(section)s, %(chunk_idx)s,
    %(fiscal_year_end)s, %(filing_date)s, %(emb_model)s, %(text)s, %(embedding)s::vector
)
ON CONFLICT (chunk_id) DO NOTHING
"""


def row_to_params(row: dict) -> dict:
    meta = row["metadata"]
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": meta.get("doc_id"),
        "stock_symbol": meta.get("stock symbol"),
        "part": meta.get("part"),
        "section": meta.get("section"),
        "chunk_idx": meta.get("chunk_idx"),
        "fiscal_year_end": meta.get("fiscal_year_end"),
        "filing_date": meta.get("filing_date"),
        "emb_model": meta.get("emb_model"),
        "text": row["text"],
        "embedding": "[" + ",".join(str(x) for x in row["embedding"]) + "]",
    }


def main():
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur, open(CHUNKS_PATH) as f:
            batch = []
            total = 0
            for line in f:
                batch.append(row_to_params(json.loads(line)))
                if len(batch) >= BATCH_SIZE:
                    cur.executemany(INSERT_SQL, batch)
                    total += len(batch)
                    print(f"inserted {total}", end="\r")
                    batch = []
            if batch:
                cur.executemany(INSERT_SQL, batch)
                total += len(batch)
        conn.commit()
    print(f"\ndone: {total} rows")


if __name__ == "__main__":
    main()
