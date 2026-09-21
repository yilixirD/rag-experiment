"""Chunk+embed only the doc_ids not yet in chunks_collection.jsonl / Postgres, then ingest+append."""

import json
from pathlib import Path

from chunk_embed import ChunkEmbed, PROCESSED_PATH, OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMB_MODEL
from ingest_pgvector import row_to_params, INSERT_SQL, DSN

import os
from dotenv import load_dotenv
import psycopg

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

MISSING_DOC_IDS = {
    "cost-20250831",
    "goog-20251103",
    "intc-20241228",
    "isrg-20241231",
    "tsla-20241231",
    "unh-20241231",
}


def main():
    processor = ChunkEmbed(
        folder_path=PROCESSED_PATH,
        key=OPENAI_API_KEY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        emb_model=EMB_MODEL,
    )
    for text_path in sorted(PROCESSED_PATH.glob("*.txt")):
        if text_path.stem not in MISSING_DOC_IDS:
            continue
        print(f"Processing {text_path}...")
        processor.chunking(text_path)

    new_chunks = processor.chunks_collection
    print(f"Chunked+embedded {len(new_chunks)} new chunks")

    # metadata key here is "stock_symbol" (chunk_embed.py), unlike the older
    # jsonl which used "stock symbol" -- row_to_params handles the old key only,
    # so normalize before reuse.
    for c in new_chunks:
        if "stock symbol" not in c["metadata"]:
            c["metadata"]["stock symbol"] = c["metadata"].get("stock_symbol")

    # Append to the master JSONL (don't overwrite existing rows).
    jsonl_path = OUTPUT_DIR / "chunks_collection.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        for c in new_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Appended {len(new_chunks)} chunks to {jsonl_path}")

    # Insert into Postgres.
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            params = [row_to_params(c) for c in new_chunks]
            cur.executemany(INSERT_SQL, params)
        conn.commit()
    print(f"Inserted {len(new_chunks)} rows into Postgres")


if __name__ == "__main__":
    main()
