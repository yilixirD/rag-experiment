from pathlib import Path
from dotenv import load_dotenv
import os
import re
import json

# langchain imports
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# loads variables from .env into environment
load_dotenv()  
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

PROCESSED_PATH=Path("data/text/")
OUTPUT_DIR = Path("data/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMB_MODEL = "text-embedding-3-small"

# Maps each section header to its parent Part for metadata tagging.
# "FORM 10-K" (cover page) is excluded — it's low-value boilerplate.
PARTS = {
    "Part I":   ["Item 1.", "Item 1A.", "Item 1B.", "Item 1C.", "Item 2.", "Item 3.", "Item 4."],
    "Part II":  ["Item 5.", "Item 6.", "Item 7.", "Item 7A.", "Item 8.", "Item 9.", "Item 9A.", "Item 9B.", "Item 9C."],
    "Part III": ["Item 10.", "Item 11.", "Item 12.", "Item 13.", "Item 14."],
    "Part IV":  ["Item 15.", "Item 16."],
}

# Ordered list of section headers used to split each 10-K into sections.
# "FORM 10-K" removed: it matched the old (newline-based) text format and
# only captured the boilerplate cover page — not useful for RAG retrieval.
SECTION_HEADERS = [
    "Item 1.", "Item 1A.", "Item 1B.", "Item 1C.",
    "Item 2.", "Item 3.", "Item 4.",
    "Item 5.", "Item 6.", "Item 7.", "Item 7A.", "Item 8.",
    "Item 9.", "Item 9A.", "Item 9B.", "Item 9C.",
    "Item 10.", "Item 11.", "Item 12.", "Item 13.", "Item 14.",
    "Item 15.", "Item 16.",
]



class ChunkEmbed:
    def __init__(self, folder_path: Path, key: str, chunk_size: int = 1000, chunk_overlap: int = 200,
                 emb_model: str = "text-embedding-3-small"):
        self.folder_path = folder_path
        self.OPENAIkey = key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.emb_model = emb_model
        self.chunks_collection = []
        self.embeddings = self.init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                            chunk_overlap=self.chunk_overlap)

    def init_embeddings(self) -> OpenAIEmbeddings: 
        """initialize the OpenAI embeddings model."""
        embeddings = OpenAIEmbeddings(model=self.emb_model,
            openai_api_key=self.OPENAIkey)
        return embeddings
    
    def chunk_all(self) -> None:
        """iterate through all text files in the folder and chunk them."""
        for text_path in self.folder_path.glob("*.txt"):
            print(f"Processing {text_path}...")
            self.chunking(text_path)


    def find_part(self, header: str) -> str:
        # find part
        part = "Unknown"
        for p, sl in PARTS.items():
            if header in sl:
                part = p
                break
        return part
    
    def chunking(self, path: Path) -> None:
        """
        Split a 10-K text file into sections, then chunk each section by length.

        Section boundaries are found by searching for SECTION_HEADERS in order.
        Searches are done sequentially (each starting after the previous hit) so
        that the Table of Contents — which lists all headers near the top — is
        skipped automatically: the first real hit for "Item 1." after the TOC's
        "Item 1." entry becomes the anchor, and every subsequent header is found
        after that point.

        Each chunk is stored as a dict with keys:
          chunk_id  — unique identifier: "{doc_id}_section_{i}_chunk_{j}"
          text      — the raw chunk text
          embedding — embedding vector from OpenAI
          metadata  — doc_id, stock symbol, part (I/II/III/IV), section header,
                      chunk position, source path, and embedding model name
        """
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        stock_sym = path.stem.split("-")[0]
        # Filename format is "{ticker}-{YYYYMMDD}". For 10-K filings the date
        # in the filename is the fiscal year end date (period of report).
        # We also use it as a proxy for the filing date since the actual filing
        # date would require parsing the HTML cover page.
        raw_date = path.stem.split("-")[1]   # e.g. "20250927"
        fiscal_year_end = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"  # "2025-09-27"
        filing_date = fiscal_year_end  # same source; update if you parse the real filing date later

        # Skip the Table of Contents by finding the SECOND occurrence of the
        # first section header. The TOC lists all Item headers near the top of
        # the file; the real content appears later with the same header text.
        first_header = SECTION_HEADERS[0]
        toc_hit = text.find(first_header)
        content_start = text.find(first_header, toc_hit + 1) if toc_hit != -1 else 0

        # Find the start position of each section, searching forward from the
        # previous section's position so we never re-match the TOC.
        section_positions = []
        prev_pos = content_start
        for header in SECTION_HEADERS:
            pos = text.find(header, prev_pos)
            if pos == -1:
                continue
            section_positions.append((header, pos))
            prev_pos = pos + 1

        # Chunk each section: text runs from this section's start to the next's start
        # (or end-of-file for the last section).
        chunk_idx = 0
        for i, (header, start) in enumerate(section_positions):
            end = section_positions[i + 1][1] if i + 1 < len(section_positions) else len(text)
            section = text[start:end]
            part = self.find_part(header)
            chunks = self.text_splitter.split_text(section)
            for j, c in enumerate(chunks):
                chunk_id = f"{doc_id}_section_{i + 1}_chunk_{j + 1}"
                metadata = {
                    "chunk_idx": chunk_idx,
                    "source": str(path),
                    "doc_id": doc_id,
                    "emb_model": self.emb_model,
                    "stock symbol": stock_sym,
                    "fiscal_year_end": fiscal_year_end,
                    "filing_date": filing_date,
                    "part": part,
                    "section": header,
                    "chunk": j + 1,
                }
                emb = self.embeddings.embed_query(c)
                chunk = {"chunk_id": chunk_id, "text": c, "embedding": emb, "metadata": metadata}
                self.chunks_collection.append(chunk)
                chunk_idx += 1

    def save_chunks(self, output_dir: Path):
        """save the chunks collection to a file."""
        output_path = output_dir / "chunks_collection.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks_collection:
                json_line = json.dumps(chunk, ensure_ascii=False)
                f.write(json_line + "\n")
        print(f"Saved {len(self.chunks_collection)} chunks to {output_path}")


def main() -> None:
    processor = ChunkEmbed(folder_path=PROCESSED_PATH, key=OPENAI_API_KEY, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    processor.chunk_all()
    processor.save_chunks(OUTPUT_DIR)


if __name__ == "__main__":
    main()



