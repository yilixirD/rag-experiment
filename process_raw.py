import re
from pathlib import Path

from bs4 import BeautifulSoup


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/text")
START_MARKER = "SECURITIES AND EXCHANGE COMMISSION"


def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text, trimming to start after the SEC marker."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script / style elements that are usually noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    #Keep only text after the first occurrence of the marker (case-insensitive)
    upper_text = text.upper()
    marker_upper = START_MARKER.upper()
    idx = upper_text.find(marker_upper)
    if idx != -1:
        text = text[idx + len(START_MARKER) :]

    text = text.replace("\xa0", " ")

    # Collapse multiple spaces
    #text = re.sub(r" +", " ", text)
    # Remove space before punctuation so "word ," and "word ." become "word," and "word."
    #text = re.sub(r" ([,.;:!?)\]}\"])", r"\1", text)

    # Normalise: strip lines and drop empties, then rejoin
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n([\\(\\),.;:$])", r"\1", text)
    text = re.sub(r"([,$])\n", r"\1", text)

    text = re.sub(r"\n{2,}", "\n", text)

    return text


def convert_all_html(raw_dir: Path = RAW_DIR, out_dir: Path = OUTPUT_DIR) -> None:
    """Convert all .html files in raw_dir to .txt files in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for html_path in sorted(raw_dir.glob("*.html")):
        txt_path = out_dir / (html_path.stem + ".txt")

        html = html_path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        txt_path.write_text(text, encoding="utf-8")

        print(f"Converted {html_path} -> {txt_path}")


def main() -> None:
    convert_all_html()


if __name__ == "__main__":
    main()

