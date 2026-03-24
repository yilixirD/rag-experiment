import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/text")
START_MARKER = "SECURITIES AND EXCHANGE COMMISSION"

# Tags that represent block-level structure in HTML. When we encounter one of
# these during text extraction we insert a newline before and after its content,
# so that paragraphs, table cells, list items, etc. each end up on their own
# line. Inline tags (span, a, b, em, font, …) are NOT in this set, so their
# text flows directly into the surrounding sentence without extra line breaks.
BLOCK_TAGS = frozenset({
    "p", "div",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "tr", "td", "th",
    "li", "ul", "ol",
    "section", "article", "header", "footer",
    "blockquote", "pre",
})


def _extract_text(node) -> str:
    """
    Recursively walk the BeautifulSoup tree and build plain text.

    - Block-level elements (div, p, td, etc.) get a newline before and after
      their content, producing one logical unit per line.
    - Inline elements (span, a, b, font, etc.) are traversed without adding
      any separator, so their text merges naturally with adjacent text.
    - <br> tags emit a single newline.
    - Raw text nodes are returned as-is.
    """
    if isinstance(node, NavigableString):
        return str(node)

    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name == "br":
                parts.append("\n")
            elif child.name in BLOCK_TAGS:
                inner = _extract_text(child).strip()
                if inner:
                    parts.append("\n" + inner + "\n")
            else:
                # Inline element — recurse with no extra newlines
                parts.append(_extract_text(child))
    return "".join(parts)


def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text, trimming to start after the SEC marker."""
    soup = BeautifulSoup(html, "lxml")

    # Remove elements that are pure noise (scripts, styles, hidden XBRL metadata)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = _extract_text(soup)

    # Keep only text after the first occurrence of the SEC header marker
    idx = text.upper().find(START_MARKER.upper())
    if idx != -1:
        text = text[idx + len(START_MARKER):]

    # Normalize non-breaking spaces to regular spaces
    text = text.replace("\xa0", " ")

    # Collapse any run of whitespace that is NOT a newline into a single space.
    # This cleans up multiple spaces left by inline tag boundaries.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip trailing/leading spaces from each line, drop blank lines
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Join a newline that appears immediately before closing punctuation —
    # e.g. a sentence that was split at a "." or "," by a tag boundary.
    text = re.sub(r"\n([,.:;)])", r"\1", text)

    # Collapse any remaining runs of 2+ newlines down to one
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

