from pathlib import Path


def get_section_title(text: str) -> str:
    """Use the first Markdown heading as the document section title."""

    for line in text.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()

    return "Untitled"


def chunk_document(
    document: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, str | int]]:
    """Split one document into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    words = document["text"].split()
    step = chunk_size - overlap
    source = document["source"]
    document_name = Path(source).stem
    section = get_section_title(document["text"])

    chunks = []

    for chunk_index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            continue

        chunks.append(
            {
                "id": f"{document_name}-{chunk_index:03d}",
                "text": " ".join(chunk_words),
                "source": source,
                "section": section,
                "chunk_index": chunk_index,
            }
        )

    return chunks