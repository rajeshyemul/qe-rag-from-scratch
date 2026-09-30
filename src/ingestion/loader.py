from pathlib import Path


def load_documents(directory: str) -> list[dict[str, str]]:
    """Load Markdown documents from a directory."""

    documents = []

    for file_path in Path(directory).glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": str(file_path),
                "text": text,
            }
        )

    return documents