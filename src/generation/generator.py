from pathlib import Path

from ollama import Client # type: ignore

SYSTEM_PROMPT = """
You are a quality-engineering knowledge assistant.

Answer only from the supplied context.
Every statement must be directly supported by that context.
Do not infer benefits, add examples, or introduce facts that are not stated.

If the context does not contain enough information, say:
"I do not have enough information in the knowledge base to answer that."

Use a concise, clear explanation.
Do not include a sources section. The application adds verified sources itself.
""".strip()


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as evidence for answer generation."""

    context_blocks = []

    for chunk in chunks:
        source_name = Path(chunk["source"]).name

        context_blocks.append(
            "\n".join(
                [
                    f"[Source: {source_name}]",
                    f"[Section: {chunk['section']}]",
                    f"[Chunk: {chunk['chunk_index']}]",
                    chunk["text"],
                ]
            )
        )

    return "\n\n".join(context_blocks)

def collect_sources(chunks: list[dict]) -> list[dict[str, str]]:
    """Return unique, source-aware citations from retrieved chunks."""

    sources = []
    seen_sources = set()

    for chunk in chunks:
        source_name = Path(chunk["source"]).name

        if source_name in seen_sources:
            continue

        sources.append(
            {
                "source": source_name,
                "section": chunk["section"],
            }
        )
        seen_sources.add(source_name)

    return sources

class OllamaGenerator:
    """Generate grounded answers using a local Ollama model."""

    def __init__(self, model: str = "llama3.2:latest") -> None:
        self.model = model
        self.client = Client(host="http://localhost:11434")

    def generate(self, question: str, context: str) -> str:
        """Answer a question using only the retrieved context."""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{question}\n\n"
                        "Answer:"
                    ),
                },
            ],
            options={"temperature": 0},
        )

        return response.message.content