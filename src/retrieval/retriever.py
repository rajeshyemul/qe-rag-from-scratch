class Retriever:
    """Retrieve relevant knowledge chunks for a user question."""

    def __init__(self, embedder, vector_store, chunks: list[dict]) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """Return the top matching chunks for one question."""

        query_embedding = self.embedder.embed_query(question)

        distances, indices = self.vector_store.search(
            query_embedding,
            k=k,
        )

        results = []

        for distance, chunk_position in zip(distances, indices):
            chunk = self.chunks[int(chunk_position)]

            results.append(
                {
                    **chunk,
                    "distance": float(distance),
                }
            )

        return results