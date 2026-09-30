from sentence_transformers import SentenceTransformer


class Embedder:
    """Create semantic vector embeddings for RAG documents and questions."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks: list[dict]) :
        """Convert chunk text into a matrix of document embeddings."""

        texts = [chunk["text"] for chunk in chunks]

        return self.model.encode_document(
            texts,
            convert_to_numpy=True,
        )

    def embed_query(self, question: str):
        """Convert one user question into a query embedding."""

        return self.model.encode_query(
            question,
            convert_to_numpy=True,
        )