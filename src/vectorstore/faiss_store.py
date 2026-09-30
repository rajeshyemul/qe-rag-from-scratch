import faiss # type: ignore


class FaissStore:
    """Store and search fixed-size embedding vectors with FAISS."""

    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings) -> None:
        """Add a matrix of embeddings to the index."""

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix")

        if embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Expected embeddings with {self.index.d} dimensions, "
                f"received {embeddings.shape[1]}"
            )

        self.index.add(embeddings.astype("float32"))

    def search(self, query_embedding, k: int = 3):
        """Return the nearest stored vectors for one query embedding."""

        if self.count == 0:
            raise ValueError("Cannot search an empty index")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be a vector or a matrix")

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Expected a query with {self.index.d} dimensions, "
                f"received {query_embedding.shape[1]}"
            )

        k = min(k, self.count)

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            k,
        )

        return distances[0], indices[0]

    @property
    def count(self) -> int:
        """Return the number of vectors stored in the index."""

        return self.index.ntotal
