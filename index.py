from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_document
from src.ingestion.loader import load_documents
from src.vectorstore.faiss_store import FaissStore
from src.retrieval.retriever import Retriever
from src.generation.generator import (
    OllamaGenerator,
    build_context,
    collect_sources,
)

if __name__ == "__main__":
    documents = load_documents("data/knowledge")

    all_chunks = []

    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size=30, overlap=5))

    embedder = Embedder()
    embeddings = embedder.embed_documents(all_chunks)

    store = FaissStore(dimension=embeddings.shape[1])
    store.add(embeddings)

    print(f"Chunks indexed: {len(all_chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"FAISS vectors stored: {store.count}")


    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        chunks=all_chunks,
    )

    question = "How do Playwright fixtures help manage test dependencies?"
    results = retriever.retrieve(question, k=3)

    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks:")

    for result in results:
        print(f"\nID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text']}")

    context = build_context(results)

    generator = OllamaGenerator()

    answer = generator.generate(
        question=question,
        context=context,
    )

    print("\nGrounded answer:")
    print("-" * 50)
    print(answer)

    sources = collect_sources(results)

    print("\nSources:")
    print("-" * 50)

    for source in sources:
        print(f"- {source['source']} ({source['section']})")