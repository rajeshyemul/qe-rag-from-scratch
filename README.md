# QE Knowledge Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/RAG-From%20Scratch-0F766E)](#architecture)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-FAISS-1565C0)](https://github.com/facebookresearch/faiss)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20Local-111111?logo=ollama&logoColor=white)](https://ollama.com/)
[![Status](https://img.shields.io/badge/Status-Part%201%20Complete-2E7D32)](#part-1-completion-criteria)

A small, standalone Retrieval-Augmented Generation (RAG) system for quality-engineering documentation. It is designed to make the RAG mechanics visible: documents are loaded, chunked, embedded, indexed with FAISS, retrieved for a question, and supplied as evidence to a local Ollama model.

This is Part 1 of a larger learning path. It intentionally does not include agents, LangChain, MCP, hybrid search, reranking, or a production vector database.

## What Problem Does RAG Solve?

Language models can answer from their training data, but they do not know this project's QE documentation. RAG retrieves relevant evidence at question time and adds it to the model's context.

RAG does not permanently teach the model the documents. The knowledge remains in the document set and is retrieved only when it is relevant to a question.

## Architecture

The system has two distinct pipelines.

```text
OFFLINE / INDEXING

Markdown documents
        |
        v
Document loader
        |
        v
Chunker + metadata
        |
        v
Sentence Transformer embeddings
        |
        v
FAISS vector index


ONLINE / QUERY

User question
        |
        v
Query embedding
        |
        v
FAISS similarity search
        |
        v
Top-K chunks with metadata
        |
        v
Context builder
        |
        v
Local Ollama model
        |
        v
Grounded answer + verified sources
```

The vector store performs retrieval. Ollama performs generation over the retrieved context. Those are separate responsibilities.

## Data Flow

Each stage changes the shape of the data in a deliberate way:

```text
Markdown file
  -> {source, text}
  -> {id, source, section, chunk_index, text}
  -> embedding vector with 384 dimensions
  -> FAISS vector position
  -> retrieved chunk records with L2 distance
  -> source-labelled context string
  -> generated answer and deterministic source list
```

FAISS stores vectors only. The application keeps chunk text and metadata in a Python list. A vector's FAISS position must remain aligned with the chunk at the same list position.

## Project Structure

```text
.
|-- data/
|   `-- knowledge/
|       |-- api-testing.md
|       |-- ci-cd.md
|       `-- playwright.md
|-- src/
|   |-- embeddings/
|   |   `-- embedder.py       # Sentence Transformer wrapper
|   |-- generation/
|   |   `-- generator.py      # Context, sources, and Ollama generation
|   |-- ingestion/
|   |   |-- chunker.py        # Overlapping word-based chunks and metadata
|   |   `-- loader.py         # Markdown document loading
|   |-- retrieval/
|   |   `-- retriever.py      # Question-to-chunk retrieval workflow
|   `-- vectorstore/
|       `-- faiss_store.py    # FAISS IndexFlatL2 wrapper
|-- index.py                  # End-to-end demonstration pipeline
|-- query.py                  # Reserved for a future query CLI
|-- requirements.txt
`-- README.md
```

## Components

| Component | Responsibility |
| --- | --- |
| `loader.py` | Reads Markdown files into `{source, text}` records. |
| `chunker.py` | Splits documents into overlapping word chunks and adds stable IDs and source metadata. |
| `embedder.py` | Uses `sentence-transformers/all-MiniLM-L6-v2` to create document and query embeddings. |
| `faiss_store.py` | Stores `float32` vectors in an exact L2-distance FAISS index and returns nearest positions. |
| `retriever.py` | Embeds a question, searches FAISS, and reconnects positions to complete chunk records. |
| `generator.py` | Builds labelled context, calls a local Ollama model, and derives sources from retrieved chunks. |
| `index.py` | Wires the components together to demonstrate the entire RAG flow. |

## Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running locally
- A pulled Ollama model. The default configuration uses `llama3.2:latest`.

## Setup

Create and activate a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Pull the default local model if it is not already available:

```bash
ollama pull llama3.2
```

Confirm that Ollama can see the model:

```bash
ollama list
```

If you use the Ollama desktop application, its local service normally starts with the application. Otherwise, start the service before running the project:

```bash
ollama serve
```

## Run the Demonstration

```bash
./.venv/bin/python index.py
```

The first run downloads the Sentence Transformer model if it is not cached locally. The demonstration then prints:

1. The number of chunks and embedding dimensions.
2. The user question.
3. The top three retrieved chunks and their L2 distances.
4. The Ollama-generated answer.
5. A deterministic list of source documents used for retrieval.

The default question is in `index.py`:

```python
question = "How do Playwright fixtures help manage test dependencies?"
```

Change this string and run the command again to experiment with the supplied knowledge base.

## Key Design Decisions

### Simple, inspectable stack

This project uses direct library calls rather than a framework abstraction:

```text
Python + Sentence Transformers + FAISS + Ollama
```

The aim is to understand the mechanics before introducing higher-level RAG or agent frameworks.

### Chunking and metadata

The chunker defaults to 500 words with 50 words of overlap. Every chunk keeps:

```text
id, text, source, section, chunk_index
```

Overlap preserves context when an important idea crosses a chunk boundary. Metadata enables the displayed source attribution and provides a basis for future filtering by document type, version, team, or date.

### Embeddings and similarity search

`all-MiniLM-L6-v2` creates 384-dimensional semantic vectors. FAISS `IndexFlatL2` performs exact nearest-neighbour search using L2 distance, where smaller values indicate closer vectors.

`IndexFlatL2` is intentionally not optimized for very large collections. It is an excellent learning tool because its behaviour is straightforward: every query is compared with every indexed vector.

### Grounding and sources

The Ollama system prompt instructs the model to use only retrieved context. The application derives sources from the retrieved chunks rather than trusting the model to cite documents accurately.

This is an important distinction: source attribution proves which documents were retrieved, but it does not mathematically guarantee that every generated sentence is supported by the evidence. Stronger evidence validation is a future enhancement.

## Intentional Limitations

Part 1 stays small on purpose. It currently has these limitations:

- The FAISS index lives only in memory and is rebuilt for each run.
- `index.py` contains a fixed demonstration question; `query.py` is not yet a user-facing query command.
- Chunking is word-based rather than token-aware or structure-aware.
- Retrieval uses dense vectors and exact L2 search only; there is no BM25, hybrid retrieval, metadata filtering, or reranking.
- The sample knowledge base is intentionally small.
- The LLM can still make unsupported inferences despite prompt instructions.
- There are no automated tests, persistence, observability, authentication, or evaluation harnesses yet.

These are deliberate boundaries, not hidden production claims.

## Part 1 Completion Criteria

Part 1 is complete. The project demonstrates a full, end-to-end RAG loop:

```text
QE Markdown knowledge
  -> document loading
  -> chunking and metadata
  -> embeddings
  -> FAISS indexing
  -> query embedding
  -> top-K retrieval
  -> labelled context
  -> local LLM generation
  -> source attribution
```

You should now be able to explain the following on a whiteboard:

- Why RAG retrieves evidence rather than retraining an LLM.
- Why indexing and querying are different pipelines.
- Why chunks and metadata matter.
- How semantic embeddings make similarity search possible.
- Why FAISS returns vector positions, not application documents.
- How retrieved context is augmented into an LLM prompt.
- Why retrieval quality and evidence boundaries determine answer quality.

## Out of Scope for Part 1

The following belong to a future Part 2, where RAG is connected to an Agentic QE architecture:

- Atlas orchestration and agents
- RAG as an agent tool
- Jira, framework, test-case, and failure-history knowledge sources
- Query rewriting, reranking, and hybrid retrieval
- Evidence validation and RAG evaluation
- MCP integration
- Persistent vector stores and production operations

## Technology Stack

- Python 3
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- `sentence-transformers/all-MiniLM-L6-v2`
- `llama3.2:latest` by default, configurable in `OllamaGenerator`
