import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "./data/documents"
PERSIST_DIRECTORY = "./data/vectorstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
TOP_K = 3

# Hybrid search weights: BM25 (keyword) + Vector (semantic)
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


# -----------------------------
# Initialize Local Vector DB
# -----------------------------

def initialize_local_vector_db():
    """
    Builds and persists a local Chroma vector database.
    Safe for production and prevents unnecessary re-indexing.
    """

    print("--- LOG: Initializing Knowledge Base (Local RAG) ---")

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # If DB already exists, skip rebuild
    if os.listdir(PERSIST_DIRECTORY):
        print("--- INFO: Existing vector database detected. Skipping re-index. ---")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=HuggingFaceEmbeddings(model_name=MODEL_NAME)
        )

    # -----------------------------
    # Load PDFs
    # -----------------------------
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"--- Loaded: {file} ---")
            except Exception as e:
                print(f"--- WARNING: Failed to load {file}: {e} ---")

    if not documents:
        print("--- ERROR: No valid PDF documents found to index. ---")
        return None

    # -----------------------------
    # Split Documents
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    splits = splitter.split_documents(documents)

    # -----------------------------
    # Embeddings (Local Model)
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # -----------------------------
    # Create & Persist Vector DB
    # -----------------------------
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"--- SUCCESS: Indexed {len(splits)} chunks ---")
    print(f"--- Vector DB persisted at: {PERSIST_DIRECTORY} ---")

    return vector_db


# -----------------------------
# Retriever Access
# -----------------------------

def get_retriever():
    """
    Loads existing persisted Chroma index.
    Raises error if DB not initialized.
    """

    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        raise RuntimeError(
            "Vector database not initialized. Run initialize_local_vector_db() first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    return vector_db.as_retriever(search_kwargs={"k": TOP_K})


# -----------------------------
# Hybrid Retriever (BM25 + Vector)
# -----------------------------

def get_hybrid_retriever(
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
):
    """
    EnsembleRetriever that fuses BM25 keyword search with Chroma semantic search.

    BM25 excels at exact-term matches (names, codes, technical jargon).
    Vector search excels at semantic similarity (paraphrasing, concepts).
    Combining both covers gaps each method has alone.

    Falls back to pure vector retriever if rank-bm25 is not installed or the
    corpus is empty.
    """
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        raise RuntimeError(
            "Vector database not initialized. Run initialize_local_vector_db() first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})

    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers.ensemble import EnsembleRetriever
        from langchain_core.documents import Document

        # Rebuild corpus from the persisted Chroma collection
        chroma_data = vector_db.get()
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(
                chroma_data.get("documents", []),
                chroma_data.get("metadatas", []),
            )
            if text
        ]

        if not docs:
            print("--- Hybrid: corpus empty, using pure vector retriever ---")
            return vector_retriever

        bm25_retriever = BM25Retriever.from_documents(docs, k=TOP_K)

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight],
        )

    except ImportError:
        print("--- rank-bm25 not installed, falling back to vector-only retriever ---")
        return vector_retriever


# -----------------------------
# CLI Entry
# -----------------------------

if __name__ == "__main__":
    initialize_local_vector_db()