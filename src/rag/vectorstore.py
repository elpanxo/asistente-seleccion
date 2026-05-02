import sys
from pathlib import Path
from typing import List, Optional
 
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)
 
 
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Embeddings locales con sentence-transformers.
    El modelo se descarga automáticamente la primera vez (~90 MB).
    No requiere API key ni conexión después de la descarga inicial.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
 
 
def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Crea o reconstruye el vectorstore desde una lista de chunks."""
    print(f"[vectorstore] Generando embeddings locales para {len(chunks)} chunks...")
    print("[vectorstore] Primera vez: descargando modelo (~90MB)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[vectorstore] ✓ Vectorstore guardado en '{CHROMA_PERSIST_DIR}'")
    return vectorstore
 
 
def load_vectorstore() -> Optional[Chroma]:
    """Carga un vectorstore persistido previamente. Retorna None si no existe."""
    import os
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None
    vs = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[vectorstore] ✓ Vectorstore cargado desde '{CHROMA_PERSIST_DIR}'")
    return vs
 
 
def add_documents(vectorstore: Chroma, chunks: List[Document]) -> None:
    """Agrega nuevos chunks a un vectorstore existente sin reconstruirlo."""
    vectorstore.add_documents(chunks)
    print(f"[vectorstore] ✓ {len(chunks)} chunks agregados")
 
 
def get_retriever(vectorstore: Chroma, k: int = 6):
    """Retriever con búsqueda MMR para diversidad de resultados."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20},
    )
 
 
def filter_by_candidate(
    vectorstore: Chroma,
    candidate_id: str,
    query: str,
    k: int = 6,
) -> List[Document]:
    """Recupera fragmentos filtrados por candidate_id."""
    return vectorstore.similarity_search(
        query=query,
        k=k,
        filter={"candidate_id": candidate_id},
    )
 
 
def get_all_candidate_ids(vectorstore: Chroma) -> List[str]:
    """Retorna la lista de candidate_ids únicos indexados en ChromaDB."""
    try:
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        ids = set()
        for meta in results.get("metadatas", []):
            if meta and "candidate_id" in meta:
                ids.add(meta["candidate_id"])
        return sorted(ids)
    except Exception:
        return []