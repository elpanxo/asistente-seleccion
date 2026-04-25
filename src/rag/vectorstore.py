import sys
from pathlib import Path
from typing import List, Optional
 
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)
 
 
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
 
 
def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Crea o reconstruye el vectorstore desde una lista de chunks."""
    print(f"[vectorstore] Generando embeddings para {len(chunks)} chunks...")
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
    print(f"[vectorstore] ✓ {len(chunks)} chunks agregados al vectorstore")
 
 
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
    """
    Retorna la lista de candidate_ids únicos indexados en ChromaDB.
    Útil para saber qué candidatos están disponibles sin leer el disco.
    """
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