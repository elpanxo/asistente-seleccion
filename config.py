import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Rutas base
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CVS_DIR = DATA_DIR / "cvs"
EVALUACIONES_DIR = DATA_DIR / "evaluaciones"
DOCS_DIR = BASE_DIR / "docs"

# Crear directorios si no existen
for d in [CVS_DIR, EVALUACIONES_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = 0.0

# ChromaDB
CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION: str = "candidatos"

# Embeddings
EMBEDDING_MODEL: str = "models/embeddings-001"

# Chunking
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100

# Criterios de evaluación ponderados (deben sumar 1.0)
EVALUATION_CRITERIA: dict[str, float] = {
    "experiencia_relevante": 0.30,
    "habilidades_tecnicas": 0.25,
    "formacion_academica": 0.15,
    "proyectos_destacados": 0.15,
    "diversidad_y_inclusion": 0.10,
    "comunicacion_liderazgo": 0.05,
}

# Auditoria
AUDIT_LOG_FILE: str = str(BASE_DIR / "docs" / "audit_log.jsonl")

# Tipos de archivos soportados
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"}