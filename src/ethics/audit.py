import json
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Agregar raíz del proyecto al path para importar la configuración
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from config import AUDIT_LOG_FILE

def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Registra un evento en el archivo de auditoría JSONL.
    
    Tipos de evento:
      - file_uploaded       → cuando se sube un archivo
      - profile_extraction  → extracción de perfil de candidato
      - candidate_evaluation → evaluación con puntuaciones
      - candidate_ranking   → ranking comparativo
      - qa_query            → consulta en lenguaje natural
    """
    entry = {
        "event_id":  str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        **payload,
    }
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
 
 
def get_audit_log() -> list[Dict]:
    """Retorna todos los registros de auditoría como lista de dicts."""
    path = Path(AUDIT_LOG_FILE)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
 
 
def safe_parse_json(text: str) -> Dict:
    """
    Intenta parsear JSON desde la respuesta del LLM.
    Elimina bloques markdown si el modelo los incluyó.
    """
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"raw_response": text, "parse_error": True}
