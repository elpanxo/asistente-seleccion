"""
src/memory/long_term.py — Memoria de largo plazo del agente.

Persiste decisiones, evaluaciones y contexto relevante entre sesiones
en un archivo JSON en disco. A diferencia de la memoria de corto plazo,
esta información sobrevive al cierre de la aplicación.

Casos de uso:
  - Recordar que un candidato ya fue evaluado en un proceso anterior
  - Mantener el historial de recomendaciones por proceso de selección
  - Guardar preferencias del reclutador entre sesiones

Ubicación: asistente-seleccion/src/memory/long_term.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_DIR

LONG_TERM_FILE = BASE_DIR / "docs" / "long_term_memory.json"


class LongTermMemory:
    """
    Memoria de largo plazo persistida en disco (JSON).

    Estructura del archivo:
    {
      "processes": {
        "<job_title>": {
          "created_at": "...",
          "candidates_evaluated": ["id1", "id2"],
          "decisions": [
            { "candidate_id": "...", "recommendation": "...", "score": ..., "date": "..." }
          ],
          "recruiter_notes": "..."
        }
      },
      "global_context": {
        "total_evaluations": 0,
        "last_active": "..."
      }
    }
    """

    def __init__(self):
        self._data = self._load()

    def _load(self) -> Dict:
        """Carga la memoria desde disco. Si no existe, inicializa vacía."""
        if LONG_TERM_FILE.exists():
            with open(LONG_TERM_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "processes": {},
            "global_context": {
                "total_evaluations": 0,
                "last_active": None,
            }
        }

    def _save(self) -> None:
        """Persiste la memoria en disco."""
        LONG_TERM_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LONG_TERM_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Gestión de procesos de selección ─────────────────────────────────

    def register_process(self, job_title: str) -> None:
        """Registra un nuevo proceso de selección si no existe."""
        if job_title not in self._data["processes"]:
            self._data["processes"][job_title] = {
                "created_at": datetime.utcnow().isoformat(),
                "candidates_evaluated": [],
                "decisions": [],
                "recruiter_notes": "",
            }
            self._save()

    def record_decision(
        self,
        job_title: str,
        candidate_id: str,
        recommendation: str,
        score: float,
        notes: str = "",
    ) -> None:
        """
        Registra la decisión de evaluación de un candidato.
        Si el proceso no existe lo crea automáticamente.
        """
        self.register_process(job_title)
        process = self._data["processes"][job_title]

        # Evitar duplicados — actualizar si ya existe
        existing = next(
            (d for d in process["decisions"] if d["candidate_id"] == candidate_id),
            None
        )
        entry = {
            "candidate_id": candidate_id,
            "recommendation": recommendation,
            "score": score,
            "notes": notes,
            "date": datetime.utcnow().isoformat(),
        }
        if existing:
            process["decisions"].remove(existing)
        process["decisions"].append(entry)

        if candidate_id not in process["candidates_evaluated"]:
            process["candidates_evaluated"].append(candidate_id)

        self._data["global_context"]["total_evaluations"] += 1
        self._data["global_context"]["last_active"] = datetime.utcnow().isoformat()
        self._save()

    def save_recruiter_notes(self, job_title: str, notes: str) -> None:
        """Guarda notas del reclutador para un proceso específico."""
        self.register_process(job_title)
        self._data["processes"][job_title]["recruiter_notes"] = notes
        self._save()

    # ── Consultas ─────────────────────────────────────────────────────────

    def get_process(self, job_title: str) -> Optional[Dict]:
        """Retorna toda la información de un proceso de selección."""
        return self._data["processes"].get(job_title)

    def get_candidate_history(self, candidate_id: str) -> List[Dict]:
        """
        Retorna todas las evaluaciones previas de un candidato
        en cualquier proceso de selección.
        """
        history = []
        for job_title, process in self._data["processes"].items():
            for decision in process["decisions"]:
                if decision["candidate_id"] == candidate_id:
                    history.append({
                        "job_title": job_title,
                        **decision,
                    })
        return history

    def was_already_evaluated(self, job_title: str, candidate_id: str) -> bool:
        """Verifica si un candidato ya fue evaluado en un proceso específico."""
        process = self.get_process(job_title)
        if not process:
            return False
        return candidate_id in process["candidates_evaluated"]

    def get_top_candidates(self, job_title: str, n: int = 3) -> List[Dict]:
        """Retorna los n candidatos con mayor puntaje en un proceso."""
        process = self.get_process(job_title)
        if not process:
            return []
        sorted_decisions = sorted(
            process["decisions"],
            key=lambda d: d.get("score", 0),
            reverse=True,
        )
        return sorted_decisions[:n]

    def get_summary(self) -> Dict:
        """Retorna un resumen del estado de la memoria de largo plazo."""
        return {
            "total_processes": len(self._data["processes"]),
            "total_evaluations": self._data["global_context"]["total_evaluations"],
            "last_active": self._data["global_context"]["last_active"],
            "processes": list(self._data["processes"].keys()),
        }

    def clear_process(self, job_title: str) -> None:
        """Elimina un proceso de selección de la memoria."""
        if job_title in self._data["processes"]:
            del self._data["processes"][job_title]
            self._save()