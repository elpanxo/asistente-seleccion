"""
src/observability/metrics.py — Observabilidad y métricas del agente RecruitAI.

IL3.1: Herramientas de Observabilidad y Métricas

Implementa:
  - Logging estructurado de todas las operaciones
  - Medición de tiempo de respuesta por operación
  - Contador de interacciones por tipo
  - Métricas de rendimiento exportables
  - Dashboard de métricas para Streamlit

Basado en el patrón del curso RA3/IL3.1/1-observability_tools.py
"""

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BASE_DIR, AUDIT_LOG_FILE

# ── Logging estructurado (IL3.1) ──────────────────────────────────────────────
LOG_FILE = BASE_DIR / "docs" / "agent_metrics.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("recruitai.metrics")


# ── Clase principal de métricas (IL3.1) ───────────────────────────────────────

class AgentMetrics:
    """
    Sistema de observabilidad del agente RecruitAI.

    Registra y expone métricas de rendimiento:
      - Tiempo de respuesta por operación
      - Contador de interacciones por tipo
      - Tasa de errores
      - Herramientas más usadas
      - Historial de métricas en sesión

    Patrón del curso:
        agent.counter += 1
        duration = time.time() - start
        logging.info(f"Duración: {duration:.4f}s")
    """

    def __init__(self):
        self.interaction_counter = 0
        self.error_counter       = 0
        self.response_times: List[float] = []
        self.tool_usage: Dict[str, int] = defaultdict(int)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.session_start = datetime.utcnow()
        self._history: List[Dict] = []

    def record_interaction(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        tool_used: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> None:
        """
        Registra una interacción del agente con sus métricas.
        Equivalente al patrón del curso:
            self.counter += 1
            logging.info(f"Duración: {duration:.4f}s")
        """
        self.interaction_counter += 1
        self.response_times.append(duration)
        self.operation_counts[operation] += 1

        if tool_used:
            self.tool_usage[tool_used] += 1
        if not success:
            self.error_counter += 1

        entry = {
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "operation":    operation,
            "duration_ms":  round(duration * 1000, 2),
            "success":      success,
            "tool_used":    tool_used,
            "candidate_id": candidate_id,
            "interaction_n": self.interaction_counter,
        }
        self._history.append(entry)

        # Log estructurado
        level = logging.INFO if success else logging.ERROR
        logger.log(level,
            f"[{operation}] duration={duration*1000:.1f}ms "
            f"tool={tool_used or '-'} "
            f"candidate={candidate_id or '-'} "
            f"success={success} "
            f"total_interactions={self.interaction_counter}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen de métricas de la sesión actual."""
        times = self.response_times
        uptime = (datetime.utcnow() - self.session_start).seconds

        return {
            "total_interactions":   self.interaction_counter,
            "total_errors":         self.error_counter,
            "error_rate_pct":       round(
                (self.error_counter / self.interaction_counter * 100)
                if self.interaction_counter > 0 else 0, 1
            ),
            "avg_response_ms":      round(
                (sum(times) / len(times) * 1000) if times else 0, 1
            ),
            "min_response_ms":      round(min(times) * 1000, 1) if times else 0,
            "max_response_ms":      round(max(times) * 1000, 1) if times else 0,
            "session_uptime_sec":   uptime,
            "operations":           dict(self.operation_counts),
            "tool_usage":           dict(self.tool_usage),
        }

    def get_history(self, last_n: int = 20) -> List[Dict]:
        """Retorna las últimas n interacciones registradas."""
        return self._history[-last_n:]

    def get_top_tools(self, n: int = 5) -> List[Dict]:
        """Retorna las herramientas más usadas."""
        sorted_tools = sorted(
            self.tool_usage.items(), key=lambda x: x[1], reverse=True
        )
        return [{"tool": t, "count": c} for t, c in sorted_tools[:n]]

    def reset(self) -> None:
        """Reinicia las métricas de sesión."""
        self.__init__()
        logger.info("Métricas de sesión reiniciadas.")


# ── Decorador de medición (IL3.1) ─────────────────────────────────────────────

def measure_time(operation: str, metrics_instance: "AgentMetrics"):
    """
    Decorador que mide el tiempo de ejecución de una función
    y registra la métrica automáticamente.

    Uso:
        @measure_time("evaluate_candidate", metrics)
        def evaluate_candidate(...):
            ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"[{operation}] Error: {e}")
                raise
            finally:
                duration = time.time() - start
                metrics_instance.record_interaction(
                    operation=operation,
                    duration=duration,
                    success=success,
                )
        return wrapper
    return decorator


# ── Lector del audit_log (IL3.2 base) ────────────────────────────────────────

def analyze_audit_log() -> Dict[str, Any]:
    """
    Analiza el audit_log.jsonl existente para extraer métricas históricas.
    Cubre IL3.2: análisis de trazabilidad y logs.
    """
    path = Path(AUDIT_LOG_FILE)
    if not path.exists():
        return {"error": "No hay audit log disponible aún."}

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not events:
        return {"total_events": 0}

    # Conteo por tipo de evento
    by_type: Dict[str, int] = defaultdict(int)
    for e in events:
        by_type[e.get("event_type", "unknown")] += 1

    # Candidatos más evaluados
    candidate_counts: Dict[str, int] = defaultdict(int)
    for e in events:
        cid = e.get("candidate_id")
        if cid:
            candidate_counts[cid] += 1

    # Recomendaciones dadas
    recommendations: Dict[str, int] = defaultdict(int)
    for e in events:
        rec = e.get("recomendacion")
        if rec:
            recommendations[rec] += 1

    # Primero y último evento
    timestamps = [e.get("timestamp", "") for e in events if e.get("timestamp")]
    first = min(timestamps) if timestamps else None
    last  = max(timestamps) if timestamps else None

    return {
        "total_events":           len(events),
        "by_event_type":          dict(by_type),
        "candidates_evaluated":   dict(candidate_counts),
        "recommendations_given":  dict(recommendations),
        "first_event":            first,
        "last_event":             last,
        "recent_events":          events[-10:],
    }


# ── Instancia global de métricas ──────────────────────────────────────────────
# Se importa desde app.py y agent.py para compartir estado en la sesión
session_metrics = AgentMetrics()