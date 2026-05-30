"""
src/planning/planner.py — Planificación y toma de decisiones del agente.

El planificador analiza la consulta del reclutador y decide:
  1. Qué herramientas necesita usar y en qué orden (IE5)
  2. Cómo adaptar el comportamiento según el contexto (IE6)
  3. Qué información priorizar de la memoria de largo plazo

Estrategia de planificación: ReAct (Reason + Act)
  - El agente razona sobre qué herramienta usar
  - Ejecuta la herramienta
  - Observa el resultado
  - Decide si necesita más herramientas o puede responder

Ubicación: asistente-seleccion/src/planning/planner.py
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TaskType(Enum):
    """Tipos de tarea que puede resolver el agente."""
    SEARCH        = "buscar_informacion"
    EVALUATE      = "evaluar_candidato"
    RANK          = "rankear_candidatos"
    COMPARE       = "comparar_candidatos"
    SUMMARIZE     = "resumir_proceso"
    UNKNOWN       = "desconocido"


class Priority(Enum):
    """Niveles de prioridad para la ejecución de tareas."""
    HIGH   = 1   # Requiere respuesta inmediata
    MEDIUM = 2   # Flujo normal
    LOW    = 3   # Puede diferirse


@dataclass
class TaskPlan:
    """
    Representa el plan de ejecución para una consulta.

    Atributos:
        task_type:     tipo de tarea detectada
        priority:      nivel de prioridad asignado
        tools_needed:  herramientas a usar en orden
        candidate_ids: candidatos identificados en la consulta
        reasoning:     justificación del plan
        use_long_term: si debe consultar memoria de largo plazo
    """
    task_type:     TaskType
    priority:      Priority
    tools_needed:  List[str]
    candidate_ids: List[str]
    reasoning:     str
    use_long_term: bool = False


class RecruitmentPlanner:
    """
    Planificador de tareas para el agente de selección de personal.

    Analiza la intención del reclutador y genera un plan de ejecución
    con las herramientas necesarias en el orden correcto.
    """

    # Palabras clave por tipo de tarea
    _KEYWORDS = {
        TaskType.EVALUATE: [
            "evalúa", "evalua", "evaluación", "evaluacion", "puntaje",
            "puntuación", "califica", "analiza", "análisis", "score",
        ],
        TaskType.RANK: [
            "rankea", "ranking", "ordena", "clasifica", "compara todos",
            "mejor candidato", "quién es mejor", "top",
        ],
        TaskType.COMPARE: [
            "compara", "diferencia", "versus", "vs", "entre",
            "cuál es mejor", "cual es mejor",
        ],
        TaskType.SUMMARIZE: [
            "resumen", "resume", "qué se ha hecho", "que se ha hecho",
            "estado del proceso", "cuántos candidatos", "cuantos",
        ],
        TaskType.SEARCH: [
            "busca", "encuentra", "quién", "quien", "qué candidatos",
            "que candidatos", "tiene experiencia", "conoce", "sabe",
            "habilidades", "trabaj", "estudió", "estudió",
        ],
    }

    def analyze(
        self,
        query: str,
        available_candidates: List[str],
        job_title: str,
    ) -> TaskPlan:
        """
        Analiza la consulta y genera un plan de ejecución.

        Proceso de razonamiento:
        1. Detectar tipo de tarea por palabras clave
        2. Identificar candidatos mencionados
        3. Determinar herramientas necesarias en orden
        4. Asignar prioridad según urgencia
        5. Decidir si consultar memoria de largo plazo
        """
        query_lower = query.lower()
        task_type   = self._detect_task_type(query_lower)
        candidates  = self._extract_candidates(query_lower, available_candidates)
        tools       = self._select_tools(task_type, candidates, available_candidates)
        priority    = self._assign_priority(task_type, query_lower)
        use_lt      = self._needs_long_term(task_type, query_lower)
        reasoning   = self._build_reasoning(task_type, candidates, tools, job_title)

        return TaskPlan(
            task_type=task_type,
            priority=priority,
            tools_needed=tools,
            candidate_ids=candidates,
            reasoning=reasoning,
            use_long_term=use_lt,
        )

    def _detect_task_type(self, query: str) -> TaskType:
        """Detecta el tipo de tarea por coincidencia de palabras clave."""
        scores = {task: 0 for task in TaskType}
        for task_type, keywords in self._KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    scores[task_type] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else TaskType.SEARCH

    def _extract_candidates(
        self, query: str, available: List[str]
    ) -> List[str]:
        """Identifica qué candidatos se mencionan en la consulta."""
        mentioned = []
        for cid in available:
            # Busca el id completo o partes del nombre
            parts = cid.replace("_", " ").lower().split()
            if any(part in query for part in parts) or cid.lower() in query:
                mentioned.append(cid)
        return mentioned

    def _select_tools(
        self,
        task_type: TaskType,
        candidates: List[str],
        all_candidates: List[str],
    ) -> List[str]:
        """
        Selecciona y ordena las herramientas necesarias.
        El orden importa: primero buscar, luego evaluar, luego rankear.
        """
        tools = []

        if task_type == TaskType.SEARCH:
            tools = ["buscar_candidatos"]

        elif task_type == TaskType.EVALUATE:
            if candidates:
                tools = ["buscar_candidatos", "evaluar_candidato"]
            else:
                tools = ["buscar_candidatos"]

        elif task_type in (TaskType.RANK, TaskType.COMPARE):
            target = candidates if candidates else all_candidates
            if len(target) >= 2:
                tools = ["buscar_candidatos", "evaluar_candidato", "rankear_candidatos"]
            else:
                tools = ["buscar_candidatos", "evaluar_candidato"]

        elif task_type == TaskType.SUMMARIZE:
            tools = ["buscar_candidatos"]

        return tools

    def _assign_priority(self, task_type: TaskType, query: str) -> Priority:
        """
        Asigna prioridad según el tipo de tarea y urgencia detectada.
        Palabras como "urgente", "hoy", "ahora" elevan la prioridad.
        """
        urgent_words = ["urgente", "inmediato", "hoy", "ahora", "rápido"]
        if any(w in query for w in urgent_words):
            return Priority.HIGH
        if task_type in (TaskType.RANK, TaskType.EVALUATE):
            return Priority.HIGH
        if task_type == TaskType.SEARCH:
            return Priority.MEDIUM
        return Priority.LOW

    def _needs_long_term(self, task_type: TaskType, query: str) -> bool:
        """Decide si consultar la memoria de largo plazo."""
        lt_keywords = [
            "antes", "anterior", "historia", "historial", "proceso previo",
            "ya fue evaluado", "primera vez", "visto antes"
        ]
        return (
            task_type in (TaskType.EVALUATE, TaskType.RANK)
            or any(kw in query for kw in lt_keywords)
        )

    def _build_reasoning(
        self,
        task_type: TaskType,
        candidates: List[str],
        tools: List[str],
        job_title: str,
    ) -> str:
        """Genera una explicación legible del plan para mostrar en la UI."""
        cand_str = ", ".join(candidates) if candidates else "todos los disponibles"
        tools_str = " → ".join(tools) if tools else "ninguna"
        return (
            f"Tarea detectada: {task_type.value} | "
            f"Candidatos: {cand_str} | "
            f"Cargo: {job_title} | "
            f"Herramientas: {tools_str}"
        )

    def format_plan_for_display(self, plan: TaskPlan) -> dict:
        """Formatea el plan para mostrarlo en la interfaz Streamlit."""
        priority_labels = {
            Priority.HIGH:   "🔴 Alta",
            Priority.MEDIUM: "🟡 Media",
            Priority.LOW:    "🟢 Baja",
        }
        task_labels = {
            TaskType.SEARCH:    "Búsqueda de información",
            TaskType.EVALUATE:  "Evaluación de candidato",
            TaskType.RANK:      "Ranking comparativo",
            TaskType.COMPARE:   "Comparación de candidatos",
            TaskType.SUMMARIZE: "Resumen del proceso",
            TaskType.UNKNOWN:   "Tarea general",
        }
        return {
            "tipo_tarea":      task_labels.get(plan.task_type, "Desconocido"),
            "prioridad":       priority_labels.get(plan.priority, "—"),
            "herramientas":    " → ".join(plan.tools_needed) if plan.tools_needed else "Ninguna",
            "candidatos":      ", ".join(plan.candidate_ids) if plan.candidate_ids else "Todos",
            "memoria_lt":      "Sí" if plan.use_long_term else "No",
            "razonamiento":    plan.reasoning,
        }