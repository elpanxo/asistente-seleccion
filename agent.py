"""
agent.py — Agente funcional de selección de personal.

Integra:
  - Herramientas de consulta, escritura y razonamiento (IE1, IE2)
  - Memoria de corto plazo (conversación activa) (IE3)
  - Memoria de largo plazo (persistencia entre sesiones) (IE3)
  - Planificador de tareas con prioridades (IE5)
  - Toma de decisiones adaptativas según contexto (IE6)

Arquitectura: LangGraph ReAct Agent (compatible con LangChain >= 1.0)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parent))

from config import GITHUB_TOKEN, GITHUB_ENDPOINT, LLM_MODEL, LLM_TEMPERATURE
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.planning.planner import RecruitmentPlanner

# ── Estado global ─────────────────────────────────────────────────────────────
_vectorstore     = None
_job_title       = "Desarrollador/a Backend Senior"
_job_description = "Buscamos un profesional con experiencia en Python, APIs REST y trabajo en equipo ágil."
_short_memory    = ShortTermMemory(k=10)
_long_memory     = LongTermMemory()
_planner         = RecruitmentPlanner()


def init_agent(vectorstore, job_title: str, job_description: str) -> None:
    """Inicializa el contexto global del agente."""
    global _vectorstore, _job_title, _job_description
    _vectorstore     = vectorstore
    _job_title       = job_title
    _job_description = job_description
    _long_memory.register_process(job_title)


# ── Herramientas ──────────────────────────────────────────────────────────────

@tool
def buscar_candidatos(query: str) -> str:
    """
    Herramienta de CONSULTA: busca información sobre candidatos
    en la base de datos interna usando búsqueda semántica RAG.
    Úsala para responder preguntas generales sobre los candidatos.
    """
    from src.evaluation.rag_pipeline import query_candidates
    return query_candidates(_vectorstore, query)


@tool
def evaluar_candidato(candidate_id: str) -> str:
    """
    Herramienta de RAZONAMIENTO: genera evaluación completa de un candidato
    con puntuaciones por criterio y justificación detallada.
    El parámetro candidate_id debe ser el identificador exacto, ej: ana_lopez.
    """
    from src.evaluation.rag_pipeline import evaluate_candidate
    result = evaluate_candidate(_vectorstore, candidate_id, _job_title, _job_description)
    if not result.get("parse_error"):
        _long_memory.record_decision(
            job_title=_job_title,
            candidate_id=candidate_id,
            recommendation=result.get("recomendacion", "desconocida"),
            score=result.get("puntuacion_ponderada_validada",
                             result.get("puntuacion_ponderada", 0)),
            notes=result.get("justificacion_etica", ""),
        )
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def rankear_candidatos(candidate_ids_str: str) -> str:
    """
    Herramienta de ESCRITURA: compara y rankea múltiples candidatos.
    El parámetro candidate_ids_str debe ser IDs separados por coma.
    Ejemplo: ana_lopez,roberto_sanchez,camila_torres
    """
    from src.evaluation.rag_pipeline import rank_candidates
    ids = [cid.strip() for cid in candidate_ids_str.split(",")]
    result = rank_candidates(_vectorstore, ids, _job_title, _job_description)
    for ev in result.get("evaluaciones_detalladas", []):
        if not ev.get("parse_error"):
            _long_memory.record_decision(
                job_title=_job_title,
                candidate_id=ev.get("candidate_id", "desconocido"),
                recommendation=ev.get("recomendacion", "desconocida"),
                score=ev.get("puntuacion_ponderada_validada",
                             ev.get("puntuacion_ponderada", 0)),
            )
    return json.dumps(result["ranking"], ensure_ascii=False, indent=2)


@tool
def consultar_historial(candidate_id: str) -> str:
    """
    Herramienta de MEMORIA: recupera evaluaciones previas de un candidato
    desde la memoria de largo plazo entre sesiones.
    """
    history = _long_memory.get_candidate_history(candidate_id)
    if not history:
        return f"No hay evaluaciones previas registradas para '{candidate_id}'."
    return json.dumps(history, ensure_ascii=False, indent=2)


@tool
def resumen_proceso(job_title: str) -> str:
    """
    Herramienta de ESCRITURA: genera resumen del proceso de selección
    con los mejores candidatos identificados hasta ahora.
    """
    process = _long_memory.get_process(job_title)
    if not process:
        return f"No hay datos registrados para el proceso '{job_title}'."
    top = _long_memory.get_top_candidates(job_title, n=3)
    summary = {
        "cargo": job_title,
        "candidatos_evaluados": len(process["candidates_evaluated"]),
        "top_3_candidatos": top,
        "notas_reclutador": process.get("recruiter_notes", "Sin notas"),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ── LLM ───────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GITHUB_TOKEN,
        base_url=GITHUB_ENDPOINT,
    )


# ── Ejecución del agente ──────────────────────────────────────────────────────

def run_agent(query: str, available_candidates: List[str]) -> Dict[str, Any]:
    """
    Ejecuta el agente con planificación previa y memoria integrada.

    Flujo:
      1. Planificador analiza la consulta y genera un plan
      2. Si hay memoria de largo plazo relevante, se agrega al contexto
      3. El agente ReAct ejecuta con historial de conversación
      4. La respuesta se guarda en memoria de corto plazo
    """
    # 1. Planificar
    plan = _planner.analyze(query, available_candidates, _job_title)
    plan_display = _planner.format_plan_for_display(plan)

    # 2. Enriquecer con memoria de largo plazo si aplica
    enriched_query = query
    memory_used = False
    if plan.use_long_term:
        process_data = _long_memory.get_process(_job_title)
        if process_data and process_data["decisions"]:
            top = _long_memory.get_top_candidates(_job_title, 3)
            enriched_query = (
                query +
                f"\n\n[CONTEXTO DE MEMORIA]: Ya se han evaluado "
                f"{len(process_data['candidates_evaluated'])} candidatos. "
                f"Top actuales: {json.dumps(top, ensure_ascii=False)}"
            )
            memory_used = True

    # 3. Construir historial para el agente
    chat_history = _short_memory.get_history()

    # 4. Crear y ejecutar agente ReAct con LangGraph
    tools = [
        buscar_candidatos,
        evaluar_candidato,
        rankear_candidatos,
        consultar_historial,
        resumen_proceso,
    ]

    SYSTEM_PROMPT = """Eres un agente especializado en selección de personal.
Tienes acceso a herramientas para buscar, evaluar y rankear candidatos.
Usa las herramientas necesarias en el orden correcto para responder.
Siempre justifica tus decisiones basándote en evidencia de los documentos.
Nunca discrimines por género, edad, etnia o religión."""

    agent = create_react_agent(
        model=_get_llm(),
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    # Construir mensajes con historial
    messages = []
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": enriched_query})

    result = agent.invoke({"messages": messages})

    # Extraer respuesta y pasos intermedios
    output = result["messages"][-1].content
    steps = [
        (msg.name, msg.content)
        for msg in result["messages"]
        if hasattr(msg, "name") and msg.name
    ]

    # 5. Guardar en memoria de corto plazo
    _short_memory.add_interaction(query, output)

    return {
        "output":       output,
        "plan":         plan_display,
        "steps":        steps,
        "memory_used":  memory_used,
        "short_memory": _short_memory.message_count,
    }


# ── Accesores de memoria ──────────────────────────────────────────────────────

def get_short_memory() -> ShortTermMemory:
    return _short_memory

def get_long_memory() -> LongTermMemory:
    return _long_memory

def clear_short_memory() -> None:
    _short_memory.clear()