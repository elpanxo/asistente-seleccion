"""
agent.py — Agente funcional de selección de personal.

Basado en el material del curso RA2 (IL2.1, IL2.2, IL2.3).

Integra:
  - Herramientas de consulta, escritura y razonamiento (IE1, IE2)
  - Memoria de corto plazo con ConversationBufferWindowMemory (IE3)
  - Memoria de largo plazo persistida en JSON (IE3)
  - Planificador jerárquico reactivo (IE5)
  - Toma de decisiones adaptativas ReAct (IE6)

Stack: LangChain 0.2.x + AgentExecutor + create_openai_tools_agent
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain import hub
from langchain.agents import tool, create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent))

from config import GITHUB_TOKEN, GITHUB_ENDPOINT, LLM_MODEL, LLM_TEMPERATURE
from src.memory.long_term import LongTermMemory
from src.planning.planner import RecruitmentPlanner

# ── Estado global ─────────────────────────────────────────────────────────────
_vectorstore     = None
_job_title       = "Desarrollador/a Backend Senior"
_job_description = "Buscamos un profesional con experiencia en Python, APIs REST y trabajo en equipo ágil."
_long_memory     = LongTermMemory()
_planner         = RecruitmentPlanner()

# Memoria de corto plazo — ConversationBufferWindowMemory (IL2.2)
# Guarda los últimos 10 intercambios, igual que el notebook 2-memory-agent-advanced
_short_memory = ConversationBufferWindowMemory(
    k=10,
    memory_key="chat_history",
    return_messages=True,
)
_interaction_count = 0


def init_agent(vectorstore, job_title: str, job_description: str) -> None:
    """Inicializa el contexto global del agente."""
    global _vectorstore, _job_title, _job_description
    _vectorstore     = vectorstore
    _job_title       = job_title
    _job_description = job_description
    _long_memory.register_process(job_title)


# ── Herramientas (IL2.1) ──────────────────────────────────────────────────────

@tool
def buscar_candidatos(query: str) -> str:
    """
    Herramienta de CONSULTA: busca información sobre candidatos
    en la base de datos interna usando búsqueda semántica RAG.
    Úsala para responder preguntas generales sobre los candidatos.
    El parámetro query debe ser la pregunta en lenguaje natural.
    """
    from src.evaluation.rag_pipeline import query_candidates
    return query_candidates(_vectorstore, query)


@tool
def evaluar_candidato(candidate_id: str) -> str:
    """
    Herramienta de RAZONAMIENTO: genera evaluación completa de un candidato
    con puntuaciones por criterio, fortalezas y justificación.
    El parámetro candidate_id debe ser el identificador exacto del candidato.
    Ejemplo: ana_lopez, roberto_sanchez, camila_torres.
    """
    from src.evaluation.rag_pipeline import evaluate_candidate
    result = evaluate_candidate(
        _vectorstore, candidate_id, _job_title, _job_description
    )
    # Guardar en memoria de largo plazo automáticamente
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
    Herramienta de ESCRITURA: compara y rankea múltiples candidatos
    generando un informe comparativo con posiciones justificadas.
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
    Herramienta de MEMORIA LARGO PLAZO: recupera evaluaciones previas
    de un candidato guardadas entre sesiones.
    Úsala cuando el reclutador pregunte si un candidato ya fue evaluado.
    El parámetro candidate_id debe ser el identificador del candidato.
    """
    history = _long_memory.get_candidate_history(candidate_id)
    if not history:
        return f"No hay evaluaciones previas registradas para '{candidate_id}'."
    return json.dumps(history, ensure_ascii=False, indent=2)


@tool
def resumen_proceso(job_title: str) -> str:
    """
    Herramienta de ESCRITURA: genera resumen del estado actual del
    proceso de selección con los mejores candidatos hasta ahora.
    El parámetro job_title debe ser el nombre exacto del cargo.
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


# ── Construcción del agente (IL2.1 + IL2.2) ──────────────────────────────────

def _build_agent_executor() -> AgentExecutor:
    """
    Construye el AgentExecutor siguiendo el patrón del curso RA2.

    Igual que en 3-langchain-agent.ipynb y 2-memory-agent-advanced.ipynb:
      1. Definir LLM
      2. Definir herramientas con @tool
      3. Descargar prompt de LangChain Hub
      4. create_openai_tools_agent(llm, tools, prompt)
      5. AgentExecutor(agent, tools, verbose=True)
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_base=GITHUB_ENDPOINT,
        openai_api_key=GITHUB_TOKEN,
    )

    tools = [
        buscar_candidatos,
        evaluar_candidato,
        rankear_candidatos,
        consultar_historial,
        resumen_proceso,
    ]

    # Prompt del hub — mismo que usa el curso en todos los notebooks
    prompt = hub.pull("hwchase17/openai-tools-agent")

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ── Función principal (IL2.2 + IL2.3) ────────────────────────────────────────

def run_agent(query: str, available_candidates: List[str]) -> Dict[str, Any]:
    """
    Ejecuta el agente con planificación y memoria integradas.

    Flujo (basado en 2-memory-agent-advanced.ipynb):
      1. Planificador analiza la consulta → plan de ejecución
      2. Cargar historial desde ConversationBufferWindowMemory
      3. Enriquecer consulta con memoria de largo plazo si aplica
      4. Invocar AgentExecutor con chat_history
      5. Guardar respuesta en memoria de corto plazo
    """
    global _interaction_count

    # 1. Planificar (IL2.3)
    plan = _planner.analyze(query, available_candidates, _job_title)
    plan_display = _planner.format_plan_for_display(plan)

    # 2. Cargar historial de corto plazo (IL2.2)
    chat_history = _short_memory.load_memory_variables({})["chat_history"]

    # 3. Enriquecer con memoria de largo plazo si aplica (IL2.2)
    enriched_query = query
    memory_used = False
    if plan.use_long_term:
        process_data = _long_memory.get_process(_job_title)
        if process_data and process_data["decisions"]:
            top = _long_memory.get_top_candidates(_job_title, 3)
            enriched_query = (
                query +
                f"\n\n[CONTEXTO DE MEMORIA LARGO PLAZO]: "
                f"Ya se han evaluado {len(process_data['candidates_evaluated'])} "
                f"candidatos en este proceso. "
                f"Top actuales: {json.dumps(top, ensure_ascii=False)}"
            )
            memory_used = True

    # 4. Ejecutar agente
    agent_executor = _build_agent_executor()
    result = agent_executor.invoke({
        "input": enriched_query,
        "chat_history": chat_history,
    })

    output = result.get("output", "")
    steps  = result.get("intermediate_steps", [])

    # 5. Guardar en memoria de corto plazo (IL2.2)
    _short_memory.save_context(
        {"input": query},
        {"output": output},
    )
    _interaction_count += 1

    return {
        "output":       output,
        "plan":         plan_display,
        "steps":        steps,
        "memory_used":  memory_used,
        "short_memory": _interaction_count,
    }


# ── Accesores para la UI ──────────────────────────────────────────────────────

def get_short_memory_count() -> int:
    return _interaction_count

def get_long_memory() -> LongTermMemory:
    return _long_memory

def clear_short_memory() -> None:
    global _interaction_count
    _short_memory.clear()
    _interaction_count = 0