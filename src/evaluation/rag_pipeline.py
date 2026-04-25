import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
 
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EVALUATION_CRITERIA, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY
from src.ethics.audit import log_event, safe_parse_json
from src.evaluation.prompts import EXTRACTION_PROMPT, QA_PROMPT, RANKING_PROMPT, SCORING_PROMPT
from src.rag.vectorstore import filter_by_candidate, get_retriever
 
 
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )
 
 
# 1. Extracción de perfil 
 
def extract_candidate_profile(vectorstore: Chroma, candidate_id: str) -> Dict:
    """Recupera info del candidato desde ChromaDB y extrae perfil estructurado."""
    docs = filter_by_candidate(
        vectorstore, candidate_id,
        query="experiencia habilidades educación proyectos"
    )
    if not docs:
        return {"error": f"No se encontró información para '{candidate_id}'"}
 
    context = "\n\n---\n\n".join([
        f"[Fuente: {d.metadata.get('source_type', '?')}]\n{d.page_content}"
        for d in docs
    ])
 
    chain = EXTRACTION_PROMPT | _get_llm() | StrOutputParser()
    raw = chain.invoke({"context": context, "candidate_id": candidate_id})
    profile = safe_parse_json(raw)
 
    log_event("profile_extraction", {
        "candidate_id": candidate_id,
        "docs_retrieved": len(docs),
        "sources_used": [d.metadata.get("source_type") for d in docs],
    })
    return profile
 
 
# 2. Evaluación individual
 
def evaluate_candidate(
    vectorstore: Chroma,
    candidate_id: str,
    job_title: str,
    job_description: str,
) -> Dict:
    """Evalúa un candidato con puntuaciones ponderadas por criterio."""
    profile = extract_candidate_profile(vectorstore, candidate_id)
 
    external_docs = filter_by_candidate(
        vectorstore, candidate_id,
        query="github linkedin proyectos open source contribuciones",
        k=4,
    )
    external_context = "\n\n".join([
        f"[{d.metadata.get('source_type')}]: {d.page_content}"
        for d in external_docs
        if d.metadata.get("source_type") in ("linkedin", "github")
    ]) or "No se encontró información de fuentes externas."
 
    chain = SCORING_PROMPT | _get_llm() | StrOutputParser()
    raw = chain.invoke({
        "candidate_id": candidate_id,
        "job_title": job_title,
        "candidate_profile": json.dumps(profile, ensure_ascii=False, indent=2),
        "external_context": external_context,
        "job_description": job_description,
    })
    evaluation = safe_parse_json(raw)
 
    # Validar puntuación ponderada localmente
    if "puntuaciones" in evaluation and not evaluation.get("parse_error"):
        scores = evaluation["puntuaciones"]
        weighted = sum(
            scores.get(c, 0) * w for c, w in EVALUATION_CRITERIA.items()
        )
        evaluation["puntuacion_ponderada_validada"] = round(weighted, 2)
 
    log_event("candidate_evaluation", {
        "candidate_id": candidate_id,
        "job_title": job_title,
        "puntuacion_ponderada": evaluation.get("puntuacion_ponderada"),
        "recomendacion": evaluation.get("recomendacion"),
        "confianza": evaluation.get("confianza_evaluacion"),
    })
    return evaluation
 
 
# 3. Ranking comparativo
 
def rank_candidates(
    vectorstore: Chroma,
    candidate_ids: List[str],
    job_title: str,
    job_description: str,
) -> Dict:
    """Evalúa múltiples candidatos y genera ranking comparativo."""
    evaluations = []
    for cid in candidate_ids:
        ev = evaluate_candidate(vectorstore, cid, job_title, job_description)
        evaluations.append(ev)
 
    chain = RANKING_PROMPT | _get_llm() | StrOutputParser()
    raw = chain.invoke({
        "job_title": job_title,
        "evaluations_json": json.dumps(evaluations, ensure_ascii=False, indent=2),
    })
    ranking = safe_parse_json(raw)
 
    log_event("candidate_ranking", {
        "job_title": job_title,
        "candidates_evaluated": candidate_ids,
        "top_candidate": ranking.get("ranking", [{}])[0].get("candidate_id")
            if ranking.get("ranking") else None,
    })
    return {"ranking": ranking, "evaluaciones_detalladas": evaluations}
 
 
# 4. Q&A libre
 
def query_candidates(
    vectorstore: Chroma,
    question: str,
    candidate_id: Optional[str] = None,
    k: int = 6,
) -> str:
    """Responde preguntas en lenguaje natural sobre los candidatos."""
    if candidate_id:
        docs = filter_by_candidate(vectorstore, candidate_id, query=question, k=k)
    else:
        retriever = get_retriever(vectorstore, k=k)
        docs = retriever.invoke(question)
 
    context = "\n\n---\n\n".join([
        f"[Candidato: {d.metadata.get('candidate_id', '?')} | "
        f"Fuente: {d.metadata.get('source_type', '?')}]\n{d.page_content}"
        for d in docs
    ])
 
    chain = QA_PROMPT | _get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
 
    log_event("qa_query", {
        "question": question,
        "candidate_id_filter": candidate_id,
        "docs_retrieved": len(docs),
    })
    return answer