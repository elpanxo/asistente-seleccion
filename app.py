import json
import sys
import tempfile
from pathlib import Path
 
import streamlit as st
 
# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
 
from config import CVS_DIR, EVALUACIONES_DIR
from src.rag.ingestion import load_single_file, split_documents
from src.rag.vectorstore import (
    add_documents,
    build_vectorstore,
    get_all_candidate_ids,
    load_vectorstore,
)
from src.evaluation.rag_pipeline import (
    evaluate_candidate,
    query_candidates,
    rank_candidates,
)
from src.ethics.audit import get_audit_log
 
# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente de Selección de Personal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #7c3aed;
    }
    .recommend-avanzar  { color: #22c55e; font-weight: bold; }
    .recommend-espera   { color: #f59e0b; font-weight: bold; }
    .recommend-descartar{ color: #ef4444; font-weight: bold; }
    .ethics-box {
        background: #1a2e1a;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 10px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ── Estado de sesión ──────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "candidate_ids": [],
        "job_title": "Desarrollador/a Backend Senior",
        "job_description": (
            "Buscamos un/a Desarrollador/a Backend Senior con experiencia en Python, "
            "APIs REST, bases de datos y trabajo en equipo ágil. "
            "Valoramos arquitecturas de microservicios, cloud (AWS/GCP) y diversidad."
        ),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
init_session()
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def get_vs():
    """Carga o retorna el vectorstore de la sesión."""
    if st.session_state.vectorstore is None:
        vs = load_vectorstore()
        if vs:
            st.session_state.vectorstore = vs
            st.session_state.candidate_ids = get_all_candidate_ids(vs)
    return st.session_state.vectorstore
 
 
def recommendation_badge(rec: str) -> str:
    colors = {
        "avanzar":   ("✅", "#22c55e"),
        "en_espera": ("⏳", "#f59e0b"),
        "descartar": ("❌", "#ef4444"),
    }
    icon, color = colors.get(rec, ("❓", "#888"))
    return f'<span style="color:{color};font-weight:bold">{icon} {rec.upper()}</span>'
 
 
def score_color(score: float) -> str:
    if score >= 7.5:  return "#22c55e"
    if score >= 5.0:  return "#f59e0b"
    return "#ef4444"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 Asistente de Selección")
    st.divider()
 
    # ── Configuración del cargo ───────────────────────────────────────────────
    st.subheader("📋 Configuración del Cargo")
    st.session_state.job_title = st.text_input(
        "Nombre del cargo",
        value=st.session_state.job_title,
    )
    st.session_state.job_description = st.text_area(
        "Descripción del cargo",
        value=st.session_state.job_description,
        height=120,
    )
    st.divider()
 
    # ── Subida de archivos ────────────────────────────────────────────────────
    st.subheader("📁 Subir Documentos de Candidato")
 
    candidate_name = st.text_input(
        "ID del candidato",
        placeholder="ej: ana_lopez",
        help="Identificador único sin espacios. Usa _ en lugar de espacios.",
    )
 
    source_type = st.selectbox(
        "Tipo de documento",
        ["curriculum", "feedback_entrevista", "linkedin", "github", "evaluacion_previa"],
    )
 
    uploaded_file = st.file_uploader(
        "Selecciona el archivo",
        type=["pdf", "txt", "json", "docx"],
        help="PDF, TXT, JSON o Word (.docx)",
    )
 
    if st.button("⬆️ Subir e Indexar", use_container_width=True, type="primary"):
        if not candidate_name:
            st.error("⚠️ Ingresa un ID de candidato.")
        elif not uploaded_file:
            st.error("⚠️ Selecciona un archivo.")
        else:
            with st.spinner("Procesando e indexando..."):
                # Guardar en disco
                ext = Path(uploaded_file.name).suffix
                save_dir = CVS_DIR / candidate_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{source_type}{ext}"
 
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
 
                # Cargar y chunkear
                docs = load_single_file(str(save_path), candidate_name, source_type)
                chunks = split_documents(docs)
 
                if not chunks:
                    st.error("No se pudo procesar el archivo.")
                else:
                    vs = get_vs()
                    if vs is None:
                        vs = build_vectorstore(chunks)
                    else:
                        add_documents(vs, chunks)
                    st.session_state.vectorstore = vs
                    st.session_state.candidate_ids = get_all_candidate_ids(vs)
                    st.success(f"✅ {uploaded_file.name} indexado para **{candidate_name}**")
                    st.rerun()
 
    st.divider()
 
    # ── Candidatos indexados ──────────────────────────────────────────────────
    st.subheader("👥 Candidatos Indexados")
    vs = get_vs()
    if vs:
        ids = get_all_candidate_ids(vs)
        if ids:
            for cid in ids:
                st.markdown(f"• `{cid}`")
        else:
            st.caption("Aún no hay candidatos indexados.")
    else:
        st.caption("Sube documentos para comenzar.")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_eval, tab_rank, tab_audit = st.tabs([
    "💬 Chat con el Asistente",
    "📊 Evaluar Candidato",
    "🏆 Ranking",
    "🔍 Auditoría",
])
 
 
# ── TAB 1: CHAT ───────────────────────────────────────────────────────────────
with tab_chat:
    st.header("💬 Chat con el Asistente")
    st.caption("Haz preguntas sobre los candidatos en lenguaje natural.")
 
    vs = get_vs()
    if vs is None:
        st.info("📂 Sube documentos desde el panel izquierdo para empezar.")
    else:
        # Filtro opcional por candidato
        ids = get_all_candidate_ids(vs)
        filter_options = ["Todos los candidatos"] + ids
        selected_filter = st.selectbox("🔎 Filtrar por candidato", filter_options)
        candidate_filter = None if selected_filter == "Todos los candidatos" else selected_filter
 
        # Mostrar historial de chat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
 
        # Input del usuario
        if prompt := st.chat_input("Escribe tu pregunta sobre los candidatos..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
 
            with st.chat_message("assistant"):
                with st.spinner("Consultando..."):
                    answer = query_candidates(
                        vs, prompt,
                        candidate_id=candidate_filter,
                    )
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
 
        if st.session_state.chat_history:
            if st.button("🗑️ Limpiar chat"):
                st.session_state.chat_history = []
                st.rerun()
 
 
# ── TAB 2: EVALUACIÓN INDIVIDUAL ─────────────────────────────────────────────
with tab_eval:
    st.header("📊 Evaluación Individual de Candidato")
 
    vs = get_vs()
    if vs is None:
        st.info("📂 Sube documentos desde el panel izquierdo para empezar.")
    else:
        ids = get_all_candidate_ids(vs)
        if not ids:
            st.warning("No hay candidatos indexados aún.")
        else:
            selected_candidate = st.selectbox("Selecciona un candidato", ids)
 
            if st.button("🔍 Evaluar Candidato", type="primary"):
                with st.spinner(f"Evaluando a {selected_candidate}..."):
                    result = evaluate_candidate(
                        vs,
                        selected_candidate,
                        st.session_state.job_title,
                        st.session_state.job_description,
                    )
 
                if result.get("parse_error"):
                    st.error("Error al procesar la evaluación. Revisa el log de auditoría.")
                    st.json(result)
                else:
                    # ── Cabecera ──────────────────────────────────────────
                    col1, col2, col3 = st.columns(3)
                    score = result.get("puntuacion_ponderada_validada",
                                       result.get("puntuacion_ponderada", 0))
                    rec   = result.get("recomendacion", "?")
                    conf  = result.get("confianza_evaluacion", "?")
 
                    with col1:
                        st.metric("Puntuación ponderada", f"{score:.1f} / 10")
                    with col2:
                        st.markdown(f"**Recomendación**<br>{recommendation_badge(rec)}",
                                    unsafe_allow_html=True)
                    with col3:
                        st.metric("Confianza", conf.upper())
 
                    st.divider()
 
                    # ── Puntuaciones por criterio ─────────────────────────
                    st.subheader("Puntuaciones por Criterio")
                    scores = result.get("puntuaciones", {})
                    razon  = result.get("razonamiento", {})
 
                    criteria_labels = {
                        "experiencia_relevante":  "Experiencia Relevante (30%)",
                        "habilidades_tecnicas":   "Habilidades Técnicas (25%)",
                        "formacion_academica":    "Formación Académica (15%)",
                        "proyectos_destacados":   "Proyectos Destacados (15%)",
                        "diversidad_e_inclusion": "Diversidad e Inclusión (10%)",
                        "comunicacion_liderazgo": "Comunicación y Liderazgo (5%)",
                    }
 
                    for key, label in criteria_labels.items():
                        s = scores.get(key, 0)
                        color = score_color(s)
                        with st.expander(f"{label} — **{s:.1f}/10**"):
                            st.progress(s / 10)
                            st.markdown(razon.get(key, "Sin análisis disponible."))
 
                    st.divider()
 
                    # ── Fortalezas y áreas de mejora ──────────────────────
                    col_f, col_m = st.columns(2)
                    with col_f:
                        st.subheader("✅ Fortalezas")
                        for f in result.get("fortalezas", []):
                            st.markdown(f"• {f}")
                    with col_m:
                        st.subheader("⚠️ Áreas de Mejora")
                        for a in result.get("areas_de_mejora", []):
                            st.markdown(f"• {a}")
 
                    # ── Declaración ética ─────────────────────────────────
                    st.divider()
                    st.markdown(
                        f'<div class="ethics-box">🛡️ <b>Declaración ética:</b> '
                        f'{result.get("justificacion_etica", "")}</div>',
                        unsafe_allow_html=True,
                    )
 
                    # ── JSON completo (expandible) ────────────────────────
                    with st.expander("Ver JSON completo de la evaluación"):
                        st.json(result)
 
 
# ── TAB 3: RANKING ───────────────────────────────────────────────────────────
with tab_rank:
    st.header("🏆 Ranking Comparativo de Candidatos")
 
    vs = get_vs()
    if vs is None:
        st.info("📂 Sube documentos desde el panel izquierdo para empezar.")
    else:
        ids = get_all_candidate_ids(vs)
        if len(ids) < 2:
            st.warning("Necesitas al menos 2 candidatos indexados para generar un ranking.")
        else:
            selected_for_rank = st.multiselect(
                "Selecciona candidatos a comparar",
                ids,
                default=ids,
            )
 
            if st.button("🏆 Generar Ranking", type="primary"):
                if len(selected_for_rank) < 2:
                    st.error("Selecciona al menos 2 candidatos.")
                else:
                    with st.spinner("Evaluando y rankeando candidatos..."):
                        result = rank_candidates(
                            vs,
                            selected_for_rank,
                            st.session_state.job_title,
                            st.session_state.job_description,
                        )
 
                    ranking_data = result["ranking"].get("ranking", [])
 
                    # ── Tabla de ranking ──────────────────────────────────
                    st.subheader(f"Ranking para: {st.session_state.job_title}")
 
                    for r in ranking_data:
                        pos   = r.get("posicion", "?")
                        name  = r.get("nombre", r.get("candidate_id", "?"))
                        score = r.get("puntuacion_ponderada", 0)
                        rec   = r.get("recomendacion", "?")
                        motivo = r.get("motivo_principal", "")
 
                        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"#{pos}")
                        color = score_color(score)
 
                        with st.container():
                            col1, col2, col3 = st.columns([1, 4, 2])
                            with col1:
                                st.markdown(f"## {medal}")
                            with col2:
                                st.markdown(f"**{name}**")
                                st.caption(motivo)
                            with col3:
                                st.markdown(
                                    f'<span style="color:{color};font-size:1.4em;font-weight:bold">'
                                    f'{score:.1f}/10</span>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(recommendation_badge(rec), unsafe_allow_html=True)
                            st.divider()
 
                    # ── Análisis de diversidad ────────────────────────────
                    diversity = result["ranking"].get("analisis_diversidad_equipo", "")
                    if diversity:
                        st.info(f"🌍 **Análisis de diversidad:** {diversity}")
 
                    # ── Advertencias éticas ───────────────────────────────
                    warnings = result["ranking"].get("advertencias_eticas", [])
                    for w in warnings:
                        st.warning(f"⚠️ {w}")
 
                    # ── Detalle expandible ────────────────────────────────
                    with st.expander("Ver evaluaciones detalladas (JSON)"):
                        st.json(result["evaluaciones_detalladas"])
 
 
# ── TAB 4: AUDITORÍA ─────────────────────────────────────────────────────────
with tab_audit:
    st.header("🔍 Registro de Auditoría")
    st.caption("Trazabilidad completa de todas las operaciones del sistema.")
 
    log = get_audit_log()
 
    if not log:
        st.info("No hay eventos registrados aún. Comienza evaluando candidatos.")
    else:
        # Métricas resumen
        col1, col2, col3, col4 = st.columns(4)
        event_counts = {}
        for entry in log:
            t = entry.get("event_type", "otro")
            event_counts[t] = event_counts.get(t, 0) + 1
 
        with col1:
            st.metric("Total eventos", len(log))
        with col2:
            st.metric("Evaluaciones", event_counts.get("candidate_evaluation", 0))
        with col3:
            st.metric("Rankings", event_counts.get("candidate_ranking", 0))
        with col4:
            st.metric("Consultas Q&A", event_counts.get("qa_query", 0))
 
        st.divider()
 
        # Filtro por tipo de evento
        event_types = ["Todos"] + list(event_counts.keys())
        filter_type = st.selectbox("Filtrar por tipo de evento", event_types)
 
        filtered_log = log if filter_type == "Todos" else [
            e for e in log if e.get("event_type") == filter_type
        ]
 
        # Mostrar eventos en orden inverso (más reciente primero)
        for entry in reversed(filtered_log):
            with st.expander(
                f"[{entry.get('timestamp', '?')[:19]}] "
                f"{entry.get('event_type', '?')} — "
                f"{entry.get('candidate_id', entry.get('question', ''))[:40]}"
            ):
                st.json(entry)
 
        # Descargar log completo
        st.download_button(
            "⬇️ Descargar audit_log.jsonl",
            data="\n".join(json.dumps(e, ensure_ascii=False) for e in log),
            file_name="audit_log.jsonl",
            mime="application/json",
        )