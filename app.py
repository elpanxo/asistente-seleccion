"""
app.py — Interfaz web Streamlit del Asistente de Selección de Personal.
Diseño: profesional/formal inspirado en ChatGPT — sidebar oscuro, área de chat limpia.
Ejecutar con: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CVS_DIR
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

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecruitAI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #212121 !important;
    color: #ececec !important;
    font-family: 'Sora', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * { font-family: 'Sora', sans-serif !important; color: #ececec !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 1.2rem; border-bottom: 1px solid #2a2a2a; margin-bottom: 1.2rem;
}
.sidebar-logo-icon {
    width: 32px; height: 32px; background: #10a37f; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700; color: #fff; flex-shrink: 0;
}
.sidebar-logo-text { font-size: 1rem; font-weight: 600; color: #ececec; letter-spacing: -0.02em; }
.sidebar-logo-sub  { font-size: 0.68rem; color: #555; margin-top: 1px; }
.sidebar-section-title {
    font-size: 0.7rem; font-weight: 600; color: #555;
    letter-spacing: 0.08em; text-transform: uppercase; margin: 1.2rem 0 0.5rem 0;
}
.candidate-chip {
    display: flex; align-items: center; gap: 6px;
    background: #2a2a2a; border: 1px solid #333; border-radius: 6px;
    padding: 5px 10px; font-size: 0.78rem; color: #ccc; margin: 3px 0; width: 100%;
}
.candidate-chip-dot { width: 6px; height: 6px; background: #10a37f; border-radius: 50%; flex-shrink: 0; }

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #2a2a2a !important; border: 1px solid #333 !important;
    border-radius: 8px !important; color: #ececec !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.85rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #10a37f !important; box-shadow: 0 0 0 2px rgba(16,163,127,0.15) !important;
}

[data-testid="stButton"] button[kind="primary"] {
    background-color: #10a37f !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: 'Sora', sans-serif !important;
    font-weight: 500 !important; font-size: 0.85rem !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { background-color: #0d8f6f !important; }
[data-testid="stButton"] button[kind="secondary"] {
    background-color: transparent !important; color: #aaa !important;
    border: 1px solid #333 !important; border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.82rem !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 1px solid #2a2a2a !important; gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #666 !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.85rem !important; font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] { color: #ececec !important; border-bottom-color: #10a37f !important; }

[data-testid="stMetric"] {
    background: #262626; border: 1px solid #2f2f2f; border-radius: 10px; padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #666 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #ececec !important; font-size: 1.5rem !important; font-weight: 600 !important; font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stExpander"] {
    background: #262626 !important; border: 1px solid #2f2f2f !important;
    border-radius: 10px !important; margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary { color: #bbb !important; font-size: 0.85rem !important; font-weight: 500 !important; }

[data-testid="stProgressBar"] > div > div { background-color: #10a37f !important; }
[data-testid="stProgressBar"] > div { background-color: #2a2a2a !important; border-radius: 4px !important; }

[data-testid="stFileUploader"] {
    background: #262626 !important; border: 1px dashed #3a3a3a !important; border-radius: 10px !important;
}

[data-testid="stChatInput"] {
    background: #2a2a2a !important; border: 1px solid #333 !important; border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #ececec !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.9rem !important; border: none !important;
}

hr { border-color: #2a2a2a !important; }
h1, h2, h3 { font-family: 'Sora', sans-serif !important; font-weight: 600 !important; color: #ececec !important; letter-spacing: -0.02em !important; }

.rec-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}
.rec-avanzar   { background: rgba(16,163,127,0.15); color: #10a37f; border: 1px solid rgba(16,163,127,0.3); }
.rec-espera    { background: rgba(245,158,11,0.12);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.rec-descartar { background: rgba(239,68,68,0.12);   color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

.ethics-box {
    background: #1a2420; border: 1px solid #1f4035; border-left: 3px solid #10a37f;
    border-radius: 8px; padding: 12px 16px; font-size: 0.82rem; color: #7ec8a8; line-height: 1.5;
}
.rank-card {
    background: #262626; border: 1px solid #2f2f2f; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 8px; display: flex; align-items: center; gap: 16px;
}
.rank-pos { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: #555; min-width: 40px; }
.rank-pos.gold   { color: #f59e0b; }
.rank-pos.silver { color: #94a3b8; }
.rank-pos.bronze { color: #a07850; }
.rank-name   { font-weight: 600; font-size: 0.95rem; color: #ececec; }
.rank-reason { font-size: 0.78rem; color: #666; margin-top: 2px; }
.rank-score  { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; margin-left: auto; margin-right: 12px; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1.5rem !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ── Estado de sesión ──────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "vectorstore": None,
        "chat_history": [],
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


def get_vs():
    if st.session_state.vectorstore is None:
        vs = load_vectorstore()
        if vs:
            st.session_state.vectorstore = vs
    return st.session_state.vectorstore


def score_color(s: float) -> str:
    if s >= 7.5: return "#10a37f"
    if s >= 5.0: return "#f59e0b"
    return "#ef4444"


def rec_badge(rec: str) -> str:
    cls   = {"avanzar": "rec-avanzar", "en_espera": "rec-espera", "descartar": "rec-descartar"}
    label = {"avanzar": "Avanzar", "en_espera": "En espera", "descartar": "Descartar"}
    return f'<span class="rec-badge {cls.get(rec,"")}">{label.get(rec, rec)}</span>'


def rank_pos_class(pos: int) -> str:
    return {1: "gold", 2: "silver", 3: "bronze"}.get(pos, "")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">◈</div>
        <div>
            <div class="sidebar-logo-text">RecruitAI</div>
            <div class="sidebar-logo-sub">ISY0101 — Selección de Personal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Cargo</div>', unsafe_allow_html=True)
    st.session_state.job_title = st.text_input(
        "Cargo", value=st.session_state.job_title,
        label_visibility="collapsed", placeholder="Nombre del cargo"
    )
    st.session_state.job_description = st.text_area(
        "Descripción", value=st.session_state.job_description,
        label_visibility="collapsed", height=90, placeholder="Descripción del cargo..."
    )

    st.markdown('<div class="sidebar-section-title">Subir Documento</div>', unsafe_allow_html=True)
    candidate_name = st.text_input("ID", placeholder="ej: ana_lopez", label_visibility="collapsed")
    source_type = st.selectbox(
        "Tipo", ["curriculum", "feedback_entrevista", "linkedin", "github", "evaluacion_previa"],
        label_visibility="collapsed",
    )
    uploaded_file = st.file_uploader("Archivo", type=["pdf", "txt", "json", "docx"], label_visibility="collapsed")

    if st.button("Subir e Indexar", use_container_width=True, type="primary"):
        if not candidate_name:
            st.error("Ingresa un ID de candidato.")
        elif not uploaded_file:
            st.error("Selecciona un archivo.")
        else:
            with st.spinner("Indexando..."):
                ext = Path(uploaded_file.name).suffix
                save_dir = CVS_DIR / candidate_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{source_type}{ext}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                docs   = load_single_file(str(save_path), candidate_name, source_type)
                chunks = split_documents(docs)
                if chunks:
                    vs = get_vs()
                    if vs is None:
                        vs = build_vectorstore(chunks)
                    else:
                        add_documents(vs, chunks)
                    st.session_state.vectorstore = vs
                    st.success("Indexado correctamente.")
                    st.rerun()
                else:
                    st.error("No se pudo procesar el archivo.")

    vs = get_vs()
    if vs:
        ids = get_all_candidate_ids(vs)
        if ids:
            st.markdown('<div class="sidebar-section-title">Candidatos</div>', unsafe_allow_html=True)
            for cid in ids:
                st.markdown(
                    f'<div class="candidate-chip"><div class="candidate-chip-dot"></div>{cid}</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_eval, tab_rank, tab_audit, tab_agent = st.tabs(["Chat", "Evaluación", "Ranking", "Auditoría", "Agente EP2"])


# ── CHAT ─────────────────────────────────────────────────────────────────────
with tab_chat:
    vs = get_vs()
    if vs is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#555">
            <div style="font-size:2rem;margin-bottom:12px">◈</div>
            <div style="font-size:1rem;font-weight:500;color:#777;margin-bottom:6px">RecruitAI</div>
            <div style="font-size:0.82rem;color:#555">Sube documentos desde el panel izquierdo para comenzar.</div>
        </div>""", unsafe_allow_html=True)
    else:
        ids = get_all_candidate_ids(vs)
        col_f, col_c = st.columns([4, 1])
        with col_f:
            sel = st.selectbox("Filtrar", ["Todos los candidatos"] + ids, label_visibility="collapsed")
        with col_c:
            if st.button("Limpiar", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

        candidate_filter = None if sel == "Todos los candidatos" else sel

        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#444">
                <div style="font-size:0.85rem;line-height:1.8">
                    Haz una pregunta sobre los candidatos indexados.<br>
                    <span style="color:#555;font-size:0.78rem">Ej: "¿Quién tiene más experiencia con Python?"</span>
                </div>
            </div>""", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Escribe tu consulta sobre los candidatos..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    answer = query_candidates(vs, prompt, candidate_id=candidate_filter)
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ── EVALUACIÓN ───────────────────────────────────────────────────────────────
with tab_eval:
    vs = get_vs()
    if vs is None:
        st.info("Sube documentos para comenzar.")
    else:
        ids = get_all_candidate_ids(vs)
        if not ids:
            st.warning("No hay candidatos indexados.")
        else:
            col_s, col_b = st.columns([3, 1])
            with col_s:
                selected = st.selectbox("Candidato", ids, label_visibility="collapsed")
            with col_b:
                run = st.button("Evaluar", type="primary", use_container_width=True)

            if run:
                with st.spinner(f"Analizando {selected}..."):
                    result = evaluate_candidate(
                        vs, selected,
                        st.session_state.job_title,
                        st.session_state.job_description,
                    )

                if result.get("parse_error"):
                    st.error("Error al procesar.")
                    st.json(result)
                else:
                    score = result.get("puntuacion_ponderada_validada", result.get("puntuacion_ponderada", 0))
                    rec   = result.get("recomendacion", "?")
                    conf  = result.get("confianza_evaluacion", "?")

                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Puntuación", f"{score:.1f} / 10")
                    with c2: st.metric("Confianza", conf.upper())
                    with c3:
                        st.markdown(
                            f"<div style='padding-top:8px'>"
                            f"<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
                            f"letter-spacing:0.05em;margin-bottom:8px'>Recomendación</div>"
                            f"{rec_badge(rec)}</div>", unsafe_allow_html=True,
                        )

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
                        "letter-spacing:0.08em;margin-bottom:10px'>Criterios</div>",
                        unsafe_allow_html=True,
                    )

                    scores = result.get("puntuaciones", {})
                    razon  = result.get("razonamiento", {})
                    criteria = {
                        "experiencia_relevante":  ("Experiencia Relevante", "30%"),
                        "habilidades_tecnicas":   ("Habilidades Técnicas",  "25%"),
                        "formacion_academica":    ("Formación Académica",   "15%"),
                        "proyectos_destacados":   ("Proyectos Destacados",  "15%"),
                        "diversidad_e_inclusion": ("Diversidad e Inclusión","10%"),
                        "comunicacion_liderazgo": ("Comunicación y Liderazgo","5%"),
                    }
                    for key, (label, weight) in criteria.items():
                        s = scores.get(key, 0)
                        with st.expander(f"{label}  ·  {weight}  —  {s:.1f} / 10"):
                            st.progress(s / 10)
                            st.markdown(
                                f"<div style='font-size:0.83rem;color:#aaa;margin-top:6px;line-height:1.6'>"
                                f"{razon.get(key, 'Sin análisis.')}</div>",
                                unsafe_allow_html=True,
                            )

                    cf, cm = st.columns(2)
                    with cf:
                        st.markdown("<div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>Fortalezas</div>", unsafe_allow_html=True)
                        for f in result.get("fortalezas", []):
                            st.markdown(f"<div style='font-size:0.85rem;color:#ccc;padding:5px 0;border-bottom:1px solid #2a2a2a'><span style='color:#10a37f;margin-right:8px'>+</span>{f}</div>", unsafe_allow_html=True)
                    with cm:
                        st.markdown("<div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>Áreas de Mejora</div>", unsafe_allow_html=True)
                        for a in result.get("areas_de_mejora", []):
                            st.markdown(f"<div style='font-size:0.85rem;color:#ccc;padding:5px 0;border-bottom:1px solid #2a2a2a'><span style='color:#f59e0b;margin-right:8px'>△</span>{a}</div>", unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="ethics-box" style="margin-top:16px">'
                        f'<strong>Declaración ética</strong><br>'
                        f'{result.get("justificacion_etica","")}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("JSON completo"):
                        st.json(result)


# ── RANKING ──────────────────────────────────────────────────────────────────
with tab_rank:
    vs = get_vs()
    if vs is None:
        st.info("Sube documentos para comenzar.")
    else:
        ids = get_all_candidate_ids(vs)
        if len(ids) < 2:
            st.warning("Necesitas al menos 2 candidatos para generar un ranking.")
        else:
            col_ms, col_rb = st.columns([3, 1])
            with col_ms:
                sel_rank = st.multiselect("Candidatos", ids, default=ids, label_visibility="collapsed")
            with col_rb:
                run_rank = st.button("Generar ranking", type="primary", use_container_width=True)

            if run_rank:
                if len(sel_rank) < 2:
                    st.error("Selecciona al menos 2 candidatos.")
                else:
                    with st.spinner("Evaluando y rankeando..."):
                        result = rank_candidates(
                            vs, sel_rank,
                            st.session_state.job_title,
                            st.session_state.job_description,
                        )

                    ranking_data = result["ranking"].get("ranking", [])
                    st.markdown(
                        f"<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
                        f"letter-spacing:0.08em;margin-bottom:12px'>{st.session_state.job_title}</div>",
                        unsafe_allow_html=True,
                    )

                    for r in ranking_data:
                        pos    = r.get("posicion", 0)
                        name   = r.get("nombre", r.get("candidate_id", "?"))
                        score  = r.get("puntuacion_ponderada", 0)
                        rec    = r.get("recomendacion", "?")
                        motivo = r.get("motivo_principal", "")
                        medal  = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"#{pos}")
                        pc     = rank_pos_class(pos)
                        sc     = score_color(score)

                        st.markdown(f"""
                        <div class="rank-card">
                            <div class="rank-pos {pc}">{medal}</div>
                            <div style="flex:1">
                                <div class="rank-name">{name}</div>
                                <div class="rank-reason">{motivo}</div>
                            </div>
                            <div class="rank-score" style="color:{sc}">{score:.1f}</div>
                            {rec_badge(rec)}
                        </div>""", unsafe_allow_html=True)

                    diversity = result["ranking"].get("analisis_diversidad_equipo", "")
                    if diversity:
                        st.markdown(f'<div class="ethics-box" style="margin-top:12px"><strong>Análisis de diversidad</strong><br>{diversity}</div>', unsafe_allow_html=True)

                    for w in result["ranking"].get("advertencias_eticas", []):
                        st.warning(w)

                    with st.expander("Evaluaciones detalladas (JSON)"):
                        st.json(result["evaluaciones_detalladas"])


# ── AUDITORÍA ────────────────────────────────────────────────────────────────
with tab_audit:
    log = get_audit_log()
    if not log:
        st.markdown("<div style='text-align:center;padding:60px 0;color:#555;font-size:0.85rem'>No hay eventos registrados aún.</div>", unsafe_allow_html=True)
    else:
        counts = {}
        for e in log:
            t = e.get("event_type", "otro")
            counts[t] = counts.get(t, 0) + 1

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total eventos", len(log))
        with c2: st.metric("Evaluaciones", counts.get("candidate_evaluation", 0))
        with c3: st.metric("Rankings", counts.get("candidate_ranking", 0))
        with c4: st.metric("Consultas", counts.get("qa_query", 0))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col_f, col_d = st.columns([3, 1])
        with col_f:
            ft = st.selectbox("Filtrar", ["Todos"] + list(counts.keys()), label_visibility="collapsed")
        with col_d:
            st.download_button(
                "Descargar log", use_container_width=True,
                data="\n".join(json.dumps(e, ensure_ascii=False) for e in log),
                file_name="audit_log.jsonl", mime="application/json",
            )

        filtered = log if ft == "Todos" else [e for e in log if e.get("event_type") == ft]

        for entry in reversed(filtered):
            ts     = entry.get("timestamp", "")[:19].replace("T", " ")
            etype  = entry.get("event_type", "?")
            detail = entry.get("candidate_id", entry.get("question", ""))
            detail = (detail[:50] + "…") if detail and len(detail) > 50 else (detail or "")
            with st.expander(f"{ts}  ·  {etype}  ·  {detail}"):
                st.json(entry)


# ── AGENTE EP2 ────────────────────────────────────────────────────────────────
with tab_agent:
    vs = get_vs()

    if vs is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#555">
            <div style="font-size:2rem;margin-bottom:12px">◈</div>
            <div style="font-size:0.85rem;color:#666">Sube documentos primero para activar el agente.</div>
        </div>""", unsafe_allow_html=True)
    else:
        from agent import init_agent, run_agent, get_short_memory, get_long_memory, clear_short_memory
        from src.rag.vectorstore import get_all_candidate_ids

        # Inicializar agente con contexto actual
        init_agent(vs, st.session_state.job_title, st.session_state.job_description)
        available_ids = get_all_candidate_ids(vs)

        st.markdown(
            "<div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px'>"
            "Agente funcional con memoria y planificación — EP2</div>",
            unsafe_allow_html=True,
        )

        # ── Memoria de largo plazo ────────────────────────────────────────
        lt = get_long_memory()
        summary = lt.get_summary()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Procesos en memoria", summary["total_processes"])
        with c2:
            st.metric("Evaluaciones guardadas", summary["total_evaluations"])
        with c3:
            st.metric("Mensajes sesión actual", get_short_memory().message_count)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Panel de herramientas disponibles ─────────────────────────────
        with st.expander("Herramientas del agente"):
            tools_info = [
                ("buscar_candidatos",  "Consulta",            "Búsqueda semántica RAG sobre documentos indexados"),
                ("evaluar_candidato",  "Razonamiento",        "Genera evaluación ponderada con chain-of-thought"),
                ("rankear_candidatos", "Escritura",           "Ranking comparativo con análisis de diversidad"),
                ("consultar_historial","Memoria largo plazo", "Recupera evaluaciones previas del candidato"),
                ("resumen_proceso",    "Escritura",           "Resumen del estado actual del proceso de selección"),
            ]
            for name, tipo, desc in tools_info:
                st.markdown(
                    f"<div style='display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #2a2a2a'>"
                    f"<code style='color:#10a37f;min-width:180px'>{name}</code>"
                    f"<span style='color:#666;min-width:120px;font-size:0.8rem'>{tipo}</span>"
                    f"<span style='color:#aaa;font-size:0.8rem'>{desc}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Historial del chat del agente ─────────────────────────────────
        if "agent_history" not in st.session_state:
            st.session_state.agent_history = []

        if not st.session_state.agent_history:
            st.markdown("""
            <div style="text-align:center;padding:30px 0;color:#444;font-size:0.83rem;line-height:1.8">
                El agente puede razonar, planificar y usar múltiples herramientas.<br>
                <span style="color:#555;font-size:0.78rem">
                Ej: "Evalúa a ana_lopez y dime si ya fue evaluada antes"<br>
                "Rankea todos los candidatos y dame un resumen del proceso"
                </span>
            </div>""", unsafe_allow_html=True)

        for msg in st.session_state.agent_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("plan"):
                    with st.expander("Plan de ejecución del agente"):
                        for k, v in msg["plan"].items():
                            st.markdown(
                                f"<div style='font-size:0.8rem;padding:3px 0'>"
                                f"<span style='color:#555;min-width:140px;display:inline-block'>{k}</span>"
                                f"<span style='color:#ccc'>{v}</span></div>",
                                unsafe_allow_html=True,
                            )
                if msg.get("steps"):
                    with st.expander(f"Razonamiento ReAct ({len(msg['steps'])} pasos)"):
                        for i, (action, observation) in enumerate(msg["steps"], 1):
                            st.markdown(
                                f"<div style='font-size:0.78rem;color:#888;margin-bottom:4px'>"
                                f"<span style='color:#10a37f'>Paso {i}</span> · "
                                f"Herramienta: <code>{action.tool}</code></div>",
                                unsafe_allow_html=True,
                            )

        # ── Input del agente ──────────────────────────────────────────────
        col_input, col_clear = st.columns([4, 1])
        with col_clear:
            if st.button("Limpiar", type="secondary", key="clear_agent"):
                st.session_state.agent_history = []
                clear_short_memory()
                st.rerun()

        if prompt := st.chat_input("Escribe tu consulta al agente...", key="agent_input"):
            st.session_state.agent_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("El agente está razonando..."):
                    result = run_agent(prompt, available_ids)

                st.markdown(result["output"])

                # Mostrar plan del planificador
                with st.expander("Plan de ejecución del agente"):
                    for k, v in result["plan"].items():
                        st.markdown(
                            f"<div style='font-size:0.8rem;padding:3px 0'>"
                            f"<span style='color:#555;min-width:140px;display:inline-block'>{k}</span>"
                            f"<span style='color:#ccc'>{v}</span></div>",
                            unsafe_allow_html=True,
                        )

                # Mostrar pasos de razonamiento ReAct
                if result["steps"]:
                    with st.expander(f"Razonamiento ReAct ({len(result['steps'])} pasos)"):
                        for i, (action, observation) in enumerate(result["steps"], 1):
                            st.markdown(
                                f"<div style='font-size:0.78rem;color:#888;margin-bottom:4px'>"
                                f"<span style='color:#10a37f'>Paso {i}</span> · "
                                f"Herramienta: <code>{action.tool}</code></div>",
                                unsafe_allow_html=True,
                            )

                if result["memory_used"]:
                    st.markdown(
                        '<div style="font-size:0.75rem;color:#10a37f;margin-top:8px">'
                        '🧠 Memoria de largo plazo consultada en esta respuesta</div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.agent_history.append({
                "role":    "assistant",
                "content": result["output"],
                "plan":    result["plan"],
                "steps":   result["steps"],
            })

        # ── Memoria de largo plazo del proceso ────────────────────────────
        process_data = lt.get_process(st.session_state.job_title)
        if process_data and process_data["decisions"]:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
                "letter-spacing:0.08em;margin-bottom:8px'>Memoria de largo plazo — Proceso actual</div>",
                unsafe_allow_html=True,
            )
            top = lt.get_top_candidates(st.session_state.job_title)
            for d in top:
                sc = {"avanzar": "#10a37f", "en_espera": "#f59e0b", "descartar": "#ef4444"}
                color = sc.get(d.get("recommendation", ""), "#888")
                st.markdown(
                    f"<div style='background:#262626;border:1px solid #2f2f2f;border-radius:8px;"
                    f"padding:10px 14px;margin-bottom:6px;display:flex;gap:12px;align-items:center'>"
                    f"<span style='color:#ccc;font-weight:500;flex:1'>{d['candidate_id']}</span>"
                    f"<span style='font-family:monospace;color:{color}'>{d.get('score',0):.1f}</span>"
                    f"<span style='color:{color};font-size:0.75rem'>{d.get('recommendation','')}</span>"
                    f"<span style='color:#555;font-size:0.72rem'>{d.get('date','')[:10]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )