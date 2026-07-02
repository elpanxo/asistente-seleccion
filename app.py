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
# RA3 — Observabilidad y Seguridad (importación dinámica)
import importlib.util as _ilu, os as _os

def _load_module(rel_path, module_name):
    base = _os.path.dirname(_os.path.abspath(__file__))
    full = _os.path.join(base, rel_path)
    spec = _ilu.spec_from_file_location(module_name, full)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_metrics_mod = _load_module("src/observability/metrics.py", "metrics")
_guard_mod   = _load_module("src/security/guard.py",        "guard")

session_metrics   = _metrics_mod.session_metrics
analyze_audit_log = _metrics_mod.analyze_audit_log
input_guard       = _guard_mod.input_guard
rate_limiter      = _guard_mod.rate_limiter
ethics_monitor    = _guard_mod.ethics_monitor
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
/* Ocultar texto duplicado del botón upload */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span { display: none !important; }
[data-testid="stFileUploader"] small { display: none !important; }
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] span { display: none !important; }

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
[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 0.5rem !important; max-width: 100% !important; }
/* Ocultar botón de colapsar sidebar — siempre visible */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
<script>
// Forzar sidebar expandido al cargar
window.addEventListener("load", function() {
    const btn = document.querySelector("[data-testid=\"stSidebarCollapseButton\"]");
    if (btn) btn.style.display = "flex";
});
</script>
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
        label_visibility="collapsed", placeholder="Nombre del cargo", key="sb_job_title"
    )
    st.session_state.job_description = st.text_area(
        "Descripción", value=st.session_state.job_description,
        label_visibility="collapsed", height=90, placeholder="Descripción del cargo...", key="sb_job_desc"
    )

    st.markdown('<div class="sidebar-section-title">Subir Documento</div>', unsafe_allow_html=True)
    candidate_name = st.text_input("ID", placeholder="ej: ana_lopez", label_visibility="collapsed", key="sb_candidate_id")
    source_type = st.selectbox(
        "Tipo", ["curriculum", "feedback_entrevista", "linkedin", "github", "evaluacion_previa"],
        label_visibility="collapsed", key="sb_source_type"
    )
    uploaded_files = st.file_uploader("Archivos", type=["pdf", "txt", "json", "docx"], label_visibility="collapsed", accept_multiple_files=True, help="Sube hasta 3 archivos a la vez", key="sb_uploader")

    if st.button("Subir e Indexar", use_container_width=True, type="primary", key="sb_upload_btn"):
        if not candidate_name:
            st.error("Ingresa un ID de candidato.")
        elif not uploaded_files:
            st.error("Selecciona un archivo.")
        else:
            with st.spinner("Indexando..."):
                save_dir = CVS_DIR / candidate_name
                save_dir.mkdir(parents=True, exist_ok=True)
                all_chunks = []
                nombres = []
                for uf in uploaded_files[:3]:
                    ext = Path(uf.name).suffix
                    save_path = save_dir / f"{source_type}_{uf.name}"
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    docs   = load_single_file(str(save_path), candidate_name, source_type)
                    chunks = split_documents(docs)
                    all_chunks.extend(chunks)
                    nombres.append(uf.name)
                if all_chunks:
                    vs = get_vs()
                    if vs is None:
                        vs = build_vectorstore(all_chunks)
                    else:
                        add_documents(vs, all_chunks)
                    st.session_state.vectorstore = vs
                    st.success(f"✓ {len(nombres)} archivo(s) indexado(s): {', '.join(nombres)}")
                    st.rerun()
                else:
                    st.error("No se pudo procesar ningún archivo.")

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
# Botón flotante para abrir el sidebar si está colapsado
st.markdown("""
<style>
.sidebar-toggle-btn {
    position: fixed; top: 12px; left: 12px; z-index: 999999;
    background: #262626; border: 1px solid #10a37f;
    border-radius: 8px; padding: 6px 10px;
    color: #10a37f; font-size: 1rem; cursor: pointer;
    display: none;
}
[data-testid="stSidebar"][aria-expanded="false"] ~ * .sidebar-toggle-btn,
body:has([data-testid="stSidebar"][aria-expanded="false"]) .sidebar-toggle-btn {
    display: block !important;
}
/* Forzar sidebar siempre visible */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: none !important;
    width: 21rem !important;
    min-width: 21rem !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Forzar apertura del sidebar via JS
st.components.v1.html("""
<script>
(function forceSidebar() {
    function expand() {
        // Buscar y hacer clic en el botón de expandir si existe
        var btns = parent.document.querySelectorAll('[data-testid="collapsedControl"] button, [data-testid="stSidebarCollapseButton"]');
        btns.forEach(function(btn) { btn.click(); });
        
        // Remover clase de colapsado del sidebar
        var sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.style.transform = 'none';
            sidebar.style.width = '21rem';
            sidebar.setAttribute('aria-expanded', 'true');
        }
        
        // Limpiar localStorage de Streamlit
        try {
            var keys = Object.keys(parent.localStorage);
            keys.forEach(function(k) {
                if (k.includes('sidebar') || k.includes('Sidebar')) {
                    parent.localStorage.removeItem(k);
                }
            });
        } catch(e) {}
    }
    expand();
    setTimeout(expand, 500);
    setTimeout(expand, 1500);
})();
</script>
""", height=0)

tab_chat, tab_eval, tab_rank, tab_audit, tab_obs = st.tabs(["Chat", "Evaluación", "Ranking", "Auditoría", "Observabilidad"])


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
        from agent import init_agent, run_agent, get_short_memory_count, get_long_memory, clear_short_memory
        from src.rag.vectorstore import get_all_candidate_ids

        ids = get_all_candidate_ids(vs)
        init_agent(vs, st.session_state.job_title, st.session_state.job_description)

        lt = get_long_memory()

        # ── Barra superior compacta ───────────────────────────────────────
        summary = lt.get_summary()
        col_f, col_info, col_c = st.columns([3, 4, 1])
        with col_f:
            sel = st.selectbox("Filtrar", ["Todos los candidatos"] + ids,
                               label_visibility="collapsed", key="chat_filter")
        with col_info:
            st.markdown(
                f"<div style='display:flex;gap:16px;align-items:center;height:38px;padding:0 8px'>"
                f"<span style='font-size:0.75rem;color:#555'>"
                f"🧠 <span style='color:#888'>{summary['total_processes']}</span> proceso(s) &nbsp;·&nbsp; "
                f"<span style='color:#888'>{summary['total_evaluations']}</span> evaluacion(es) &nbsp;·&nbsp; "
                f"<span style='color:#888'>{get_short_memory_count()}</span> mensaje(s) en sesión"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        with col_c:
            if st.button("Limpiar", type="secondary", key="chat_clear"):
                st.session_state.chat_history = []
                clear_short_memory()
                st.rerun()

        candidate_filter = None if sel == "Todos los candidatos" else sel

        st.markdown("""
        <style>
        .chat-msg-user { display:flex; justify-content:flex-end; margin:12px 0; }
        .chat-msg-user .bubble {
            background:#2f2f2f; color:#ececec; padding:12px 18px;
            border-radius:18px 18px 4px 18px; max-width:75%;
            font-size:0.9rem; line-height:1.6;
        }
        .chat-empty { text-align:center; padding:40px 0; color:#555; }
        .chat-empty .icon { font-size:2rem; margin-bottom:12px; color:#333; }
        .chat-empty .title { font-size:1rem; font-weight:600; color:#777; margin-bottom:10px; }
        </style>
        """, unsafe_allow_html=True)

        # ── Inicializar session state ─────────────────────────────────────
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_pending" not in st.session_state:
            st.session_state.chat_pending = None

        # ── Pantalla vacía con sugerencias ───────────────────────────────
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:60px 0 24px 0;color:#555">
                <div style="font-size:1.8rem;margin-bottom:10px;color:#333">◈</div>
                <div style="font-size:1rem;font-weight:600;color:#777;margin-bottom:4px">¿En qué puedo ayudarte?</div>
                <div style="font-size:0.8rem;color:#4a4a4a">Selecciona una sugerencia o escribe tu consulta</div>
            </div>""", unsafe_allow_html=True)

            sug1, sug2, sug3 = st.columns(3)
            suggestions = [
                ("sug1", "¿Quién tiene más experiencia con Python?"),
                ("sug2", "¿Qué candidatos conocen Docker?"),
                ("sug3", "¿Cuál es el nivel de inglés de los candidatos?"),
            ]
            for col, (key, text) in zip([sug1, sug2, sug3], suggestions):
                with col:
                    if st.button(text, key=key, use_container_width=True):
                        st.session_state.chat_pending = text
                        st.rerun()

        # ── Mostrar historial ─────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-msg-user">
                    <div class="bubble">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                col_av, col_txt = st.columns([0.04, 0.96])
                with col_av:
                    st.markdown(
                        '<div style="width:30px;height:30px;border-radius:50%;background:#10a37f;'
                        'display:flex;align-items:center;justify-content:center;'
                        'font-size:13px;font-weight:700;color:#fff;margin-top:4px">R</div>',
                        unsafe_allow_html=True)
                with col_txt:
                    st.markdown(msg["content"])
                    # Mostrar plan y pasos si existen
                    if msg.get("plan"):
                        with st.expander("Plan de ejecución"):
                            for k, v in msg["plan"].items():
                                st.markdown(
                                    f"<div style='font-size:0.78rem;padding:2px 0'>"
                                    f"<span style='color:#555;min-width:130px;display:inline-block'>{k}</span>"
                                    f"<span style='color:#ccc'>{v}</span></div>",
                                    unsafe_allow_html=True)
                    if msg.get("steps") and len(msg["steps"]) > 0:
                        with st.expander(f"Razonamiento ReAct ({len(msg['steps'])} pasos)"):
                            for i, step in enumerate(msg["steps"], 1):
                                tool_name = step[0] if isinstance(step, tuple) else str(step)
                                st.markdown(
                                    f"<div style='font-size:0.78rem;color:#888;margin-bottom:4px'>"
                                    f"<span style='color:#10a37f'>Paso {i}</span> · "
                                    f"<code>{tool_name}</code></div>",
                                    unsafe_allow_html=True)
                    if msg.get("memory_used"):
                        st.markdown(
                            '<span style="font-size:0.72rem;color:#10a37f">'
                            '🧠 Memoria de largo plazo consultada</span>',
                            unsafe_allow_html=True)



        # ── Procesar mensaje pendiente (sugerencias) o input manual ───────
        def process_message(user_input: str):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("El agente está razonando..."):
                result = run_agent(user_input, ids)
            st.session_state.chat_history.append({
                "role":        "assistant",
                "content":     result["output"],
                "plan":        result.get("plan", {}),
                "steps":       result.get("steps", []),
                "memory_used": result.get("memory_used", False),
            })

        # Procesar sugerencia pendiente
        if st.session_state.chat_pending:
            pending = st.session_state.chat_pending
            st.session_state.chat_pending = None
            process_message(pending)
            st.rerun()

        # Input manual
        if prompt := st.chat_input("Escribe tu consulta...", key="chat_input_main"):
            process_message(prompt)
            st.rerun()


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
                selected = st.selectbox("Candidato", ids, label_visibility="collapsed", key="eval_candidate")
            with col_b:
                run = st.button("Evaluar", type="primary", use_container_width=True, key="eval_btn")

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
                sel_rank = st.multiselect("Candidatos", ids, default=ids, label_visibility="collapsed", key="rank_select")
            with col_rb:
                run_rank = st.button("Generar ranking", type="primary", use_container_width=True, key="rank_btn")

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
            ft = st.selectbox("Filtrar", ["Todos"] + list(counts.keys()), label_visibility="collapsed", key="audit_filter")
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



# ── OBSERVABILIDAD ───────────────────────────────────────────────────────────
with tab_obs:
    st.markdown("### 📊 Métricas de Rendimiento")

    summary = session_metrics.get_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Interacciones", summary["total_interactions"])
    with c2:
        st.metric("Errores", summary["total_errors"])
    with c3:
        st.metric("Tasa de error", f"{summary['error_rate_pct']}%")
    with c4:
        st.metric("Tiempo prom.", f"{summary['avg_response_ms']} ms")

    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("Tiempo mín.", f"{summary['min_response_ms']} ms")
    with c6:
        st.metric("Tiempo máx.", f"{summary['max_response_ms']} ms")
    with c7:
        st.metric("Uptime sesión", f"{summary['session_uptime_sec']} s")

    # Herramientas más usadas
    top_tools = session_metrics.get_top_tools()
    if top_tools:
        st.markdown(
            "<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
            "letter-spacing:0.08em;margin:12px 0 8px 0'>Herramientas más usadas</div>",
            unsafe_allow_html=True,
        )
        for t in top_tools:
            pct = (t["count"] / summary["total_interactions"] * 100) if summary["total_interactions"] > 0 else 0
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px'>"
                f"<code style='color:#10a37f;min-width:180px;font-size:0.8rem'>{t['tool']}</code>"
                f"<div style='flex:1;background:#2a2a2a;border-radius:4px;height:8px'>"
                f"<div style='background:#10a37f;width:{min(pct,100):.0f}%;height:8px;border-radius:4px'></div></div>"
                f"<span style='color:#888;font-size:0.78rem;min-width:40px'>{t['count']}x</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Historial de interacciones
    history = session_metrics.get_history(10)
    if history:
        with st.expander("Historial de interacciones (últimas 10)"):
            for h in reversed(history):
                color = "#10a37f" if h["success"] else "#ef4444"
                st.markdown(
                    f"<div style='font-size:0.78rem;padding:4px 0;border-bottom:1px solid #2a2a2a'>"
                    f"<span style='color:{color}'>●</span> "
                    f"<span style='color:#888'>{h['timestamp'][11:19]}</span> &nbsp;"
                    f"<span style='color:#ccc'>{h['operation']}</span> &nbsp;"
                    f"<span style='color:#555'>{h['duration_ms']} ms</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if st.button("Reiniciar métricas de sesión", key="reset_metrics"):
        session_metrics.reset()
        st.success("Métricas reiniciadas.")
        st.rerun()

    st.divider()

    st.divider()
    st.markdown("### 🔍 Trazabilidad e Historial")

    audit_analysis = analyze_audit_log()

    if "error" in audit_analysis:
        st.info(audit_analysis["error"])
    else:
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            st.metric("Total eventos", audit_analysis["total_events"])
        with ca2:
            evals = audit_analysis["by_event_type"].get("candidate_evaluation", 0)
            st.metric("Evaluaciones", evals)
        with ca3:
            queries = audit_analysis["by_event_type"].get("qa_query", 0)
            st.metric("Consultas Q&A", queries)

        # Distribución de eventos
        by_type = audit_analysis.get("by_event_type", {})
        if by_type:
            st.markdown(
                "<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
                "letter-spacing:0.08em;margin:12px 0 8px 0'>Distribución de eventos históricos</div>",
                unsafe_allow_html=True,
            )
            total_ev = sum(by_type.values())
            for etype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_ev * 100 if total_ev > 0 else 0
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px'>"
                    f"<span style='color:#888;min-width:200px;font-size:0.8rem'>{etype}</span>"
                    f"<div style='flex:1;background:#2a2a2a;border-radius:4px;height:8px'>"
                    f"<div style='background:#1a56a0;width:{min(pct,100):.0f}%;height:8px;border-radius:4px'></div></div>"
                    f"<span style='color:#888;font-size:0.78rem;min-width:50px'>{count} ({pct:.0f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Candidatos más evaluados
        cand_counts = audit_analysis.get("candidates_evaluated", {})
        if cand_counts:
            with st.expander("Candidatos más evaluados"):
                for cid, cnt in sorted(cand_counts.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(
                        f"<div style='font-size:0.83rem;padding:4px 0;border-bottom:1px solid #2a2a2a'>"
                        f"<span style='color:#ccc'>{cid}</span>"
                        f"<span style='color:#555;float:right'>{cnt} evaluacion(es)</span></div>",
                        unsafe_allow_html=True,
                    )

        # Recomendaciones históricas
        recs = audit_analysis.get("recommendations_given", {})
        if recs:
            total_r = sum(recs.values())
            st.markdown(
                "<div style='font-size:0.7rem;color:#555;text-transform:uppercase;"
                "letter-spacing:0.08em;margin:12px 0 8px 0'>Distribución de recomendaciones históricas</div>",
                unsafe_allow_html=True,
            )
            colors_rec = {"avanzar": "#10a37f", "en_espera": "#f59e0b", "descartar": "#ef4444"}
            for rec, cnt in recs.items():
                pct = cnt / total_r * 100 if total_r > 0 else 0
                color = colors_rec.get(rec, "#888")
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px'>"
                    f"<span style='color:{color};min-width:120px;font-size:0.8rem'>{rec}</span>"
                    f"<div style='flex:1;background:#2a2a2a;border-radius:4px;height:8px'>"
                    f"<div style='background:{color};width:{min(pct,100):.0f}%;height:8px;border-radius:4px;opacity:0.7'></div></div>"
                    f"<span style='color:#888;font-size:0.78rem'>{cnt} ({pct:.0f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    st.divider()
    st.markdown("### 🔒 Seguridad y Ética")

    cs1, cs2 = st.columns(2)
    with cs1:
        remaining = rate_limiter.remaining("session")
        st.metric("Consultas disponibles / min", remaining)
    with cs2:
        st.metric("Límite por minuto", rate_limiter.rpm)

    # Reporte ético
    audit_events = []
    audit_path = __import__('pathlib').Path(__import__('config').AUDIT_LOG_FILE)
    if audit_path.exists():
        import json as _json
        with open(audit_path, 'r', encoding='utf-8') as _f:
            audit_events = [_json.loads(l) for l in _f if l.strip()]

    ethics_report = ethics_monitor.generate_ethics_report(audit_events)
    st.markdown(
        f'<div class="ethics-box" style="margin-top:8px">'
        f'<strong>Reporte Ético del Sistema</strong><br><br>'
        f'Evaluaciones totales: <strong>{ethics_report.get("total_evaluaciones", 0)}</strong><br>'
        f'Candidatos únicos: <strong>{ethics_report.get("candidatos_unicos", 0)}</strong><br>',
        unsafe_allow_html=True,
    )
    dist = ethics_report.get("distribucion_recomendaciones", {})
    if dist:
        for k, v in dist.items():
            st.markdown(
                f"<span style='color:#aaa;font-size:0.82rem'>&nbsp;&nbsp;{k}: {v}</span><br>",
                unsafe_allow_html=True,
            )
    st.markdown(
        f"<br><em style='color:#7ec8a8;font-size:0.8rem'>"
        f"{ethics_report.get('nota_etica', '')}</em></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.divider()
    st.markdown("### 📈 Escalabilidad")

    st.markdown("""
    <div style='background:#1e1e1e;border:1px solid #2a2a2a;border-radius:10px;padding:16px;font-size:0.83rem;color:#aaa;line-height:1.8'>
    <strong style='color:#ececec'>Estrategias de escalabilidad implementadas:</strong><br>
    <span style='color:#10a37f'>✓</span> Embeddings locales (sentence-transformers) — sin dependencia de APIs externas<br>
    <span style='color:#10a37f'>✓</span> ChromaDB persistido en disco — escala hasta miles de candidatos sin servidor<br>
    <span style='color:#10a37f'>✓</span> Chunking configurable — ajustable según volumen de documentos<br>
    <span style='color:#10a37f'>✓</span> Rate limiting — protege contra sobrecarga del sistema<br>
    <span style='color:#10a37f'>✓</span> Reintentos con backoff exponencial — resiliencia ante fallas de API<br>
    <span style='color:#10a37f'>✓</span> Memoria LP en JSON — portable y sin dependencias de base de datos<br>
    <br>
    <strong style='color:#ececec'>Recomendaciones para mayor escala:</strong><br>
    <span style='color:#555'>→</span> Migrar ChromaDB a Pinecone o Weaviate para escala distribuida<br>
    <span style='color:#555'>→</span> Agregar cola de mensajes (Redis/RabbitMQ) para evaluaciones asíncronas<br>
    <span style='color:#555'>→</span> Containerizar con Docker para despliegue horizontal<br>
    <span style='color:#555'>→</span> Implementar caché de evaluaciones para candidatos ya procesados<br>
    <span style='color:#555'>→</span> Usar modelos locales (Ollama) para reducir costos de API a escala<br>
    </div>
    """, unsafe_allow_html=True)