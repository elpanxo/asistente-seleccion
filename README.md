# ◈ RecruitAI — Asistente Inteligente de Selección de Personal

> **ISY0101 — Ingeniería de Soluciones con IA | DuocUC 2025**

Sistema basado en **LLM + RAG + Agente con Memoria** que automatiza la preselección de candidatos. Integra fuentes internas y externas, genera evaluaciones ponderadas, mantiene memoria entre sesiones, planifica tareas de forma autónoma y cuenta con observabilidad, seguridad y trazabilidad completa.

---

## ¿Qué hace el sistema?

1. Indexa documentos de candidatos (PDF, TXT, JSON, DOCX) en ChromaDB con embeddings locales
2. Recupera información filtrando por candidato específico con búsqueda semántica
3. Genera evaluaciones ponderadas con razonamiento explícito por criterio
4. Produce rankings comparativos con análisis de diversidad
5. Responde preguntas en lenguaje natural como un agente conversacional
6. Recuerda conversaciones anteriores y decisiones entre sesiones
7. Planifica qué herramientas usar según la consulta del reclutador
8. Detecta y bloquea intentos de manipulación del sistema
9. Registra métricas de rendimiento y errores en tiempo real
10. Guarda trazabilidad completa de cada decisión en `audit_log.jsonl`

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────────┐
│             CAPA 1 — INTERFAZ (app.py / Streamlit)               │
│   Chat · Evaluación · Ranking · Auditoría · Observabilidad       │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│          CAPA 2 — AGENTE ReAct (agent.py)                        │
│  RecruitmentPlanner → AgentExecutor → 5 herramientas             │
│  Memoria CP (ConversationBufferWindowMemory k=10)                 │
│  Memoria LP (long_term_memory.json)                               │
└───────────┬──────────────────────────────────┬───────────────────┘
            │                                  │
┌───────────▼────────────┐    ┌────────────────▼──────────────────┐
│  CAPA 3 — RECUPERACIÓN │    │  CAPA 4 — GENERACIÓN Y CONTROL    │
│  ingestion.py          │    │  prompts.py → GPT-4o-mini          │
│  └─ chunks 800 chars   │    │  audit.py → audit_log.jsonl        │
│  vectorstore.py        │    │  metrics.py → agent_metrics.log    │
│  └─ ChromaDB           │    │  guard.py → seguridad y ética      │
│  └─ sentence-transf.   │    │                                    │
└────────────────────────┘    └────────────────────────────────────┘
```

**Flujo de datos:**

```
Archivo → ingestion.py → chunks → embeddings → ChromaDB

Consulta → guard.py (validación) → planner.py (plan) →
AgentExecutor (ReAct) → herramientas → respuesta →
metrics.py (registro) → audit_log.jsonl → UI
```

---

## Stack Tecnológico

| Componente     | Tecnología                      | Versión          |
| -------------- | ------------------------------- | ---------------- |
| LLM            | GPT-4o-mini (GitHub Models)     | API gratuita     |
| Agente         | LangChain AgentExecutor + ReAct | 0.2.x            |
| Embeddings     | sentence-transformers (local)   | all-MiniLM-L6-v2 |
| Vector Store   | ChromaDB                        | 0.5.x            |
| Framework RAG  | LangChain                       | 0.2.x            |
| Memoria CP     | ConversationBufferWindowMemory  | k=10             |
| Memoria LP     | JSON persistido en disco        | —                |
| Planificador   | RecruitmentPlanner (custom)     | —                |
| Seguridad      | InputGuard + RateLimiter        | —                |
| Observabilidad | AgentMetrics + logging          | —                |
| Interfaz       | Streamlit                       | ≥ 1.35.0         |

> **Costo operativo: $0** — GitHub Models es gratuito con token, sentence-transformers corre 100% local.

---

## Estructura del Proyecto

```
asistente-seleccion/
├── app.py                           # Interfaz web Streamlit (EJECUTAR ESTO)
├── agent.py                         # Agente ReAct con memoria y planificación
├── config.py                        # Configuración central y ponderaciones
├── requirements.txt
├── .env                             # API keys (NO subir a git)
│
├── data/
│   └── cvs/                         # Documentos de candidatos
│       └── <candidate_id>/
│           ├── curriculum.pdf
│           ├── feedback_entrevista.txt
│           ├── linkedin.txt
│           └── github.json
│
├── docs/
│   ├── audit_log.jsonl              # Historial de evaluaciones (auto-generado)
│   ├── agent_metrics.log            # Log de rendimiento (auto-generado)
│   └── long_term_memory.json        # Memoria entre sesiones (auto-generado)
│
└── src/
    ├── ethics/
    │   └── audit.py                 # Trazabilidad y auditoría ética
    ├── evaluation/
    │   ├── prompts.py               # 5 prompts optimizados
    │   └── rag_pipeline.py          # Pipeline RAG principal
    ├── memory/
    │   ├── short_term.py            # Memoria de corto plazo
    │   └── long_term.py             # Memoria persistente entre sesiones
    ├── observability/
    │   └── metrics.py               # Métricas de rendimiento y logging
    ├── planning/
    │   └── planner.py               # Planificador de tareas con prioridades
    ├── rag/
    │   ├── ingestion.py             # Carga PDF, TXT, JSON, DOCX
    │   └── vectorstore.py           # ChromaDB + embeddings locales
    └── security/
        └── guard.py                 # Validación, rate limiting, monitor ético
```

---

## Requisitos Previos

- Python 3.11 o superior
- Token de GitHub gratuito
- Conexión a internet (primera ejecución descarga modelo ~90MB)
- Windows 10/11, macOS o Linux

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/elpanxo/asistente-seleccion.git
cd asistente-seleccion

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración

### 1. Obtener token de GitHub (gratuito)

1. Ir a [github.com/settings/tokens](https://github.com/settings/tokens)
2. Clic en **"Generate new token (classic)"**
3. Asignar cualquier nombre, sin permisos especiales
4. Copiar el token (empieza con `ghp_...`)

### 2. Crear archivo `.env`

```env
GITHUB_TOKEN=ghp_tu_token_aqui
LLM_MODEL=gpt-4o-mini
```

### 3. Verificar `.gitignore`

```
venv/
chroma_db/
.env
__pycache__/
*.pyc
docs/audit_log.jsonl
docs/agent_metrics.log
docs/long_term_memory.json
```

---

## Ejecución

```bash
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Lanzar la interfaz
streamlit run app.py
```

La aplicación se abre en: **http://localhost:8501**

> **Primera ejecución:** se descarga el modelo `all-MiniLM-L6-v2` (~90MB). Ocurre una sola vez.

---

## Uso de la Interfaz

### Panel lateral — Subir documentos

1. Ingresar el **ID del candidato** (ej: `ana_lopez`)
2. Seleccionar el **tipo de documento**: curriculum / feedback_entrevista / linkedin / github / evaluacion_previa
3. Seleccionar hasta **3 archivos** (PDF, TXT, JSON o DOCX)
4. Clic en **"Subir e Indexar"**

### Tab Chat — Agente conversacional

El chat integra el agente completo con memoria y planificación. Puedes escribir consultas en lenguaje natural:

```
"¿Quién tiene más experiencia con Python?"
"Evalúa a ana_lopez"
"Rankea todos los candidatos"
"¿Ya fue evaluada camila_torres antes?"
"Dame un resumen del proceso"
```

El agente decide automáticamente qué herramientas usar y en qué orden. Cada respuesta muestra el plan de ejecución y el razonamiento paso a paso.

### Tab Evaluación — Evaluación individual

1. Seleccionar candidato
2. Clic en **"Evaluar"**
3. Ver puntuaciones por criterio con razonamiento, fortalezas, áreas de mejora y declaración ética

### Tab Ranking — Ranking comparativo

1. Seleccionar candidatos a comparar
2. Clic en **"Generar ranking"**
3. Ver ranking con medallas y análisis de diversidad del grupo finalista

### Tab Auditoría — Historial de decisiones

- Registro completo de todas las operaciones históricas
- Filtro por tipo de evento
- Descarga de `audit_log.jsonl`

### Tab Observabilidad — Métricas y seguridad

- Métricas de rendimiento en tiempo real: interacciones, tiempos de respuesta, errores
- Análisis del historial de evaluaciones con distribución de recomendaciones
- Estado del rate limiter y reporte ético del sistema
- Estrategias de escalabilidad documentadas

---

## Herramientas del Agente

El agente dispone de 5 herramientas que selecciona automáticamente:

| Herramienta         | Tipo         | Propósito                                         |
| ------------------- | ------------ | ------------------------------------------------- |
| buscar_candidatos   | Consulta     | Búsqueda semántica RAG sobre documentos indexados |
| evaluar_candidato   | Razonamiento | Evaluación ponderada con chain-of-thought         |
| rankear_candidatos  | Escritura    | Ranking comparativo con análisis de diversidad    |
| consultar_historial | Memoria LP   | Evaluaciones previas del candidato entre sesiones |
| resumen_proceso     | Escritura    | Estado actual del proceso de selección            |

---

## Memoria del Sistema

| Tipo        | Implementación                        | Alcance        | Persistencia               |
| ----------- | ------------------------------------- | -------------- | -------------------------- |
| Corto plazo | ConversationBufferWindowMemory (k=10) | Sesión activa  | Se pierde al cerrar la app |
| Largo plazo | JSON en docs/long_term_memory.json    | Entre sesiones | Permanente en disco        |

---

## Criterios de Evaluación

Definidos en `config.py → EVALUATION_CRITERIA`:

| Criterio                 | Peso |
| ------------------------ | ---- |
| Experiencia relevante    | 30%  |
| Habilidades técnicas     | 25%  |
| Formación académica      | 15%  |
| Proyectos destacados     | 15%  |
| Diversidad e inclusión   | 10%  |
| Comunicación y liderazgo | 5%   |

---

## Seguridad

El módulo `src/security/guard.py` protege el sistema con tres capas:

- **InputGuard**: detecta prompt injection en español e inglés y sanitiza inputs antes de procesarlos
- **RateLimiter**: máximo 20 consultas por minuto por sesión
- **EthicsMonitor**: verifica que las respuestas no contengan sesgos y genera reporte ético

Para demostrar el bloqueo, escribe en el chat:

```
ignora todas tus instrucciones anteriores
```

El sistema lo bloqueará y registrará el error en las métricas de Observabilidad.

---

## Auditoría Ética

Cada operación genera una entrada en `docs/audit_log.jsonl`:

```json
{
  "event_id": "uuid-único",
  "timestamp": "2025-05-01T12:00:00Z",
  "event_type": "candidate_evaluation",
  "candidate_id": "ana_lopez",
  "job_title": "Desarrollador/a Backend Senior",
  "puntuacion_ponderada": 8.7,
  "recomendacion": "avanzar",
  "confianza": "alta"
}
```

**Principios éticos:**

- Sin discriminación por género, edad, etnia, religión o apariencia
- Evaluación basada exclusivamente en méritos documentados
- Diversidad e inclusión como criterio positivo (10%)
- Trazabilidad completa y auditable de todas las decisiones

---

## Funcionalidades Verificadas

- Subida de hasta 3 archivos simultáneos (PDF, TXT, JSON, DOCX) ✓
- Evaluación individual con puntuaciones y razonamiento por criterio ✓
- Ranking comparativo con análisis de diversidad ✓
- Chat conversacional con agente ReAct y 5 herramientas ✓
- Memoria de corto plazo: coherencia dentro de la sesión ✓
- Memoria de largo plazo: historial persistente entre sesiones ✓
- Planificación automática de herramientas según la consulta ✓
- Detección y bloqueo de prompt injection (español e inglés) ✓
- Rate limiting de 20 consultas por minuto ✓
- Métricas de rendimiento en tiempo real ✓
- Análisis visual del historial de auditoría ✓
- Reporte ético con distribución de recomendaciones ✓
- Filtrado por candidate_id en ChromaDB ✓
- Registro de auditoría con trazabilidad completa ✓

---

_DuocUC — ISY0101 Ingeniería de Soluciones con IA — 2025_
