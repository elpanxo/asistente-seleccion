# ◈ Asistente Inteligente de Selección de Personal

Sistema basado en **LLM + RAG** que automatiza la preselección de candidatos integrando fuentes internas (currículos, feedback de entrevistas) y externas (LinkedIn, GitHub), generando evaluaciones ponderadas, rankings comparativos y trazabilidad ética completa.

---

Agente RAG que:

1. Indexa documentos de candidatos (PDF, TXT, JSON, DOCX) en ChromaDB con embeddings locales
2. Recupera información relevante filtrando por candidato específico
3. Genera evaluaciones ponderadas con razonamiento explícito (chain-of-thought)
4. Produce rankings comparativos con análisis de diversidad
5. Responde preguntas en lenguaje natural citando fuentes
6. Registra cada decisión en `audit_log.jsonl` para trazabilidad

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│            CAPA 1 — INTERFAZ (app.py / Streamlit)               │
│     Chat · Evaluación · Ranking · Auditoría · Subida archivos   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│         CAPA 2 — ORQUESTACIÓN RAG (rag_pipeline.py)             │
│  extract_profile() · evaluate_candidate() · rank_candidates()   │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
┌──────────────▼────────────┐    ┌──────────────▼─────────────────┐
│  CAPA 3 — RECUPERACIÓN    │    │   CAPA 4 — GENERACIÓN          │
│  ingestion.py             │    │   prompts.py                   │
│  └─ RecursiveTextSplitter │    │   └─ GPT-4o-mini               │
│  vectorstore.py           │    │      (GitHub Models)           │
│  └─ ChromaDB              │    │   audit.py                     │
│  └─ sentence-transformers │    │   └─ audit_log.jsonl           │
└───────────────────────────┘    └────────────────────────────────┘
```

**Flujo de datos:**

```
Archivo → ingestion.py → chunks (800 chars) →
sentence-transformers → embeddings (384 dims) → ChromaDB

Consulta → filter_by_candidate() → contexto → prompt →
GPT-4o-mini → JSON evaluación → audit_log.jsonl → UI
```

---

## Stack Tecnológico

| Componente    | Tecnología                    | Versión          |
| ------------- | ----------------------------- | ---------------- |
| LLM           | GPT-4o-mini (GitHub Models)   | API gratuita     |
| Embeddings    | sentence-transformers (local) | all-MiniLM-L6-v2 |
| Vector Store  | ChromaDB                      | ≥ 0.5.0          |
| Framework RAG | LangChain                     | ≥ 0.3.0          |
| Interfaz      | Streamlit                     | ≥ 1.35.0         |
| Auditoría     | JSONL nativo                  | —                |

> **Costo operativo: $0** — GitHub Models es gratuito con token, sentence-transformers corre 100% local.

---

## Estructura del Proyecto

```
asistente-seleccion/
├── app.py                          # Interfaz web Streamlit (EJECUTAR ESTO)
├── config.py                       # Configuración central y ponderaciones
├── requirements.txt                # Dependencias Python
├── .env                            # API keys (NO subir a git)
├── .gitignore
├── README.md
│
│
├── docs/
│   └── audit_log.jsonl             # Registro de auditoría (auto-generado)
│
├── src/
│   ├── __init__.py
│   ├── ethics/
│   │   ├── __init__.py
│   │   └── audit.py                # Trazabilidad y auditoría ética
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── prompts.py              # 5 prompts optimizados
│   │   └── rag_pipeline.py         # Pipeline RAG principal
│   └── rag/
│       ├── __init__.py
│       ├── ingestion.py            # Carga PDF, TXT, JSON, DOCX
│       └── vectorstore.py          # ChromaDB + embeddings locales
│
└── chroma_db/                      # Base vectorial (auto-generado, no subir a git)
```

---

## Requisitos Previos

- Python 3.11 o superior
- Token de GitHub gratuito — ver [Configuración](#configuración)
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
3. Asignar cualquier nombre, sin permisos especiales necesarios
4. Copiar el token generado (empieza con `ghp_...`)

### 2. Crear archivo `.env`

Crear el archivo `.env` en la raíz del proyecto:

```env
GITHUB_TOKEN=ghp_tu_token_aqui
LLM_MODEL=gpt-4o-mini
```

### 3. Verificar `.gitignore`

Asegurarse de que `.gitignore` contenga:

```
venv/
chroma_db/
.env
__pycache__/
*.pyc
docs/audit_log.jsonl
```

---

## Ejecución

```bash
# Activar entorno virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Lanzar la interfaz web
streamlit run app.py
```

La aplicación se abre automáticamente en: **http://localhost:8501**

> **Primera ejecución:** al subir el primer archivo, se descarga el modelo `all-MiniLM-L6-v2` (~90MB). Ocurre una sola vez y queda en caché local.

---

## Uso de la Interfaz

### Panel lateral — Configurar y subir documentos

1. **Cargo:** ingresar título y descripción del cargo a cubrir
2. **ID del candidato:** identificador sin espacios (ej: `ana_lopez`)
3. **Tipo de documento:** curriculum / feedback_entrevista / linkedin / github / evaluacion_previa
4. **Archivo:** seleccionar PDF, TXT, JSON o DOCX
5. Clic en **"Subir e Indexar"**

Repetir para cada documento de cada candidato.

### Tab Chat — Preguntas en lenguaje natural

```
Ejemplos:
  "¿Qué candidatos tienen experiencia con Docker?"
  "¿Cuál es el nivel de inglés de ana_lopez?"
  "¿Quién ha contribuido a proyectos open source?"
```

### Tab Evaluación — Evaluación individual

1. Seleccionar candidato del menú
2. Clic en **"Evaluar"**
3. Ver puntuaciones por criterio, razonamiento, fortalezas y declaración ética

### Tab Ranking — Ranking comparativo

1. Seleccionar candidatos a comparar
2. Clic en **"Generar ranking"**
3. Ver ranking con medallas, puntuaciones y análisis de diversidad

### Tab Auditoría — Trazabilidad

- Registro completo de todas las operaciones
- Filtro por tipo de evento
- Descarga de `audit_log.jsonl`

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

Para ajustar los pesos, editar `config.py`. La suma debe ser siempre `1.0`.

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

**Principios éticos del sistema:**

- Sin discriminación por género, edad, etnia, religión o apariencia
- Evaluación basada exclusivamente en méritos documentados
- Diversidad e inclusión como criterio de valor positivo (10%)
- Trazabilidad completa y auditable de todas las decisiones

---

**Funcionalidades verificadas:**

- Subida de archivos PDF, TXT y JSON ✓
- Evaluación individual con puntuaciones por criterio ✓
- Ranking comparativo de múltiples candidatos ✓
- Chat con preguntas sobre habilidades específicas ✓
- Registro de auditoría con trazabilidad completa ✓
- Filtrado por candidate_id en ChromaDB ✓
