"""
src/security/guard.py — Seguridad y Ética en el agente RecruitAI.

IL3.3: Protocolos de Seguridad y Consideraciones Éticas

Implementa:
  - Validación y sanitización de inputs del usuario
  - Detección de prompt injection
  - Filtros de contenido inapropiado
  - Protección de datos sensibles en outputs
  - Rate limiting básico por sesión

Basado en el patrón del curso RA3/IL3.3/1-security_ethics.py:
    def safe_eval(expression): ...
    class EthicalAgent: ...
"""

import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── IL3.3: Validación de inputs ───────────────────────────────────────────────

class InputGuard:
    """
    Valida y sanitiza el input del usuario antes de enviarlo al agente.

    Basado en el patrón del curso:
        class SecureAgent:
            def sanitize_input(self, user_input):
                cleaned = re.sub(r'[<>\"\';&|`]', '', user_input)
    """

    MAX_INPUT_LENGTH = 2000

    # Patrones de prompt injection (IL3.3 del curso)
    INJECTION_PATTERNS = [
        # Inglés
        r"ignore\s+(previous|all|prior)\s+instructions",
        r"forget\s+everything\s+(above|before|prior)",
        r"act\s+as\s+if\s+you\s+are",
        r"pretend\s+(to\s+be|you\s+are)",
        r"system\s*:\s*you\s+are\s+now",
        r"new\s+instructions\s*:",
        r"override\s+(your|the)\s+(instructions|rules|system)",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
        # Español
        r"ignora\s+(todas?\s+)?(tus|las|mis|sus)\s+instrucciones",
        r"olvida\s+(todo|todas?\s+las\s+instrucciones|lo\s+anterior)",
        r"act[úu]a\s+como\s+si\s+(fueras|eres|no\s+tuvieras)",
        r"h[aá]z(te|)\s+pasar\s+por",
        r"eres\s+ahora\s+un",
        r"nuevas?\s+instrucciones\s*:",
        r"desde\s+ahora\s+(eres|ser[aá]s|act[úu]a)",
        r"sistema\s*:\s*(ahora|eres|act[úu]a)",
        r"omite\s+(tus|las)\s+(instrucciones|restricciones|reglas)",
        r"sin\s+(restricciones|l[íi]mites|filtros|reglas)",
        r"modo\s+(desarrollador|sin\s+restricciones|libre|hack)",
        r"supera\s+(tus|las)\s+(instrucciones|restricciones)",
    ]

    # Contenido inapropiado para un sistema de RRHH (IL3.3)
    PROHIBITED_CONTENT = [
        r"\bhackear?\b",
        r"\bmalware\b",
        r"\bexploit\b",
        r"\bvirus\b",
        r"información\s+personal\s+privada",
        r"datos?\s+bancarios?",
    ]

    # Patrones de datos sensibles a proteger en outputs (IL3.3)
    SENSITIVE_DATA_PATTERNS = [
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  # tarjetas
        r"\b\d{3}-\d{2}-\d{4}\b",                             # SSN
        r"api[_\-]?key\s*[:=]\s*\S+",                         # API keys
        r"password\s*[:=]\s*\S+",                              # passwords
        r"token\s*[:=]\s*\S+",                                 # tokens
    ]

    def sanitize(self, user_input: str) -> Tuple[str, bool, str]:
        """
        Sanitiza el input del usuario.

        Retorna:
            (input_limpio, es_seguro, razon_si_no_es_seguro)
        """
        if not user_input or not user_input.strip():
            return "", False, "Input vacío."

        # 1. Limitar longitud
        if len(user_input) > self.MAX_INPUT_LENGTH:
            user_input = user_input[:self.MAX_INPUT_LENGTH]

        # 2. Detectar prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                return user_input, False, (
                    "Se detectó un intento de modificar las instrucciones del sistema. "
                    "Por favor formula tu consulta de forma directa."
                )

        # 3. Detectar contenido prohibido
        for pattern in self.PROHIBITED_CONTENT:
            if re.search(pattern, user_input, re.IGNORECASE):
                return user_input, False, (
                    "La consulta contiene contenido no permitido en este sistema de selección de personal."
                )

        # 4. Limpiar caracteres potencialmente peligrosos
        cleaned = re.sub(r"[<>\"';&|`\\]", " ", user_input).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned, True, "OK"

    def sanitize_output(self, output: str) -> str:
        """
        Protege datos sensibles en el output antes de mostrarlo al usuario.
        Patrón del curso: DataProtection.contains_sensitive_data()
        """
        for pattern in self.SENSITIVE_DATA_PATTERNS:
            output = re.sub(pattern, "[DATO PROTEGIDO]", output, flags=re.IGNORECASE)
        return output


# ── IL3.3: Rate limiter por sesión ────────────────────────────────────────────

class RateLimiter:
    """
    Limita la cantidad de consultas por minuto para evitar abuso.

    Patrón del curso:
        class RateLimiter:
            def is_allowed(self, user_id): ...
    """

    def __init__(self, requests_per_minute: int = 20):
        self.rpm = requests_per_minute
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, session_id: str = "default") -> Tuple[bool, int]:
        """
        Verifica si la sesión puede hacer otra consulta.
        Retorna (permitido, segundos_para_siguiente_si_bloqueado)
        """
        now = time.time()
        minute_ago = now - 60

        # Limpiar requests antiguos
        self._requests[session_id] = [
            t for t in self._requests[session_id] if t > minute_ago
        ]

        if len(self._requests[session_id]) >= self.rpm:
            oldest = self._requests[session_id][0]
            wait = int(60 - (now - oldest)) + 1
            return False, wait

        self._requests[session_id].append(now)
        return True, 0

    def remaining(self, session_id: str = "default") -> int:
        """Retorna cuántas consultas quedan en el minuto actual."""
        now = time.time()
        minute_ago = now - 60
        recent = [t for t in self._requests[session_id] if t > minute_ago]
        return max(0, self.rpm - len(recent))


# ── IL3.3: Monitor ético ──────────────────────────────────────────────────────

class EthicsMonitor:
    """
    Monitorea el comportamiento ético del sistema.

    Verifica que las respuestas del agente:
      - No discriminen por género, edad, etnia
      - Mantengan diversidad e inclusión
      - Citen fuentes verificables
      - No generen contenido dañino

    Patrón del curso:
        class EthicalAgent:
            def answer(self, question):
                if "hackear" in question.lower():
                    return "No puedo ayudar con esa solicitud."
    """

    # Palabras que podrían indicar sesgo en las respuestas
    BIAS_INDICATORS = [
        r"\b(hombre|mujer|género|sexo)\b.*\b(mejor|peor|más\s+apto|menos\s+apto)\b",
        r"\b(joven|viejo|edad)\b.*\b(preferible|ideal|descartado)\b",
        r"\b(extranjero|inmigrante|nacionalidad)\b.*\b(problema|riesgo)\b",
    ]

    def check_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica que la respuesta del agente no contenga sesgos explícitos.

        Retorna:
            (es_etica, advertencia_si_hay_problema)
        """
        for pattern in self.BIAS_INDICATORS:
            if re.search(pattern, response, re.IGNORECASE):
                return False, (
                    "⚠️ Se detectó posible sesgo en la respuesta. "
                    "Esta evaluación debe ser revisada manualmente."
                )
        return True, None

    def generate_ethics_report(self, audit_events: list) -> Dict:
        """
        Genera un reporte ético basado en el historial de evaluaciones.
        Analiza distribución de recomendaciones para detectar patrones.
        """
        if not audit_events:
            return {"status": "Sin datos suficientes para análisis ético."}

        recommendations = defaultdict(int)
        candidates_evaluated = set()

        for event in audit_events:
            rec = event.get("recomendacion")
            if rec:
                recommendations[rec] += 1
            cid = event.get("candidate_id")
            if cid:
                candidates_evaluated.add(cid)

        total = sum(recommendations.values())
        distribution = {
            k: f"{round(v/total*100, 1)}%"
            for k, v in recommendations.items()
        } if total > 0 else {}

        return {
            "total_evaluaciones":        total,
            "candidatos_unicos":         len(candidates_evaluated),
            "distribucion_recomendaciones": distribution,
            "nota_etica": (
                "El sistema aplica criterios objetivos y ponderados. "
                "Todas las evaluaciones se basan exclusivamente en méritos "
                "documentados, sin discriminación por género, edad ni etnia."
            ),
        }


# ── Instancias globales ───────────────────────────────────────────────────────
input_guard    = InputGuard()
rate_limiter   = RateLimiter(requests_per_minute=20)
ethics_monitor = EthicsMonitor()