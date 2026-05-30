"""
src/memory/short_term.py — Memoria de corto plazo del agente.

Mantiene el historial de la conversación activa en memoria RAM.
Se pierde al cerrar la sesión (corto plazo).
Compatible con LangChain >= 1.0 usando mensajes nativos.
"""

import sys
from pathlib import Path
from typing import List, Dict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ShortTermMemory:
    """
    Memoria de corto plazo con ventana deslizante de k mensajes.
    Guarda los últimos k intercambios para no exceder el contexto del LLM.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self._messages: List[BaseMessage] = []
        self._message_count = 0

    def add_interaction(self, human_input: str, ai_output: str) -> None:
        """Registra un intercambio humano-agente."""
        self._messages.append(HumanMessage(content=human_input))
        self._messages.append(AIMessage(content=ai_output))
        # Mantener solo los últimos k*2 mensajes (k intercambios)
        if len(self._messages) > self.k * 2:
            self._messages = self._messages[-(self.k * 2):]
        self._message_count += 1

    def get_history(self) -> List[BaseMessage]:
        """Retorna el historial de mensajes para el agente."""
        return self._messages.copy()

    def get_history_as_text(self) -> str:
        """Retorna el historial como texto legible."""
        if not self._messages:
            return "Sin historial en esta sesión."
        lines = []
        for msg in self._messages:
            role = "Reclutador" if isinstance(msg, HumanMessage) else "Agente"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Limpia la memoria de la sesión."""
        self._messages = []
        self._message_count = 0

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def is_empty(self) -> bool:
        return self._message_count == 0