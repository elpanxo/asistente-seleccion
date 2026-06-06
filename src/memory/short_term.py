"""
src/memory/short_term.py — Memoria de corto plazo del agente.

Basado en IL2.2 del curso: 2-memory-agent-advanced.ipynb
Usa ConversationBufferWindowMemory de LangChain para mantener
los últimos k intercambios de la sesión activa.

Se pierde al cerrar la sesión (corto plazo).
La memoria de largo plazo está en long_term.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importación directa para uso externo si se necesita
from langchain.memory import ConversationBufferWindowMemory


def create_short_term_memory(k: int = 10) -> ConversationBufferWindowMemory:
    """
    Crea una instancia de ConversationBufferWindowMemory.

    Parámetros:
        k: número de intercambios a recordar (default 10)

    Igual que en el notebook del curso:
        memory = ConversationBufferWindowMemory(
            k=1, memory_key="chat_history", return_messages=True
        )
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
    )