"""Ferramentas (tools) para os agentes especialistas do CrewAI (F1 a F6 do Agente A2)."""

from typing import Callable, Optional
from sinergia.rag.retrievers import BuscadorHibrido

try:
    from crewai.tools import tool
    from crewai.tools.base_tool import Tool

    if "__call__" not in Tool.__dict__:
        Tool.__call__ = lambda self, *args, **kwargs: self.run(*args, **kwargs)
except ImportError:
    # Fallback transparente quando crewai não estiver instalado no ambiente
    def tool(name_or_func: Optional[Callable] = None):
        if callable(name_or_func):
            return name_or_func

        def decorator(f: Callable):
            return f

        return decorator


_buscador_singleton: Optional[BuscadorHibrido] = None


def _get_buscador() -> BuscadorHibrido:
    """Retorna a instância singleton do Buscador Híbrido."""
    global _buscador_singleton
    if _buscador_singleton is None:
        _buscador_singleton = BuscadorHibrido()
    return _buscador_singleton


def reset_buscador_singleton() -> None:
    """Reseta a instância singleton (útil para testes)."""
    global _buscador_singleton
    _buscador_singleton = None


def consultar_base_conhecimento(query: str, id_base: int) -> str:
    """Consulta o subsistema RAG para uma base específica e retorna os trechos formatados."""
    buscador = _get_buscador()
    contexto = buscador.buscar(query=query, id_base=id_base)
    return contexto.formatar_para_prompt()


@tool("Busca Base de Pausas e Ergonomia (F1)")
def ferramenta_busca_pausas(query: str) -> str:
    """Consulta documentos normativos sobre pausas, descansos e ergonomia (NR-17)."""
    return consultar_base_conhecimento(query=query, id_base=1)


@tool("Busca Base de Direitos e Benefícios (F2)")
def ferramenta_busca_direitos(query: str) -> str:
    """Consulta legislação trabalhista (CLT) sobre direitos, benefícios e faltas justificadas."""
    return consultar_base_conhecimento(query=query, id_base=2)


@tool("Busca Base de Procedimentos e Treinamentos (F3)")
def ferramenta_busca_procedimentos(query: str) -> str:
    """Consulta procedimentos operacionais padrão (POPs) e cronogramas de capacitação da empresa."""
    return consultar_base_conhecimento(query=query, id_base=3)


@tool("Busca Base de Saúde Ocupacional (F4)")
def ferramenta_busca_saude_ocupacional(query: str) -> str:
    """Consulta normas e diretrizes do PCMSO, exames periódicos e programas de saúde do colaborador."""
    return consultar_base_conhecimento(query=query, id_base=4)


@tool("Busca Base de EPIs e Fichas de CA (F5/F6)")
def ferramenta_busca_epis(query: str) -> str:
    """Consulta fichas técnicas de EPIs, números de CA (NR-6), orientações de uso e prazos de substituição."""
    return consultar_base_conhecimento(query=query, id_base=5)

