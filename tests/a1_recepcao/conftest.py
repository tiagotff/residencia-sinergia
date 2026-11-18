"""Fixtures compartilhadas da suíte de testes do Agente 1 (`tests/a1_recepcao/`).

Fornece:
- `repositorio_memoria_a1`: instância limpa de `RepositorioA1Memoria` com a
  empresa fictícia canônica (Carla Souza, Campinas/SP, `rh.ana`, CRM/CRO/CRP).
- `mock_llm_a1`: instância de `MockLLM` para testes offline determinísticos.
- `catalogo_atestados_sinteticos`: garante a criação dos 12 pares de atestados
  sintéticos (`.png` + `.pdf`) com gabarito em `.storage/atestados/`.
"""

from __future__ import annotations

import pytest

from sinergia.agentes.a1_recepcao import (
    GabaritoAtestadoSintetico,
    MockLLM,
    RepositorioA1Memoria,
    garantir_atestados_sinteticos,
)


@pytest.fixture
def repositorio_memoria_a1() -> RepositorioA1Memoria:
    """Retorna um repositório em memória isolado e pré-populado."""
    return RepositorioA1Memoria()


@pytest.fixture
def mock_llm_a1() -> MockLLM:
    """Retorna um cliente `MockLLM` determinístico para testes offline."""
    return MockLLM()


@pytest.fixture(scope="session")
def catalogo_atestados_sinteticos() -> list[GabaritoAtestadoSintetico]:
    """Garante os 12 atestados sintéticos e retorna o catálogo com gabarito."""
    return garantir_atestados_sinteticos()
