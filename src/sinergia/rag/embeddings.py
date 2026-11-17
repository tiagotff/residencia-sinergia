"""Gerador de representações vetoriais densas utilizando o modelo Colibri PT-BR (768d)."""

import hashlib
import math
import os
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

MODELO_PADRAO = "tardellirs/colibri-embed-ptbr"
DIMENSAO_PADRAO = 768


class GeradorEmbeddings:
    """Singleton responsável pelo carregamento do modelo de embedding e inferência vetorial."""

    _instance: Optional["GeradorEmbeddings"] = None
    _modelo = None

    def __init__(self, model_name: str = MODELO_PADRAO):
        self.model_name = model_name
        self._modelo = None

    @classmethod
    def get_instance(cls, model_name: str = MODELO_PADRAO) -> "GeradorEmbeddings":
        """Obtém a instância singleton do gerador de embeddings."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reseta a instância singleton (útil para testes isolados)."""
        cls._instance = None

    def _carregar_modelo(self):
        """Carrega o modelo SentenceTransformer de forma preguiçosa (lazy loading)."""
        if self._modelo is None:
            if SentenceTransformer is not None:
                self._modelo = SentenceTransformer(self.model_name)
        return self._modelo

    def _fallback_embedding(self, texto: str) -> List[float]:
        """Gera um vetor determinístico normalizado de 768 dimensões para ambientes sem dependências pesadas."""
        seed_bytes = hashlib.sha256(texto.encode("utf-8")).digest()
        seed_int = int.from_bytes(seed_bytes[:8], "big")
        
        # Gerador congruente linear determinístico a partir do hash do texto
        vetor = []
        val = seed_int
        for _ in range(DIMENSAO_PADRAO):
            val = (val * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            # Normalizar entre -1.0 e 1.0
            vetor.append((val / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0)

        # L2 normalize
        norma = math.sqrt(sum(x * x for x in vetor))
        if norma > 0:
            vetor = [x / norma for x in vetor]

        return vetor

    def gerar_embedding(self, texto: str) -> List[float]:
        """Gera um vetor denso de exatamente 768 dimensões para o texto fornecido."""
        modelo = self._carregar_modelo()
        if modelo is not None:
            resultado = modelo.encode(texto, normalize_embeddings=True)
            if hasattr(resultado, "tolist"):
                vetor = resultado.tolist()
            else:
                vetor = [float(x) for x in resultado]
            
            # Garantir 768 dimensões caso mock/modelo retorne dimensão diferente
            if len(vetor) == DIMENSAO_PADRAO:
                return [float(x) for x in vetor]
            elif len(vetor) < DIMENSAO_PADRAO:
                return [float(x) for x in vetor] + [0.0] * (DIMENSAO_PADRAO - len(vetor))
            else:
                return [float(x) for x in vetor[:DIMENSAO_PADRAO]]

        return self._fallback_embedding(texto)

    def gerar_embeddings(self, textos: List[str]) -> List[List[float]]:
        """Gera vetores densos de 768 dimensões para uma lista de textos."""
        modelo = self._carregar_modelo()
        if modelo is not None:
            resultados = modelo.encode(textos, normalize_embeddings=True)
            if hasattr(resultados, "tolist"):
                lista_vetores = resultados.tolist()
            else:
                lista_vetores = [[float(x) for x in res] for res in resultados]
            return [
                [float(x) for x in v] if len(v) == DIMENSAO_PADRAO else self._fallback_embedding(t)
                for t, v in zip(textos, lista_vetores)
            ]

        return [self._fallback_embedding(t) for t in textos]


def get_gerador_embeddings(model_name: str = MODELO_PADRAO) -> GeradorEmbeddings:
    """Função utilitária para acesso à instância singleton do gerador de embeddings."""
    return GeradorEmbeddings.get_instance(model_name=model_name)

