"""Módulo de recuperação híbrida (Dense + Lexical) com fusão Reciprocal Rank Fusion (RRF)."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from sinergia.rag.embeddings import GeradorEmbeddings, get_gerador_embeddings
from sinergia.rag.interfaces import ContextoRAG, MetadadosTrecho, TrechoRecuperado


def calcular_rrf(
    densos: List[TrechoRecuperado],
    lexicos: List[TrechoRecuperado],
    k: int = 60,
    top_k: int = 4,
) -> List[TrechoRecuperado]:
    """Combina ranqueamentos denso e léxico utilizando Reciprocal Rank Fusion (RRF).

    Fórmula: RRF_Score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[Any, float] = {}
    trechos_map: Dict[Any, TrechoRecuperado] = {}

    def _get_chave(t: TrechoRecuperado) -> Any:
        return t.id_trecho if t.id_trecho is not None else hash(t.conteudo)

    # Computar scores do ranqueamento denso
    for rank_idx, trecho in enumerate(densos, start=1):
        chave = _get_chave(trecho)
        trechos_map[chave] = trecho
        scores[chave] = scores.get(chave, 0.0) + (1.0 / (k + rank_idx))

    # Computar scores do ranqueamento léxico
    for rank_idx, trecho in enumerate(lexicos, start=1):
        chave = _get_chave(trecho)
        if chave not in trechos_map:
            trechos_map[chave] = trecho
        scores[chave] = scores.get(chave, 0.0) + (1.0 / (k + rank_idx))

    # Ordenar por maior pontuação RRF
    chaves_ordenadas = sorted(scores.keys(), key=lambda ch: scores[ch], reverse=True)

    resultado: List[TrechoRecuperado] = []
    for chave in chaves_ordenadas[:top_k]:
        trecho_orig = trechos_map[chave]
        # Criar nova instância com o score RRF calculado
        trecho_atualizado = TrechoRecuperado(
            id_trecho=trecho_orig.id_trecho,
            conteudo=trecho_orig.conteudo,
            score=scores[chave],
            metadados=trecho_orig.metadados,
        )
        resultado.append(trecho_atualizado)

    return resultado


def _similaridade_cosseno(v1: List[float], v2: List[float]) -> float:
    """Calcula similaridade de cosseno entre dois vetores de mesma dimensão."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class BuscadorHibrido:
    """Orquestrador de recuperação híbrida com pré-filtragem estrita por base e vigência."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        gerador_embeddings: Optional[GeradorEmbeddings] = None,
        k_rrf: int = 60,
        top_k: int = 4,
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://sinergia:sinergia@localhost:5432/sinergia"
        )
        self.gerador_embeddings = gerador_embeddings or get_gerador_embeddings()
        self.k_rrf = k_rrf
        self.top_k = top_k
        self._trechos_memoria: List[Tuple[TrechoRecuperado, List[float]]] = []

    def adicionar_trechos_memoria(self, trechos: List[TrechoRecuperado]) -> None:
        """Adiciona trechos à memória com vetorização automática para testes e fallback local."""
        for t in trechos:
            vetor = self.gerador_embeddings.gerar_embedding(t.conteudo)
            self._trechos_memoria.append((t, vetor))

    def _buscar_memoria(self, query: str, id_base: int) -> Tuple[List[TrechoRecuperado], List[TrechoRecuperado]]:
        """Realiza busca densa e léxica sobre os dados em memória aplicando filtros de base e vigência."""
        # Pré-filtro estrito: id_base e vigente
        candidatos = [
            (t, v)
            for t, v in self._trechos_memoria
            if t.metadados.id_base == id_base and t.metadados.vigente is True
        ]

        if not candidatos:
            return [], []

        vetor_query = self.gerador_embeddings.gerar_embedding(query)

        # 1. Busca Densa (cosseno)
        scores_densos = [
            (t, _similaridade_cosseno(vetor_query, v)) for t, v in candidatos
        ]
        scores_densos.sort(key=lambda x: x[1], reverse=True)
        densos = [
            TrechoRecuperado(
                id_trecho=item[0].id_trecho,
                conteudo=item[0].conteudo,
                score=float(item[1]),
                metadados=item[0].metadados,
            )
            for item in scores_densos
        ]

        # 2. Busca Léxica (overlap de termos)
        termos_query = set(query.lower().split())

        def _score_lexico(conteudo: str) -> float:
            palavras = conteudo.lower().split()
            if not palavras:
                return 0.0
            matches = sum(1 for p in palavras if p in termos_query)
            return matches / len(termos_query) if termos_query else 0.0

        scores_lexicos = [(t, _score_lexico(t.conteudo)) for t, _ in candidatos]
        scores_lexicos.sort(key=lambda x: x[1], reverse=True)
        lexicos = [
            TrechoRecuperado(
                id_trecho=item[0].id_trecho,
                conteudo=item[0].conteudo,
                score=float(item[1]),
                metadados=item[0].metadados,
            )
            for item in scores_lexicos
        ]

        return densos, lexicos

    def _buscar_banco(
        self, query: str, id_base: int
    ) -> Optional[Tuple[List[TrechoRecuperado], List[TrechoRecuperado]]]:
        """Tenta executar consulta no PostgreSQL com pgvector e Full-Text Search se disponível."""
        try:
            import psycopg
            with psycopg.connect(self.database_url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    vetor_query = self.gerador_embeddings.gerar_embedding(query)
                    vetor_str = "[" + ",".join(str(x) for x in vetor_query) + "]"

                    # 1. Busca densa pgvector com score de similaridade de cosseno
                    query_densa = """
                        SELECT t.id_trecho, t.conteudo, t.ordem,
                               d.id_documento, d.titulo, d.fonte, d.versao, d.id_base, d.vigente,
                               (1 - (t.embedding <=> %s::vector)) AS similaridade,
                               t.metadados
                        FROM rag.trecho t
                        JOIN rag.documento d ON d.id_documento = t.id_documento
                        WHERE d.id_base = %s AND d.vigente = true
                        ORDER BY t.embedding <=> %s::vector
                        LIMIT 10;
                    """
                    cur.execute(query_densa, (vetor_str, id_base, vetor_str))
                    linhas_densas = cur.fetchall()

                    densos = []
                    for row in linhas_densas:
                        meta_json = row[10] if len(row) > 10 and isinstance(row[10], dict) else {}
                        meta = MetadadosTrecho(
                            id_documento=row[3],
                            titulo=row[4],
                            fonte=row[5],
                            versao=row[6],
                            id_base=row[7],
                            ordem=row[2],
                            vigente=row[8],
                            breadcrumb=meta_json.get("breadcrumb"),
                            secao=meta_json.get("secoes_map", {}).get("h1") or meta_json.get("hierarquia", [None])[-1] if meta_json.get("hierarquia") else None,
                            pagina=meta_json.get("page_start"),
                            extra=meta_json,
                        )
                        score_sim = float(row[9]) if len(row) > 9 and row[9] is not None else 0.0
                        densos.append(
                            TrechoRecuperado(
                                id_trecho=row[0],
                                conteudo=row[1],
                                score=score_sim,
                                metadados=meta,
                            )
                        )

                    # 2. Busca léxica tsvector
                    query_lexica = """
                        SELECT t.id_trecho, t.conteudo, t.ordem,
                               d.id_documento, d.titulo, d.fonte, d.versao, d.id_base, d.vigente,
                               ts_rank(to_tsvector('portuguese', t.conteudo), plainto_tsquery('portuguese', %s)) AS rank_lex,
                               t.metadados
                        FROM rag.trecho t
                        JOIN rag.documento d ON d.id_documento = t.id_documento
                        WHERE d.id_base = %s AND d.vigente = true
                          AND (
                              to_tsvector('portuguese', t.conteudo) @@ plainto_tsquery('portuguese', %s)
                              OR t.conteudo ILIKE %s
                          )
                        ORDER BY rank_lex DESC
                        LIMIT 10;
                    """
                    termo_like = f"%{query}%"
                    cur.execute(query_lexica, (query, id_base, query, termo_like))
                    linhas_lexicas = cur.fetchall()

                    lexicos = []
                    for row in linhas_lexicas:
                        meta_json = row[10] if len(row) > 10 and isinstance(row[10], dict) else {}
                        meta = MetadadosTrecho(
                            id_documento=row[3],
                            titulo=row[4],
                            fonte=row[5],
                            versao=row[6],
                            id_base=row[7],
                            ordem=row[2],
                            vigente=row[8],
                            breadcrumb=meta_json.get("breadcrumb"),
                            secao=meta_json.get("secoes_map", {}).get("h1") or meta_json.get("hierarquia", [None])[-1] if meta_json.get("hierarquia") else None,
                            pagina=meta_json.get("page_start"),
                            extra=meta_json,
                        )
                        rank_lex = float(row[9]) if len(row) > 9 and row[9] is not None else 0.0
                        lexicos.append(
                            TrechoRecuperado(
                                id_trecho=row[0],
                                conteudo=row[1],
                                score=rank_lex,
                                metadados=meta,
                            )
                        )

                    return densos, lexicos
        except Exception:
            return None

    def buscar(self, query: str, id_base: int, limite: int = 4) -> ContextoRAG:
        """Executa busca híbrida combinada com RRF (k=60) e retorna o ContextoRAG limitado a 4 trechos com score."""
        limite_efetivo = limite if limite is not None else self.top_k

        # Se houver trechos em memória (modo mock/testes locais), prioriza a busca em memória
        if self._trechos_memoria:
            densos, lexicos = self._buscar_memoria(query, id_base)
        else:
            resultado_banco = self._buscar_banco(query, id_base)
            if resultado_banco is not None:
                densos, lexicos = resultado_banco
            else:
                densos, lexicos = self._buscar_memoria(query, id_base)

        top_trechos = calcular_rrf(densos, lexicos, k=self.k_rrf, top_k=limite_efetivo)
        top_trechos = top_trechos[:limite_efetivo]

        return ContextoRAG(
            query=query,
            id_base=id_base,
            trechos=top_trechos,
            total_encontrado=len(top_trechos),
        )

