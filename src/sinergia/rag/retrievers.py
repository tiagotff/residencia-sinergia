"""Módulo de recuperação híbrida (Dense + Lexical) com fusão Reciprocal Rank Fusion (RRF)."""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from sinergia.rag.embeddings import MODELO_PADRAO, GeradorEmbeddings, get_gerador_embeddings
from sinergia.rag.interfaces import (
    ContextoRAG,
    DetalheScoreTrecho,
    EstatisticasEmbedding,
    MetadadosTrecho,
    ResultadoBuscaInstrumentada,
    TrechoRecuperado,
)


def calcular_rrf_detalhado(
    densos: List[TrechoRecuperado],
    lexicos: List[TrechoRecuperado],
    k: int = 60,
    top_k: int = 4,
) -> Tuple[List[TrechoRecuperado], List[DetalheScoreTrecho]]:
    """Combina ranqueamentos denso e léxico com RRF, preservando scores e ranks individuais."""
    scores: Dict[Any, float] = {}
    trechos_map: Dict[Any, TrechoRecuperado] = {}
    dense_scores: Dict[Any, float] = {}
    dense_ranks: Dict[Any, int] = {}
    lex_scores: Dict[Any, float] = {}
    lex_ranks: Dict[Any, int] = {}

    def _get_chave(t: TrechoRecuperado) -> Any:
        return t.id_trecho if t.id_trecho is not None else hash(t.conteudo)

    for rank_idx, trecho in enumerate(densos, start=1):
        chave = _get_chave(trecho)
        trechos_map[chave] = trecho
        dense_scores[chave] = float(trecho.score)
        dense_ranks[chave] = rank_idx
        scores[chave] = scores.get(chave, 0.0) + (1.0 / (k + rank_idx))

    for rank_idx, trecho in enumerate(lexicos, start=1):
        chave = _get_chave(trecho)
        if chave not in trechos_map:
            trechos_map[chave] = trecho
        lex_scores[chave] = float(trecho.score)
        lex_ranks[chave] = rank_idx
        scores[chave] = scores.get(chave, 0.0) + (1.0 / (k + rank_idx))

    chaves_ordenadas = sorted(scores.keys(), key=lambda ch: scores[ch], reverse=True)

    resultado: List[TrechoRecuperado] = []
    detalhes: List[DetalheScoreTrecho] = []
    for chave in chaves_ordenadas[:top_k]:
        trecho_orig = trechos_map[chave]
        rrf_val = float(scores[chave])
        resultado.append(
            TrechoRecuperado(
                id_trecho=trecho_orig.id_trecho,
                conteudo=trecho_orig.conteudo,
                score=rrf_val,
                metadados=trecho_orig.metadados,
            )
        )
        detalhes.append(
            DetalheScoreTrecho(
                id_trecho=trecho_orig.id_trecho,
                conteudo=trecho_orig.conteudo,
                metadados=trecho_orig.metadados,
                rrf_score=rrf_val,
                dense_score=dense_scores.get(chave, 0.0),
                dense_rank=dense_ranks.get(chave),
                lex_score=lex_scores.get(chave, 0.0),
                lex_rank=lex_ranks.get(chave),
            )
        )

    return resultado, detalhes


def calcular_rrf(
    densos: List[TrechoRecuperado],
    lexicos: List[TrechoRecuperado],
    k: int = 60,
    top_k: int = 4,
) -> List[TrechoRecuperado]:
    """Combina ranqueamentos denso e léxico utilizando Reciprocal Rank Fusion (RRF).

    Fórmula: RRF_Score(d) = sum(1 / (k + rank_i(d)))
    """
    # TODO: Analisar se essa está sendo a forma mais eficaz de similaridade e combinação:
    # 1. Avaliar se RRF puro (k=60) supera ou deve ser complementado por Relative Score Fusion (RSF / min-max normalization)
    #    ou Weighted Hybrid Search (ex: alpha * dense_score + (1-alpha) * lexical_score).
    # 2. Avaliar se o valor padrão de k=60 é ideal para conjuntos pequenos de candidatos (top 10),
    #    onde valores menores de k (ex: k=20 ou k=30) ou pesos diferenciados podem priorizar melhor a semântica.
    # 3. Considerar inclusão de Re-ranking cross-encoder (ex: bge-reranker ou flashrank) pós-recuperação inicial
    #    para validar se a eficácia e precisão do Top-K aumentam significativamente.
    resultado, _ = calcular_rrf_detalhado(densos=densos, lexicos=lexicos, k=k, top_k=top_k)
    return resultado


def _calcular_estatisticas_embedding(
    vetor: List[float],
    modelo: str = MODELO_PADRAO,
    tamanho_amostra: int = 12,
) -> EstatisticasEmbedding:
    """Computa norma L2, extremos, média e amostra das primeiras dimensões de um vetor."""
    if not vetor:
        return EstatisticasEmbedding(
            modelo=modelo,
            dimensoes=0,
            norma_l2=0.0,
            min_val=0.0,
            max_val=0.0,
            mean_val=0.0,
            amostra=[],
            vetor=[],
        )

    norma = math.sqrt(sum(x * x for x in vetor))
    return EstatisticasEmbedding(
        modelo=modelo,
        dimensoes=len(vetor),
        norma_l2=round(norma, 6),
        min_val=round(min(vetor), 6),
        max_val=round(max(vetor), 6),
        mean_val=round(sum(vetor) / len(vetor), 6),
        amostra=[round(float(x), 6) for x in vetor[:tamanho_amostra]],
        vetor=[float(x) for x in vetor],
    )


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

    def _buscar_memoria(
        self, query: str, id_base: int, vetor_query: Optional[List[float]] = None
    ) -> Tuple[List[TrechoRecuperado], List[TrechoRecuperado]]:
        """Realiza busca densa e léxica sobre os dados em memória aplicando filtros de base e vigência."""
        # Pré-filtro estrito: id_base e vigente
        candidatos = [
            (t, v)
            for t, v in self._trechos_memoria
            if t.metadados.id_base == id_base and t.metadados.vigente is True
        ]

        if not candidatos:
            return [], []

        if vetor_query is None:
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
        self, query: str, id_base: int, vetor_query: Optional[List[float]] = None
    ) -> Optional[Tuple[List[TrechoRecuperado], List[TrechoRecuperado]]]:
        """Tenta executar consulta no PostgreSQL com pgvector e Full-Text Search se disponível."""
        try:
            import psycopg
            with psycopg.connect(self.database_url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    if vetor_query is None:
                        vetor_query = self.gerador_embeddings.gerar_embedding(query)
                    vetor_str = "[" + ",".join(str(x) for x in vetor_query) + "]"

                    # 1. Busca densa pgvector com score de similaridade de cosseno
                    # TODO: Analisar eficácia da métrica de similaridade:
                    # - Atualmente usa distância de cosseno (<=>). Avaliar se Inner Product (<#>) com vetores normalizados
                    #   traz maior performance/eficácia de indexação (HNSW/IVFFlat).
                    # - Analisar o corte (threshold) mínimo de similaridade para descartar ruídos irrelevantes.
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
                    # TODO: Analisar eficácia do ranqueamento léxico:
                    # - Avaliar se ts_rank com dicionário 'portuguese' cobre sinônimos jurídicos/técnicos
                    #   ou se deve ser combinado com BM25 (ex: extensão pg_search/paradedb ou biblioteca bm25s).
                    # - Avaliar se o fallback ILIKE adiciona falsos positivos nos resultados léxicos.
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

    def buscar_com_telemetria(
        self, query: str, id_base: int, limite: int = 4
    ) -> ResultadoBuscaInstrumentada:
        """Executa busca híbrida retornando ContextoRAG e artefatos completos de telemetria."""
        limite_efetivo = limite if limite is not None else self.top_k

        t0_embed = time.perf_counter()
        vetor_query = self.gerador_embeddings.gerar_embedding(query)
        embed_ms = (time.perf_counter() - t0_embed) * 1000.0

        modelo_nome = getattr(self.gerador_embeddings, "model_name", MODELO_PADRAO)
        if not isinstance(modelo_nome, str):
            modelo_nome = MODELO_PADRAO
        estatisticas_embedding = _calcular_estatisticas_embedding(vetor_query, modelo=modelo_nome)

        t0_db = time.perf_counter()
        if self._trechos_memoria:
            densos, lexicos = self._buscar_memoria(query, id_base, vetor_query=vetor_query)
        else:
            try:
                resultado_banco = self._buscar_banco(query, id_base, vetor_query=vetor_query)
            except TypeError:
                resultado_banco = self._buscar_banco(query, id_base)

            if resultado_banco is not None:
                densos, lexicos = resultado_banco
            else:
                densos, lexicos = self._buscar_memoria(query, id_base, vetor_query=vetor_query)

        top_trechos, detalhes_trechos = calcular_rrf_detalhado(
            densos=densos,
            lexicos=lexicos,
            k=self.k_rrf,
            top_k=limite_efetivo,
        )
        top_trechos = top_trechos[:limite_efetivo]
        detalhes_trechos = detalhes_trechos[:limite_efetivo]
        db_rrf_ms = (time.perf_counter() - t0_db) * 1000.0

        contexto = ContextoRAG(
            query=query,
            id_base=id_base,
            trechos=top_trechos,
            total_encontrado=len(top_trechos),
        )
        return ResultadoBuscaInstrumentada(
            contexto=contexto,
            embed_ms=round(embed_ms, 2),
            db_rrf_ms=round(db_rrf_ms, 2),
            estatisticas_embedding=estatisticas_embedding,
            detalhes_trechos=detalhes_trechos,
        )


