"""Aplicação Streamlit SinergIA - RAG Demo com roteamento autônomo e telemetria técnica."""

import json
import os
from pathlib import Path

import streamlit as st

from sinergia.rag.chunking import chunking_documento_pdf
from sinergia.rag.ingestao import MAPA_MOCKS_PADRAO, mapear_mock_para_metadados
from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.pitch.ollama_service import (
    MAPA_BASES_FERRAMENTAS,
    GroqPitchService,
    OllamaPitchService,
)
from sinergia.rag.pitch.telemetry import (
    TelemetriaTurno,
    TemposTurno,
    calcular_visao_global,
)
from sinergia.rag.pitch.theme import (
    aplicar_tema_monocromatico,
    renderizar_badge_base,
    renderizar_barra_distribuicao,
    renderizar_card_kpi,
    renderizar_citacoes_html,
    renderizar_grafico_embedding_html,
)
from sinergia.rag.retrievers import BuscadorHibrido

DIRETORIO_MOCKS = Path(__file__).resolve().parent.parent / "data" / "mocks"

PERGUNTAS_RAPIDAS: list[dict[str, str]] = [
    {
        "key": "quick_nr17",
        "label": "Pausa digitação (NR-17)",
        "query": "Qual deve ser a pausa para atividades de digitação contínua?",
    },
    {
        "key": "quick_epi",
        "label": "Código CA 12345 (NR-6)",
        "query": "Qual a validade e aplicação do EPI com CA 12345?",
    },
    {
        "key": "quick_clt",
        "label": "Falta casamento (CLT)",
        "query": "Quantos dias de falta justificada tenho direito por casamento?",
    },
]


def _banco_disponivel_com_dados(database_url: str) -> bool:
    """Verifica se o PostgreSQL está acessível e possui trechos vigentes."""
    try:
        import psycopg
    except ImportError:
        return False

    try:
        with (
            psycopg.connect(database_url, connect_timeout=1) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                """
                SELECT count(*)
                FROM rag.trecho t
                JOIN rag.documento d ON d.id_documento = t.id_documento
                WHERE d.vigente = true;
                """
            )
            row = cur.fetchone()
            return bool(row and row[0] > 0)
    except (psycopg.Error, OSError):
        return False


def carregar_trechos_fallback_mocks(diretorio_mocks: Path | None = None) -> list[TrechoRecuperado]:
    """Carrega os documentos PDF de mocks em objetos TrechoRecuperado para busca em memória."""
    pasta = diretorio_mocks or DIRETORIO_MOCKS
    trechos: list[TrechoRecuperado] = []
    proximo_id = 1

    if pasta.exists():
        for nome_arquivo in sorted(MAPA_MOCKS_PADRAO.keys()):
            caminho_pdf = pasta / nome_arquivo
            if not caminho_pdf.exists():
                continue
            meta_doc = mapear_mock_para_metadados(nome_arquivo)
            chunks = chunking_documento_pdf(
                caminho_pdf=str(caminho_pdf),
                titulo_documento=meta_doc["titulo"],
            )
            for chunk in chunks:
                secao = (
                    chunk.metadados.get("secoes_map", {}).get("h1")
                    or (
                        chunk.metadados.get("hierarquia", [None])[-1]
                        if chunk.metadados.get("hierarquia")
                        else None
                    )
                )
                trechos.append(
                    TrechoRecuperado(
                        id_trecho=proximo_id,
                        conteudo=chunk.conteudo,
                        metadados=MetadadosTrecho(
                            id_documento=meta_doc["id_base"],
                            titulo=meta_doc["titulo"],
                            fonte=meta_doc["fonte"],
                            versao=meta_doc["versao"],
                            secao=secao,
                            pagina=chunk.metadados.get("page_start"),
                            id_base=meta_doc["id_base"],
                            ordem=chunk.ordem,
                            breadcrumb=chunk.breadcrumb,
                            vigente=True,
                            extra=chunk.metadados,
                        ),
                    )
                )
                proximo_id += 1

    if not trechos:
        # Fallback mínimo sintético caso o diretório de PDFs não esteja disponível
        for id_base, info in MAPA_BASES_FERRAMENTAS.items():
            trechos.append(
                TrechoRecuperado(
                    id_trecho=id_base,
                    conteudo=f"Documento normativo de referência para {info['nome']}: {info['descricao']}",
                    metadados=MetadadosTrecho(
                        id_documento=id_base,
                        titulo=info["nome"],
                        fonte="SinergIA Mocks",
                        versao="1.0",
                        secao="Geral",
                        pagina=1,
                        id_base=id_base,
                        ordem=1,
                        breadcrumb=info["nome"],
                        vigente=True,
                    ),
                )
            )

    return trechos


@st.cache_resource
def obter_buscador_pitch() -> BuscadorHibrido:
    """Inicializa o BuscadorHibrido usando PostgreSQL ou fallback automático dos mocks."""
    buscador = BuscadorHibrido()
    if not _banco_disponivel_com_dados(buscador.database_url):
        trechos_mock = carregar_trechos_fallback_mocks()
        buscador.adicionar_trechos_memoria(trechos_mock)
    return buscador


def obter_servico_ollama() -> OllamaPitchService:
    """Retorna uma instância do serviço Ollama local."""
    return OllamaPitchService(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


def obter_servico_groq() -> GroqPitchService:
    """Retorna uma instância do serviço Groq Cloud."""
    return GroqPitchService()


def inicializar_estado_sessao() -> None:
    """Inicializa as chaves de estado da sessão em um único ponto."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "turnos" not in st.session_state:
        st.session_state.turnos = []
    if "selected_msg_id" not in st.session_state:
        st.session_state.selected_msg_id = None
    if "next_msg_id" not in st.session_state:
        st.session_state.next_msg_id = 1
    if "pending_quick_prompt" not in st.session_state:
        st.session_state.pending_quick_prompt = None


def _selecionar_turno(id_turno: int | None) -> None:
    """Callback para alternar a seleção de resposta no Raio-X Nerd."""
    if st.session_state.selected_msg_id == id_turno:
        st.session_state.selected_msg_id = None
    else:
        st.session_state.selected_msg_id = id_turno


def _disparar_pergunta_rapida(texto: str) -> None:
    """Callback dos botões de pergunta rápida no cabeçalho."""
    st.session_state.pending_quick_prompt = texto


def executar_turno_rag(
    pergunta: str,
    modelo_selecionado: str | None,
    buscador: BuscadorHibrido,
    servico_llm: OllamaPitchService | GroqPitchService,
) -> TelemetriaTurno:
    """Executa roteamento autônomo, busca híbrida instrumentada e geração na LLM ativa."""
    decisao = servico_llm.decidir_base_autonomamente(pergunta, modelo=modelo_selecionado)
    resultado_rag = buscador.buscar_com_telemetria(
        query=decisao.query_busca,
        id_base=decisao.id_base,
        limite=4,
    )
    resposta_llm = servico_llm.gerar_resposta_fundamentada(
        pergunta=pergunta,
        contexto=resultado_rag.contexto,
        modelo=modelo_selecionado,
    )

    total_ms = (
        decisao.roteamento_ms
        + resultado_rag.embed_ms
        + resultado_rag.db_rrf_ms
        + resposta_llm.llm_ms
    )
    id_turno = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1

    return TelemetriaTurno(
        id_turno=id_turno,
        pergunta=pergunta,
        resposta=resposta_llm.resposta,
        citacoes=resposta_llm.citacoes,
        roteamento=decisao,
        tempos=TemposTurno(
            total_ms=round(total_ms, 2),
            roteamento_ms=decisao.roteamento_ms,
            embed_ms=resultado_rag.embed_ms,
            db_rrf_ms=resultado_rag.db_rrf_ms,
            llm_ms=resposta_llm.llm_ms,
        ),
        estatisticas_embedding=resultado_rag.estatisticas_embedding,
        detalhes_trechos=resultado_rag.detalhes_trechos,
        metricas_ollama=resposta_llm.metricas,
        prompt_enviado=resposta_llm.prompt_enviado,
    )


def _renderizar_visao_global(turnos: list[TelemetriaTurno]) -> None:
    """Renderiza o painel direito no modo padrão Visão Global da Sessão."""
    resumo = calcular_visao_global(turnos)

    topo_esq, topo_dir = st.columns([3, 1], vertical_alignment="center")
    with topo_esq:
        st.caption("VISÃO GLOBAL DA SESSÃO")
        st.subheader("Métricas agregadas do pipeline")
    with topo_dir:
        st.badge(f"{resumo.total_consultas} consultas", color="gray")

    k1, k2 = st.columns(2)
    with k1:
        st.html(
            renderizar_card_kpi(
                "Tempo médio total",
                f"{resumo.tempo_medio_total_s:.2f} s",
                "End-to-end por resposta",
            )
        )
        st.html(
            renderizar_card_kpi(
                "Latência média LLM",
                f"{resumo.latencia_media_llm_s:.2f} s",
                f"{resumo.velocidade_media_tok_s:.1f} tok/s em média",
            )
        )
    with k2:
        st.html(
            renderizar_card_kpi(
                "Latência média RAG",
                f"{resumo.latencia_media_rag_ms:.1f} ms",
                "Embedding + Busca híbrida",
            )
        )
        st.html(
            renderizar_card_kpi(
                "Tokens gerados",
                str(resumo.total_tokens_gerados),
                "Total acumulado na sessão",
            )
        )

    with st.container(border=True):
        st.markdown("**Roteamento autônomo por base (Tool Calling)**")
        barras_html = "".join(
            renderizar_barra_distribuicao(d.codigo, d.nome, d.contagem, d.percentual)
            for d in resumo.distribuicao_bases
        )
        st.html(barras_html)

    st.markdown("**Histórico de consultas (clique para abrir o Raio-X)**")
    if not turnos:
        st.caption("Envie uma pergunta no chat ou use os botões rápidos acima para iniciar.")
    else:
        for turno in reversed(turnos):
            rot = turno.roteamento
            tempo_s = turno.tempos.total_ms / 1000.0
            rotulo = f"[{rot.codigo_base}] {turno.pergunta[:55]} · {tempo_s:.2f}s"
            st.button(
                rotulo,
                key=f"history_btn_{turno.id_turno}",
                width="stretch",
                on_click=_selecionar_turno,
                args=(turno.id_turno,),
            )


def _renderizar_raio_x(turno: TelemetriaTurno) -> None:
    """Renderiza o painel direito no modo Raio-X Nerd para uma resposta selecionada."""
    topo_esq, topo_dir = st.columns([1, 1], vertical_alignment="center")
    with topo_esq:
        st.button(
            "Voltar para Visão Global",
            icon=":material/arrow_back:",
            key="btn_voltar_visao_global",
            on_click=_selecionar_turno,
            args=(None,),
        )
    with topo_dir:
        st.html(
            renderizar_badge_base(
                turno.roteamento.codigo_base,
                turno.roteamento.nome_base,
                ativo=True,
            )
        )

    t_cols = st.columns(4)
    t_cols[0].html(renderizar_card_kpi("Total", f"{turno.tempos.total_ms / 1000.0:.2f} s"))
    t_cols[1].html(renderizar_card_kpi("Embedding", f"{turno.tempos.embed_ms:.1f} ms"))
    t_cols[2].html(renderizar_card_kpi("DB + RRF", f"{turno.tempos.db_rrf_ms:.1f} ms"))
    t_cols[3].html(renderizar_card_kpi("LLM", f"{turno.tempos.llm_ms / 1000.0:.2f} s"))

    aba_chunks, aba_emb, aba_meta, aba_tool = st.tabs(
        ["1. Chunks & RRF", "2. Embedding (768d)", "3. Metadados", "4. Tool & Prompt"]
    )

    with aba_chunks:
        if not turno.detalhes_trechos:
            st.caption("Nenhum trecho retornado para esta consulta.")
        for idx, det in enumerate(turno.detalhes_trechos, start=1):
            with st.container(border=True):
                meta = det.metadados
                st.markdown(f"**#{idx} · {meta.titulo}** ({meta.secao or 'Geral'}) — `RRF: {det.rrf_score:.4f}`")
                if meta.breadcrumb:
                    st.caption(meta.breadcrumb)
                c1, c2 = st.columns(2)
                dense_rank_str = f"#{det.dense_rank}" if det.dense_rank is not None else "N/A"
                lex_rank_str = f"#{det.lex_rank}" if det.lex_rank is not None else "N/A"
                c1.html(
                    renderizar_card_kpi(
                        "Denso (Cosseno)", f"{det.dense_score:.4f}", f"Rank: {dense_rank_str}"
                    )
                )
                c2.html(
                    renderizar_card_kpi(
                        "Léxico (ts_rank)", f"{det.lex_score:.4f}", f"Rank: {lex_rank_str}"
                    )
                )
                st.text(det.conteudo)

    with aba_emb:
        emb = turno.estatisticas_embedding
        with st.container(border=True):
            st.markdown("**Vetor da consulta (Query Embedding)**")
            e1, e2 = st.columns(2)
            e1.html(renderizar_card_kpi("Modelo", emb.modelo))
            e2.html(
                renderizar_card_kpi(
                    "Dimensões / Norma L2", f"{emb.dimensoes}d", f"||v|| = {emb.norma_l2:.4f}"
                )
            )
            s1, s2, s3 = st.columns(3)
            s1.html(renderizar_card_kpi("min", f"{emb.min_val:.4f}"))
            s2.html(renderizar_card_kpi("max", f"{emb.max_val:.4f}"))
            s3.html(renderizar_card_kpi("mean", f"{emb.mean_val:.4f}"))

            st.caption("Primeiras 12 dimensões:")
            st.html(renderizar_grafico_embedding_html(emb.amostra))
            restantes = max(0, emb.dimensoes - len(emb.amostra))
            amostra_txt = ", ".join(f"{v:.4f}" for v in emb.amostra)
            st.code(f"[{amostra_txt}, ... (+{restantes} floats)]", language="json")

    with aba_meta:
        if not turno.detalhes_trechos:
            st.caption("Sem metadados para exibir.")
        for idx, det in enumerate(turno.detalhes_trechos, start=1):
            meta = det.metadados
            with st.container(border=True):
                st.markdown(f"**Chunk #{idx} · Linhagem**")
                linhagem = {
                    "id_trecho": det.id_trecho,
                    "id_documento": meta.id_documento,
                    "id_base": meta.id_base,
                    "versao": meta.versao,
                    "pagina": meta.pagina,
                    "vigente": meta.vigente,
                    "extra": meta.extra,
                }
                st.code(json.dumps(linhagem, ensure_ascii=False, indent=2), language="json")

    with aba_tool:
        ol = turno.metricas_ollama
        with st.container(border=True):
            st.markdown("**Tool escolhida & métricas de geração**")
            st.code(
                f"{turno.roteamento.tool_name}(query={turno.roteamento.query_busca!r}) "
                f"# modo={turno.roteamento.modo_roteamento} | modelo={ol.modelo}",
                language="python",
            )
            m1, m2 = st.columns(2)
            m1.html(renderizar_card_kpi("prompt_tokens", str(ol.prompt_eval_count)))
            m2.html(renderizar_card_kpi("eval_tokens", str(ol.eval_count)))
            m3, m4 = st.columns(2)
            m3.html(renderizar_card_kpi("velocidade", f"{ol.tokens_por_segundo:.1f} tok/s"))
            m4.html(renderizar_card_kpi("load_duration", f"{ol.load_duration_ms:.1f} ms"))
            st.markdown("**Prompt formatado enviado à LLM**")
            st.code(turno.prompt_enviado, language="markdown")


# --- Execução principal da página Streamlit ---

st.set_page_config(
    page_title="SinergIA - RAG Demo",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

aplicar_tema_monocromatico()
inicializar_estado_sessao()

buscador = obter_buscador_pitch()

# Cabeçalho limpo: título, toggle Local/Cloud (Groq), seletor de modelo e botões rápidos
hdr_title, hdr_provider, hdr_model, hdr_quick = st.columns(
    [3.5, 1.5, 2.2, 3.8], vertical_alignment="center"
)
with hdr_title:
    st.html(
        '<h1 class="sinergia-header-title">SinergIA - RAG Demo</h1>'
        '<p class="sinergia-header-subtitle">'
        "A LLM decide autonomamente qual base consultar. Selecione qualquer resposta para abrir o Raio-X técnico."
        "</p>"
    )
with hdr_provider:
    usar_cloud_groq = st.toggle(
        "Cloud (Groq)",
        value=False,
        key="provider_cloud_toggle",
        help="Alternar entre inferência Local (Ollama) e Cloud (Groq)",
    )

if usar_cloud_groq:
    servico_llm: OllamaPitchService | GroqPitchService = obter_servico_groq()
    instalados_local: list[str] = []
    ativos_cloud: list[str] = servico_llm.listar_modelos_disponiveis()
    modelos_disponiveis = servico_llm.listar_modelos_com_presets()
else:
    servico_llm = obter_servico_ollama()
    instalados_local = servico_llm.listar_modelos_disponiveis()
    ativos_cloud = []
    modelos_disponiveis = servico_llm.listar_modelos_com_presets()

if (
    "ollama_model_select" in st.session_state
    and st.session_state["ollama_model_select"] not in modelos_disponiveis
    and modelos_disponiveis
):
    st.session_state["ollama_model_select"] = modelos_disponiveis[0]

with hdr_model:
    modelo_selecionado = st.selectbox(
        "Modelo LLM",
        options=modelos_disponiveis,
        key="ollama_model_select",
        label_visibility="collapsed",
    )
    if (
        not usar_cloud_groq
        and instalados_local
        and modelo_selecionado
        and modelo_selecionado not in instalados_local
    ):
        st.caption(
            f":material/download: Preset ainda não baixado. Rode: `ollama pull {modelo_selecionado}`"
        )
    elif (
        usar_cloud_groq
        and ativos_cloud
        and modelo_selecionado
        and modelo_selecionado not in ativos_cloud
    ):
        st.caption(
            f":material/info: Modelo `{modelo_selecionado}` não consta ativo nesta chave Groq (verifique permissões no Groq Console)."
        )

with hdr_quick:
    q_cols = st.columns(len(PERGUNTAS_RAPIDAS))
    for col_q, item in zip(q_cols, PERGUNTAS_RAPIDAS):
        with col_q:
            st.button(
                item["label"],
                key=item["key"],
                width="stretch",
                on_click=_disparar_pergunta_rapida,
                args=(item["query"],),
            )

# Capturar entrada do chat ou botão rápido antes de renderizar as colunas
prompt_digitado = st.chat_input("Faça uma pergunta ao colaborador...", key="chat_input_pitch")
prompt_ativo = prompt_digitado or st.session_state.pending_quick_prompt
if st.session_state.pending_quick_prompt:
    st.session_state.pending_quick_prompt = None

if prompt_ativo:
    st.session_state.messages.append({"role": "user", "content": prompt_ativo})
    turno_novo = executar_turno_rag(
        pergunta=prompt_ativo,
        modelo_selecionado=modelo_selecionado,
        buscador=buscador,
        servico_llm=servico_llm,
    )
    st.session_state.turnos.append(turno_novo)
    st.session_state.messages.append(
        {"role": "assistant", "id": turno_novo.id_turno, "turno": turno_novo}
    )
    st.session_state.selected_msg_id = None

# Layout principal em duas colunas: Chat (esquerda) + Telemetria/Raio-X (direita)
col_chat, col_panel = st.columns([7, 5], gap="large")

with col_chat:
    for idx_msg, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            turno: TelemetriaTurno = msg["turno"]
            is_selected = st.session_state.selected_msg_id == turno.id_turno
            with st.chat_message("assistant"), st.container(border=True):
                c_badge, c_btn = st.columns([3, 1], vertical_alignment="center")
                with c_badge:
                    st.html(
                        renderizar_badge_base(
                            turno.roteamento.codigo_base,
                            turno.roteamento.nome_base,
                            ativo=is_selected,
                        )
                    )
                with c_btn:
                    st.button(
                        "Inspecionando" if is_selected else "Inspecionar no Raio-X",
                        key=f"inspect_btn_{turno.id_turno}_{idx_msg}",
                        width="stretch",
                        type="primary" if is_selected else "secondary",
                        on_click=_selecionar_turno,
                        args=(turno.id_turno,),
                    )

                st.markdown(turno.resposta)
                st.caption(
                    f"Tempo total: {turno.tempos.total_ms / 1000.0:.2f} s · Modelo: {turno.metricas_ollama.modelo}"
                )
                if turno.citacoes:
                    st.html(renderizar_citacoes_html(turno.citacoes))

with col_panel:
    turno_selecionado: TelemetriaTurno | None = next(
        (t for t in st.session_state.turnos if t.id_turno == st.session_state.selected_msg_id),
        None,
    )
    if turno_selecionado is None:
        _renderizar_visao_global(st.session_state.turnos)
    else:
        _renderizar_raio_x(turno_selecionado)
