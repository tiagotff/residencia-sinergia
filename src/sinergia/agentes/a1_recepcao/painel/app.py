"""Aplicação Streamlit de configuração, simulação e homologação A0 <-> A1."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from sinergia.agentes.a1_recepcao.avaliacao_metricas import avaliar_metricas_task_32
from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.gerador_sintetico import garantir_atestados_sinteticos
from sinergia.agentes.a1_recepcao.painel.cliente_conexao import (
    ClienteConexaoA0A1,
    ModoLLMConexao,
    ModoRepositorio,
    ModoTransporte,
    PerfilInspecao,
    RegistroTrocaProtocolo,
    validar_envelope_sinergia,
)
from sinergia.agentes.a1_recepcao.painel.diagnostico import (
    ItemDiagnostico,
    executar_diagnostico_completo,
    mascarar_segredo,
    sanitizar_url_conexao,
    verificar_endpoint_http,
)
from sinergia.agentes.a1_recepcao.painel.emulador_a0 import (
    EstadoCasoSimulado,
    aplicar_resultado_ao_caso,
    calcular_campos_corrigidos,
    carregar_fixtures_oficiais,
    comparar_com_esperado,
    criar_caso_padrao,
    gerar_preview_documento_png,
    limpar_uploads_temporarios,
    montar_tarefa_baixa_fisica,
    montar_tarefa_confirmar,
    montar_tarefa_conversar,
    montar_tarefa_processar_documento,
    resumir_diff_edicao,
    salvar_upload_temporario,
)
from sinergia.agentes.a1_recepcao.painel.tema import (
    VarianteBadge,
    aplicar_estilo_painel,
    renderizar_alerta_diff,
    renderizar_badge_origem_cid,
    renderizar_badge_status,
    renderizar_barra_status_topo,
    renderizar_card_campo_revisao,
    renderizar_card_kpi,
    renderizar_card_origem_cid,
    renderizar_stepper_estados,
)
from sinergia.agentes.a1_recepcao.repositorio import RepositorioA1Memoria
from sinergia.agentes.a1_recepcao.schemas import AtestadoExtraido, DivergenciaCampo
from sinergia.agentes.a1_recepcao.validador import (
    UFS_BRASIL,
    construir_quadro_revisao_atestado,
)

_MODELOS_GROQ_SUGERIDOS: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)


def _inicializar_estado() -> None:
    cfg_padrao = obter_config_a1()
    if "painel_casos" not in st.session_state:
        caso_inicial = criar_caso_padrao(105)
        st.session_state.painel_casos = {105: caso_inicial}
    if "painel_id_caso_ativo" not in st.session_state:
        st.session_state.painel_id_caso_ativo = 105
    if "painel_repo_memoria" not in st.session_state:
        st.session_state.painel_repo_memoria = RepositorioA1Memoria()
    if "painel_cache_idempotencia" not in st.session_state:
        st.session_state.painel_cache_idempotencia = {}
    if "painel_extracoes_pendentes" not in st.session_state:
        st.session_state.painel_extracoes_pendentes = {}
    if "painel_diagnostico" not in st.session_state:
        st.session_state.painel_diagnostico = []
    if "painel_metricas_sec10" not in st.session_state:
        st.session_state.painel_metricas_sec10 = None
    if "painel_ultimo_replay" not in st.session_state:
        st.session_state.painel_ultimo_replay = None
    if "painel_health_a1" not in st.session_state:
        st.session_state.painel_health_a1 = None
    if "painel_health_a0" not in st.session_state:
        st.session_state.painel_health_a0 = None
    if "painel_resultado_fixture" not in st.session_state:
        st.session_state.painel_resultado_fixture = None
    if "painel_suite_fixtures" not in st.session_state:
        st.session_state.painel_suite_fixtures = None
    if "painel_consulta_cid" not in st.session_state:
        st.session_state.painel_consulta_cid = None
    if "painel_cfg_base" not in st.session_state:
        st.session_state.painel_cfg_base = cfg_padrao


def _renderizar_sidebar() -> tuple[ClienteConexaoA0A1, dict[str, Any]]:
    cfg_base: A1Config = st.session_state.painel_cfg_base

    with st.sidebar:
        st.markdown("### Configuração Viva A0 ↔ A1")

        st.markdown("**1. Transporte de Conexão**")
        rotulos_transporte = {
            "in_process": "In-Process Python (AgenteA1)",
            "http_a1": "HTTP FastAPI A1 (POST /executar)",
            "http_a0": "Orquestrador A0 Real (POST /chat)",
        }
        modo_transporte: ModoTransporte = st.selectbox(
            "Modo de transporte",
            options=["in_process", "http_a1", "http_a0"],
            format_func=lambda m: rotulos_transporte[m],
            key="sb_modo_transporte",
        )

        retornar_modelo = False
        url_a1_http = "http://localhost:8001"
        url_a0_http = "http://localhost:8000"
        token_jwt_a0 = ""

        if modo_transporte == "in_process":
            retornar_modelo = st.toggle(
                "Retornar modelo Pydantic (ResultadoA1)",
                value=False,
                key="sb_retornar_modelo",
                help="Quando ativo, testa AgenteA1(..., retornar_modelo=True).",
            )
        elif modo_transporte == "http_a1":
            url_a1_http = st.text_input(
                "URL HTTP do Agente A1",
                value="http://localhost:8001",
                key="sb_url_a1_http",
            )
            if st.button("Testar /health do A1", key="sb_btn_health_a1"):
                st.session_state.painel_health_a1 = verificar_endpoint_http(
                    url_a1_http, "/health", nome="FastAPI A1"
                )
            if st.session_state.painel_health_a1 is not None:
                item: ItemDiagnostico = st.session_state.painel_health_a1
                st.caption(f"[{item.status.upper()}] {item.resumo}")
        else:
            url_a0_http = st.text_input(
                "URL da API do Agente A0",
                value="http://localhost:8000",
                key="sb_url_a0_http",
            )
            token_jwt_a0 = st.text_input(
                "Token JWT (Bearer)",
                value="",
                type="password",
                key="sb_token_jwt_a0",
            )
            if st.button("Testar /health do A0", key="sb_btn_health_a0"):
                st.session_state.painel_health_a0 = verificar_endpoint_http(
                    url_a0_http, "/health", nome="API Orquestrador A0"
                )
            if st.session_state.painel_health_a0 is not None:
                item_a0: ItemDiagnostico = st.session_state.painel_health_a0
                st.caption(f"[{item_a0.status.upper()}] {item_a0.resumo}")

        st.divider()
        st.markdown("**2. Repositório de Dados**")
        modo_repo: ModoRepositorio = st.radio(
            "Persistência do A1",
            options=["memoria", "postgres"],
            format_func=lambda x: (
                "Memória (RepositorioA1Memoria)"
                if x == "memoria"
                else "PostgreSQL 16 (DATABASE_URL)"
            ),
            key="sb_modo_repo",
        )
        database_url = cfg_base.database_url
        if modo_repo == "postgres":
            database_url = st.text_input(
                "DATABASE_URL",
                value=cfg_base.database_url,
                key="sb_database_url",
            )
            st.caption(f"Ativo: `{sanitizar_url_conexao(database_url)}`")
        elif st.button("Resetar banco em memória", key="sb_btn_reset_mem"):
            st.session_state.painel_repo_memoria = RepositorioA1Memoria()
            st.session_state.painel_cache_idempotencia = {}
            st.session_state.painel_extracoes_pendentes = {}
            st.success("Repositório em memória reiniciado.")

        st.divider()
        st.markdown("**3. Motor de LLM**")
        opcoes_llm: list[ModoLLMConexao] = (
            ["groq", "mock"] if cfg_base.groq_api_key.strip() else ["mock", "groq"]
        )
        modo_llm: ModoLLMConexao = st.radio(
            "Cliente LLM",
            options=opcoes_llm,
            format_func=lambda x: (
                "Groq Cloud Real" if x == "groq" else "MockLLM Determinístico (Offline)"
            ),
            key="sb_modo_llm",
        )
        modelos_lista = list(_MODELOS_GROQ_SUGERIDOS)
        if cfg_base.llm_model and cfg_base.llm_model not in modelos_lista:
            modelos_lista.insert(0, cfg_base.llm_model)
        llm_model = st.selectbox("LLM_MODEL", options=modelos_lista, key="sb_llm_model")
        llm_timeout = st.slider(
            "LLM_TIMEOUT_SEGUNDOS",
            min_value=5,
            max_value=90,
            value=int(cfg_base.llm_timeout_segundos),
            key="sb_llm_timeout",
        )
        st.caption(f"`GROQ_API_KEY` no .env: `{mascarar_segredo(cfg_base.groq_api_key)}`")
        groq_sessao = st.text_input(
            "Sobrescrever GROQ_API_KEY na sessão (opcional)",
            value="",
            type="password",
            key="sb_groq_override",
        )
        chave_groq_efetiva = groq_sessao.strip() or cfg_base.groq_api_key

        st.divider()
        st.markdown("**4. Regras de Negócio & LGPD**")
        politica_cid_modo = st.selectbox(
            "POLITICA_CID",
            options=["extrair_restrito", "nao_extrair"],
            index=0 if cfg_base.politica_extrair_cid else 1,
            key="sb_politica_cid",
        )
        limiar_confianca = st.slider(
            "LIMIAR_CONFIANCA_CAMPO",
            min_value=0.50,
            max_value=0.99,
            value=float(cfg_base.limiar_confianca_campo),
            step=0.05,
            key="sb_limiar_conf",
        )
        prazo_dias = st.slider(
            "PRAZO_ENTREGA_FISICA_DIAS_UTEIS",
            min_value=1,
            max_value=5,
            value=int(cfg_base.prazo_entrega_fisica_dias_uteis),
            key="sb_prazo_dias",
        )
        max_paginas_pdf = st.slider(
            "MAX_PAGINAS_PDF",
            min_value=1,
            max_value=15,
            value=int(cfg_base.max_paginas_pdf),
            key="sb_max_paginas",
        )

        st.divider()
        st.markdown("**5. API CID-11/CID-10 (OMS / WHO)**")
        who_icd_enabled = st.toggle(
            "WHO_ICD_ENABLED (Fallback API OMS)",
            value=cfg_base.who_icd_enabled,
            key="sb_who_icd_enabled",
            help=(
                "Quando o CID não existir em afastamento.cid, consulta a API externa "
                "da OMS (https://id.who.int) e salva em cache-aside no banco."
            ),
        )
        st.caption(
            f"Endpoint: `{cfg_base.who_icd_base_url}` ({cfg_base.who_icd_release_11}/{cfg_base.who_icd_language})"
        )
        st.caption(
            f"`WHO_ICD_CLIENT_ID`: `{mascarar_segredo(cfg_base.who_icd_client_id)}` · "
            f"`SECRET`: `{mascarar_segredo(cfg_base.who_icd_client_secret)}`"
        )

        config_ativa = replace(
            cfg_base,
            groq_api_key=chave_groq_efetiva,
            llm_model=llm_model,
            llm_timeout_segundos=float(llm_timeout),
            database_url=database_url,
            politica_extrair_cid=(politica_cid_modo == "extrair_restrito"),
            politica_cid_modo=politica_cid_modo,
            limiar_confianca_campo=float(limiar_confianca),
            prazo_entrega_fisica_dias_uteis=int(prazo_dias),
            max_paginas_pdf=int(max_paginas_pdf),
            who_icd_enabled=who_icd_enabled,
        )

        st.divider()
        if st.button("Limpar uploads temporários", key="sb_btn_limpar_uploads"):
            removidos = limpar_uploads_temporarios(config_ativa.atestados_storage_dir)
            st.info(f"{removidos} arquivo(s) removido(s) de uploads_painel/.")

        with st.expander("Gerar snippet .env (sem segredos)"):
            snippet = "\n".join(
                [
                    f"LLM_MODEL={config_ativa.llm_model}",
                    f"LLM_TIMEOUT_SEGUNDOS={int(config_ativa.llm_timeout_segundos)}",
                    f"POLITICA_EXTRAIR_CID={str(config_ativa.politica_extrair_cid).lower()}",
                    f"POLITICA_CID={config_ativa.politica_cid_modo}",
                    f"LIMIAR_CONFIANCA_CAMPO={config_ativa.limiar_confianca_campo}",
                    f"PRAZO_ENTREGA_FISICA_DIAS_UTEIS={config_ativa.prazo_entrega_fisica_dias_uteis}",
                    f"ATESTADOS_STORAGE_DIR={config_ativa.atestados_storage_dir}",
                    f"MAX_PAGINAS_PDF={config_ativa.max_paginas_pdf}",
                    f"WHO_ICD_ENABLED={str(config_ativa.who_icd_enabled).lower()}",
                    f"WHO_ICD_BASE_URL={config_ativa.who_icd_base_url}",
                ]
            )
            st.code(snippet, language="ini")

    cliente = ClienteConexaoA0A1(
        modo_transporte=modo_transporte,
        config=config_ativa,
        modo_repo=modo_repo,
        modo_llm=modo_llm,
        retornar_modelo=retornar_modelo,
        url_a1_http=url_a1_http,
        url_a0_http=url_a0_http,
        token_jwt_a0=token_jwt_a0,
        repositorio_memoria=st.session_state.painel_repo_memoria,
        cache_idempotencia=st.session_state.painel_cache_idempotencia,
        extracoes_pendentes=st.session_state.painel_extracoes_pendentes,
    )
    return cliente, {
        "modo_transporte": modo_transporte,
        "modo_repo": modo_repo,
        "modo_llm": modo_llm,
        "config": config_ativa,
        "retornar_modelo": retornar_modelo,
    }


def _obter_caso_ativo() -> EstadoCasoSimulado:
    casos: dict[int, EstadoCasoSimulado] = st.session_state.painel_casos
    id_ativo: int = st.session_state.painel_id_caso_ativo
    if id_ativo not in casos:
        id_ativo = next(iter(casos.keys()))
        st.session_state.painel_id_caso_ativo = id_ativo
    return casos[id_ativo]


def _limpar_widgets_edicao_caso(id_caso: int) -> None:
    prefixos = (
        f"hitl_tipo_{id_caso}",
        f"hitl_nome_{id_caso}",
        f"hitl_ini_{id_caso}",
        f"hitl_fim_{id_caso}",
        f"hitl_emissao_{id_caso}",
        f"hitl_cons_{id_caso}",
        f"hitl_num_{id_caso}",
        f"hitl_uf_{id_caso}",
        f"hitl_prof_nome_{id_caso}",
        f"hitl_mun_{id_caso}",
        f"hitl_cid_{id_caso}",
        f"toggle_editar_{id_caso}",
    )
    for chave in prefixos:
        st.session_state.pop(chave, None)


def _registrar_troca_no_caso(troca: RegistroTrocaProtocolo) -> None:
    caso = _obter_caso_ativo()
    atualizado = aplicar_resultado_ao_caso(caso, troca)
    st.session_state.painel_casos[atualizado.id_caso] = atualizado
    st.session_state.painel_ultimo_replay = None
    if troca.acao == "a1.processar_documento":
        _limpar_widgets_edicao_caso(atualizado.id_caso)


def _parse_data_iso(valor: Any) -> date | None:
    if isinstance(valor, date) and not isinstance(valor, datetime):
        return valor
    if not valor:
        return None
    try:
        return date.fromisoformat(str(valor).strip())
    except ValueError:
        return None


def _montar_atestado_limpo(extraido: dict[str, Any]) -> dict[str, Any]:
    ext_per = extraido.get("periodo") or {}
    ext_prof = extraido.get("profissional") or {}
    return {
        "tipo_documento": extraido.get("tipo_documento", "atestado_medico"),
        "legibilidade": "ok",
        "nome_no_documento": extraido.get("nome_no_documento"),
        "periodo": {
            "inicio": ext_per.get("inicio"),
            "fim": ext_per.get("fim"),
        },
        "data_emissao": extraido.get("data_emissao"),
        "profissional": {
            "conselho": ext_prof.get("conselho"),
            "numero": ext_prof.get("numero"),
            "uf": ext_prof.get("uf"),
            "nome": ext_prof.get("nome"),
        },
        "cid": extraido.get("cid"),
        "municipio": extraido.get("municipio"),
    }


def _obter_ou_gerar_revisao(
    caso: EstadoCasoSimulado,
    cfg: A1Config,
) -> dict[str, Any]:
    if isinstance(caso.ultima_revisao_colaborador, dict):
        return caso.ultima_revisao_colaborador
    ext_obj = AtestadoExtraido.model_validate(caso.ultimo_atestado_extraido)
    divs_obj = [DivergenciaCampo.model_validate(d) for d in caso.ultimas_divergencias]
    return construir_quadro_revisao_atestado(
        ext_obj,
        divs_obj,
        politica_extrair_cid=cfg.politica_extrair_cid,
    ).model_dump(mode="json")


def _aplicar_cenario_rapido(
    caso: EstadoCasoSimulado,
    *,
    amostra_label: str,
    inicio: date,
    dias: int,
    tipo_solicitacao: str = "atestado_medico",
) -> None:
    caso.inicio_informado = inicio
    caso.dias_informados = dias
    caso.tipo_solicitacao = tipo_solicitacao
    st.session_state[f"reg_ini_{caso.id_caso}"] = inicio
    st.session_state[f"reg_dias_{caso.id_caso}"] = dias
    st.session_state[f"reg_tipo_{caso.id_caso}"] = tipo_solicitacao
    st.session_state[f"sel_amostra_{caso.id_caso}"] = amostra_label
    st.rerun()


def _renderizar_aba_simulador(
    cliente: ClienteConexaoA0A1,
    meta: dict[str, Any],
) -> None:
    casos: dict[int, EstadoCasoSimulado] = st.session_state.painel_casos
    cfg: A1Config = meta["config"]

    pendente_sel = st.session_state.pop("_painel_selecionar_caso", None)
    if pendente_sel is not None and pendente_sel in casos:
        st.session_state.painel_id_caso_ativo = pendente_sel
        st.session_state["sim_select_caso"] = pendente_sel

    topo_c1, topo_c2, topo_c3 = st.columns([2.2, 1.0, 1.0])
    with topo_c1:
        ids_casos = sorted(casos.keys())
        id_selecionado = st.selectbox(
            "Caso ativo (desambiguação por id_caso / thread_id)",
            options=ids_casos,
            index=ids_casos.index(st.session_state.painel_id_caso_ativo),
            format_func=lambda cid: (
                f"Caso #{cid} ({casos[cid].thread_id}) — {casos[cid].estado_atual}"
            ),
            key="sim_select_caso",
        )
        if id_selecionado != st.session_state.painel_id_caso_ativo:
            st.session_state.painel_id_caso_ativo = id_selecionado
            st.rerun()
    with topo_c2:
        novo_id = max(ids_casos) + 1
        if st.button(f"+ Novo Caso (#{novo_id})", key="sim_btn_novo_caso"):
            casos[novo_id] = criar_caso_padrao(novo_id)
            st.session_state.painel_id_caso_ativo = novo_id
            st.session_state["_painel_selecionar_caso"] = novo_id
            st.rerun()
    with topo_c3:
        if st.button("Limpar histórico do caso", key="sim_btn_limpar_caso"):
            cid_atual = st.session_state.painel_id_caso_ativo
            casos[cid_atual] = criar_caso_padrao(cid_atual)
            _limpar_widgets_edicao_caso(cid_atual)
            st.session_state.painel_ultimo_replay = None
            st.rerun()

    caso = _obter_caso_ativo()

    amostras = garantir_atestados_sinteticos(cfg.atestados_storage_dir)
    opcoes_amostras = {
        f"{a.id_amostra} (PNG)": (a.arquivo_png, "image/png", a.sha256_png)
        for a in amostras
    }
    opcoes_amostras.update(
        {
            f"{a.id_amostra} (PDF)": (a.arquivo_pdf, "application/pdf", a.sha256_pdf)
            for a in amostras
        }
    )

    st.caption("Cenários rápidos de demonstração (pré-configuram registro inicial + amostra):")
    cq1, cq2, cq3, cq4 = st.columns(4)
    with cq1:
        if st.button(
            "Carla 105 (3 vs 5 dias)",
            key=f"cenario_carla_{caso.id_caso}",
            width="stretch",
        ):
            _aplicar_cenario_rapido(
                caso,
                amostra_label="atestado_carla_105 (PNG)",
                inicio=date(2026, 11, 9),
                dias=5,
            )
    with cq2:
        if st.button(
            "30 dias (INSS + NR-7)",
            key=f"cenario_30d_{caso.id_caso}",
            width="stretch",
        ):
            _aplicar_cenario_rapido(
                caso,
                amostra_label="atestado_inss_nr7_30dias (PNG)",
                inicio=date(2026, 11, 1),
                dias=30,
            )
    with cq3:
        if st.button(
            "Atestado com ruído",
            key=f"cenario_ruido_{caso.id_caso}",
            width="stretch",
        ):
            _aplicar_cenario_rapido(
                caso,
                amostra_label="atestado_ruido_leve_manual (PNG)",
                inicio=date(2026, 12, 14),
                dias=4,
            )
    with cq4:
        if st.button(
            "Borrado (Ilegível)",
            key=f"cenario_ileg_{caso.id_caso}",
            width="stretch",
        ):
            _aplicar_cenario_rapido(
                caso,
                amostra_label="atestado_borrado_ilegivel (PNG)",
                inicio=date(2026, 11, 9),
                dias=3,
            )

    if meta["modo_transporte"] == "http_a0":
        with st.container(border=True):
            st.markdown("#### Conexão Direta com Orquestrador A0 (`POST /chat`)")
            msg_a0 = st.text_input(
                "Mensagem para o endpoint do A0",
                value="Preciso enviar um atestado médico de 3 dias.",
                key="sim_a0_msg",
            )
            if st.button("Enviar para API do A0", key="sim_btn_enviar_a0"):
                resp_a0 = cliente.enviar_chat_a0(
                    msg_a0,
                    thread_id=caso.thread_id,
                    id_caso=caso.id_caso,
                )
                st.json(resp_a0)

    modo_visao = meta.get("modo_visao", "engenharia")
    if modo_visao == "engenharia":
        col_esq, col_dir = st.columns([1.22, 1.0], gap="large")
    else:
        col_esq = st.container()
        col_dir = None

    with col_esq:
        with st.expander("Registro Inicial do Caso (Anexo B — enviado pelo A0)", expanded=False):
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                nome_inf = st.text_input(
                    "Nome no cadastro", value=caso.nome_colaborador, key=f"reg_nome_{caso.id_caso}"
                )
                inicio_inf = st.date_input(
                    "Início informado",
                    value=caso.inicio_informado,
                    key=f"reg_ini_{caso.id_caso}",
                )
            with c_r2:
                tipo_inf = st.selectbox(
                    "Tipo de solicitação",
                    options=["atestado_medico", "declaracao_comparecimento", "outro"],
                    index=["atestado_medico", "declaracao_comparecimento", "outro"].index(
                        caso.tipo_solicitacao
                    )
                    if caso.tipo_solicitacao in {"atestado_medico", "declaracao_comparecimento", "outro"}
                    else 0,
                    key=f"reg_tipo_{caso.id_caso}",
                )
                dias_inf = st.number_input(
                    "Dias informados",
                    min_value=1,
                    max_value=180,
                    value=int(caso.dias_informados),
                    key=f"reg_dias_{caso.id_caso}",
                )
            caso.nome_colaborador = nome_inf
            caso.inicio_informado = inicio_inf
            caso.dias_informados = int(dias_inf)
            caso.tipo_solicitacao = tipo_inf

        st.markdown("#### 1. Conversa com o Colaborador (`a1.conversar`)")
        if caso.mensagens_chat:
            with st.container(border=True):
                for m in caso.mensagens_chat:
                    eh_colab = m["papel"] == "colaborador"
                    with st.chat_message("user" if eh_colab else "assistant"):
                        if not eh_colab:
                            st.caption(f"A1 → {m.get('destinatario', 'colaborador')} ({m.get('acao', '')})")
                        st.markdown(m["texto"])

        c_txt, c_btn = st.columns([3.0, 1.1])
        with c_txt:
            txt_conv = st.text_input(
                "Mensagem do colaborador no chat",
                value="Preciso enviar um atestado médico de 5 dias.",
                key=f"conv_txt_{caso.id_caso}",
            )
        with c_btn:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            if st.button("Enviar a1.conversar", key=f"btn_conv_{caso.id_caso}", width="stretch"):
                tarefa = montar_tarefa_conversar(caso, txt_conv)
                troca = cliente.enviar_tarefa(tarefa)
                _registrar_troca_no_caso(troca)
                st.rerun()

        msg_chat_nativo = st.chat_input(
            "Ou digite aqui e pressione Enter para enviar ao A1...",
            key=f"chat_in_{caso.id_caso}",
        )
        if msg_chat_nativo:
            tarefa_chat = montar_tarefa_conversar(caso, msg_chat_nativo)
            troca_chat = cliente.enviar_tarefa(tarefa_chat)
            _registrar_troca_no_caso(troca_chat)
            st.rerun()

        st.markdown("#### 2. Documento do Atestado (`a1.processar_documento`)")
        arquivo_up = st.file_uploader(
            "Upload real de atestado (PNG, JPG ou PDF)",
            type=["png", "jpg", "jpeg", "pdf"],
            key=f"up_doc_{caso.id_caso}",
        )
        amostra_sel = st.selectbox(
            "Ou selecione uma amostra sintética oficial (.storage/atestados/)",
            options=list(opcoes_amostras.keys()),
            key=f"sel_amostra_{caso.id_caso}",
        )

        caminho_prev = opcoes_amostras[amostra_sel][0]
        if arquivo_up is None:
            preview_bytes = gerar_preview_documento_png(caminho_prev)
            if preview_bytes is not None:
                with st.expander("Preview do documento selecionado", expanded=False):
                    st.image(preview_bytes, caption=Path(caminho_prev).name, width=360)

        if st.button(
            "Processar Documento (a1.processar_documento)",
            key=f"btn_proc_{caso.id_caso}",
            type="secondary" if caso.tem_documento else "primary",
        ):
            if arquivo_up is not None:
                uri, mime, sha256 = salvar_upload_temporario(
                    arquivo_up.name,
                    arquivo_up.getvalue(),
                    cfg.atestados_storage_dir,
                )
            else:
                uri, mime, sha256 = opcoes_amostras[amostra_sel]
            tarefa_doc = montar_tarefa_processar_documento(caso, uri, mime, sha256)
            troca_doc = cliente.enviar_tarefa(tarefa_doc)
            _registrar_troca_no_caso(troca_doc)
            st.rerun()

        if caso.ultimo_atestado_extraido is not None:
            ext = caso.ultimo_atestado_extraido
            revisao = _obter_ou_gerar_revisao(caso, cfg)
            legibilidade = str(revisao.get("legibilidade", ext.get("legibilidade", "ok")))

            st.markdown(f"#### 3. 🪪 {revisao.get('titulo', 'Espelho de Leitura da IA')}")
            with st.container(border=True):
                st.caption(revisao.get("instrucao_colaborador", ""))

                if legibilidade == "ilegivel":
                    st.error(
                        "Documento ilegível ou sem conteúdo suficiente. Nenhum dado foi gravado no "
                        "banco (`0` registros em `afastamento.atestado`). Envie uma foto ou PDF mais nítido."
                    )

                sp_doc, sp_dados = st.columns([0.85, 1.45], gap="medium")
                with sp_doc:
                    st.markdown("**Documento enviado**")
                    preview_env = (
                        gerar_preview_documento_png(caso.ultimo_arquivo_uri)
                        if caso.ultimo_arquivo_uri
                        else None
                    )
                    if preview_env is not None:
                        st.image(
                            preview_env,
                            caption=Path(str(caso.ultimo_arquivo_uri)).name,
                            width="stretch",
                        )
                    elif caso.ultimo_arquivo_uri:
                        st.caption(Path(str(caso.ultimo_arquivo_uri)).name)
                    else:
                        st.caption("Sem preview disponível.")

                with sp_dados:
                    st.markdown("**O que o Agente A1 leu no documento**")
                    if caso.ultimo_diagnostico_cid is not None:
                        st.markdown(
                            renderizar_card_origem_cid(caso.ultimo_diagnostico_cid),
                            unsafe_allow_html=True,
                        )
                    if caso.ultimas_divergencias:
                        for div in caso.ultimas_divergencias:
                            st.warning(
                                f"Divergência em **{div.get('campo')}**: "
                                f"no documento = `{div.get('no_documento')}` vs informado antes = `{div.get('informado')}`"
                            )
                    itens_revisao = list(revisao.get("itens") or [])
                    baixos = [
                        it["rotulo"]
                        for it in itens_revisao
                        if it.get("status") == "baixa_confianca"
                    ]
                    if baixos:
                        st.info(
                            f"Campos com confiança < {cfg.limiar_confianca_campo:.2f} "
                            f"(esvaziados para preenchimento manual): {', '.join(baixos)}"
                        )

                    grid_cols = st.columns(2)
                    for idx_item, item in enumerate(itens_revisao):
                        with grid_cols[idx_item % 2]:
                            st.markdown(
                                renderizar_card_campo_revisao(item),
                                unsafe_allow_html=True,
                            )

                if caso.aguardando_hitl and legibilidade == "ok":
                    st.divider()
                    atestado_direto = _montar_atestado_limpo(ext)
                    permite_direto = bool(revisao.get("permite_confirmar_direto", True))
                    act_c1, act_c2 = st.columns([1.4, 1.2])
                    with act_c1:
                        if st.button(
                            "✅ Validar e Confirmar Dados Lidos",
                            key=f"btn_conf_{caso.id_caso}",
                            type="primary",
                            disabled=not permite_direto,
                            width="stretch",
                        ):
                            t_conf = montar_tarefa_confirmar(caso, atestado_direto, [])
                            troca_conf = cliente.enviar_tarefa(t_conf)
                            _registrar_troca_no_caso(troca_conf)
                            st.rerun()
                    with act_c2:
                        modo_editar = st.toggle(
                            "✏️ Alterar / Corrigir informações lidas",
                            value=False,
                            key=f"toggle_editar_{caso.id_caso}",
                        )

                    if not permite_direto:
                        st.warning(
                            "Preencha ou corrija as datas obrigatórias do afastamento antes de confirmar."
                        )

                    if modo_editar:
                        ext_per = ext.get("periodo") or {}
                        ext_prof = ext.get("profissional") or {}
                        opcoes_tipo = ["atestado_medico", "declaracao_comparecimento", "outro"]
                        tipo_atual = ext.get("tipo_documento", "atestado_medico")
                        idx_tipo = opcoes_tipo.index(tipo_atual) if tipo_atual in opcoes_tipo else 0

                        opcoes_cons = ["", "CRM", "CRO", "CRP"]
                        cons_atual = str(ext_prof.get("conselho") or "").upper()
                        if cons_atual and cons_atual not in opcoes_cons:
                            opcoes_cons.append(cons_atual)
                        idx_cons = opcoes_cons.index(cons_atual) if cons_atual in opcoes_cons else 0

                        opcoes_uf = ["", *sorted(UFS_BRASIL)]
                        uf_atual = str(ext_prof.get("uf") or "").upper()
                        idx_uf = opcoes_uf.index(uf_atual) if uf_atual in opcoes_uf else 0

                        h_c1, h_c2 = st.columns(2)
                        with h_c1:
                            ed_tipo = st.selectbox(
                                "Tipo de documento",
                                options=opcoes_tipo,
                                index=idx_tipo,
                                key=f"hitl_tipo_{caso.id_caso}",
                            )
                            ed_nome = st.text_input(
                                "Paciente no documento",
                                value=ext.get("nome_no_documento") or "",
                                key=f"hitl_nome_{caso.id_caso}",
                            )
                            ed_ini = st.date_input(
                                "Início do afastamento",
                                value=_parse_data_iso(ext_per.get("inicio")),
                                format="DD/MM/YYYY",
                                key=f"hitl_ini_{caso.id_caso}",
                            )
                            ed_fim = st.date_input(
                                "Término do afastamento",
                                value=_parse_data_iso(ext_per.get("fim")),
                                format="DD/MM/YYYY",
                                key=f"hitl_fim_{caso.id_caso}",
                            )
                            if ed_ini is not None and ed_fim is not None:
                                dias_prev = (ed_fim - ed_ini).days + 1
                                if dias_prev > 0:
                                    st.caption(f"Duração calculada: **{dias_prev} dia(s)**")
                                else:
                                    st.error("A data de término está anterior à de início.")
                            ed_emissao = st.date_input(
                                "Data de emissão",
                                value=_parse_data_iso(ext.get("data_emissao")),
                                format="DD/MM/YYYY",
                                key=f"hitl_emissao_{caso.id_caso}",
                            )
                        with h_c2:
                            ed_conselho = st.selectbox(
                                "Conselho profissional",
                                options=opcoes_cons,
                                index=idx_cons,
                                key=f"hitl_cons_{caso.id_caso}",
                            )
                            ed_numero = st.text_input(
                                "Número do registro",
                                value=ext_prof.get("numero") or "",
                                key=f"hitl_num_{caso.id_caso}",
                            )
                            ed_uf = st.selectbox(
                                "UF do profissional",
                                options=opcoes_uf,
                                index=idx_uf,
                                key=f"hitl_uf_{caso.id_caso}",
                            )
                            ed_prof_nome = st.text_input(
                                "Nome do profissional",
                                value=ext_prof.get("nome") or "",
                                key=f"hitl_prof_nome_{caso.id_caso}",
                            )
                            ed_mun = st.text_input(
                                "Município",
                                value=ext.get("municipio") or "",
                                key=f"hitl_mun_{caso.id_caso}",
                            )
                            if cfg.politica_extrair_cid:
                                ed_cid = st.text_input(
                                    "CID-11 / CID-10 (opcional)",
                                    value=ext.get("cid") or "",
                                    key=f"hitl_cid_{caso.id_caso}",
                                    help="Validado primeiro em afastamento.cid e, se ausente, na API externa WHO ICD (cache-aside).",
                                )
                            else:
                                ed_cid = ""
                                st.caption("Política LGPD (`nao_extrair`): CID não coletado.")

                        editado = {
                            "tipo_documento": ed_tipo,
                            "legibilidade": "ok",
                            "nome_no_documento": ed_nome.strip() or None,
                            "periodo": {
                                "inicio": ed_ini.isoformat() if ed_ini is not None else None,
                                "fim": ed_fim.isoformat() if ed_fim is not None else None,
                            },
                            "data_emissao": (
                                ed_emissao.isoformat() if ed_emissao is not None else None
                            ),
                            "profissional": {
                                "conselho": ed_conselho.strip() or None,
                                "numero": ed_numero.strip() or None,
                                "uf": ed_uf.strip() or None,
                                "nome": ed_prof_nome.strip() or None,
                            },
                            "cid": ed_cid.strip() or None,
                            "municipio": ed_mun.strip() or None,
                        }
                        diffs_edicao = resumir_diff_edicao(ext, editado)
                        campos_corr = calcular_campos_corrigidos(ext, editado)
                        if diffs_edicao:
                            st.markdown(
                                renderizar_alerta_diff(diffs_edicao),
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                f"Alterações detectadas (`campos_corrigidos={campos_corr}`) "
                                "→ origem registrada será **MANUAL**."
                            )
                        else:
                            st.caption("Nenhuma alteração em relação à leitura da IA → origem **IA**.")

                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button(
                                "💾 Salvar Alterações e Confirmar (a1.confirmar)",
                                key=f"btn_conf_editado_{caso.id_caso}",
                                type="primary",
                                width="stretch",
                            ):
                                t_conf_ed = montar_tarefa_confirmar(caso, editado, campos_corr)
                                troca_conf_ed = cliente.enviar_tarefa(t_conf_ed)
                                _registrar_troca_no_caso(troca_conf_ed)
                                st.rerun()
                        with b2:
                            if st.button(
                                "Simular datas inválidas (fim < inicio)",
                                key=f"btn_inv_{caso.id_caso}",
                                width="stretch",
                            ):
                                inv = copy.deepcopy(editado)
                                inv["periodo"] = {"inicio": "2026-11-11", "fim": "2026-11-09"}
                                t_inv = montar_tarefa_confirmar(caso, inv, ["periodo.fim"])
                                troca_inv = cliente.enviar_tarefa(t_inv)
                                _registrar_troca_no_caso(troca_inv)
                                st.rerun()

        with st.expander(
            "4. Baixa Física pelo RH (`a1.registrar_baixa_fisica`)",
            expanded=(caso.estado_atual == "VALIDADO" and not caso.baixa_registrada),
        ):
            bf1, bf2 = st.columns(2)
            with bf1:
                id_at_baixa = st.number_input(
                    "id_atestado",
                    min_value=1,
                    value=int(caso.id_atestado_confirmado or 311),
                    key=f"baixa_id_{caso.id_caso}",
                )
                perfil_baixa = st.selectbox(
                    "Perfil chamador (teste RBAC)",
                    options=["rh", "admin", "colaborador"],
                    key=f"baixa_perfil_{caso.id_caso}",
                )
            with bf2:
                rec_por = st.text_input(
                    "Recebido por (login ou ID)",
                    value="rh.ana",
                    key=f"baixa_por_{caso.id_caso}",
                )
                data_baixa = st.text_input(
                    "Data/hora ISO 8601",
                    value=datetime.now(UTC).isoformat(),
                    key=f"baixa_dt_{caso.id_caso}",
                )
            usar_aliases = st.checkbox(
                "Usar aliases Guia v0.7 (`recebido_por` / `data_hora`)",
                value=False,
                key=f"baixa_alias_{caso.id_caso}",
            )
            if st.button("Registrar Baixa Física", key=f"btn_baixa_{caso.id_caso}"):
                t_baixa = montar_tarefa_baixa_fisica(
                    caso,
                    id_atestado=int(id_at_baixa),
                    recebido_por=rec_por,
                    data_hora_iso=data_baixa,
                    perfil=perfil_baixa,
                    usar_aliases_v07=usar_aliases,
                )
                troca_baixa = cliente.enviar_tarefa(t_baixa)
                _registrar_troca_no_caso(troca_baixa)
                st.rerun()

    if col_dir is not None:
        with col_dir:
            st.markdown("#### Raio-X do Protocolo `sinergia/1`")
            st.markdown(
                renderizar_stepper_estados(
                    caso.estado_atual,
                    caso.aguardando_hitl,
                    caso.baixa_registrada,
                ),
                unsafe_allow_html=True,
            )

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(
                    renderizar_card_kpi("Estado", caso.estado_atual, caso.thread_id),
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    renderizar_card_kpi(
                        "id_atestado",
                        str(caso.id_atestado_confirmado or "—"),
                        f"Prazo: {caso.prazo_entrega_fisica or '—'}",
                    ),
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    renderizar_card_kpi(
                        "Marcos Legais",
                        ", ".join(caso.prazos_legais) if caso.prazos_legais else "Nenhum",
                        f"{len(caso.trocas)} troca(s)",
                    ),
                    unsafe_allow_html=True,
                )

            if caso.trocas:
                ultima = caso.trocas[-1]
                if st.button(
                    "🔁 Reenviar último id_mensagem (Idempotência)",
                    key=f"btn_replay_{caso.id_caso}",
                ):
                    replay = cliente.enviar_tarefa(ultima.tarefa, eh_replay=True)
                    st.session_state.painel_ultimo_replay = {
                        "identico": replay.resultado == ultima.resultado,
                        "latencia_replay_ms": replay.latencia_ms,
                        "id_mensagem": ultima.id_mensagem_tarefa,
                        "resultado_replay": replay.resultado,
                    }

                if st.session_state.painel_ultimo_replay is not None:
                    rep = st.session_state.painel_ultimo_replay
                    if rep["identico"]:
                        st.success(
                            f"Idempotência 100% íntegra para `id_mensagem={rep['id_mensagem']}` "
                            f"(replay em {rep['latencia_replay_ms']} ms)."
                        )
                    else:
                        st.error("Divergência detectada no replay idempotente!")

                st.markdown("##### Histórico de Trocas")
                for idx, troca in enumerate(reversed(caso.trocas), start=1):
                    res = troca.resultado
                    status_res = str(res.get("status", ""))
                    variante = (
                        "ok"
                        if status_res == "OK"
                        else ("aviso" if status_res == "PRECISA_HUMANO" else "erro")
                    )
                    titulo_exp = (
                        f"#{len(caso.trocas) - idx + 1} · {troca.acao} · "
                        f"{status_res} ({troca.latencia_ms} ms)"
                    )
                    with st.expander(titulo_exp, expanded=(idx == 1)):
                        schema_ok = (
                            troca.schema_tarefa_valido and troca.schema_resultado_valido
                        )
                        badges = (
                            renderizar_badge_status(status_res, variante)
                            + " "
                            + renderizar_badge_status(
                                "Schema OK" if schema_ok else "Schema Inválido",
                                "ok" if schema_ok else "erro",
                            )
                        )
                        if troca.retornou_modelo_pydantic:
                            badges += " " + renderizar_badge_status("ResultadoA1", "neutro")
                        if troca.diagnostico_cid is not None:
                            badges += " " + renderizar_badge_origem_cid(troca.diagnostico_cid)
                        st.markdown(badges, unsafe_allow_html=True)

                        if troca.erros_schema:
                            st.error(" | ".join(troca.erros_schema))
                        if res.get("decisao"):
                            st.caption(f"Decisão: `{res['decisao']}`")
                        if res.get("eventos"):
                            st.caption(f"Eventos: `{res['eventos']}`")

                        tab_t, tab_r = st.tabs(
                            ["TAREFA (A0 -> A1)", "RESULTADO (A1 -> A0)"]
                        )
                        with tab_t:
                            st.json(troca.tarefa)
                        with tab_r:
                            st.json(troca.resultado)
            else:
                st.info(
                    "Envie uma conversa, documento ou fixture para inspecionar o protocolo `sinergia/1`."
                )


def _renderizar_aba_contratos(
    cliente: ClienteConexaoA0A1,
    meta: dict[str, Any],
) -> None:
    st.markdown("#### Laboratório de Contratos & Fixtures Oficiais (`Anexo D.2 / F`)")
    fixtures = carregar_fixtures_oficiais()
    if not fixtures:
        st.warning("Nenhuma fixture encontrada em `tests/a1_recepcao/fixtures/`.")
        return

    c1, c2 = st.columns([1.5, 1.0])
    with c1:
        chave_sel = st.selectbox(
            "Selecione uma fixture oficial",
            options=list(fixtures.keys()),
            key="lab_select_fixture",
        )
    with c2:
        if st.button("Rodar todas as 5 fixtures oficiais", key="lab_btn_suite"):
            relatorio_suite: list[dict[str, Any]] = []
            for nome_fx, (fx_t, fx_e) in fixtures.items():
                fx_copia = copy.deepcopy(fx_t)
                fx_copia["id_mensagem"] = f"lab-suite-{nome_fx}-{datetime.now(UTC).timestamp()}"
                troca_fx = cliente.enviar_tarefa(fx_copia)
                diffs_fx = comparar_com_esperado(
                    troca_fx.resultado,
                    {**fx_e, "em_resposta_a": fx_copia["id_mensagem"]},
                    modo_llm=meta["modo_llm"],
                )
                relatorio_suite.append(
                    {
                        "fixture": nome_fx,
                        "status": troca_fx.resultado.get("status"),
                        "schema_ok": troca_fx.schema_resultado_valido,
                        "latencia_ms": troca_fx.latencia_ms,
                        "divergencias": diffs_fx,
                    }
                )
            st.session_state.painel_suite_fixtures = relatorio_suite

    if st.session_state.painel_suite_fixtures is not None:
        st.dataframe(st.session_state.painel_suite_fixtures, width="stretch")

    tarefa_fx, esperado_fx = fixtures[chave_sel]
    col_ed, col_esp = st.columns(2)
    with col_ed:
        json_editado = st.text_area(
            "Envelope TAREFA editável (JSON)",
            value=json.dumps(tarefa_fx, ensure_ascii=False, indent=2),
            height=380,
            key=f"lab_editor_{chave_sel}",
        )
        if st.button("Validar e Executar TAREFA", key="lab_btn_exec"):
            try:
                tarefa_dict = json.loads(json_editado)
            except json.JSONDecodeError as exc:
                st.error(f"JSON inválido: {exc}")
            else:
                ok_t, errs_t = validar_envelope_sinergia(tarefa_dict, "TarefaA1")
                troca = cliente.enviar_tarefa(tarefa_dict)
                diffs = comparar_com_esperado(
                    troca.resultado,
                    {**esperado_fx, "em_resposta_a": tarefa_dict.get("id_mensagem")},
                    modo_llm=meta["modo_llm"],
                )
                st.session_state.painel_resultado_fixture = {
                    "ok_tarefa": ok_t,
                    "erros_tarefa": errs_t,
                    "troca": troca.model_dump(mode="json"),
                    "diffs_esperado": diffs,
                }
    with col_esp:
        st.markdown("**RESULTADO esperado (`.esperado.json`)**")
        st.json(esperado_fx)

    if st.session_state.painel_resultado_fixture is not None:
        res_fx = st.session_state.painel_resultado_fixture
        st.markdown("##### Resultado Obtido")
        if res_fx["diffs_esperado"]:
            st.warning("Diferenças vs `.esperado.json`: " + "; ".join(res_fx["diffs_esperado"]))
        else:
            st.success("RESULTADO 100% compatível com o `.esperado.json`!")
        st.json(res_fx["troca"]["resultado"])


def _renderizar_aba_inspetor(cliente: ClienteConexaoA0A1) -> None:
    st.markdown("#### Inspetor das 5 Tabelas `afastamento.*` & Governança LGPD (Art. 11)")
    perfil_visao: PerfilInspecao = st.radio(
        "Simular visão de acesso ao banco (RLS)",
        options=["A1_RH", "GESTOR_A2"],
        horizontal=True,
        format_func=lambda p: (
            "Perfil A1 / RH (acesso completo + cid_restrito)"
            if p == "A1_RH"
            else "Perfil Gestor / A2 (vw_atestado_sem_cid + RLS bloqueia cid_restrito)"
        ),
        key="insp_perfil_rls",
    )

    tabelas = cliente.inspecionar_tabelas_afastamento(perfil_acesso=perfil_visao)
    if perfil_visao == "GESTOR_A2":
        st.info(
            "Visão LGPD Art. 11 ativa: coluna `codigo_cid` removida e tabela "
            "`afastamento.cid_restrito` isolada por RLS."
        )

    nomes = (
        ("atestado", "1. afastamento.atestado"),
        ("atestado_arquivo", "2. afastamento.atestado_arquivo (SHA-256)"),
        ("atestado_campo", "3. afastamento.atestado_campo (IA vs MANUAL)"),
        ("baixa_fisica", "4. afastamento.baixa_fisica"),
        ("cid_restrito", "5. afastamento.cid_restrito (RLS Art. 11)"),
    )
    for chave, titulo in nomes:
        registros = tabelas.get(chave, [])
        with st.expander(f"{titulo} ({len(registros)} registro(s))", expanded=(chave in {"atestado", "cid_restrito"})):
            if registros:
                st.dataframe(registros, width="stretch")
            else:
                st.caption("Nenhum registro visível para este perfil.")

    st.divider()
    st.markdown("#### 6. Consulta & Catálogo CID (`afastamento.cid` + Fallback API Externa OMS)")
    with st.container(border=True):
        st.caption(
            "Consulte um código CID-11 ou CID-10 para verificar em tempo real se ele foi "
            "resolvido no **Banco Local** ou buscado na **API Externa da OMS (cache-aside)**."
        )
        c_cid_in, c_cid_btn = st.columns([2.2, 1.0], vertical_alignment="bottom")
        with c_cid_in:
            codigo_teste = st.text_input(
                "Código CID-11 ou CID-10",
                value="M54.5",
                placeholder="Ex.: M54.5, J06.9, ME84.2, CA07.0",
                key="insp_cid_consulta",
            )
        with c_cid_btn:
            if st.button(
                "Validar CID (Banco → API OMS)",
                key="insp_btn_consultar_cid",
                type="primary",
                width="stretch",
            ):
                st.session_state.painel_consulta_cid = cliente.consultar_cid(codigo_teste)

        if st.session_state.painel_consulta_cid is not None:
            st.markdown(
                renderizar_card_origem_cid(st.session_state.painel_consulta_cid),
                unsafe_allow_html=True,
            )

        with st.expander("Catálogo local `afastamento.cid`", expanded=False):
            filtro_cid = st.text_input(
                "Filtrar por código ou descrição",
                value="",
                placeholder="Ex.: ME84, M54, lombar",
                key="insp_filtro_cid",
            )
            cids_catalogo = cliente.listar_catalogo_cids(
                filtro=filtro_cid or None, limite=200
            )
            if cids_catalogo:
                st.dataframe(cids_catalogo, width="stretch")
            else:
                st.caption("Nenhum CID encontrado para o filtro informado.")


def _renderizar_aba_diagnostico_e_metricas(meta: dict[str, Any]) -> None:
    cfg: A1Config = meta["config"]
    st.markdown("#### Diagnóstico Pré-Voo (Ambiente A0 ↔ A1)")

    d1, d2, d3 = st.columns([1.4, 1.4, 1.0])
    with d1:
        url_a1_diag = st.text_input(
            "URL HTTP A1 para diagnóstico", value="http://localhost:8001", key="diag_url_a1"
        )
    with d2:
        url_a0_diag = st.text_input(
            "URL HTTP A0 para diagnóstico (opcional)", value="", key="diag_url_a0"
        )
    with d3:
        testar_groq = st.checkbox("Testar ping real na Groq", value=False, key="diag_ping_groq")
        testar_who = st.checkbox("Testar consulta real na API WHO", value=False, key="diag_ping_who")

    if st.button("Executar Diagnóstico Completo", key="diag_btn_run"):
        st.session_state.painel_diagnostico = executar_diagnostico_completo(
            config=cfg,
            url_a1_http=url_a1_diag,
            url_a0_http=url_a0_diag,
            testar_groq_rede=testar_groq,
            testar_who_rede=testar_who,
        )

    itens_diag: list[ItemDiagnostico] = st.session_state.painel_diagnostico
    if itens_diag:
        for item in itens_diag:
            with st.container(border=True):
                st.markdown(
                    f"{renderizar_badge_status(item.status.upper(), item.status)} **{item.nome}** — {item.resumo}",
                    unsafe_allow_html=True,
                )
                if item.detalhes:
                    with st.expander("Detalhes técnicos"):
                        st.json(item.detalhes)

    st.divider()
    st.markdown("#### Avaliador Oficial das Métricas da Seção 10 (Task 3.2)")
    m1, m2 = st.columns(2)
    with m1:
        modo_eval = st.radio(
            "Modo de LLM da bateria",
            options=["mock", "groq"],
            horizontal=True,
            key="eval_modo_llm",
        )
    with m2:
        fmts_eval = st.multiselect(
            "Formatos avaliados",
            options=["png", "pdf"],
            default=["png", "pdf"],
            key="eval_formatos",
        )

    if st.button("Executar Bateria da Seção 10", key="eval_btn_run"):
        if not fmts_eval:
            st.error("Selecione ao menos um formato ('png' ou 'pdf').")
        else:
            with st.spinner("Executando bateria completa de métricas da Seção 10..."):
                st.session_state.painel_metricas_sec10 = avaliar_metricas_task_32(
                    cfg=cfg,
                    modo_llm=modo_eval,  # type: ignore[arg-type]
                    formatos=tuple(fmts_eval),  # type: ignore[arg-type]
                )

    rel = st.session_state.painel_metricas_sec10
    if rel is not None:
        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        with c_k1:
            st.markdown(
                renderizar_card_kpi(
                    "Extração Bruta Exata",
                    f"{rel['taxa_extracao_bruta_exata_pct']:.1f}%",
                    f"{rel['legiveis_exatos']}/{rel['legiveis_execucoes_total']} (meta >= 80%)",
                ),
                unsafe_allow_html=True,
            )
        with c_k2:
            st.markdown(
                renderizar_card_kpi(
                    "Confirmados IA Direta",
                    f"{rel['taxa_confirmados_origem_ia_pct']:.1f}%",
                    f"{rel['confirmados_origem_ia']}/{rel['confirmados_total']}",
                ),
                unsafe_allow_html=True,
            )
        with c_k3:
            st.markdown(
                renderizar_card_kpi(
                    "Rejeição Ilegíveis",
                    f"{rel['taxa_rejeicao_ilegiveis_pct']:.1f}%",
                    f"{rel['ilegiveis_rejeitados']}/{rel['ilegiveis_execucoes_total']} (0 BD)",
                ),
                unsafe_allow_html=True,
            )
        with c_k4:
            st.markdown(
                renderizar_card_kpi(
                    "Baixa Física no Prazo",
                    f"{rel['taxa_baixa_nominal_pct']:.1f}%",
                    f"Estendido: {rel['cenario_estendido_baixa']['taxa_no_prazo_pct']:.1f}% (meta >= 90%)",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("##### Acurácia por Campo (`AtestadoExtraido`)")
        linhas_campos = [
            {
                "campo": campo,
                "acertos": info["acertos"],
                "total": info["total"],
                "taxa_pct": info["taxa_pct"],
            }
            for campo, info in rel["acuracia_por_campo"].items()
        ]
        st.dataframe(linhas_campos, width="stretch")


def main() -> None:
    """Ponto de entrada da aplicação Streamlit de homologação A0 <-> A1."""
    st.set_page_config(
        page_title="SinergIA — Painel de Integração A0 ↔ A1",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    aplicar_estilo_painel()
    _inicializar_estado()

    st.markdown(
        '<h1 class="sinergia-painel-titulo">SinergIA — Painel de Configuração e Homologação A0 ↔ A1</h1>'
        '<p class="sinergia-painel-sub">'
        "Simulação em tempo real do protocolo <code>sinergia/1</code>, pausa humana HITL, "
        "inspeção das 5 tabelas v0.7 (LGPD Art. 11) e diagnóstico pré-voo."
        "</p>",
        unsafe_allow_html=True,
    )

    cliente, meta = _renderizar_sidebar()
    caso_topo = _obter_caso_ativo()
    cfg_ativa: A1Config = meta["config"]

    rotulos_transp_curto = {
        "in_process": "In-Process",
        "http_a1": "HTTP A1",
        "http_a0": "HTTP A0",
    }
    var_llm: VarianteBadge = "ok" if meta["modo_llm"] == "groq" else "neutro"
    var_cid: VarianteBadge = "ok" if cfg_ativa.politica_extrair_cid else "neutro"
    who_cred_ok = bool(
        cfg_ativa.who_icd_client_id.strip() and cfg_ativa.who_icd_client_secret.strip()
    )
    if cfg_ativa.who_icd_enabled and who_cred_ok:
        rotulo_who, var_who = "Ativa (OMS)", "api"
    elif cfg_ativa.who_icd_enabled:
        rotulo_who, var_who = "Somente Banco", "neutro"
    else:
        rotulo_who, var_who = "Desativada", "neutro"
    var_estado: VarianteBadge = (
        "aviso"
        if caso_topo.aguardando_hitl
        else ("ok" if caso_topo.estado_atual == "VALIDADO" else "neutro")
    )
    itens_barra: list[tuple[str, str, VarianteBadge]] = [
        ("Transporte", rotulos_transp_curto[meta["modo_transporte"]], "neutro"),
        ("LLM", "Groq Cloud" if meta["modo_llm"] == "groq" else "MockLLM", var_llm),
        ("Banco", "PostgreSQL 16" if meta["modo_repo"] == "postgres" else "Memória", "neutro"),
        ("Política CID", cfg_ativa.politica_cid_modo, var_cid),
        ("API CID WHO", rotulo_who, var_who),  # type: ignore[arg-type]
        (f"Caso #{caso_topo.id_caso}", caso_topo.estado_atual, var_estado),
    ]
    st.markdown(renderizar_barra_status_topo(itens_barra), unsafe_allow_html=True)

    modo_visao = st.radio(
        "Modo de visualização",
        options=["engenharia", "apresentacao"],
        horizontal=True,
        format_func=lambda m: (
            "🛠️ Modo Engenharia (Split A0 ↔ A1)"
            if m == "engenharia"
            else "🧑‍💼 Modo Apresentação (Foco Colaborador)"
        ),
        key="painel_modo_visao",
    )
    meta["modo_visao"] = modo_visao

    aba_sim, aba_contratos, aba_insp, aba_diag = st.tabs(
        [
            "💬 Simulador A0 ↔ A1 (HITL)",
            "🧪 Contratos & Fixtures (sinergia/1)",
            "🗄️ Inspetor de Tabelas (5 Tabelas v0.7 + LGPD)",
            "🩺 Diagnóstico Pré-Voo & Métricas (Seção 10)",
        ]
    )
    with aba_sim:
        _renderizar_aba_simulador(cliente, meta)
    with aba_contratos:
        _renderizar_aba_contratos(cliente, meta)
    with aba_insp:
        _renderizar_aba_inspetor(cliente)
    with aba_diag:
        _renderizar_aba_diagnostico_e_metricas(meta)


if __name__ == "__main__":
    main()
