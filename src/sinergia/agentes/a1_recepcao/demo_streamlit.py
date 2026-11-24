"""Site de demonstração em Streamlit do Agente 1 (OCR + Formulário Manual + CID WHO).

Permite testar de forma interativa:
1. Envio de atestado via OCR (PNG/JPG/PDF ou PDF sintético de teste) com extração
   estruturada, validação de CID (PostgreSQL ``afastamento.cid`` -> fallback API
   externa ``https://id.who.int`` com cache-aside) e formulário de revisão/confirmação.
2. Formulário manual completo (fallback quando o OCR falha ou o arquivo é ilegível).
3. Consulta pontual e catálogo de CIDs salvos em ``afastamento.cid``.
4. Inspeção dos atestados persistidos no PostgreSQL Docker (com toggle LGPD para
   ``afastamento.cid_restrito``).

Execução:
    .venv/bin/streamlit run src/sinergia/agentes/a1_recepcao/demo_streamlit.py
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pymupdf
import streamlit as st

from sinergia.agentes.a1_recepcao.cliente_cid_who import ClienteCidWHO
from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.extrator_llm import MockLLM
from sinergia.agentes.a1_recepcao.ocr import processar_arquivo_atestado
from sinergia.agentes.a1_recepcao.repositorio import (
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    RepositorioA1Protocol,
)
from sinergia.agentes.a1_recepcao.servico import AgenteA1
from sinergia.agentes.a1_recepcao.validador import UFS_BRASIL

st.set_page_config(
    page_title="SinergIA — Demo Agente 1 (OCR + CID WHO)",
    page_icon=":material/clinical_notes:",
    layout="wide",
)

_SECOES = {
    "ocr": "Envio por OCR",
    "manual": "Formulário manual",
    "catalogo": "Consulta e catálogo CID",
    "atestados": "Atestados no banco",
}
_UFS_ORDENADAS = sorted(UFS_BRASIL)
_CONSELHOS_OPCOES = ["CRM", "CRO", "CRP"]
_TIPOS_DOC_LABELS = {
    "atestado_medico": "Atestado médico",
    "declaracao_comparecimento": "Declaração de comparecimento",
    "outro": "Outro",
}


def _verificar_postgres(database_url: str) -> tuple[bool, str]:
    if not database_url.strip():
        return False, "DATABASE_URL não configurada"
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=1) as conn:
            row = conn.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'afastamento' AND table_name = 'cid'"
            ).fetchone()
            if not row or int(row[0]) < 1:
                return False, "Schema afastamento ainda não aplicado"
            return True, "Conectado (PostgreSQL + pgvector)"
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return False, f"Indisponível ({type(exc).__name__})"
    except Exception as exc:  # noqa: BLE001 - psycopg.Error pode vir de import lazy
        return False, f"Indisponível ({type(exc).__name__})"


def _rapidocr_disponivel() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


@st.cache_resource(max_entries=4)
def _obter_repositorio_memoria() -> RepositorioA1Memoria:
    return RepositorioA1Memoria()


@st.cache_resource(max_entries=4)
def _obter_repositorio_postgres(database_url: str) -> RepositorioA1Postgres:
    return RepositorioA1Postgres(database_url)


def _obter_repositorio_ativo(usar_postgres: bool, database_url: str) -> RepositorioA1Protocol:
    if usar_postgres:
        return _obter_repositorio_postgres(database_url)
    return _obter_repositorio_memoria()


def _construir_registro_inicial(
    repo: RepositorioA1Protocol,
    *,
    id_colaborador: int = 42,
    id_caso: int = 900,
    inicio_informado: date | None = None,
    dias_informados: int | None = None,
    tem_documento: bool = True,
) -> dict[str, Any]:
    colab = repo.conferir_colaborador_ativo(id_colaborador)
    nome = colab.nome if colab else "Carla Souza"
    matricula = colab.matricula if colab else "00123"
    id_area = colab.id_area if colab else 2
    id_setor = colab.id_setor if colab else 7
    id_cargo = colab.id_cargo if colab else 15
    municipio = colab.municipio_nome if colab else "Campinas"
    uf = colab.sigla_uf if colab else "SP"
    return {
        "id_caso": id_caso,
        "thread_id": f"streamlit-{id_colaborador}-{id_caso}",
        "aberto_em": datetime.now(UTC).isoformat(),
        "canal": "chat",
        "id_colaborador": id_colaborador,
        "nome": nome,
        "matricula": matricula,
        "id_area": id_area,
        "id_setor": id_setor,
        "id_cargo": id_cargo,
        "jornada_inicio": "08:00",
        "jornada_fim": "17:00",
        "municipio": municipio,
        "uf": uf,
        "tipo_solicitacao": "atestado_medico",
        "inicio_informado": inicio_informado.isoformat() if inicio_informado else None,
        "dias_informados": dias_informados,
        "tem_documento": tem_documento,
    }


def _gerar_pdf_sintetico_atestado(
    *,
    destino: Path,
    nome: str,
    inicio: date,
    dias: int,
    cid: str | None,
    conselho: str = "CRM",
    numero: str = "123456",
    uf: str = "SP",
    municipio: str = "Campinas",
) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fim = inicio + timedelta(days=max(dias, 1) - 1)
    linha_cid = f"CID: {cid}\n" if cid else ""
    texto = (
        "ATESTADO MEDICO\n\n"
        f"Atesto para os devidos fins que a Sra. {nome} necessita de {dias} dias\n"
        f"de afastamento de suas atividades laborais no periodo de "
        f"{inicio:%d/%m/%Y} a {fim:%d/%m/%Y}.\n\n"
        f"{linha_cid}"
        f"{municipio} - {uf}, {inicio:%d/%m/%Y}.\n"
        f"Dr. Marcos Lima - {conselho}/{uf} {numero}\n"
    )
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=420)
    page.insert_text((48, 64), texto, fontsize=14)
    doc.save(destino)
    doc.close()
    return destino


def _renderizar_badge_diagnostico_cid(diag: dict[str, Any] | None) -> None:
    if not diag:
        return
    origem = diag.get("origem")
    cod = diag.get("codigo_cid") or ""
    desc = diag.get("descricao") or ""
    cap = diag.get("capitulo") or ""

    if origem == "BANCO_LOCAL":
        st.badge(
            f"CID {cod} validado no PostgreSQL (afastamento.cid)",
            icon=":material/database:",
            color="green",
        )
        if desc:
            st.caption(f"Descrição no banco: **{desc}** (capítulo `{cap}`)")
    elif origem == "API_WHO_EXTERNA_CACHE_ASIDE":
        st.badge(
            f"CID {cod} validado na API externa OMS e salvo em afastamento.cid",
            icon=":material/cloud_done:",
            color="blue",
        )
        if desc:
            st.caption(
                f"Descrição obtida da OMS ({diag.get('versao_cid', 'CID')}): "
                f"**{desc}** (capítulo `{cap}`)"
            )
    elif origem in {"API_WHO_NAO_ENCONTRADO", "SINTAXE_INVALIDA"}:
        st.badge(
            f"CID {cod} não encontrado na base local nem na API oficial da OMS",
            icon=":material/error:",
            color="red",
        )
        if diag.get("detalhe"):
            st.caption(str(diag["detalhe"]))
    elif origem in {"API_WHO_INDISPONIVEL", "NAO_ENCONTRADO_SEM_API"}:
        st.badge(
            f"CID {cod} ausente no banco local (API externa indisponível ou sem credenciais)",
            icon=":material/cloud_off:",
            color="orange",
        )
        if diag.get("detalhe"):
            st.caption(str(diag["detalhe"]))


def _inicializar_estado() -> None:
    padroes: dict[str, Any] = {
        "a1_demo_secao": "ocr",
        "a1_demo_ultimo_ocr": None,
        "a1_demo_ultimo_resultado_doc": None,
        "a1_demo_ultimo_diag_cid": None,
        "a1_demo_arquivo_atual": None,
        "a1_demo_caso_seq": 900,
    }
    for chave, valor in padroes.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


_inicializar_estado()
cfg_base = obter_config_a1()
pg_ok, pg_msg = _verificar_postgres(cfg_base.database_url)
rapidocr_ok = _rapidocr_disponivel()
who_cred_ok = bool(cfg_base.who_icd_client_id and cfg_base.who_icd_client_secret)

with st.sidebar:
    st.subheader("Ambiente do Agente 1", anchor=False)
    if pg_ok:
        st.badge("PostgreSQL Docker online", icon=":material/database:", color="green")
    else:
        st.badge("PostgreSQL offline", icon=":material/database_off:", color="orange")
    st.caption(pg_msg)

    if who_cred_ok:
        st.badge("API externa OMS configurada", icon=":material/cloud_done:", color="blue")
    else:
        st.badge("API externa OMS sem credenciais", icon=":material/key_off:", color="orange")
    st.caption(f"Fallback externo: `{cfg_base.who_icd_base_url}`")

    if rapidocr_ok:
        st.badge("OCR imagem + PDF pronto", icon=":material/document_scanner:", color="green")
    else:
        st.badge("OCR PDF nativo (RapidOCR ausente)", icon=":material/picture_as_pdf:", color="orange")

    st.divider()
    opcoes_repo = ["PostgreSQL Docker", "Memória local"]
    repo_escolha = st.segmented_control(
        "Persistência",
        options=opcoes_repo,
        default="PostgreSQL Docker" if pg_ok else "Memória local",
        selection_mode="single",
        required=True,
        key="cfg_repo_escolha",
    )
    usar_postgres = repo_escolha == "PostgreSQL Docker" and pg_ok
    if repo_escolha == "PostgreSQL Docker" and not pg_ok:
        st.warning(
            "O container `db` não respondeu em `localhost:5432`. "
            "Suba com `docker compose up -d` ou use o modo Memória local.",
            title="PostgreSQL inacessível",
        )

    opcoes_llm = ["MockLLM (offline)"]
    if cfg_base.groq_api_key.strip():
        opcoes_llm.append("ChatGroq (LLM real)")
    llm_escolha = st.segmented_control(
        "Motor de extração",
        options=opcoes_llm,
        default="MockLLM (offline)",
        selection_mode="single",
        required=True,
        key="cfg_llm_escolha",
    )

    extrair_cid = st.toggle(
        "Extrair e validar CID (LGPD restrito)",
        value=cfg_base.politica_extrair_cid,
        key="cfg_extrair_cid",
    )
    habilitar_fallback_who = st.toggle(
        "Consultar API externa OMS em cache-miss",
        value=cfg_base.who_icd_enabled,
        key="cfg_fallback_who",
    )

cfg_ativa: A1Config = replace(
    cfg_base,
    politica_extrair_cid=bool(extrair_cid),
    politica_cid_modo="extrair_restrito" if extrair_cid else "nao_extrair",
    who_icd_enabled=bool(habilitar_fallback_who),
)
repo_ativo = _obter_repositorio_ativo(usar_postgres, cfg_ativa.database_url)
llm_ativo: Any | None = None if llm_escolha == "ChatGroq (LLM real)" else MockLLM()
cliente_who_ativo = (
    ClienteCidWHO(config=cfg_ativa)
    if (cfg_ativa.who_icd_enabled and who_cred_ok)
    else None
)


def _criar_agente() -> AgenteA1:
    return AgenteA1(
        repositorio=repo_ativo,
        llm=llm_ativo,
        config=cfg_ativa,
        cliente_cid=cliente_who_ativo,
    )


st.title("SinergIA — Demonstração do Agente 1", anchor=False)
st.caption(
    "Recepção de atestados com OCR, formulário manual de fallback, validação de CID "
    "(PostgreSQL `afastamento.cid` → fallback API externa OMS `id.who.int`) e isolamento LGPD."
)

secao = st.segmented_control(
    "Navegação",
    options=list(_SECOES.keys()),
    format_func=lambda k: _SECOES[k],
    default=st.session_state["a1_demo_secao"],
    selection_mode="single",
    required=True,
    label_visibility="collapsed",
    key="nav_secao_principal",
)
st.session_state["a1_demo_secao"] = secao

# ============================================================================
# SEÇÃO 1: Envio por OCR (Imagem/PDF) + Revisão / Confirmação
# ============================================================================
if secao == "ocr":
    with st.container(border=True):
        st.subheader("1. Enviar atestado para leitura por OCR", anchor=False)
        arquivo_upload = st.file_uploader(
            "Envie uma foto ou PDF do atestado médico",
            type=["png", "jpg", "jpeg", "pdf"],
            key="ocr_uploader",
        )

        st.caption("Ou gere um atestado PDF sintético instantâneo para testar cada cenário de CID:")
        with st.container(horizontal=True):
            gerar_cid10 = st.button(
                "Gerar PDF com CID-10 (M54.5 — no banco)",
                icon=":material/description:",
                key="btn_gen_cid10",
            )
            gerar_cid11 = st.button(
                "Gerar PDF com CID-11 (ME84.2 — fallback API)",
                icon=":material/cloud_sync:",
                key="btn_gen_cid11",
            )
            gerar_cid_inv = st.button(
                "Gerar PDF com CID inválido (ZZ99.9)",
                icon=":material/report:",
                key="btn_gen_cid_inv",
            )

        storage_demo = cfg_ativa.atestados_storage_dir / "demo_streamlit"
        storage_demo.mkdir(parents=True, exist_ok=True)
        caminho_para_processar: Path | None = None
        mime_para_processar: str | None = None

        if gerar_cid10:
            caminho_para_processar = _gerar_pdf_sintetico_atestado(
                destino=storage_demo / f"sintetico_m545_{uuid4().hex[:6]}.pdf",
                nome="Carla Souza",
                inicio=date.today(),
                dias=3,
                cid="M54.5",
            )
            mime_para_processar = "application/pdf"
        elif gerar_cid11:
            caminho_para_processar = _gerar_pdf_sintetico_atestado(
                destino=storage_demo / f"sintetico_me842_{uuid4().hex[:6]}.pdf",
                nome="Carla Souza",
                inicio=date.today(),
                dias=2,
                cid="ME84.2",
            )
            mime_para_processar = "application/pdf"
        elif gerar_cid_inv:
            caminho_para_processar = _gerar_pdf_sintetico_atestado(
                destino=storage_demo / f"sintetico_zz999_{uuid4().hex[:6]}.pdf",
                nome="Carla Souza",
                inicio=date.today(),
                dias=2,
                cid="ZZ99.9",
            )
            mime_para_processar = "application/pdf"
        elif arquivo_upload is not None and st.button(
            "Processar documento com OCR + Agente 1",
            type="primary",
            icon=":material/document_scanner:",
            key="btn_executar_ocr",
        ):
            ext = Path(arquivo_upload.name).suffix.lower() or ".pdf"
            destino_up = storage_demo / f"upload_{uuid4().hex[:8]}{ext}"
            destino_up.write_bytes(arquivo_upload.getvalue())
            caminho_para_processar = destino_up
            mime_para_processar = arquivo_upload.type or None

        if caminho_para_processar is not None:
            with st.status("Executando OCR, extração estruturada e validação de CID...", expanded=True):
                try:
                    res_ocr = processar_arquivo_atestado(
                        str(caminho_para_processar),
                        mime=mime_para_processar,
                    )
                    st.session_state["a1_demo_ultimo_ocr"] = res_ocr.model_dump(mode="json")
                    st.session_state["a1_demo_arquivo_atual"] = str(caminho_para_processar)

                    st.session_state["a1_demo_caso_seq"] += 1
                    id_caso = int(st.session_state["a1_demo_caso_seq"])
                    sha = hashlib.sha256(caminho_para_processar.read_bytes()).hexdigest()
                    tarefa_doc = {
                        "protocolo": "sinergia/1",
                        "id_mensagem": f"demo-ocr-{uuid4().hex[:10]}",
                        "tipo": "TAREFA",
                        "acao": "a1.processar_documento",
                        "origem": "A0",
                        "destino": "A1",
                        "correlacao": {
                            "id_caso": id_caso,
                            "thread_id": f"streamlit-42-{id_caso}",
                            "id_tarefa": f"t-doc-{id_caso}",
                        },
                        "contexto": {
                            "id_colaborador": 42,
                            "perfil": "colaborador",
                            "canal": "chat",
                            "data_hora": datetime.now(UTC).isoformat(),
                        },
                        "registro_inicial": _construir_registro_inicial(
                            repo_ativo, id_colaborador=42, id_caso=id_caso
                        ),
                        "payload": {
                            "arquivo_uri": str(caminho_para_processar),
                            "mime": res_ocr.mime or mime_para_processar,
                            "sha256": sha,
                        },
                    }
                    agente = _criar_agente()
                    res_doc = agente.executar(tarefa_doc)
                    st.session_state["a1_demo_ultimo_resultado_doc"] = res_doc
                    st.session_state["a1_demo_ultimo_diag_cid"] = agente.ultimo_diagnostico_cid
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    st.error(
                        f"Não foi possível executar o OCR no arquivo enviado: {exc}. "
                        "Use a aba **Formulário manual** como fallback.",
                        title="Falha na leitura por OCR",
                    )

    ultimo_res = st.session_state.get("a1_demo_ultimo_resultado_doc")
    ultimo_ocr = st.session_state.get("a1_demo_ultimo_ocr")
    if ultimo_res is not None:
        col_esq, col_dir = st.columns([1, 1], gap="medium")
        with col_esq, st.container(border=True):
            st.subheader("2. Resultado da leitura (OCR + LLM)", anchor=False)
            arquivo_atual = st.session_state.get("a1_demo_arquivo_atual")
            if arquivo_atual and Path(arquivo_atual).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(
                    arquivo_atual,
                    caption="Imagem enviada para OCR",
                    alt="Prévia do atestado médico enviado para OCR",
                    width="stretch",
                )
            if ultimo_ocr:
                st.caption(
                    f"Legibilidade OCR: `{ultimo_ocr.get('legibilidade')}` | "
                    f"Confiança média: `{ultimo_ocr.get('confianca_media')}`"
                )
                with st.expander("Ver texto bruto extraído pelo OCR"):
                    st.code(ultimo_ocr.get("texto_extraido") or "(vazio)", language="text")

            _renderizar_badge_diagnostico_cid(st.session_state.get("a1_demo_ultimo_diag_cid"))

            if ultimo_res.get("saidas"):
                st.info(
                    ultimo_res["saidas"][0]["texto"],
                    icon=":material/chat:",
                    title="Mensagem do Agente 1",
                )

            divergencias = ultimo_res.get("payload", {}).get("divergencias") or []
            if divergencias:
                st.warning(
                    "Foram detectadas divergências para conferência humana antes de confirmar.",
                    title="Divergências encontradas",
                )
                st.dataframe(
                    divergencias,
                    hide_index=True,
                    alt="Tabela de divergências detectadas pelo Agente 1",
                )

        with col_dir, st.container(border=True):
            st.subheader("3. Conferir ou corrigir dados e confirmar", anchor=False)
            extraido = ultimo_res.get("payload", {}).get("atestado_extraido") or {}
            if extraido.get("legibilidade") == "ilegivel":
                st.warning(
                    "O OCR classificou o documento como ilegível. Preencha os campos abaixo "
                    "ou use a aba **Formulário manual** para registrar o atestado.",
                    title="Documento ilegível — fallback manual",
                )

            periodo_ext = extraido.get("periodo") or {}
            prof_ext = extraido.get("profissional") or {}
            inicio_def = (
                date.fromisoformat(periodo_ext["inicio"])
                if periodo_ext.get("inicio")
                else date.today()
            )
            fim_def = (
                date.fromisoformat(periodo_ext["fim"])
                if periodo_ext.get("fim")
                else inicio_def
            )
            emissao_def = (
                date.fromisoformat(extraido["data_emissao"])
                if extraido.get("data_emissao")
                else inicio_def
            )
            conselho_def = (
                prof_ext.get("conselho")
                if prof_ext.get("conselho") in _CONSELHOS_OPCOES
                else "CRM"
            )
            uf_def = prof_ext.get("uf") if prof_ext.get("uf") in _UFS_ORDENADAS else "SP"

            with st.form("form_confirmacao_ocr"):
                c1, c2 = st.columns(2)
                inicio_conf = c1.date_input("Início do afastamento", value=inicio_def)
                fim_conf = c2.date_input("Fim do afastamento", value=fim_def)

                c3, c4 = st.columns(2)
                emissao_conf = c3.date_input("Data de emissão", value=emissao_def)
                cid_conf = c4.text_input(
                    "Código CID (opcional)",
                    value=extraido.get("cid") or "",
                    placeholder="Ex.: ME84.2 ou M54.5",
                )

                c5, c6, c7 = st.columns(3)
                conselho_conf = c5.selectbox(
                    "Conselho",
                    options=_CONSELHOS_OPCOES,
                    index=_CONSELHOS_OPCOES.index(conselho_def),
                )
                numero_conf = c6.text_input(
                    "Número registro",
                    value=prof_ext.get("numero") or "123456",
                )
                uf_conf = c7.selectbox(
                    "UF",
                    options=_UFS_ORDENADAS,
                    index=_UFS_ORDENADAS.index(uf_def),
                )

                municipio_conf = st.text_input(
                    "Município de emissão",
                    value=extraido.get("municipio") or "Campinas",
                )
                submeter_conf = st.form_submit_button(
                    "Confirmar e salvar no banco",
                    type="primary",
                    icon=":material/check_circle:",
                    width="stretch",
                )

            if submeter_conf:
                corrigidos: list[str] = []
                if inicio_conf != inicio_def:
                    corrigidos.append("periodo.inicio")
                if fim_conf != fim_def:
                    corrigidos.append("periodo.fim")
                if (cid_conf.strip().upper() or None) != extraido.get("cid"):
                    corrigidos.append("cid")

                id_caso_conf = ultimo_res.get("correlacao", {}).get("id_caso") or 900
                tarefa_conf = {
                    "protocolo": "sinergia/1",
                    "id_mensagem": f"demo-conf-{uuid4().hex[:10]}",
                    "tipo": "TAREFA",
                    "acao": "a1.confirmar",
                    "origem": "A0",
                    "destino": "A1",
                    "correlacao": {
                        "id_caso": id_caso_conf,
                        "thread_id": f"streamlit-42-{id_caso_conf}",
                        "id_tarefa": f"t-conf-{id_caso_conf}",
                    },
                    "contexto": {
                        "id_colaborador": 42,
                        "perfil": "colaborador",
                        "canal": "chat",
                        "data_hora": datetime.now(UTC).isoformat(),
                    },
                    "registro_inicial": _construir_registro_inicial(
                        repo_ativo, id_colaborador=42, id_caso=id_caso_conf
                    ),
                    "payload": {
                        "atestado_confirmado": {
                            "tipo_documento": extraido.get("tipo_documento") or "atestado_medico",
                            "nome_no_documento": extraido.get("nome_no_documento") or "Carla Souza",
                            "periodo": {
                                "inicio": inicio_conf.isoformat(),
                                "fim": fim_conf.isoformat(),
                            },
                            "data_emissao": emissao_conf.isoformat(),
                            "profissional": {
                                "conselho": conselho_conf,
                                "numero": numero_conf.strip(),
                                "uf": uf_conf,
                            },
                            "cid": cid_conf.strip().upper() or None,
                            "municipio": municipio_conf.strip() or "Campinas",
                        },
                        "campos_corrigidos": corrigidos,
                    },
                }
                agente_conf = _criar_agente()
                res_conf = agente_conf.executar(tarefa_conf)
                _renderizar_badge_diagnostico_cid(agente_conf.ultimo_diagnostico_cid)
                if res_conf.get("estado_proposto") == "VALIDADO":
                    pl = res_conf["payload"]
                    st.success(
                        f"Atestado **#{pl['id_atestado']}** confirmado (`origem={pl['origem']}`). "
                        f"Prazo de entrega física: **{pl['prazo_entrega_fisica']}**.",
                        icon=":material/task_alt:",
                        title="Atestado registrado no banco",
                    )
                else:
                    msg_err = (
                        res_conf["saidas"][0]["texto"]
                        if res_conf.get("saidas")
                        else res_conf.get("erro", {}).get("mensagem", "Dados inválidos.")
                    )
                    st.error(msg_err, title="Validação recusada")

# ============================================================================
# SEÇÃO 2: Formulário Manual (Fallback quando OCR falha)
# ============================================================================
elif secao == "manual":
    with st.container(border=True):
        st.subheader("Registro manual de atestado (fallback de OCR)", anchor=False)
        st.caption(
            "Use este formulário quando a foto/PDF estiver ilegível ou o OCR falhar. "
            "O CID informado é verificado em `afastamento.cid` e, se ausente, consultado "
            "na API externa da OMS (`id.who.int`) com gravação automática em cache-aside."
        )

        with st.form("form_manual_atestado"):
            m1, m2 = st.columns(2)
            tipo_doc_m = m1.selectbox(
                "Tipo de documento",
                options=list(_TIPOS_DOC_LABELS.keys()),
                format_func=lambda k: _TIPOS_DOC_LABELS[k],
            )
            nome_m = m2.text_input("Nome do colaborador no documento", value="Carla Souza")

            m3, m4, m5 = st.columns(3)
            inicio_m = m3.date_input("Início do afastamento", value=date.today())
            fim_m = m4.date_input("Fim do afastamento", value=date.today() + timedelta(days=2))
            emissao_m = m5.date_input("Data de emissão", value=date.today())

            m6, m7, m8 = st.columns(3)
            conselho_m = m6.selectbox("Conselho profissional", options=_CONSELHOS_OPCOES)
            numero_m = m7.text_input("Número do registro", value="123456")
            uf_m = m8.selectbox("UF do registro", options=_UFS_ORDENADAS, index=_UFS_ORDENADAS.index("SP"))

            m9, m10 = st.columns(2)
            municipio_m = m9.text_input("Município de emissão", value="Campinas")
            cid_m = m10.text_input(
                "Código CID-11 ou CID-10 (opcional)",
                value="",
                placeholder="Ex.: ME84.2, CA07.0, M54.5 ou em branco",
            )

            enviar_manual = st.form_submit_button(
                "Validar CID e registrar atestado manualmente",
                type="primary",
                icon=":material/save:",
                width="stretch",
            )

        if enviar_manual:
            st.session_state["a1_demo_caso_seq"] += 1
            id_caso_m = int(st.session_state["a1_demo_caso_seq"])
            tarefa_manual = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"demo-manual-{uuid4().hex[:10]}",
                "tipo": "TAREFA",
                "acao": "a1.confirmar",
                "origem": "A0",
                "destino": "A1",
                "correlacao": {
                    "id_caso": id_caso_m,
                    "thread_id": f"streamlit-manual-{id_caso_m}",
                    "id_tarefa": f"t-manual-{id_caso_m}",
                },
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "colaborador",
                    "canal": "chat",
                    "data_hora": datetime.now(UTC).isoformat(),
                },
                "registro_inicial": _construir_registro_inicial(
                    repo_ativo,
                    id_colaborador=42,
                    id_caso=id_caso_m,
                    inicio_informado=inicio_m,
                    dias_informados=(fim_m - inicio_m).days + 1 if fim_m >= inicio_m else None,
                    tem_documento=False,
                ),
                "payload": {
                    "atestado_confirmado": {
                        "tipo_documento": tipo_doc_m,
                        "nome_no_documento": nome_m.strip() or "Carla Souza",
                        "periodo": {
                            "inicio": inicio_m.isoformat(),
                            "fim": fim_m.isoformat(),
                        },
                        "data_emissao": emissao_m.isoformat(),
                        "profissional": {
                            "conselho": conselho_m,
                            "numero": numero_m.strip(),
                            "uf": uf_m,
                        },
                        "cid": cid_m.strip().upper() or None,
                        "municipio": municipio_m.strip() or "Campinas",
                    },
                    "campos_corrigidos": [
                        "tipo_documento",
                        "nome_no_documento",
                        "periodo.inicio",
                        "periodo.fim",
                        "data_emissao",
                        "profissional.conselho",
                        "profissional.numero",
                        "profissional.uf",
                        "municipio",
                        "cid",
                    ],
                },
            }
            agente_m = _criar_agente()
            res_m = agente_m.executar(tarefa_manual)
            _renderizar_badge_diagnostico_cid(agente_m.ultimo_diagnostico_cid)

            if res_m.get("estado_proposto") == "VALIDADO":
                pl = res_m["payload"]
                marcos = ", ".join(pl.get("prazos_legais") or []) or "nenhum marco especial"
                st.success(
                    f"Atestado **#{pl['id_atestado']}** salvo com `origem={pl['origem']}`. "
                    f"Prazo físico: **{pl['prazo_entrega_fisica']}** | Marcos legais: `{marcos}`.",
                    icon=":material/task_alt:",
                    title="Registro manual confirmado",
                )
            else:
                msg_m = (
                    res_m["saidas"][0]["texto"]
                    if res_m.get("saidas")
                    else res_m.get("erro", {}).get("mensagem", "Erro ao validar formulário.")
                )
                st.error(msg_m, title="Não foi possível registrar o atestado")

# ============================================================================
# SEÇÃO 3: Consulta e Catálogo de CIDs (afastamento.cid)
# ============================================================================
elif secao == "catalogo":
    with st.container(border=True):
        st.subheader("Testar verificação de CID (Banco → Fallback API externa OMS)", anchor=False)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            codigo_teste = st.text_input(
                "Código CID-11 ou CID-10",
                placeholder="Ex.: ME84.2, CA07.0, M54.5, ZZ99.9",
                key="input_teste_cid",
            )
            btn_consultar_cid = st.button(
                "Verificar CID",
                type="primary",
                icon=":material/search:",
                key="btn_consultar_cid",
            )

        if btn_consultar_cid and codigo_teste.strip():
            agente_cid = _criar_agente()
            agente_cid._resolver_cid_com_fallback(codigo_teste.strip())
            _renderizar_badge_diagnostico_cid(agente_cid.ultimo_diagnostico_cid)

    with st.container(border=True):
        st.subheader("Catálogo salvo em `afastamento.cid`", anchor=False)
        filtro_cat = st.text_input(
            "Filtrar por código ou descrição",
            placeholder="Digite um código ou palavra-chave",
            key="input_filtro_catalogo",
        )
        cids_salvos = repo_ativo.listar_cids(filtro=filtro_cat, limite=500)
        st.caption(f"Exibindo {len(cids_salvos)} código(s) em `afastamento.cid`.")
        st.dataframe(
            cids_salvos,
            hide_index=True,
            alt="Catálogo de códigos CID salvos na tabela afastamento.cid",
        )

# ============================================================================
# SEÇÃO 4: Atestados Salvos no Banco
# ============================================================================
else:
    with st.container(border=True):
        st.subheader("Atestados persistidos no banco de dados", anchor=False)
        mostrar_cid_restrito = st.toggle(
            "Exibir CID restrito (visão autorizada A1 / RH — LGPD Art. 11)",
            value=False,
            key="toggle_mostrar_cid_restrito",
        )
        atestados_db = repo_ativo.listar_atestados_recentes(limite=100)
        if not mostrar_cid_restrito:
            linhas_visiveis = [
                {
                    k: v
                    for k, v in item.items()
                    if k not in {"codigo_cid_restrito", "cid_descricao"}
                }
                for item in atestados_db
            ]
        else:
            linhas_visiveis = atestados_db

        st.caption(f"{len(linhas_visiveis)} registro(s) encontrado(s) em `afastamento.atestado`.")
        st.dataframe(
            linhas_visiveis,
            hide_index=True,
            alt="Tabela de atestados médicos registrados no PostgreSQL",
        )
