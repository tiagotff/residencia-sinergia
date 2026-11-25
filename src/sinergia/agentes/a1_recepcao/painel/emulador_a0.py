"""Emulador headless do Orquestrador A0 para testes e simulações no painel Streamlit."""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, Field

from sinergia.agentes.a1_recepcao.painel.cliente_conexao import RegistroTrocaProtocolo

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None

UPLOADS_PAINEL_SUBDIR = "uploads_painel"
_RAIZ_REPO = Path(__file__).resolve().parents[5]
FIXTURES_DIR = _RAIZ_REPO / "tests" / "a1_recepcao" / "fixtures"

_MIME_POR_EXTENSAO = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
}


class EstadoCasoSimulado(BaseModel):
    """Estado de um caso simulado pelo A0 para testes multi-thread e HITL."""

    id_caso: int
    thread_id: str
    estado_atual: str = "NOVO"
    aguardando_hitl: bool = False
    id_colaborador: int = 42
    nome_colaborador: str = "Carla S."
    matricula: str = "00123"
    id_area: int = 2
    id_setor: int = 7
    id_cargo: int = 15
    jornada_inicio: str = "07:00"
    jornada_fim: str = "16:00"
    municipio: str = "Campinas"
    uf: str = "SP"
    tipo_solicitacao: str = "atestado_medico"
    inicio_informado: date = date(2026, 11, 9)
    dias_informados: int = 5
    tem_documento: bool = False
    ultimo_arquivo_uri: str | None = None
    ultimo_atestado_extraido: dict[str, Any] | None = None
    ultima_revisao_colaborador: dict[str, Any] | None = None
    ultimas_divergencias: list[dict[str, Any]] = Field(default_factory=list)
    ultimo_diagnostico_cid: dict[str, Any] | None = None
    id_atestado_confirmado: int | None = None
    prazo_entrega_fisica: str | None = None
    prazos_legais: list[str] = Field(default_factory=list)
    baixa_registrada: bool = False
    mensagens_chat: list[dict[str, Any]] = Field(default_factory=list)
    trocas: list[RegistroTrocaProtocolo] = Field(default_factory=list)


def criar_caso_padrao(
    id_caso: int = 105,
    *,
    id_colaborador: int = 42,
    nome_colaborador: str = "Carla S.",
    inicio_informado: date = date(2026, 11, 9),
    dias_informados: int = 5,
    tipo_solicitacao: str = "atestado_medico",
) -> EstadoCasoSimulado:
    """Cria um `EstadoCasoSimulado` padrão alinhado ao caso canônico 105."""
    return EstadoCasoSimulado(
        id_caso=id_caso,
        thread_id=f"chat-{id_colaborador}-{id_caso}",
        id_colaborador=id_colaborador,
        nome_colaborador=nome_colaborador,
        inicio_informado=inicio_informado,
        dias_informados=dias_informados,
        tipo_solicitacao=tipo_solicitacao,
    )


def salvar_upload_temporario(
    nome_arquivo: str,
    conteudo_bytes: bytes,
    storage_dir: Path,
) -> tuple[str, str, str]:
    """Salva upload em `<storage_dir>/uploads_painel/` e retorna `(caminho, mime, sha256)`."""
    nome_base = Path(nome_arquivo).name
    extensao = Path(nome_base).suffix.lower()
    if extensao not in _MIME_POR_EXTENSAO:
        raise ValueError(
            f"Extensão não suportada: {extensao or '(sem extensão)'}. Use PNG, JPG ou PDF."
        )
    if not conteudo_bytes:
        raise ValueError("Arquivo enviado está vazio.")

    stem_limpo = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(nome_base).stem).strip("._-")
    stem_final = stem_limpo[:60] or "atestado"

    pasta_uploads = storage_dir / UPLOADS_PAINEL_SUBDIR
    pasta_uploads.mkdir(parents=True, exist_ok=True)
    caminho_destino = pasta_uploads / f"{uuid4().hex[:8]}_{stem_final}{extensao}"
    caminho_destino.write_bytes(conteudo_bytes)

    sha256 = hashlib.sha256(conteudo_bytes).hexdigest()
    mime = _MIME_POR_EXTENSAO[extensao]
    return str(caminho_destino), mime, sha256


def limpar_uploads_temporarios(storage_dir: Path) -> int:
    """Remove apenas arquivos temporários em `<storage_dir>/uploads_painel/`."""
    pasta_uploads = storage_dir / UPLOADS_PAINEL_SUBDIR
    if not pasta_uploads.is_dir():
        return 0
    removidos = 0
    for item in pasta_uploads.iterdir():
        if item.is_file():
            item.unlink()
            removidos += 1
    return removidos


def gerar_preview_documento_png(caminho_arquivo: str | Path) -> bytes | None:
    """Gera bytes PNG para preview visual de imagens (PNG/JPG) ou 1ª página de PDF."""
    caminho = Path(caminho_arquivo)
    if not caminho.is_file():
        return None

    ext = caminho.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(caminho) as im:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
        except (OSError, ValueError, TypeError):
            return None

    if ext == ".pdf" and pymupdf is not None:
        try:
            doc = pymupdf.open(caminho)
            try:
                if doc.page_count == 0:
                    return None
                pixmap = doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                return pixmap.tobytes("png")
            finally:
                doc.close()
        except (OSError, ValueError, RuntimeError):
            return None
    return None


def montar_registro_inicial(caso: EstadoCasoSimulado) -> dict[str, Any]:
    """Constrói o bloco `registro_inicial` (Anexo B) a partir do estado do caso."""
    return {
        "id_caso": caso.id_caso,
        "thread_id": caso.thread_id,
        "aberto_em": datetime.now(UTC).isoformat(),
        "canal": "chat",
        "id_colaborador": caso.id_colaborador,
        "nome": caso.nome_colaborador,
        "matricula": caso.matricula,
        "id_area": caso.id_area,
        "id_setor": caso.id_setor,
        "id_cargo": caso.id_cargo,
        "jornada_inicio": caso.jornada_inicio,
        "jornada_fim": caso.jornada_fim,
        "municipio": caso.municipio,
        "uf": caso.uf,
        "tipo_solicitacao": caso.tipo_solicitacao,
        "inicio_informado": caso.inicio_informado.isoformat(),
        "dias_informados": caso.dias_informados,
        "tem_documento": caso.tem_documento,
    }


def _envelope_base(
    caso: EstadoCasoSimulado,
    *,
    acao: str,
    perfil: str = "colaborador",
    incluir_registro_inicial: bool = True,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocolo": "sinergia/1",
        "id_mensagem": str(uuid4()),
        "tipo": "TAREFA",
        "acao": acao,
        "origem": "A0",
        "destino": "A1",
        "correlacao": {
            "id_caso": caso.id_caso,
            "thread_id": caso.thread_id,
            "id_tarefa": f"t-{caso.id_caso}-{len(caso.trocas) + 1:02d}",
        },
        "contexto": {
            "id_colaborador": caso.id_colaborador,
            "id_empresa": 1,
            "perfil": perfil,
            "canal": "chat",
            "data_hora": datetime.now(UTC).isoformat(),
        },
        "registro_inicial": montar_registro_inicial(caso)
        if incluir_registro_inicial
        else None,
        "payload": payload,
    }


def montar_tarefa_conversar(
    caso: EstadoCasoSimulado,
    texto: str,
    *,
    historico_resumo: str | None = None,
    perfil: str = "colaborador",
) -> dict[str, Any]:
    """Monta envelope `TAREFA` para `a1.conversar`."""
    return _envelope_base(
        caso,
        acao="a1.conversar",
        perfil=perfil,
        payload={
            "texto": texto,
            "historico_resumo": historico_resumo,
        },
    )


def montar_tarefa_processar_documento(
    caso: EstadoCasoSimulado,
    arquivo_uri: str,
    mime: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Monta envelope `TAREFA` para `a1.processar_documento`."""
    payload: dict[str, Any] = {"arquivo_uri": arquivo_uri}
    if mime:
        payload["mime"] = mime
    if sha256:
        payload["sha256"] = sha256
    return _envelope_base(
        caso,
        acao="a1.processar_documento",
        perfil="colaborador",
        payload=payload,
    )


def _normalizar_valor_campo(valor: Any) -> str:
    if valor is None:
        return ""
    return str(valor).strip()


_ROTULOS_DIFF: dict[str, str] = {
    "tipo_documento": "Tipo de documento",
    "nome_no_documento": "Paciente no documento",
    "periodo.inicio": "Início do afastamento",
    "periodo.fim": "Término do afastamento",
    "data_emissao": "Data de emissão",
    "profissional.conselho": "Conselho profissional",
    "profissional.numero": "Número do registro",
    "profissional.uf": "UF do registro",
    "profissional.nome": "Nome do profissional",
    "cid": "Código CID-10",
    "municipio": "Município de emissão",
}


def _pares_comparacao_atestado(
    extraido: dict[str, Any],
    editado: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    ext_periodo = extraido.get("periodo") or {}
    edi_periodo = editado.get("periodo") or {}
    ext_prof = extraido.get("profissional") or {}
    edi_prof = editado.get("profissional") or {}
    return [
        ("tipo_documento", extraido.get("tipo_documento"), editado.get("tipo_documento")),
        ("nome_no_documento", extraido.get("nome_no_documento"), editado.get("nome_no_documento")),
        ("periodo.inicio", ext_periodo.get("inicio"), edi_periodo.get("inicio")),
        ("periodo.fim", ext_periodo.get("fim"), edi_periodo.get("fim")),
        ("data_emissao", extraido.get("data_emissao"), editado.get("data_emissao")),
        ("profissional.conselho", ext_prof.get("conselho"), edi_prof.get("conselho")),
        ("profissional.numero", ext_prof.get("numero"), edi_prof.get("numero")),
        ("profissional.uf", ext_prof.get("uf"), edi_prof.get("uf")),
        ("profissional.nome", ext_prof.get("nome"), edi_prof.get("nome")),
        ("cid", extraido.get("cid"), editado.get("cid")),
        ("municipio", extraido.get("municipio"), editado.get("municipio")),
    ]


def calcular_campos_corrigidos(
    extraido: dict[str, Any] | None,
    editado: dict[str, Any],
) -> list[str]:
    """Compara `extraido` vs `editado` e retorna exatamente os campos alterados."""
    if extraido is None:
        return []

    alterados: list[str] = []
    for nome_campo, v_ext, v_edi in _pares_comparacao_atestado(extraido, editado):
        if _normalizar_valor_campo(v_ext) != _normalizar_valor_campo(v_edi):
            alterados.append(nome_campo)
    return alterados


def resumir_diff_edicao(
    extraido: dict[str, Any] | None,
    editado: dict[str, Any],
) -> list[dict[str, str]]:
    """Retorna lista estruturada de alterações Antes (IA) -> Depois (Manual)."""
    if extraido is None:
        return []

    diffs: list[dict[str, str]] = []
    for nome_campo, v_ext, v_edi in _pares_comparacao_atestado(extraido, editado):
        str_ext = _normalizar_valor_campo(v_ext)
        str_edi = _normalizar_valor_campo(v_edi)
        if str_ext != str_edi:
            diffs.append(
                {
                    "campo": nome_campo,
                    "rotulo": _ROTULOS_DIFF.get(nome_campo, nome_campo),
                    "antes_ia": str_ext or "Não identificado",
                    "depois_manual": str_edi or "(vazio)",
                }
            )
    return diffs


def montar_tarefa_confirmar(
    caso: EstadoCasoSimulado,
    atestado_confirmado: dict[str, Any],
    campos_corrigidos: list[str],
) -> dict[str, Any]:
    """Monta envelope `TAREFA` para `a1.confirmar`."""
    return _envelope_base(
        caso,
        acao="a1.confirmar",
        perfil="colaborador",
        payload={
            "atestado_confirmado": atestado_confirmado,
            "campos_corrigidos": list(campos_corrigidos),
        },
    )


def montar_tarefa_baixa_fisica(
    caso: EstadoCasoSimulado,
    *,
    id_atestado: int,
    recebido_por: str,
    data_hora_iso: str,
    perfil: str = "rh",
    usar_aliases_v07: bool = False,
) -> dict[str, Any]:
    """Monta envelope `TAREFA` para `a1.registrar_baixa_fisica`."""
    if usar_aliases_v07:
        payload: dict[str, Any] = {
            "id_atestado": id_atestado,
            "recebido_por": recebido_por,
            "data_hora": data_hora_iso,
        }
    else:
        payload = {
            "id_atestado": id_atestado,
            "recebido_fisico_por": recebido_por,
            "recebido_fisico_em": data_hora_iso,
        }
    return _envelope_base(
        caso,
        acao="a1.registrar_baixa_fisica",
        perfil=perfil,
        incluir_registro_inicial=False,
        payload=payload,
    )


def aplicar_resultado_ao_caso(
    caso: EstadoCasoSimulado,
    troca: RegistroTrocaProtocolo,
) -> EstadoCasoSimulado:
    """Atualiza a máquina de estados local do emulador A0 com o RESULTADO do A1."""
    novo = caso.model_copy(deep=True)
    novo.trocas.append(troca)
    if troca.diagnostico_cid is not None:
        novo.ultimo_diagnostico_cid = dict(troca.diagnostico_cid)

    res = troca.resultado
    payload = res.get("payload") or {}
    status = res.get("status")
    estado_proposto = res.get("estado_proposto")

    if troca.acao == "a1.conversar":
        texto_usuario = (troca.tarefa.get("payload") or {}).get("texto")
        if texto_usuario:
            novo.mensagens_chat.append(
                {"papel": "colaborador", "texto": str(texto_usuario), "acao": troca.acao}
            )
    elif troca.acao == "a1.processar_documento":
        uri = (troca.tarefa.get("payload") or {}).get("arquivo_uri")
        novo.ultimo_arquivo_uri = uri
        novo.mensagens_chat.append(
            {
                "papel": "colaborador",
                "texto": f"[Documento anexado: {Path(str(uri)).name}]",
                "acao": troca.acao,
            }
        )
        if status != "ERRO" and "atestado_extraido" in payload:
            novo.ultimo_atestado_extraido = payload["atestado_extraido"]
            novo.ultima_revisao_colaborador = payload.get("revisao_colaborador")
            novo.ultimas_divergencias = list(payload.get("divergencias") or [])
            if novo.ultimo_atestado_extraido.get("legibilidade") == "ok":
                novo.tem_documento = True
    elif troca.acao == "a1.confirmar":
        if status == "OK" and estado_proposto == "VALIDADO":
            novo.id_atestado_confirmado = payload.get("id_atestado")
            novo.prazo_entrega_fisica = payload.get("prazo_entrega_fisica")
            novo.prazos_legais = list(payload.get("prazos_legais") or [])
            novo.aguardando_hitl = False
        elif status == "OK" and estado_proposto == "NOVO":
            novo.aguardando_hitl = False
    elif troca.acao == "a1.registrar_baixa_fisica" and status == "OK":
        novo.baixa_registrada = True

    if estado_proposto:
        novo.estado_atual = str(estado_proposto)

    if status == "PRECISA_HUMANO" and estado_proposto == "RECEBIDO":
        novo.aguardando_hitl = True
    elif status == "PRECISA_HUMANO" and estado_proposto == "NOVO":
        novo.aguardando_hitl = False

    for saida in res.get("saidas") or []:
        texto_saida = saida.get("texto")
        if texto_saida:
            dest = (saida.get("para") or {}).get("perfil", "colaborador")
            novo.mensagens_chat.append(
                {
                    "papel": "agente",
                    "destinatario": dest,
                    "texto": str(texto_saida),
                    "acao": troca.acao,
                }
            )
    return novo


def carregar_fixtures_oficiais(
    fixtures_dir: Path | None = None,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Carrega todos os pares `.tarefa.json` e `.esperado.json` de `tests/a1_recepcao/fixtures`."""
    raiz = fixtures_dir or FIXTURES_DIR
    pares: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    if not raiz.is_dir():
        return pares

    for caminho_tarefa in sorted(raiz.rglob("*.tarefa.json")):
        caminho_esperado = caminho_tarefa.with_name(
            caminho_tarefa.name.replace(".tarefa.json", ".esperado.json")
        )
        if not caminho_esperado.is_file():
            continue
        chave = f"{caminho_tarefa.parent.name}/{caminho_tarefa.name.replace('.tarefa.json', '')}"
        pares[chave] = (
            json.loads(caminho_tarefa.read_text(encoding="utf-8")),
            json.loads(caminho_esperado.read_text(encoding="utf-8")),
        )
    return pares


def comparar_com_esperado(
    obtido: dict[str, Any],
    esperado: dict[str, Any],
    *,
    modo_llm: str = "mock",
) -> list[str]:
    """Compara o RESULTADO obtido com a fixture `.esperado.json` e lista divergências."""
    diffs: list[str] = []
    for campo in ("protocolo", "tipo", "acao", "origem", "destino", "em_resposta_a", "status", "estado_proposto"):
        if obtido.get(campo) != esperado.get(campo):
            diffs.append(
                f"{campo}: obtido={obtido.get(campo)!r} != esperado={esperado.get(campo)!r}"
            )
    if modo_llm == "mock":
        if obtido.get("payload") != esperado.get("payload"):
            diffs.append("payload difere do esperado.")
    else:
        obt_payload = obtido.get("payload") or {}
        esp_payload = esperado.get("payload") or {}
        if "atestado_extraido" in esp_payload:
            obt_ext = obt_payload.get("atestado_extraido") or {}
            esp_ext = esp_payload.get("atestado_extraido") or {}
            for chave_ext in (
                "tipo_documento",
                "legibilidade",
                "nome_no_documento",
                "periodo",
                "data_emissao",
                "cid",
                "municipio",
            ):
                if obt_ext.get(chave_ext) != esp_ext.get(chave_ext):
                    diffs.append(
                        f"atestado_extraido.{chave_ext}: obtido={obt_ext.get(chave_ext)!r} != esperado={esp_ext.get(chave_ext)!r}"
                    )
            obt_prof = obt_ext.get("profissional") or {}
            esp_prof = esp_ext.get("profissional") or {}
            for chave_prof in ("conselho", "numero", "uf"):
                if obt_prof.get(chave_prof) != esp_prof.get(chave_prof):
                    diffs.append(
                        f"atestado_extraido.profissional.{chave_prof}: obtido={obt_prof.get(chave_prof)!r} != esperado={esp_prof.get(chave_prof)!r}"
                    )
            if obt_payload.get("divergencias") != esp_payload.get("divergencias"):
                diffs.append("payload.divergencias difere do esperado.")
            esp_rev = esp_payload.get("revisao_colaborador")
            obt_rev = obt_payload.get("revisao_colaborador")
            if esp_rev is not None and (
                not isinstance(obt_rev, dict)
                or obt_rev.get("resumo_markdown") != esp_rev.get("resumo_markdown")
            ):
                diffs.append("payload.revisao_colaborador difere do esperado.")
        elif obt_payload != esp_payload:
            diffs.append("payload difere do esperado.")
    if obtido.get("eventos") != esperado.get("eventos"):
        diffs.append("eventos diferem do esperado.")
    dec_obt = (obtido.get("decisao") or {}).get("usou_llm")
    dec_esp = (esperado.get("decisao") or {}).get("usou_llm")
    if dec_obt != dec_esp:
        diffs.append(f"decisao.usou_llm: obtido={dec_obt!r} != esperado={dec_esp!r}")
    if modo_llm == "mock" and obtido.get("saidas") != esperado.get("saidas"):
        diffs.append("saidas diferem do esperado (modo mock).")
    return diffs
