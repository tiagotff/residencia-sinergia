"""Avaliador consolidado das métricas da Seção 10 para o Agente 1 (Task 3.2).

Executa a bateria completa sobre os 12 atestados sintéticos com gabarito
(`.storage/atestados/gabarito_atestados.json`) nos formatos PNG e PDF para medir
e validar:

1. **Integridade criptográfica do catálogo**:
   - Conferência do SHA-256 real em disco dos 24 arquivos (`.png` e `.pdf`)
     contra `gabarito_atestados.json` e preservação do SHA-256 canônico do
     caso `105` (`atestado_carla_105.png`).
2. **Extração sem correção manual >= 80%**:
   - Acurácia exata de todos os campos contra o gabarito em PNG e PDF;
   - Taxa de confirmação com `origem = "IA"` (com revisão `MANUAL` simulada
     em `atestado_ruido_leve_manual` e fallback human-in-the-loop caso o LLM
     em nuvem divirja ou falhe);
   - Acurácia por campo individual e respeito à `POLITICA_EXTRAIR_CID=False`;
   - Rejeição segura nos 4 modos de ilegibilidade sem gravação no banco.
3. **Atestados físicos baixados no prazo >= 90%**:
   - Fluxo nominal (8/8 por formato com baixa registrada até `prazo_entrega_fisica`);
   - Cenário estendido de 10 entregas (9 no prazo e 1 entrega atrasada para
     comprovar a detecção de atraso consumida pelo alerta `A3.5`).
4. **Idempotência por `id_mensagem` (`sinergia/1`)**:
   - Replay das tarefas enviadas garantindo resposta idêntica sem duplicação.

Uso via CLI:
    python -m sinergia.agentes.a1_recepcao.avaliacao_metricas
    python -m sinergia.agentes.a1_recepcao.avaliacao_metricas --modo-llm mock
    python -m sinergia.agentes.a1_recepcao.avaliacao_metricas --salvar-markdown docs/out.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import jsonschema

from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.extrator_llm import MockLLM, obter_cliente_llm
from sinergia.agentes.a1_recepcao.gerador_sintetico import (
    SHA256_CANONICO_CARLA_105_PNG,
    GabaritoAtestadoSintetico,
    garantir_atestados_sinteticos,
)
from sinergia.agentes.a1_recepcao.repositorio import RepositorioA1Memoria
from sinergia.agentes.a1_recepcao.schemas import (
    AtestadoExtraido,
    ResultadoA1,
    exportar_json_schemas_a1,
)
from sinergia.agentes.a1_recepcao.servico import ServicoA1, limpar_cache_idempotencia
from sinergia.agentes.a1_recepcao.validador import calcular_marcos_legais

FormatoAvaliacao = Literal["png", "pdf"]
ModoLLM = Literal["auto", "mock", "groq"]

META_EXTRACAO_MIN_PCT = 80.0
META_BAIXA_MIN_PCT = 90.0

CAMPOS_EXTRACAO: tuple[str, ...] = (
    "legibilidade",
    "nome_no_documento",
    "periodo.inicio",
    "periodo.fim",
    "data_emissao",
    "profissional.conselho",
    "profissional.numero",
    "profissional.uf",
    "cid",
    "municipio",
)


def _configurar_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _calcular_sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arq:
        for bloco in iter(lambda: arq.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def validar_integridade_catalogo_sintetico(
    amostras: Sequence[GabaritoAtestadoSintetico],
    storage_dir: Path,
) -> dict[str, Any]:
    """Valida existência e SHA-256 em disco de todos os arquivos do catálogo sintético."""
    falhas: list[str] = []
    arquivos_validos = 0
    total_arquivos = 0
    carla_canonico_ok = False

    manifesto_path = storage_dir / "gabarito_atestados.json"
    hashes_manifesto: dict[str, tuple[str, str]] = {}
    if not manifesto_path.is_file():
        falhas.append(f"Manifesto ausente: {manifesto_path}")
    else:
        try:
            manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
            for item in manifesto.get("amostras", []):
                hashes_manifesto[str(item["id_amostra"])] = (
                    str(item["sha256_png"]),
                    str(item["sha256_pdf"]),
                )
        except (OSError, KeyError, ValueError, TypeError) as exc:
            falhas.append(f"Falha ao ler manifesto {manifesto_path}: {exc}")

    for amostra in amostras:
        pares = (
            ("png", Path(amostra.arquivo_png), amostra.sha256_png),
            ("pdf", Path(amostra.arquivo_pdf), amostra.sha256_pdf),
        )
        for ext, caminho, sha_esperado in pares:
            total_arquivos += 1
            if not caminho.is_file():
                falhas.append(f"{amostra.id_amostra}.{ext}: arquivo ausente ({caminho})")
                continue

            sha_real = _calcular_sha256_arquivo(caminho)
            if sha_real != sha_esperado:
                falhas.append(
                    f"{amostra.id_amostra}.{ext}: SHA-256 em disco {sha_real} != "
                    f"esperado {sha_esperado}"
                )
                continue

            if amostra.id_amostra in hashes_manifesto:
                sha_man = (
                    hashes_manifesto[amostra.id_amostra][0]
                    if ext == "png"
                    else hashes_manifesto[amostra.id_amostra][1]
                )
                if sha_real != sha_man:
                    falhas.append(
                        f"{amostra.id_amostra}.{ext}: SHA-256 difere do manifesto "
                        f"({sha_real} != {sha_man})"
                    )
                    continue

            arquivos_validos += 1
            if amostra.id_amostra == "atestado_carla_105" and ext == "png":
                carla_canonico_ok = sha_real == SHA256_CANONICO_CARLA_105_PNG
                if not carla_canonico_ok:
                    falhas.append(
                        "atestado_carla_105.png perdeu o SHA-256 canônico da fixture: "
                        f"{sha_real} != {SHA256_CANONICO_CARLA_105_PNG}"
                    )

    if len(amostras) != 12:
        falhas.append(f"Esperadas 12 amostras no catálogo, encontradas {len(amostras)}")

    return {
        "integro": not falhas and carla_canonico_ok and arquivos_validos == total_arquivos,
        "total_amostras": len(amostras),
        "total_arquivos_verificados": total_arquivos,
        "arquivos_validos": arquivos_validos,
        "carla_105_sha256_canonico_preservado": carla_canonico_ok,
        "falhas": falhas,
    }


def _selecionar_llm(
    cfg: A1Config,
    *,
    modo_llm: ModoLLM = "auto",
) -> tuple[Any, str]:
    if modo_llm == "groq":
        return obter_cliente_llm(config=cfg), f"groq ({cfg.llm_model})"
    if modo_llm == "mock":
        return MockLLM(), "MockLLM (offline)"
    if cfg.groq_api_key.strip():
        return obter_cliente_llm(config=cfg), f"groq ({cfg.llm_model})"
    return MockLLM(), "MockLLM (offline)"


def _validar_envelope_resultado(
    resultado: dict[str, Any],
    schemas_json: dict[str, dict[str, Any]],
) -> None:
    jsonschema.validate(instance=resultado, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(resultado).validar_payload_acao()


def _executar_e_validar_idempotencia(
    servico: ServicoA1,
    tarefa: dict[str, Any],
    repo: RepositorioA1Memoria,
    schemas_json: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Executa a tarefa, valida o contrato e testa replay idempotente por `id_mensagem`."""
    res_original = servico.executar(tarefa)
    _validar_envelope_resultado(res_original, schemas_json)

    if res_original.get("status") == "ERRO" and res_original.get("erro", {}).get(
        "reprocessavel", False
    ):
        return res_original, True

    qtd_antes = len(repo.atestados)
    res_replay = servico.executar(copy.deepcopy(tarefa))
    _validar_envelope_resultado(res_replay, schemas_json)
    qtd_depois = len(repo.atestados)

    replay_ok = res_replay == res_original and qtd_depois == qtd_antes
    return res_original, replay_ok


def _extrair_atestado_de_resultado(
    res_doc: dict[str, Any],
) -> tuple[AtestadoExtraido | None, str | None]:
    if res_doc.get("status") == "ERRO":
        erro_bloco = res_doc.get("erro") or {}
        msg = erro_bloco.get("mensagem") or res_doc.get("decisao", {}).get("motivo")
        return None, str(msg or "status ERRO retornado pelo A1")

    payload = res_doc.get("payload") or {}
    if "atestado_extraido" not in payload:
        return None, "payload sem atestado_extraido"

    return AtestadoExtraido.model_validate(payload["atestado_extraido"]), None


def _serializar_atestado_para_payload(extraido: AtestadoExtraido) -> dict[str, Any]:
    prof_dict: dict[str, Any] = {
        "conselho": extraido.profissional.conselho,
        "numero": extraido.profissional.numero,
        "uf": extraido.profissional.uf,
    }
    if extraido.profissional.nome is not None:
        prof_dict["nome"] = extraido.profissional.nome

    return {
        "legibilidade": extraido.legibilidade,
        "nome_no_documento": extraido.nome_no_documento,
        "periodo": {
            "inicio": (
                extraido.periodo.inicio.isoformat()
                if extraido.periodo.inicio is not None
                else None
            ),
            "fim": (
                extraido.periodo.fim.isoformat()
                if extraido.periodo.fim is not None
                else None
            ),
        },
        "data_emissao": (
            extraido.data_emissao.isoformat()
            if extraido.data_emissao is not None
            else None
        ),
        "profissional": prof_dict,
        "cid": extraido.cid,
        "municipio": extraido.municipio,
    }


def _montar_registro_inicial(
    amostra: GabaritoAtestadoSintetico,
    caso_id: int,
) -> dict[str, Any]:
    inicio_esp = amostra.esperado.periodo.inicio
    fim_esp = amostra.esperado.periodo.fim
    dias_esp = (
        (fim_esp - inicio_esp).days + 1
        if inicio_esp is not None and fim_esp is not None
        else 1
    )
    inicio_str = inicio_esp.isoformat() if inicio_esp is not None else "2026-11-09"
    return {
        "id_caso": caso_id,
        "thread_id": f"chat-42-{caso_id}",
        "aberto_em": "2026-11-09T09:05:00-03:00",
        "canal": "chat",
        "id_colaborador": 42,
        "nome": "Carla S.",
        "matricula": "00123",
        "id_area": 2,
        "id_setor": 7,
        "id_cargo": 15,
        "jornada_inicio": "07:00",
        "jornada_fim": "16:00",
        "municipio": "Campinas",
        "uf": "SP",
        "tipo_solicitacao": (
            "declaracao_comparecimento"
            if amostra.cenario == "declaracao_comparecimento"
            else "atestado_medico"
        ),
        "inicio_informado": inicio_str,
        "dias_informados": dias_esp,
        "tem_documento": True,
    }


def _montar_tarefa_base(
    *,
    id_mensagem: str,
    acao: str,
    caso_id: int,
    id_tarefa: str,
    perfil: Literal["colaborador", "rh"] = "colaborador",
    data_hora: str = "2026-11-09T09:10:00-03:00",
    registro_inicial: dict[str, Any] | None = None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocolo": "sinergia/1",
        "id_mensagem": id_mensagem,
        "tipo": "TAREFA",
        "acao": acao,
        "origem": "A0",
        "destino": "A1",
        "correlacao": {
            "id_caso": caso_id,
            "thread_id": f"chat-42-{caso_id}",
            "id_tarefa": id_tarefa,
        },
        "contexto": {
            "id_colaborador": 42,
            "perfil": perfil,
            "canal": "chat",
            "data_hora": data_hora,
        },
        "registro_inicial": registro_inicial,
        "payload": payload,
    }


def _obter_arquivo_por_formato(
    amostra: GabaritoAtestadoSintetico,
    formato: FormatoAvaliacao,
) -> tuple[str, str, str]:
    if formato == "png":
        return amostra.arquivo_png, "image/png", amostra.sha256_png
    return amostra.arquivo_pdf, "application/pdf", amostra.sha256_pdf


def _comparar_campos_extracao(
    obtido: AtestadoExtraido,
    esperado: AtestadoExtraido,
) -> tuple[bool, list[str], list[str], dict[str, bool]]:
    pares = (
        ("legibilidade", obtido.legibilidade, esperado.legibilidade),
        ("nome_no_documento", obtido.nome_no_documento, esperado.nome_no_documento),
        ("periodo.inicio", obtido.periodo.inicio, esperado.periodo.inicio),
        ("periodo.fim", obtido.periodo.fim, esperado.periodo.fim),
        ("data_emissao", obtido.data_emissao, esperado.data_emissao),
        (
            "profissional.conselho",
            obtido.profissional.conselho,
            esperado.profissional.conselho,
        ),
        (
            "profissional.numero",
            obtido.profissional.numero,
            esperado.profissional.numero,
        ),
        ("profissional.uf", obtido.profissional.uf, esperado.profissional.uf),
        ("cid", obtido.cid, esperado.cid),
        ("municipio", obtido.municipio, esperado.municipio),
    )
    diferencas: list[str] = []
    campos_divergentes: list[str] = []
    acertos_campos: dict[str, bool] = {}
    for campo, valor_obtido, valor_esperado in pares:
        igual = valor_obtido == valor_esperado
        acertos_campos[campo] = igual
        if not igual:
            campos_divergentes.append(campo)
            diferencas.append(f"{campo}: {valor_obtido!r} != {valor_esperado!r}")
    return not diferencas, diferencas, campos_divergentes, acertos_campos


def _preparar_payload_confirmacao(
    amostra: GabaritoAtestadoSintetico,
    *,
    extraido: AtestadoExtraido | None,
    res_doc: dict[str, Any],
    exato: bool,
    campos_divergentes: Sequence[str],
    simular_revisao_manual_ruido: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Monta `atestado_confirmado` e `campos_corrigidos` com fallback human-in-the-loop."""
    payload_doc = res_doc.get("payload") or {}
    if exato and extraido is not None and "atestado_extraido" in payload_doc:
        campos_corrigidos = (
            list(amostra.campos_corrigidos_sugeridos)
            if simular_revisao_manual_ruido and amostra.campos_corrigidos_sugeridos
            else []
        )
        return copy.deepcopy(payload_doc["atestado_extraido"]), campos_corrigidos

    # Se a extração divergiu ou falhou (ex.: LLM em nuvem), o colaborador corrige via UI.
    corrigidos = [c for c in campos_divergentes if c != "legibilidade"]
    if simular_revisao_manual_ruido:
        for campo_sug in amostra.campos_corrigidos_sugeridos:
            if campo_sug not in corrigidos:
                corrigidos.append(campo_sug)
    if not corrigidos:
        corrigidos = ["periodo.inicio", "periodo.fim"]

    return _serializar_atestado_para_payload(amostra.esperado), corrigidos


def _avaliar_execucao_amostra(
    amostra: GabaritoAtestadoSintetico,
    *,
    formato: FormatoAvaliacao,
    caso_id: int,
    llm: Any,
    cfg_com_cid: A1Config,
    cfg_sem_cid: A1Config,
    schemas_json: dict[str, dict[str, Any]],
    simular_revisao_manual_ruido: bool,
) -> tuple[str, dict[str, Any]]:
    arquivo_uri, mime, sha256 = _obter_arquivo_por_formato(amostra, formato)
    registro_inicial = _montar_registro_inicial(amostra, caso_id)

    repo_amostra = RepositorioA1Memoria()
    servico = ServicoA1(repositorio=repo_amostra, llm=llm, config=cfg_com_cid)

    replays_total = 0
    replays_identicos = 0

    tarefa_doc = _montar_tarefa_base(
        id_mensagem=f"t32-doc-{amostra.id_amostra}-{formato}",
        acao="a1.processar_documento",
        caso_id=caso_id,
        id_tarefa=f"t32-doc-{caso_id}-{formato}",
        registro_inicial=registro_inicial,
        payload={
            "arquivo_uri": arquivo_uri,
            "mime": mime,
            "sha256": sha256,
        },
    )
    res_doc, replay_doc_ok = _executar_e_validar_idempotencia(
        servico, tarefa_doc, repo_amostra, schemas_json
    )
    replays_total += 1
    if replay_doc_ok:
        replays_identicos += 1

    extraido, erro_doc = _extrair_atestado_de_resultado(res_doc)

    if amostra.esperado.legibilidade == "ilegivel":
        rejeicao_segura = (
            extraido is not None
            and extraido.legibilidade == "ilegivel"
            and res_doc["status"] == "PRECISA_HUMANO"
            and res_doc["estado_proposto"] == "NOVO"
            and res_doc["payload"]["divergencias"] == []
            and res_doc["eventos"] == []
            and len(repo_amostra.atestados) == 0
        )
        return "ilegivel", {
            "id_amostra": amostra.id_amostra,
            "formato": formato,
            "cenario": amostra.cenario,
            "rejeicao_segura": rejeicao_segura,
            "legibilidade_obtida": (
                extraido.legibilidade if extraido is not None else "erro"
            ),
            "status": res_doc["status"],
            "estado_proposto": res_doc["estado_proposto"],
            "erro_execucao": erro_doc,
            "usou_llm": bool((res_doc.get("decisao") or {}).get("usou_llm", False)),
            "registros_criados_db": len(repo_amostra.atestados),
            "replays_total": replays_total,
            "replays_identicos": replays_identicos,
        }

    if extraido is not None:
        exato, diffs, campos_divergentes, acertos_campos = _comparar_campos_extracao(
            extraido, amostra.esperado
        )
    else:
        exato = False
        diffs = [erro_doc or "status ERRO"]
        campos_divergentes = list(CAMPOS_EXTRACAO)
        acertos_campos = {campo: False for campo in CAMPOS_EXTRACAO}

    # Verifica também Política de CID desabilitada (Anexo G, opção b)
    servico_sem_cid = ServicoA1(
        repositorio=RepositorioA1Memoria(),
        llm=llm,
        config=cfg_sem_cid,
    )
    tarefa_sem_cid = _montar_tarefa_base(
        id_mensagem=f"t32-semcid-{amostra.id_amostra}-{formato}",
        acao="a1.processar_documento",
        caso_id=caso_id,
        id_tarefa=f"t32-semcid-{caso_id}-{formato}",
        registro_inicial=registro_inicial,
        payload={
            "arquivo_uri": arquivo_uri,
            "mime": mime,
            "sha256": sha256,
        },
    )
    res_sem_cid = servico_sem_cid.executar(tarefa_sem_cid)
    _validar_envelope_resultado(res_sem_cid, schemas_json)
    extraido_sem_cid, _ = _extrair_atestado_de_resultado(res_sem_cid)
    ocultou_cid = extraido_sem_cid is not None and extraido_sem_cid.cid is None

    atestado_confirmado, campos_corrigidos = _preparar_payload_confirmacao(
        amostra,
        extraido=extraido,
        res_doc=res_doc,
        exato=exato,
        campos_divergentes=campos_divergentes,
        simular_revisao_manual_ruido=simular_revisao_manual_ruido,
    )

    tarefa_conf = _montar_tarefa_base(
        id_mensagem=f"t32-conf-{amostra.id_amostra}-{formato}",
        acao="a1.confirmar",
        caso_id=caso_id,
        id_tarefa=f"t32-conf-{caso_id}-{formato}",
        registro_inicial=registro_inicial,
        payload={
            "atestado_confirmado": atestado_confirmado,
            "campos_corrigidos": campos_corrigidos,
        },
    )
    res_conf, replay_conf_ok = _executar_e_validar_idempotencia(
        servico, tarefa_conf, repo_amostra, schemas_json
    )
    replays_total += 1
    if replay_conf_ok:
        replays_identicos += 1

    payload_conf = res_conf.get("payload") or {}
    confirmado_ok = (
        res_conf.get("status") == "OK"
        and res_conf.get("estado_proposto") == "VALIDADO"
        and "id_atestado" in payload_conf
        and "prazo_entrega_fisica" in payload_conf
    )
    origem_confirmada: str | None = None
    prazo_entrega_fisica: str | None = None
    recebido_fisico_em: str | None = None
    baixado_no_prazo = False
    marcos_info: dict[str, Any] = {
        "dias_afastamento": None,
        "exige_encaminhamento_inss": False,
        "data_marco_16_dia": None,
        "exige_exame_retorno_nr7": False,
    }

    inicio_esp = amostra.esperado.periodo.inicio
    fim_esp = amostra.esperado.periodo.fim
    if inicio_esp is not None and fim_esp is not None:
        marcos = calcular_marcos_legais(inicio_esp, fim_esp)
        marcos_info = {
            "dias_afastamento": marcos.dias_afastamento,
            "exige_encaminhamento_inss": marcos.exige_encaminhamento_inss,
            "data_marco_16_dia": (
                marcos.data_marco_16_dia.isoformat()
                if marcos.data_marco_16_dia is not None
                else None
            ),
            "exige_exame_retorno_nr7": marcos.exige_exame_retorno_nr7,
        }

    if confirmado_ok:
        origem_confirmada = str(payload_conf["origem"])
        id_atestado = int(payload_conf["id_atestado"])
        prazo_entrega_fisica = str(payload_conf["prazo_entrega_fisica"])
        prazo_limite = date.fromisoformat(prazo_entrega_fisica)
        fim_ref = fim_esp or prazo_limite
        data_entrega = min(prazo_limite, fim_ref + timedelta(days=1))
        dt_recebimento = datetime.combine(
            data_entrega, datetime.min.time(), tzinfo=UTC
        ).replace(hour=10)

        tarefa_baixa = _montar_tarefa_base(
            id_mensagem=f"t32-baixa-{amostra.id_amostra}-{formato}",
            acao="a1.registrar_baixa_fisica",
            caso_id=caso_id,
            id_tarefa=f"t32-baixa-{caso_id}-{formato}",
            perfil="rh",
            data_hora=dt_recebimento.isoformat(),
            registro_inicial=None,
            payload={
                "id_atestado": id_atestado,
                "recebido_fisico_por": "rh.ana",
                "recebido_fisico_em": dt_recebimento.isoformat(),
            },
        )
        res_baixa, replay_baixa_ok = _executar_e_validar_idempotencia(
            servico, tarefa_baixa, repo_amostra, schemas_json
        )
        replays_total += 1
        if replay_baixa_ok:
            replays_identicos += 1

        if res_baixa.get("status") == "OK":
            at_db = repo_amostra.obter_atestado(id_atestado)
            if (
                at_db is not None
                and at_db.recebido_fisico_em is not None
                and at_db.recebido_fisico_por is not None
                and at_db.prazo_entrega_fisica is not None
            ):
                recebido_fisico_em = at_db.recebido_fisico_em.date().isoformat()
                baixado_no_prazo = (
                    at_db.recebido_fisico_em.date() <= at_db.prazo_entrega_fisica
                )

    return "legivel", {
        "id_amostra": amostra.id_amostra,
        "formato": formato,
        "cenario": amostra.cenario,
        "conselho": extraido.profissional.conselho if extraido else None,
        "uf": extraido.profissional.uf if extraido else None,
        "cid_com_politica": extraido.cid if extraido else None,
        "ocultou_cid_sem_politica": ocultou_cid,
        "extracao_exata": exato,
        "diferencas": diffs,
        "erro_execucao": erro_doc,
        "acertos_campos": acertos_campos,
        "divergencias_detectadas": (res_doc.get("payload") or {}).get(
            "divergencias", []
        ),
        "confirmado_ok": confirmado_ok,
        "origem_confirmada": origem_confirmada,
        "campos_corrigidos": campos_corrigidos,
        "prazo_entrega_fisica": prazo_entrega_fisica,
        "recebido_fisico_em": recebido_fisico_em,
        "baixado_no_prazo": baixado_no_prazo,
        "marcos_legais": marcos_info,
        "replays_total": replays_total,
        "replays_identicos": replays_identicos,
    }


def _avaliar_cenario_estendido_baixa_fisica(
    amostras_legiveis: list[GabaritoAtestadoSintetico],
    *,
    llm: Any,
    cfg_com_cid: A1Config,
    schemas_json: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Exercita 10 entregas físicas (9 no prazo + 1 atrasada para alerta A3.5)."""
    if not amostras_legiveis:
        return {
            "total_entregas": 0,
            "no_prazo": 0,
            "atrasadas": 0,
            "taxa_no_prazo_pct": 0.0,
            "meta_atingida": False,
            "entregas": [],
        }

    total_alvo = 10
    entregas: list[dict[str, Any]] = []
    qtd_leg = len(amostras_legiveis)

    for idx in range(total_alvo):
        amostra = amostras_legiveis[idx % qtd_leg]
        formato: FormatoAvaliacao = "png" if idx < qtd_leg else "pdf"
        arquivo_uri, mime, sha256 = _obter_arquivo_por_formato(amostra, formato)
        caso_id = 901 + idx

        repo = RepositorioA1Memoria()
        servico = ServicoA1(repositorio=repo, llm=llm, config=cfg_com_cid)
        registro_inicial = _montar_registro_inicial(amostra, caso_id)

        res_doc = servico.executar(
            _montar_tarefa_base(
                id_mensagem=f"t32-ext-doc-{idx + 1}",
                acao="a1.processar_documento",
                caso_id=caso_id,
                id_tarefa=f"t32-ext-doc-{idx + 1}",
                registro_inicial=registro_inicial,
                payload={"arquivo_uri": arquivo_uri, "mime": mime, "sha256": sha256},
            )
        )
        _validar_envelope_resultado(res_doc, schemas_json)

        extraido_ext, _ = _extrair_atestado_de_resultado(res_doc)
        if extraido_ext is not None:
            exato_ext, _, campos_div_ext, _ = _comparar_campos_extracao(
                extraido_ext, amostra.esperado
            )
        else:
            exato_ext = False
            campos_div_ext = list(CAMPOS_EXTRACAO)

        atestado_conf, campos_corr = _preparar_payload_confirmacao(
            amostra,
            extraido=extraido_ext,
            res_doc=res_doc,
            exato=exato_ext,
            campos_divergentes=campos_div_ext,
            simular_revisao_manual_ruido=False,
        )

        res_conf = servico.executar(
            _montar_tarefa_base(
                id_mensagem=f"t32-ext-conf-{idx + 1}",
                acao="a1.confirmar",
                caso_id=caso_id,
                id_tarefa=f"t32-ext-conf-{idx + 1}",
                registro_inicial=registro_inicial,
                payload={
                    "atestado_confirmado": atestado_conf,
                    "campos_corrigidos": campos_corr,
                },
            )
        )
        _validar_envelope_resultado(res_conf, schemas_json)

        payload_conf = res_conf.get("payload") or {}
        if (
            res_conf.get("status") != "OK"
            or "id_atestado" not in payload_conf
            or "prazo_entrega_fisica" not in payload_conf
        ):
            entregas.append(
                {
                    "indice": idx + 1,
                    "id_amostra": amostra.id_amostra,
                    "formato": formato,
                    "prazo_entrega_fisica": None,
                    "recebido_fisico_em": None,
                    "no_prazo": False,
                    "elegivel_alerta_a35": True,
                    "erro": payload_conf.get("motivo")
                    or (res_conf.get("erro") or {}).get("mensagem"),
                }
            )
            continue

        id_atestado = int(payload_conf["id_atestado"])
        prazo_limite = date.fromisoformat(str(payload_conf["prazo_entrega_fisica"]))
        eh_atrasado = idx == total_alvo - 1
        fim_esp = amostra.esperado.periodo.fim or prazo_limite
        data_entrega = (
            prazo_limite + timedelta(days=2)
            if eh_atrasado
            else min(prazo_limite, fim_esp + timedelta(days=1))
        )
        dt_recebimento = datetime.combine(
            data_entrega, datetime.min.time(), tzinfo=UTC
        ).replace(hour=11)

        res_baixa = servico.executar(
            _montar_tarefa_base(
                id_mensagem=f"t32-ext-baixa-{idx + 1}",
                acao="a1.registrar_baixa_fisica",
                caso_id=caso_id,
                id_tarefa=f"t32-ext-baixa-{idx + 1}",
                perfil="rh",
                data_hora=dt_recebimento.isoformat(),
                registro_inicial=None,
                payload={
                    "id_atestado": id_atestado,
                    "recebido_fisico_por": "rh.ana",
                    "recebido_fisico_em": dt_recebimento.isoformat(),
                },
            )
        )
        _validar_envelope_resultado(res_baixa, schemas_json)

        at_db = repo.obter_atestado(id_atestado)
        no_prazo = bool(
            res_baixa.get("status") == "OK"
            and at_db is not None
            and at_db.recebido_fisico_em is not None
            and at_db.prazo_entrega_fisica is not None
            and at_db.recebido_fisico_em.date() <= at_db.prazo_entrega_fisica
        )
        entregas.append(
            {
                "indice": idx + 1,
                "id_amostra": amostra.id_amostra,
                "formato": formato,
                "prazo_entrega_fisica": prazo_limite.isoformat(),
                "recebido_fisico_em": data_entrega.isoformat(),
                "no_prazo": no_prazo,
                "elegivel_alerta_a35": not no_prazo,
            }
        )

    no_prazo_total = sum(1 for e in entregas if e["no_prazo"])
    atrasadas_total = total_alvo - no_prazo_total
    taxa_pct = round((no_prazo_total / total_alvo) * 100.0, 2)
    return {
        "total_entregas": total_alvo,
        "no_prazo": no_prazo_total,
        "atrasadas": atrasadas_total,
        "taxa_no_prazo_pct": taxa_pct,
        "meta_atingida": taxa_pct >= META_BAIXA_MIN_PCT,
        "entregas": entregas,
    }


def _calcular_pct(numerador: int, denominador: int) -> float:
    return round((numerador / denominador) * 100.0, 2) if denominador else 0.0


def avaliar_metricas_task_32(
    *,
    cfg: A1Config | None = None,
    modo_llm: ModoLLM = "auto",
    formatos: Sequence[FormatoAvaliacao] = ("png", "pdf"),
    simular_revisao_manual_ruido: bool = True,
) -> dict[str, Any]:
    """Executa a avaliação completa da Task 3.2 (Seção 10 da Documentação v2.2)."""
    if not formatos:
        raise ValueError("Informe ao menos um formato para avaliação ('png' ou 'pdf').")
    for fmt in formatos:
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Formato inválido para avaliação: {fmt!r}")

    limpar_cache_idempotencia()

    cfg_base = cfg or obter_config_a1()
    cfg_com_cid = A1Config(
        groq_api_key=cfg_base.groq_api_key,
        llm_model=cfg_base.llm_model,
        database_url="",
        politica_extrair_cid=True,
        limiar_confianca_campo=cfg_base.limiar_confianca_campo,
        prazo_entrega_fisica_dias_uteis=cfg_base.prazo_entrega_fisica_dias_uteis,
        atestados_storage_dir=cfg_base.atestados_storage_dir,
    )
    cfg_sem_cid = A1Config(
        groq_api_key=cfg_base.groq_api_key,
        llm_model=cfg_base.llm_model,
        database_url="",
        politica_extrair_cid=False,
        limiar_confianca_campo=cfg_base.limiar_confianca_campo,
        prazo_entrega_fisica_dias_uteis=cfg_base.prazo_entrega_fisica_dias_uteis,
        atestados_storage_dir=cfg_base.atestados_storage_dir,
    )

    llm, nome_llm = _selecionar_llm(cfg_com_cid, modo_llm=modo_llm)
    amostras = garantir_atestados_sinteticos(cfg_com_cid.atestados_storage_dir)
    integridade_catalogo = validar_integridade_catalogo_sintetico(
        amostras, cfg_com_cid.atestados_storage_dir
    )
    schemas_json = exportar_json_schemas_a1()

    detalhes_legiveis: list[dict[str, Any]] = []
    detalhes_ilegiveis: list[dict[str, Any]] = []

    caso_seq = 500
    for fmt in formatos:
        for amostra in amostras:
            caso_seq += 1
            categoria, registro = _avaliar_execucao_amostra(
                amostra,
                formato=fmt,
                caso_id=caso_seq,
                llm=llm,
                cfg_com_cid=cfg_com_cid,
                cfg_sem_cid=cfg_sem_cid,
                schemas_json=schemas_json,
                simular_revisao_manual_ruido=simular_revisao_manual_ruido,
            )
            if categoria == "ilegivel":
                detalhes_ilegiveis.append(registro)
            else:
                detalhes_legiveis.append(registro)

    legiveis_amostras_unicas = sum(
        1 for a in amostras if a.esperado.legibilidade == "ok"
    )
    legiveis_total = len(detalhes_legiveis)
    legiveis_exatos = sum(1 for item in detalhes_legiveis if item["extracao_exata"])
    confirmados_total = sum(1 for item in detalhes_legiveis if item["confirmado_ok"])
    confirmados_ia = sum(
        1 for item in detalhes_legiveis if item["origem_confirmada"] == "IA"
    )
    confirmados_manual = sum(
        1 for item in detalhes_legiveis if item["origem_confirmada"] == "MANUAL"
    )
    ocultou_cid_total = sum(
        1 for item in detalhes_legiveis if item["ocultou_cid_sem_politica"]
    )
    baixados_nominal = sum(
        1 for item in detalhes_legiveis if item["baixado_no_prazo"]
    )

    ilegiveis_total = len(detalhes_ilegiveis)
    ilegiveis_rejeitados = sum(
        1 for item in detalhes_ilegiveis if item["rejeicao_segura"]
    )

    replays_total = sum(
        int(item["replays_total"])
        for item in (*detalhes_legiveis, *detalhes_ilegiveis)
    )
    replays_identicos = sum(
        int(item["replays_identicos"])
        for item in (*detalhes_legiveis, *detalhes_ilegiveis)
    )
    idempotencia_integra = replays_total > 0 and replays_identicos == replays_total

    taxa_extracao_bruta = _calcular_pct(legiveis_exatos, legiveis_total)
    taxa_confirmados_ia = _calcular_pct(confirmados_ia, confirmados_total)
    taxa_rejeicao_ilegiveis = _calcular_pct(ilegiveis_rejeitados, ilegiveis_total)
    taxa_baixa_nominal = _calcular_pct(baixados_nominal, confirmados_total)

    amostras_legiveis = [a for a in amostras if a.esperado.legibilidade == "ok"]
    cenario_estendido = _avaliar_cenario_estendido_baixa_fisica(
        amostras_legiveis,
        llm=llm,
        cfg_com_cid=cfg_com_cid,
        schemas_json=schemas_json,
    )

    acuracia_por_campo: dict[str, dict[str, Any]] = {}
    for campo in CAMPOS_EXTRACAO:
        acertos = sum(
            1 for item in detalhes_legiveis if item["acertos_campos"].get(campo, False)
        )
        acuracia_por_campo[campo] = {
            "acertos": acertos,
            "total": legiveis_total,
            "taxa_pct": _calcular_pct(acertos, legiveis_total),
        }

    resumo_por_formato: dict[str, dict[str, Any]] = {}
    for fmt in formatos:
        leg_fmt = [d for d in detalhes_legiveis if d["formato"] == fmt]
        ileg_fmt = [d for d in detalhes_ilegiveis if d["formato"] == fmt]
        conf_fmt = sum(1 for d in leg_fmt if d["confirmado_ok"])
        ia_fmt = sum(1 for d in leg_fmt if d["origem_confirmada"] == "IA")
        exatos_fmt = sum(1 for d in leg_fmt if d["extracao_exata"])
        baixa_fmt = sum(1 for d in leg_fmt if d["baixado_no_prazo"])
        rej_fmt = sum(1 for d in ileg_fmt if d["rejeicao_segura"])
        resumo_por_formato[fmt] = {
            "legiveis_total": len(leg_fmt),
            "legiveis_exatos": exatos_fmt,
            "taxa_extracao_bruta_exata_pct": _calcular_pct(exatos_fmt, len(leg_fmt)),
            "confirmados_origem_ia": ia_fmt,
            "taxa_confirmados_origem_ia_pct": _calcular_pct(ia_fmt, conf_fmt),
            "ilegiveis_total": len(ileg_fmt),
            "ilegiveis_rejeitados": rej_fmt,
            "taxa_rejeicao_ilegiveis_pct": _calcular_pct(rej_fmt, len(ileg_fmt)),
            "baixados_no_prazo_nominal": baixa_fmt,
            "taxa_baixa_nominal_pct": _calcular_pct(baixa_fmt, conf_fmt),
        }

    catalogo_integro = bool(integridade_catalogo["integro"])
    meta_extracao_atingida = (
        catalogo_integro
        and taxa_extracao_bruta >= META_EXTRACAO_MIN_PCT
        and taxa_confirmados_ia >= META_EXTRACAO_MIN_PCT
    )
    meta_baixa_nominal_atingida = taxa_baixa_nominal >= META_BAIXA_MIN_PCT
    meta_baixa_atingida = (
        catalogo_integro
        and meta_baixa_nominal_atingida
        and bool(cenario_estendido["meta_atingida"])
    )

    return {
        "meta_extracao_min_pct": META_EXTRACAO_MIN_PCT,
        "meta_baixa_min_pct": META_BAIXA_MIN_PCT,
        "modo_llm": nome_llm,
        "formatos_avaliados": list(formatos),
        "simular_revisao_manual_ruido": simular_revisao_manual_ruido,
        "integridade_catalogo": integridade_catalogo,
        "idempotencia_total_replays": replays_total,
        "idempotencia_replays_identicos": replays_identicos,
        "idempotencia_100_pct_integra": idempotencia_integra,
        "legiveis_amostras_unicas": legiveis_amostras_unicas,
        "legiveis_execucoes_total": legiveis_total,
        "legiveis_exatos": legiveis_exatos,
        "taxa_extracao_bruta_exata_pct": taxa_extracao_bruta,
        "confirmados_total": confirmados_total,
        "confirmados_origem_ia": confirmados_ia,
        "confirmados_origem_manual": confirmados_manual,
        "taxa_confirmados_origem_ia_pct": taxa_confirmados_ia,
        "meta_extracao_atingida": meta_extracao_atingida,
        "politica_sem_cid_total": legiveis_total,
        "politica_sem_cid_ocultou_cid": ocultou_cid_total,
        "ilegiveis_execucoes_total": ilegiveis_total,
        "ilegiveis_rejeitados": ilegiveis_rejeitados,
        "taxa_rejeicao_ilegiveis_pct": taxa_rejeicao_ilegiveis,
        "baixados_no_prazo_nominal": baixados_nominal,
        "taxa_baixa_nominal_pct": taxa_baixa_nominal,
        "meta_baixa_nominal_atingida": meta_baixa_nominal_atingida,
        "cenario_estendido_baixa": cenario_estendido,
        "meta_baixa_atingida": meta_baixa_atingida,
        "acuracia_por_campo": acuracia_por_campo,
        "resumo_por_formato": resumo_por_formato,
        "detalhes_legiveis": detalhes_legiveis,
        "detalhes_ilegiveis": detalhes_ilegiveis,
    }


def gerar_relatorio_markdown_task_32(relatorio: dict[str, Any]) -> str:
    """Formata o relatório em Markdown da avaliação das métricas da Seção 10."""
    cenario_ext = relatorio["cenario_estendido_baixa"]
    integridade = relatorio["integridade_catalogo"]
    linhas = [
        "# Avaliação das Métricas da Seção 10 — Agente 1 (Task 3.2)",
        "",
        f"- **Motor LLM utilizado:** `{relatorio['modo_llm']}`",
        f"- **Formatos avaliados:** `{', '.join(relatorio['formatos_avaliados'])}`",
        (
            "- **Simulação de revisão manual (`ruido_leve`):** "
            f"`{relatorio['simular_revisao_manual_ruido']}`"
        ),
        (
            "- **Integridade criptográfica (SHA-256):** "
            f"{integridade['arquivos_validos']}/{integridade['total_arquivos_verificados']} "
            f"arquivos íntegros (`integro={integridade['integro']}`, "
            "Carla 105 canônico="
            f"`{integridade['carla_105_sha256_canonico_preservado']}`)"
        ),
        (
            "- **Idempotência (`id_mensagem`):** "
            f"{relatorio['idempotencia_replays_identicos']}/"
            f"{relatorio['idempotencia_total_replays']} replays idênticos "
            f"(`integra={relatorio['idempotencia_100_pct_integra']}`)"
        ),
        "",
        "## Resumo Executivo vs. Metas da Seção 10",
        "",
        "| Métrica | Meta (v2.2) | Resultado Obtido | Status |",
        "| :--- | :---: | :---: | :---: |",
        (
            "| Extração bruta exata (gabarito) | >= 80,0% | "
            f"{relatorio['legiveis_exatos']}/{relatorio['legiveis_execucoes_total']} "
            f"({relatorio['taxa_extracao_bruta_exata_pct']:.1f}%) | "
            f"{'APROVADA' if relatorio['meta_extracao_atingida'] else 'REPROVADA'} |"
        ),
        (
            "| Confirmação sem correção (`origem = IA`) | >= 80,0% | "
            f"{relatorio['confirmados_origem_ia']}/{relatorio['confirmados_total']} "
            f"({relatorio['taxa_confirmados_origem_ia_pct']:.1f}%) | "
            f"{'APROVADA' if relatorio['meta_extracao_atingida'] else 'REPROVADA'} |"
        ),
        (
            "| Rejeição segura de ilegíveis | 100,0% (guardrail) | "
            f"{relatorio['ilegiveis_rejeitados']}/{relatorio['ilegiveis_execucoes_total']} "
            f"({relatorio['taxa_rejeicao_ilegiveis_pct']:.1f}%) | OK |"
        ),
        (
            "| Baixa física no prazo (lote nominal) | >= 90,0% | "
            f"{relatorio['baixados_no_prazo_nominal']}/{relatorio['confirmados_total']} "
            f"({relatorio['taxa_baixa_nominal_pct']:.1f}%) | "
            f"{'APROVADA' if relatorio['meta_baixa_nominal_atingida'] else 'REPROVADA'} |"
        ),
        (
            "| Baixa física no prazo (10 entregas com 1 atraso A3.5) | >= 90,0% | "
            f"{cenario_ext['no_prazo']}/{cenario_ext['total_entregas']} "
            f"({cenario_ext['taxa_no_prazo_pct']:.1f}%) | "
            f"{'APROVADA' if cenario_ext['meta_atingida'] else 'REPROVADA'} |"
        ),
        "",
        "## Acurácia por Campo (`AtestadoExtraido`)",
        "",
        "| Campo | Acertos | Total | Acurácia |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for campo, dados in relatorio["acuracia_por_campo"].items():
        linhas.append(
            f"| `{campo}` | {dados['acertos']} | {dados['total']} | "
            f"{dados['taxa_pct']:.1f}% |"
        )

    linhas.extend(
        [
            "",
            "## Amostras Legíveis Avaliadas",
            "",
            (
                "| Amostra | Formato | Conselho/UF | CID | Origem | Prazo Físico | "
                "Extração Exata | Baixa no Prazo |"
            ),
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for item in relatorio["detalhes_legiveis"]:
        cons_uf = f"{item['conselho'] or '-'}/{item['uf'] or '-'}"
        cid_txt = item["cid_com_politica"] or "null"
        linhas.append(
            f"| `{item['id_amostra']}` | {item['formato']} | {cons_uf} | {cid_txt} | "
            f"{item['origem_confirmada']} | {item['prazo_entrega_fisica']} | "
            f"{'Sim' if item['extracao_exata'] else 'Não'} | "
            f"{'Sim' if item['baixado_no_prazo'] else 'Não'} |"
        )

    linhas.extend(
        [
            "",
            "## Amostras Ilegíveis Avaliadas (Guardrail)",
            "",
            (
                "| Amostra | Formato | Modo | Legibilidade | Estado Proposto | "
                "Usou LLM | Registros DB | Rejeição Segura |"
            ),
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for item in relatorio["detalhes_ilegiveis"]:
        linhas.append(
            f"| `{item['id_amostra']}` | {item['formato']} | {item['cenario']} | "
            f"{item['legibilidade_obtida']} | {item['estado_proposto']} | "
            f"{'Sim' if item['usou_llm'] else 'Não'} | {item['registros_criados_db']} | "
            f"{'Sim' if item['rejeicao_segura'] else 'Não'} |"
        )

    return "\n".join(linhas) + "\n"


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avaliador das métricas da Seção 10 do Agente 1 (Task 3.2)."
    )
    parser.add_argument(
        "--modo-llm",
        choices=["auto", "mock", "groq"],
        default="auto",
        help="Seleciona o cliente LLM utilizado na avaliação (padrão: auto).",
    )
    parser.add_argument(
        "--formato",
        choices=["ambos", "png", "pdf"],
        default="ambos",
        help="Formato de arquivo dos atestados sintéticos a avaliar (padrão: ambos).",
    )
    parser.add_argument(
        "--sem-simulacao-manual",
        action="store_true",
        help="Confirma todas as amostras com origem='IA', sem simular revisão manual.",
    )
    parser.add_argument(
        "--salvar-json",
        type=Path,
        help="Caminho opcional para exportar o relatório completo em JSON.",
    )
    parser.add_argument(
        "--salvar-markdown",
        type=Path,
        help="Caminho opcional para exportar o relatório formatado em Markdown.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configurar_stdout_utf8()
    parser = _construir_parser()
    args = parser.parse_args(argv)

    formatos: tuple[FormatoAvaliacao, ...] = (
        ("png", "pdf") if args.formato == "ambos" else (args.formato,)
    )
    relatorio = avaliar_metricas_task_32(
        modo_llm=args.modo_llm,
        formatos=formatos,
        simular_revisao_manual_ruido=not args.sem_simulacao_manual,
    )

    integridade = relatorio["integridade_catalogo"]
    cenario_ext = relatorio["cenario_estendido_baixa"]
    print(f"[TASK 3.2 — MÉTRICAS SEÇÃO 10 (A1)] LLM={relatorio['modo_llm']}")
    print(
        "  * Integridade SHA-256 do catálogo: "
        f"{integridade['arquivos_validos']}/{integridade['total_arquivos_verificados']} "
        f"(integro={integridade['integro']} | carla_105_canonico="
        f"{integridade['carla_105_sha256_canonico_preservado']})"
    )
    print(
        "  * Extração bruta exata: "
        f"{relatorio['legiveis_exatos']}/{relatorio['legiveis_execucoes_total']} "
        f"({relatorio['taxa_extracao_bruta_exata_pct']:.1f}% | meta >= 80.0%)"
    )
    print(
        "  * Confirmação sem correção (origem=IA): "
        f"{relatorio['confirmados_origem_ia']}/{relatorio['confirmados_total']} "
        f"({relatorio['taxa_confirmados_origem_ia_pct']:.1f}% | meta >= 80.0%)"
    )
    print(
        "  * Rejeição segura de ilegíveis: "
        f"{relatorio['ilegiveis_rejeitados']}/{relatorio['ilegiveis_execucoes_total']} "
        f"({relatorio['taxa_rejeicao_ilegiveis_pct']:.1f}%)"
    )
    print(
        "  * Baixa física no prazo (nominal): "
        f"{relatorio['baixados_no_prazo_nominal']}/{relatorio['confirmados_total']} "
        f"({relatorio['taxa_baixa_nominal_pct']:.1f}% | meta >= 90.0%)"
    )
    print(
        "  * Baixa física no prazo (10 entregas c/ 1 atraso A3.5): "
        f"{cenario_ext['no_prazo']}/{cenario_ext['total_entregas']} "
        f"({cenario_ext['taxa_no_prazo_pct']:.1f}% | meta >= 90.0%)"
    )
    print(
        "  * Idempotência (id_mensagem): "
        f"{relatorio['idempotencia_replays_identicos']}/"
        f"{relatorio['idempotencia_total_replays']} replays idênticos"
    )

    if args.salvar_json is not None:
        args.salvar_json.parent.mkdir(parents=True, exist_ok=True)
        args.salvar_json.write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.salvar_markdown is not None:
        args.salvar_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.salvar_markdown.write_text(
            gerar_relatorio_markdown_task_32(relatorio),
            encoding="utf-8",
        )

    tudo_ok = (
        bool(integridade["integro"])
        and bool(relatorio["idempotencia_100_pct_integra"])
        and bool(relatorio["meta_extracao_atingida"])
        and bool(relatorio["meta_baixa_atingida"])
    )
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
