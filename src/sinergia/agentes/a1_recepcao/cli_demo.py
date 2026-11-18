"""Simulador interativo e CLI do Agente 1 (`sinergia/1`) — Task 2B.2.

Permite executar e demonstrar as 4 ações oficiais do Agente A1 (`a1.conversar`,
`a1.processar_documento`, `a1.confirmar` e `a1.registrar_baixa_fisica`) no
terminal, simulando o Orquestrador A0 enviando envelopes `TAREFA`:

    python -m sinergia.agentes.a1_recepcao.cli_demo --roteiro carla
    python -m sinergia.agentes.a1_recepcao.cli_demo --avaliar-metricas
    python -m sinergia.agentes.a1_recepcao.cli_demo --gerar-atestados
    python -m sinergia.agentes.a1_recepcao.cli_demo --tarefa caminho.tarefa.json
    python -m sinergia.agentes.a1_recepcao.cli_demo
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import jsonschema
import psycopg

from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.extrator_llm import MockLLM
from sinergia.agentes.a1_recepcao.gerador_sintetico import (
    GabaritoAtestadoSintetico,
    gerar_atestados_golden,
    garantir_atestados_sinteticos,
)
from sinergia.agentes.a1_recepcao.repositorio import (
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    RepositorioA1Protocol,
)
from sinergia.agentes.a1_recepcao.schemas import (
    AtestadoExtraido,
    ResultadoA1,
    exportar_json_schemas_a1,
)
from sinergia.agentes.a1_recepcao.servico import ServicoA1

_RAIZ_REPO = Path(__file__).resolve().parents[4]
_FIXTURES_DIR = _RAIZ_REPO / "tests" / "a1_recepcao" / "fixtures"


def _configurar_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _postgres_disponivel(database_url: str) -> bool:
    if not database_url.strip():
        return False
    try:
        with psycopg.connect(database_url, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT count(*) FROM cadastro.colaborador WHERE id_colaborador = 42"
            ).fetchone()
            return bool(row and row[0] >= 1)
    except psycopg.Error:
        return False


def _selecionar_backends(
    cfg: A1Config,
    *,
    modo_repo: str = "auto",
    modo_llm: str = "auto",
) -> tuple[RepositorioA1Protocol, Any, str, str]:
    if modo_repo == "postgres":
        repositorio: RepositorioA1Protocol = RepositorioA1Postgres(cfg.database_url)
        nome_repo = "postgres"
    elif modo_repo == "memoria":
        repositorio = RepositorioA1Memoria()
        nome_repo = "memoria"
    elif _postgres_disponivel(cfg.database_url):
        repositorio = RepositorioA1Postgres(cfg.database_url)
        nome_repo = "postgres"
    else:
        repositorio = RepositorioA1Memoria()
        nome_repo = "memoria"

    if modo_llm == "groq":
        if not cfg.groq_api_key.strip():
            raise ValueError("GROQ_API_KEY não configurada para --modo-llm groq.")
        llm: Any = None
        nome_llm = f"groq ({cfg.llm_model})"
    elif modo_llm == "mock":
        llm = MockLLM()
        nome_llm = "MockLLM (offline)"
    elif cfg.groq_api_key.strip():
        llm = None
        nome_llm = f"groq ({cfg.llm_model})"
    else:
        llm = MockLLM()
        nome_llm = "MockLLM (offline)"

    return repositorio, llm, nome_repo, nome_llm


def _carregar_fixture(subcaminho: str) -> dict[str, Any]:
    caminho = _FIXTURES_DIR / subcaminho
    if not caminho.is_file():
        raise FileNotFoundError(f"Fixture não encontrada: {caminho}")
    return json.loads(caminho.read_text(encoding="utf-8"))


def _validar_resultado_contrato(
    resultado: dict[str, Any],
    schemas_json: dict[str, dict[str, Any]],
) -> None:
    jsonschema.validate(instance=resultado, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(resultado).validar_payload_acao()


def _imprimir_bloco_json(titulo: str, dados: dict[str, Any]) -> None:
    print(f"\n=== {titulo} ===")
    print(json.dumps(dados, ensure_ascii=False, indent=2))


def _registro_inicial_carla(
    *,
    id_caso: int = 105,
    inicio_informado: str = "2026-11-09",
    dias_informados: int = 5,
    tem_documento: bool = True,
) -> dict[str, Any]:
    return {
        "id_caso": id_caso,
        "thread_id": f"chat-42-{id_caso}",
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
        "tipo_solicitacao": "atestado_medico",
        "inicio_informado": inicio_informado,
        "dias_informados": dias_informados,
        "tem_documento": tem_documento,
    }


def _limpar_demo_postgres(
    database_url: str,
    id_atestado: int | None,
    id_profissional: int | None,
) -> None:
    if id_atestado is None and id_profissional is None:
        return
    with psycopg.connect(database_url) as conn:
        if id_atestado is not None:
            conn.execute(
                "DELETE FROM afastamento.atestado WHERE id_atestado = %s",
                (id_atestado,),
            )
        if id_profissional is not None:
            conn.execute(
                "DELETE FROM afastamento.profissional_saude "
                "WHERE id_profissional = %s AND NOT EXISTS ("
                "SELECT 1 FROM afastamento.atestado WHERE id_profissional = %s"
                ")",
                (id_profissional, id_profissional),
            )
        conn.commit()


def executar_roteiro_carla(
    *,
    cfg: A1Config | None = None,
    modo_repo: str = "auto",
    modo_llm: str = "auto",
    persistir: bool = False,
    politica_extrair_cid: bool = False,
) -> int:
    """Executa o caso completo 105 da Carla S. (Anexos D.1 e D.2) validando contratos."""
    cfg_base = cfg or obter_config_a1()
    cfg_demo = A1Config(
        groq_api_key=cfg_base.groq_api_key,
        llm_model=cfg_base.llm_model,
        database_url=cfg_base.database_url,
        politica_extrair_cid=politica_extrair_cid,
        limiar_confianca_campo=cfg_base.limiar_confianca_campo,
        prazo_entrega_fisica_dias_uteis=cfg_base.prazo_entrega_fisica_dias_uteis,
        atestados_storage_dir=cfg_base.atestados_storage_dir,
    )
    garantir_atestados_sinteticos(cfg_demo.atestados_storage_dir)

    repositorio, llm, nome_repo, nome_llm = _selecionar_backends(
        cfg_demo, modo_repo=modo_repo, modo_llm=modo_llm
    )

    if (
        nome_repo == "postgres"
        and modo_repo == "auto"
        and repositorio.verificar_duplicidade_arquivo(
            ".storage/atestados/atestado_carla_105.png", id_colaborador=42
        )
        is not None
    ):
        print(
            "[AVISO] Atestado de demonstração já persistido no PostgreSQL; "
            "alternando para RepositorioA1Memoria isolado para não colidir."
        )
        repositorio = RepositorioA1Memoria()
        nome_repo = "memoria"

    print(
        f"[ROTEIRO CARLA S. — CHECKPOINT 2] Repositório={nome_repo} | LLM={nome_llm} | "
        f"POLITICA_EXTRAIR_CID={cfg_demo.politica_extrair_cid}"
    )

    schemas_json = exportar_json_schemas_a1()
    servico = ServicoA1(repositorio=repositorio, llm=llm, config=cfg_demo)

    id_atestado_criado: int | None = None
    id_prof_criado: int | None = None
    sufixo = uuid4().hex[:8] if nome_repo == "postgres" else None

    try:
        # 1. a1.conversar
        t1 = copy.deepcopy(_carregar_fixture("a1.conversar/falta_arquivo.tarefa.json"))
        if sufixo:
            t1["id_mensagem"] = f"demo-conv-{sufixo}"
        _imprimir_bloco_json("1) TAREFA a1.conversar", t1)
        r1 = servico.executar(t1)
        _validar_resultado_contrato(r1, schemas_json)
        _imprimir_bloco_json("1) RESULTADO a1.conversar", r1)

        # 2. a1.processar_documento
        t2 = copy.deepcopy(
            _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
        )
        if sufixo:
            t2["id_mensagem"] = f"demo-doc-{sufixo}"
        _imprimir_bloco_json("2) TAREFA a1.processar_documento", t2)
        r2 = servico.executar(t2)
        _validar_resultado_contrato(r2, schemas_json)
        _imprimir_bloco_json("2) RESULTADO a1.processar_documento", r2)

        pend = repositorio.obter_ultimo_atestado_pendente(42)
        if pend is not None:
            id_atestado_criado = pend.id_atestado
            id_prof_criado = pend.id_profissional

        # 3. a1.confirmar (válido)
        t3 = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
        if sufixo:
            t3["id_mensagem"] = f"demo-conf-{sufixo}"
        _imprimir_bloco_json("3) TAREFA a1.confirmar (válido)", t3)
        r3 = servico.executar(t3)
        _validar_resultado_contrato(r3, schemas_json)
        _imprimir_bloco_json("3) RESULTADO a1.confirmar (válido)", r3)
        id_atestado_confirmado = int(r3["payload"]["id_atestado"])
        id_atestado_criado = id_atestado_confirmado

        # 4. a1.registrar_baixa_fisica
        t4 = copy.deepcopy(
            _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
        )
        if sufixo:
            t4["id_mensagem"] = f"demo-baixa-{sufixo}"
        t4["payload"]["id_atestado"] = id_atestado_confirmado
        _imprimir_bloco_json("4) TAREFA a1.registrar_baixa_fisica", t4)
        r4 = servico.executar(t4)
        _validar_resultado_contrato(r4, schemas_json)
        _imprimir_bloco_json("4) RESULTADO a1.registrar_baixa_fisica", r4)

        # 5. a1.confirmar (datas inválidas — Anexo D.2)
        t5 = copy.deepcopy(_carregar_fixture("a1.confirmar/datas_invalidas.tarefa.json"))
        if sufixo:
            t5["id_mensagem"] = f"demo-inv-{sufixo}"
        _imprimir_bloco_json("5) TAREFA a1.confirmar (datas inválidas)", t5)
        r5 = servico.executar(t5)
        _validar_resultado_contrato(r5, schemas_json)
        _imprimir_bloco_json("5) RESULTADO a1.confirmar (datas inválidas)", r5)

        print("\n[OK] Roteiro completo da Carla S. validado 100% no contrato sinergia/1.")
        return 0
    finally:
        if nome_repo == "postgres" and not persistir:
            _limpar_demo_postgres(
                cfg_demo.database_url, id_atestado_criado, id_prof_criado
            )


def _extracao_bate_gabarito(
    obtido: AtestadoExtraido,
    esperado: AtestadoExtraido,
) -> tuple[bool, list[str]]:
    diferencas: list[str] = []
    if obtido.legibilidade != esperado.legibilidade:
        diferencas.append(f"legibilidade: {obtido.legibilidade} != {esperado.legibilidade}")
    if obtido.nome_no_documento != esperado.nome_no_documento:
        diferencas.append(
            f"nome_no_documento: {obtido.nome_no_documento!r} != {esperado.nome_no_documento!r}"
        )
    if obtido.periodo.inicio != esperado.periodo.inicio:
        diferencas.append(f"periodo.inicio: {obtido.periodo.inicio} != {esperado.periodo.inicio}")
    if obtido.periodo.fim != esperado.periodo.fim:
        diferencas.append(f"periodo.fim: {obtido.periodo.fim} != {esperado.periodo.fim}")
    if obtido.data_emissao != esperado.data_emissao:
        diferencas.append(f"data_emissao: {obtido.data_emissao} != {esperado.data_emissao}")
    if obtido.profissional.conselho != esperado.profissional.conselho:
        diferencas.append(
            f"conselho: {obtido.profissional.conselho!r} != {esperado.profissional.conselho!r}"
        )
    if obtido.profissional.numero != esperado.profissional.numero:
        diferencas.append(
            f"numero: {obtido.profissional.numero!r} != {esperado.profissional.numero!r}"
        )
    if obtido.profissional.uf != esperado.profissional.uf:
        diferencas.append(f"uf: {obtido.profissional.uf!r} != {esperado.profissional.uf!r}")
    if obtido.cid != esperado.cid:
        diferencas.append(f"cid: {obtido.cid!r} != {esperado.cid!r}")
    if obtido.municipio != esperado.municipio:
        diferencas.append(f"municipio: {obtido.municipio!r} != {esperado.municipio!r}")
    return not diferencas, diferencas


def avaliar_metricas_secao10(
    *,
    cfg: A1Config | None = None,
    modo_llm: str = "auto",
) -> dict[str, Any]:
    """Avalia as metas da Seção 10 sobre os atestados sintéticos com gabarito."""
    cfg_base = cfg or obter_config_a1()
    cfg_eval = A1Config(
        groq_api_key=cfg_base.groq_api_key,
        llm_model=cfg_base.llm_model,
        database_url="",
        politica_extrair_cid=True,
        limiar_confianca_campo=cfg_base.limiar_confianca_campo,
        prazo_entrega_fisica_dias_uteis=cfg_base.prazo_entrega_fisica_dias_uteis,
        atestados_storage_dir=cfg_base.atestados_storage_dir,
    )
    _, llm, _, nome_llm = _selecionar_backends(
        cfg_eval, modo_repo="memoria", modo_llm=modo_llm
    )
    amostras: list[GabaritoAtestadoSintetico] = garantir_atestados_sinteticos(
        cfg_eval.atestados_storage_dir
    )

    schemas_json = exportar_json_schemas_a1()
    legiveis_total = 0
    legiveis_exatos = 0
    ilegiveis_rejeitados = 0
    ilegiveis_total = 0
    confirmados_total = 0
    baixados_no_prazo = 0

    print(f"\n[AVALIAÇÃO DE MÉTRICAS — SEÇÃO 10] LLM={nome_llm}")
    for idx, amostra in enumerate(amostras, start=1):
        repo_amostra = RepositorioA1Memoria()
        servico = ServicoA1(repositorio=repo_amostra, llm=llm, config=cfg_eval)
        caso_id = 500 + idx
        inicio_esp = amostra.esperado.periodo.inicio
        fim_esp = amostra.esperado.periodo.fim
        dias_esp = (
            (fim_esp - inicio_esp).days + 1
            if inicio_esp is not None and fim_esp is not None
            else 1
        )

        tarefa_doc = {
            "protocolo": "sinergia/1",
            "id_mensagem": f"eval-doc-{amostra.id_amostra}",
            "tipo": "TAREFA",
            "acao": "a1.processar_documento",
            "origem": "A0",
            "destino": "A1",
            "correlacao": {
                "id_caso": caso_id,
                "thread_id": f"chat-42-{caso_id}",
                "id_tarefa": f"t-eval-doc-{idx}",
            },
            "contexto": {
                "id_colaborador": 42,
                "perfil": "colaborador",
                "canal": "chat",
                "data_hora": "2026-11-09T09:10:00-03:00",
            },
            "registro_inicial": _registro_inicial_carla(
                id_caso=caso_id,
                inicio_informado=(inicio_esp.isoformat() if inicio_esp else "2026-11-09"),
                dias_informados=dias_esp,
                tem_documento=True,
            ),
            "payload": {
                "arquivo_uri": amostra.arquivo_png,
                "mime": "image/png",
                "sha256": amostra.sha256_png,
            },
        }
        res_doc = servico.executar(tarefa_doc)
        _validar_resultado_contrato(res_doc, schemas_json)
        extraido = AtestadoExtraido.model_validate(res_doc["payload"]["atestado_extraido"])

        if amostra.esperado.legibilidade == "ilegivel":
            ilegiveis_total += 1
            rejeitou = (
                extraido.legibilidade == "ilegivel"
                and res_doc["estado_proposto"] == "NOVO"
            )
            if rejeitou:
                ilegiveis_rejeitados += 1
            print(
                f"  - {amostra.id_amostra}: rejeicao_ilegivel={rejeitou} "
                f"(estado_proposto={res_doc['estado_proposto']})"
            )
            continue

        legiveis_total += 1
        exato, diffs = _extracao_bate_gabarito(extraido, amostra.esperado)
        if exato:
            legiveis_exatos += 1
        print(
            f"  - {amostra.id_amostra}: extracao_exata={exato}"
            + (f" diffs={diffs}" if diffs else "")
        )

        # Simula confirmação e baixa física dentro do prazo (1 dia após o fim)
        tarefa_conf = {
            "protocolo": "sinergia/1",
            "id_mensagem": f"eval-conf-{amostra.id_amostra}",
            "tipo": "TAREFA",
            "acao": "a1.confirmar",
            "origem": "A0",
            "destino": "A1",
            "correlacao": tarefa_doc["correlacao"],
            "contexto": tarefa_doc["contexto"],
            "registro_inicial": tarefa_doc["registro_inicial"],
            "payload": {
                "atestado_confirmado": res_doc["payload"]["atestado_extraido"],
                "campos_corrigidos": [],
            },
        }
        res_conf = servico.executar(tarefa_conf)
        _validar_resultado_contrato(res_conf, schemas_json)
        if res_conf["status"] == "OK" and res_conf["estado_proposto"] == "VALIDADO":
            confirmados_total += 1
            id_atestado = int(res_conf["payload"]["id_atestado"])
            prazo_limite = datetime.fromisoformat(
                res_conf["payload"]["prazo_entrega_fisica"]
            ).date()
            data_recebimento = datetime.combine(
                min(prazo_limite, (fim_esp or prazo_limite) + timedelta(days=1)),
                datetime.min.time(),
                tzinfo=UTC,
            ).replace(hour=10)

            tarefa_baixa = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"eval-baixa-{amostra.id_amostra}",
                "tipo": "TAREFA",
                "acao": "a1.registrar_baixa_fisica",
                "origem": "A0",
                "destino": "A1",
                "correlacao": tarefa_doc["correlacao"],
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "rh",
                    "canal": "chat",
                    "data_hora": data_recebimento.isoformat(),
                },
                "registro_inicial": None,
                "payload": {
                    "id_atestado": id_atestado,
                    "recebido_fisico_por": "rh.ana",
                    "recebido_fisico_em": data_recebimento.isoformat(),
                },
            }
            res_baixa = servico.executar(tarefa_baixa)
            _validar_resultado_contrato(res_baixa, schemas_json)
            at_db = repo_amostra.obter_atestado(id_atestado)
            if (
                at_db is not None
                and at_db.recebido_fisico_em is not None
                and at_db.prazo_entrega_fisica is not None
                and at_db.recebido_fisico_em.date() <= at_db.prazo_entrega_fisica
            ):
                baixados_no_prazo += 1

    taxa_extracao = (legiveis_exatos / legiveis_total) * 100.0 if legiveis_total else 0.0
    taxa_baixa = (baixados_no_prazo / confirmados_total) * 100.0 if confirmados_total else 0.0
    relatorio = {
        "legiveis_total": legiveis_total,
        "legiveis_exatos": legiveis_exatos,
        "taxa_extracao_sem_correcao_pct": round(taxa_extracao, 2),
        "meta_extracao_atingida": taxa_extracao >= 80.0,
        "ilegiveis_total": ilegiveis_total,
        "ilegiveis_rejeitados": ilegiveis_rejeitados,
        "confirmados_total": confirmados_total,
        "baixados_no_prazo": baixados_no_prazo,
        "taxa_baixa_no_prazo_pct": round(taxa_baixa, 2),
        "meta_baixa_atingida": taxa_baixa >= 90.0,
    }
    print("\nResumo Seção 10:")
    print(
        f"  * Extraídos sem correção manual: {legiveis_exatos}/{legiveis_total} "
        f"({taxa_extracao:.1f}% | meta >= 80%)"
    )
    print(
        f"  * Ilegíveis rejeitados: {ilegiveis_rejeitados}/{ilegiveis_total}"
    )
    print(
        f"  * Físicos baixados no prazo: {baixados_no_prazo}/{confirmados_total} "
        f"({taxa_baixa:.1f}% | meta >= 90%)"
    )
    return relatorio


def executar_arquivo_tarefa(
    caminho_tarefa: Path,
    *,
    cfg: A1Config | None = None,
    modo_repo: str = "auto",
    modo_llm: str = "auto",
) -> int:
    """Executa uma `TAREFA` carregada de arquivo JSON e imprime o `RESULTADO`."""
    cfg_base = cfg or obter_config_a1()
    repositorio, llm, nome_repo, nome_llm = _selecionar_backends(
        cfg_base, modo_repo=modo_repo, modo_llm=modo_llm
    )
    tarefa = json.loads(caminho_tarefa.read_text(encoding="utf-8"))
    servico = ServicoA1(repositorio=repositorio, llm=llm, config=cfg_base)
    resultado = servico.executar(tarefa)
    _validar_resultado_contrato(resultado, exportar_json_schemas_a1())
    print(f"[TAREFA] {caminho_tarefa} | Repositório={nome_repo} | LLM={nome_llm}")
    _imprimir_bloco_json("RESULTADO", resultado)
    return 0


def executar_menu_interativo(
    *,
    cfg: A1Config | None = None,
    modo_repo: str = "auto",
    modo_llm: str = "auto",
    input_fn: Callable[[str], str] = input,
) -> int:
    """Menu interativo no terminal simulando o A0 enviando tarefas ao A1."""
    cfg_base = cfg or obter_config_a1()
    repositorio, llm, nome_repo, nome_llm = _selecionar_backends(
        cfg_base, modo_repo=modo_repo, modo_llm=modo_llm
    )
    servico = ServicoA1(repositorio=repositorio, llm=llm, config=cfg_base)
    schemas_json = exportar_json_schemas_a1()
    garantir_atestados_sinteticos(cfg_base.atestados_storage_dir)
    ultimo_id_atestado: int | None = None

    while True:
        print(
            "\n=== SinergIA A1 — Simulador Interativo (sinergia/1) ===\n"
            f"Backends ativos: Repositório={nome_repo} | LLM={nome_llm}\n"
            "  1. a1.conversar\n"
            "  2. a1.processar_documento\n"
            "  3. a1.confirmar\n"
            "  4. a1.registrar_baixa_fisica\n"
            "  5. Rodar roteiro completo Carla S. (Checkpoint 2)\n"
            "  6. Avaliar métricas da Seção 10 (atestados sintéticos)\n"
            "  7. Gerar/listar atestados sintéticos (.storage/atestados)\n"
            "  0. Sair"
        )
        try:
            opcao = input_fn("Escolha uma opção: ").strip()
        except EOFError:
            break

        if opcao == "0":
            print("Encerrando simulador do Agente A1.")
            break
        if opcao == "1":
            texto = (
                input_fn(
                    "Mensagem do colaborador [Preciso enviar um atestado médico de 5 dias.]: "
                ).strip()
                or "Preciso enviar um atestado médico de 5 dias."
            )
            tarefa = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"cli-conv-{uuid4().hex[:8]}",
                "tipo": "TAREFA",
                "acao": "a1.conversar",
                "origem": "A0",
                "destino": "A1",
                "correlacao": {
                    "id_caso": 105,
                    "thread_id": "chat-42-105",
                    "id_tarefa": "t-cli-01",
                },
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "colaborador",
                    "canal": "chat",
                    "data_hora": datetime.now(UTC).isoformat(),
                },
                "registro_inicial": _registro_inicial_carla(tem_documento=False),
                "payload": {"texto": texto, "historico_resumo": None},
            }
            res = servico.executar(tarefa)
            _validar_resultado_contrato(res, schemas_json)
            _imprimir_bloco_json("RESULTADO a1.conversar", res)
        elif opcao == "2":
            arquivo = (
                input_fn(
                    "Caminho do arquivo [.storage/atestados/atestado_carla_105.png]: "
                ).strip()
                or ".storage/atestados/atestado_carla_105.png"
            )
            tarefa = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"cli-doc-{uuid4().hex[:8]}",
                "tipo": "TAREFA",
                "acao": "a1.processar_documento",
                "origem": "A0",
                "destino": "A1",
                "correlacao": {
                    "id_caso": 105,
                    "thread_id": "chat-42-105",
                    "id_tarefa": "t-cli-02",
                },
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "colaborador",
                    "canal": "chat",
                    "data_hora": datetime.now(UTC).isoformat(),
                },
                "registro_inicial": _registro_inicial_carla(tem_documento=True),
                "payload": {"arquivo_uri": arquivo},
            }
            res = servico.executar(tarefa)
            _validar_resultado_contrato(res, schemas_json)
            _imprimir_bloco_json("RESULTADO a1.processar_documento", res)
        elif opcao == "3":
            inicio_str = input_fn("Data início [2026-11-09]: ").strip() or "2026-11-09"
            fim_str = input_fn("Data fim [2026-11-11]: ").strip() or "2026-11-11"
            corrigidos_str = input_fn("Campos corrigidos (csv, vazio=IA) []: ").strip()
            campos_corr = (
                [c.strip() for c in corrigidos_str.split(",") if c.strip()]
                if corrigidos_str
                else []
            )
            tarefa = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"cli-conf-{uuid4().hex[:8]}",
                "tipo": "TAREFA",
                "acao": "a1.confirmar",
                "origem": "A0",
                "destino": "A1",
                "correlacao": {
                    "id_caso": 105,
                    "thread_id": "chat-42-105",
                    "id_tarefa": "t-cli-03",
                },
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "colaborador",
                    "canal": "chat",
                    "data_hora": datetime.now(UTC).isoformat(),
                },
                "registro_inicial": _registro_inicial_carla(tem_documento=True),
                "payload": {
                    "atestado_confirmado": {
                        "periodo": {"inicio": inicio_str, "fim": fim_str},
                        "data_emissao": inicio_str,
                    },
                    "campos_corrigidos": campos_corr,
                },
            }
            res = servico.executar(tarefa)
            _validar_resultado_contrato(res, schemas_json)
            if res["status"] == "OK" and "id_atestado" in res["payload"]:
                ultimo_id_atestado = int(res["payload"]["id_atestado"])
            _imprimir_bloco_json("RESULTADO a1.confirmar", res)
        elif opcao == "4":
            padrao_id = str(ultimo_id_atestado or 311)
            id_txt = input_fn(f"ID do atestado [{padrao_id}]: ").strip() or padrao_id
            rh_user = input_fn("Recebido por [rh.ana]: ").strip() or "rh.ana"
            tarefa = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"cli-baixa-{uuid4().hex[:8]}",
                "tipo": "TAREFA",
                "acao": "a1.registrar_baixa_fisica",
                "origem": "A0",
                "destino": "A1",
                "correlacao": {
                    "id_caso": 105,
                    "thread_id": "chat-42-105",
                    "id_tarefa": "t-cli-04",
                },
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "rh",
                    "canal": "chat",
                    "data_hora": datetime.now(UTC).isoformat(),
                },
                "registro_inicial": None,
                "payload": {
                    "id_atestado": int(id_txt),
                    "recebido_fisico_por": rh_user,
                    "recebido_fisico_em": datetime.now(UTC).isoformat(),
                },
            }
            res = servico.executar(tarefa)
            _validar_resultado_contrato(res, schemas_json)
            _imprimir_bloco_json("RESULTADO a1.registrar_baixa_fisica", res)
        elif opcao == "5":
            executar_roteiro_carla(
                cfg=cfg_base, modo_repo=modo_repo, modo_llm=modo_llm
            )
        elif opcao == "6":
            avaliar_metricas_secao10(cfg=cfg_base, modo_llm=modo_llm)
        elif opcao == "7":
            amostras = garantir_atestados_sinteticos(cfg_base.atestados_storage_dir)
            for am in amostras:
                print(f"  - {am.id_amostra}: {am.arquivo_png} | {am.arquivo_pdf}")
        else:
            print(f"Opção inválida: {opcao!r}")

    return 0


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulador CLI do Agente 1 (A1 — Recepção e Processamento)."
    )
    parser.add_argument(
        "--roteiro",
        choices=["carla"],
        help="Executa roteiro completo ponta a ponta (ex: carla).",
    )
    parser.add_argument(
        "--avaliar-metricas",
        action="store_true",
        help="Avalia as métricas da Seção 10 sobre os atestados sintéticos.",
    )
    parser.add_argument(
        "--gerar-atestados",
        action="store_true",
        help="Gera os 12 pares de atestados sintéticos (.png/.pdf) e o gabarito JSON.",
    )
    parser.add_argument(
        "--gerar-golden",
        action="store_true",
        help="Gera um corpus golden sintético, anonimizado e determinístico.",
    )
    parser.add_argument(
        "--golden-quantidade",
        type=int,
        default=60,
        help="Quantidade de amostras do corpus golden (padrão: 60).",
    )
    parser.add_argument(
        "--golden-seed",
        type=int,
        default=20261118,
        help="Seed determinística do corpus golden.",
    )
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="Força regeração dos arquivos não canônicos ao usar --gerar-atestados.",
    )
    parser.add_argument(
        "--tarefa",
        type=Path,
        help="Caminho para um arquivo .tarefa.json a enviar para executar(tarefa).",
    )
    parser.add_argument(
        "--modo-repo",
        choices=["auto", "memoria", "postgres"],
        default="auto",
        help="Seleciona o repositório de dados (padrão: auto).",
    )
    parser.add_argument(
        "--modo-llm",
        choices=["auto", "mock", "groq"],
        default="auto",
        help="Seleciona o cliente LLM (padrão: auto).",
    )
    parser.add_argument(
        "--persistir",
        action="store_true",
        help="Mantém o registro de demonstração no PostgreSQL ao rodar --roteiro carla.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configurar_stdout_utf8()
    parser = _construir_parser()
    args = parser.parse_args(argv)
    cfg = obter_config_a1()

    if args.gerar_golden:
        destino = Path(".storage/golden-atestados")
        amostras = gerar_atestados_golden(
            destino,
            quantidade=args.golden_quantidade,
            seed=args.golden_seed,
            sobrescrever=args.sobrescrever,
        )
        print(
            json.dumps(
                {
                    "diretorio": str(destino),
                    "seed": args.golden_seed,
                    "total_amostras": len(amostras),
                    "amostras": [a.model_dump(mode="json") for a in amostras],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.gerar_atestados:
        amostras = garantir_atestados_sinteticos(
            cfg.atestados_storage_dir, sobrescrever=args.sobrescrever
        )
        print(
            json.dumps(
                {
                    "total_amostras": len(amostras),
                    "amostras": [a.model_dump(mode="json") for a in amostras],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.roteiro == "carla":
        return executar_roteiro_carla(
            cfg=cfg,
            modo_repo=args.modo_repo,
            modo_llm=args.modo_llm,
            persistir=args.persistir,
        )

    if args.avaliar_metricas:
        relatorio = avaliar_metricas_secao10(cfg=cfg, modo_llm=args.modo_llm)
        return 0 if (relatorio["meta_extracao_atingida"] and relatorio["meta_baixa_atingida"]) else 1

    if args.tarefa is not None:
        return executar_arquivo_tarefa(
            args.tarefa,
            cfg=cfg,
            modo_repo=args.modo_repo,
            modo_llm=args.modo_llm,
        )

    return executar_menu_interativo(
        cfg=cfg, modo_repo=args.modo_repo, modo_llm=args.modo_llm
    )


if __name__ == "__main__":
    raise SystemExit(main())
