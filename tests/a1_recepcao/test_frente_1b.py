"""Testes da Frente 1B do Agente 1 (`validador.py` e `repositorio.py`).

Cobre os requisitos da Task 1B.1 e Task 1B.2 alinhados à `Documentação - SinergIA v2.2.docx`
(Seção 5.3, Anexos C.2, D.2, E.2 e G) e `db/schema/001_schema_inicial.sql`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Self

import pytest
from pydantic import ValidationError

from sinergia.agentes.a1_recepcao import (
    AtestadoExtraido,
    PayloadBaixaFisicaTarefa,
    PayloadConfirmarTarefa,
    PeriodoAfastamento,
    ProfissionalExtraido,
    RegistroAtestadoDB,
    RegistroInicialCaso,
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    RepositorioA1Protocol,
    TarefaA1,
    calcular_marcos_legais,
    calcular_prazo_entrega_fisica,
    cid_sintaxe_valida,
    detectar_divergencias,
    gerar_mensagem_confirmacao_extracao,
    mesclar_atestado_confirmado,
    nomes_compativeis,
    validar_dados_confirmados,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _carregar_fixture(relpath: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / relpath).read_text(encoding="utf-8"))


# ============================================================================
# 1. Testes do Validador (`validador.py` — Task 1B.1)
# ============================================================================


def test_nomes_compativeis_abreviacao_acentos_e_divergencias() -> None:
    assert nomes_compativeis("Carla Souza", "Carla S.") is True
    assert nomes_compativeis("Carla S.", "Carla Souza") is True
    assert nomes_compativeis("José da Silva", "Jose Silva") is True
    assert nomes_compativeis("Ana Maria de Souza", "Ana M. Souza") is True

    assert nomes_compativeis("Carla Silva", "Carla Souza") is False
    assert nomes_compativeis("Maria Souza", "Carla S.") is False
    assert nomes_compativeis("Carla", "Carla Souza") is False
    assert nomes_compativeis(None, "Carla Souza") is False


def test_detectar_divergencias_fixture_carla_dias() -> None:
    tarefa_json = _carregar_fixture(
        "a1.processar_documento/carla_divergencia_dias.tarefa.json"
    )
    esperado_json = _carregar_fixture(
        "a1.processar_documento/carla_divergencia_dias.esperado.json"
    )
    tarefa = TarefaA1.model_validate(tarefa_json)
    extraido = AtestadoExtraido.model_validate(
        esperado_json["payload"]["atestado_extraido"]
    )

    divergencias = detectar_divergencias(extraido, tarefa.registro_inicial)
    assert len(divergencias) == 1
    assert divergencias[0].model_dump() == esperado_json["payload"]["divergencias"][0]

    mensagem = gerar_mensagem_confirmacao_extracao(extraido, divergencias)
    assert mensagem == esperado_json["saidas"][0]["texto"]


def test_detectar_divergencias_inicio_e_nome() -> None:
    registro = RegistroInicialCaso(
        id_caso=200,
        thread_id="chat-42-200",
        aberto_em="2026-11-10T09:00:00-03:00",
        id_colaborador=42,
        nome="Carla Souza",
        matricula="00123",
        id_area=2,
        id_setor=7,
        id_cargo=15,
        jornada_inicio="07:00",
        jornada_fim="16:00",
        municipio="Campinas",
        uf="SP",
        inicio_informado=date(2026, 11, 10),
        dias_informados=2,
        tem_documento=True,
    )
    extraido = AtestadoExtraido(
        legibilidade="ok",
        nome_no_documento="Maria Oliveira",
        periodo=PeriodoAfastamento(inicio=date(2026, 11, 9), fim=date(2026, 11, 10)),
        data_emissao=date(2026, 11, 9),
        profissional=ProfissionalExtraido(conselho="CRM", numero="123456", uf="SP"),
    )

    divergencias = detectar_divergencias(extraido, registro)
    assert [d.campo for d in divergencias] == ["inicio", "nome"]


def test_calcular_prazo_entrega_fisica_com_fim_de_semana_e_feriado() -> None:
    # Caso canônico Anexo D.2: 11/11 (qua) -> retorno 12/11 (qui) -> +2 dias úteis = 16/11 (seg)
    assert calcular_prazo_entrega_fisica(date(2026, 11, 11)) == date(2026, 11, 16)

    # Fim na sexta 13/11 -> retorno na segunda 16/11 -> +2 dias úteis = quarta 18/11
    assert calcular_prazo_entrega_fisica(date(2026, 11, 13)) == date(2026, 11, 18)

    # Considerando feriado em 16/11 -> retorno em 17/11 -> +2 dias úteis = 19/11
    assert calcular_prazo_entrega_fisica(
        date(2026, 11, 13), feriados=[date(2026, 11, 16)]
    ) == date(2026, 11, 19)


def test_validar_dados_confirmados_fixtures_valido_manual_e_invalido() -> None:
    tarefa_ok = TarefaA1.model_validate(
        _carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json")
    )
    esperado_ok = _carregar_fixture("a1.confirmar/carla_confirmado_ia.esperado.json")
    payload_ok = tarefa_ok.validar_payload_acao()
    assert isinstance(payload_ok, PayloadConfirmarTarefa)

    res_ia = validar_dados_confirmados(
        payload_ok.atestado_confirmado,
        payload_ok.campos_corrigidos,
    )
    assert res_ia.valido is True
    assert res_ia.origem == esperado_ok["payload"]["origem"]
    assert res_ia.prazo_entrega_fisica == date.fromisoformat(
        esperado_ok["payload"]["prazo_entrega_fisica"]
    )
    assert res_ia.mensagem_usuario == esperado_ok["saidas"][0]["texto"]

    # Correção manual -> origem = MANUAL
    res_manual = validar_dados_confirmados(
        payload_ok.atestado_confirmado,
        ["periodo.fim"],
    )
    assert res_manual.valido is True
    assert res_manual.origem == "MANUAL"

    # Datas inválidas (fim < inicio) -> reproduz datas_invalidas.esperado.json
    tarefa_inv = TarefaA1.model_validate(
        _carregar_fixture("a1.confirmar/datas_invalidas.tarefa.json")
    )
    esperado_inv = _carregar_fixture("a1.confirmar/datas_invalidas.esperado.json")
    payload_inv = tarefa_inv.validar_payload_acao()
    assert isinstance(payload_inv, PayloadConfirmarTarefa)

    res_inv = validar_dados_confirmados(
        payload_inv.atestado_confirmado,
        payload_inv.campos_corrigidos,
    )
    assert res_inv.valido is False
    assert res_inv.motivo == esperado_inv["payload"]["motivo"]
    assert res_inv.codigo_evento == esperado_inv["eventos"][0]["dados"]["motivo"]
    assert res_inv.mensagem_usuario == esperado_inv["saidas"][0]["texto"]


def test_validar_regras_e2_inativo_profissional_cid_e_marcos_legais() -> None:
    base = AtestadoExtraido(
        legibilidade="ok",
        periodo=PeriodoAfastamento(inicio=date(2026, 11, 1), fim=date(2026, 11, 16)),
        data_emissao=date(2026, 11, 1),
        profissional=ProfissionalExtraido(conselho="CRM", numero="123456", uf="SP"),
        cid="M54.5",
    )

    # Colaborador inativo
    res_inativo = validar_dados_confirmados(base, colaborador_ativo=False)
    assert res_inativo.valido is False
    assert res_inativo.codigo_evento == "colaborador_inativo"

    # Conselho/UF inválido
    prof_inv = base.model_copy(
        update={
            "profissional": ProfissionalExtraido(conselho="XYZ", numero="1", uf="XX")
        }
    )
    res_prof = validar_dados_confirmados(prof_inv)
    assert res_prof.valido is False
    assert res_prof.codigo_evento == "profissional_invalido"

    # CID inválido com política ativa vs inativa
    cid_inv = base.model_copy(update={"cid": "INVALIDO"})
    assert cid_sintaxe_valida("M54.5") is True
    assert cid_sintaxe_valida("INVALIDO") is False
    assert validar_dados_confirmados(cid_inv, politica_extrair_cid=True).valido is False
    assert validar_dados_confirmados(cid_inv, politica_extrair_cid=False).valido is True

    # Marcos legais: 16 dias -> INSS; soma em 60 dias -> INSS; >= 30 dias -> NR-7
    m16 = calcular_marcos_legais(date(2026, 11, 1), date(2026, 11, 16))
    assert m16.dias_afastamento == 16
    assert m16.exige_encaminhamento_inss is True
    assert m16.data_marco_16_dia == date(2026, 11, 16)
    assert m16.exige_exame_retorno_nr7 is False

    m_soma = calcular_marcos_legais(
        date(2026, 11, 10), date(2026, 11, 15), dias_acumulados_60d=10
    )
    assert m_soma.dias_afastamento == 6
    assert m_soma.dias_totais_60d == 16
    assert m_soma.exige_encaminhamento_inss is True

    m30 = calcular_marcos_legais(date(2026, 11, 1), date(2026, 11, 30))
    assert m30.dias_afastamento == 30
    assert m30.exige_encaminhamento_inss is True
    assert m30.exige_exame_retorno_nr7 is True


# ============================================================================
# 2. Testes do Repositório (`repositorio.py` — Task 1B.2)
# ============================================================================


def test_repositorio_memoria_fluxo_completo_anexo_c2_e_d2() -> None:
    repo = RepositorioA1Memoria()
    assert isinstance(repo, RepositorioA1Protocol)

    # 1. Conferir colaboradora ativa no período (cadastro.colaborador)
    colab = repo.conferir_colaborador_ativo(
        42, inicio=date(2026, 11, 9), fim=date(2026, 11, 11)
    )
    assert colab is not None
    assert colab.ativo is True
    assert colab.id_municipio == 3509502

    # Antes da admissão (2024-03-01) deve marcar ativo=False no período
    colab_antes = repo.conferir_colaborador_ativo(
        42, inicio=date(2024, 1, 1), fim=date(2024, 1, 10)
    )
    assert colab_antes is not None
    assert colab_antes.ativo is False

    # 2. Catálogos de conselho, UF, município e CID
    assert repo.validar_conselho_profissional("crm", "sp") == 1
    assert repo.validar_conselho_profissional("cro", "rj") == 2
    assert repo.validar_conselho_profissional("crm", "XX") is None
    assert repo.buscar_id_municipio("Campinas", "SP") == 3509502
    assert repo.buscar_id_municipio("Desconhecido", None, id_colaborador_fallback=42) == 3509502
    assert repo.validar_cid("m54.5", politica_extrair_cid=True) == "M54.5"
    assert repo.validar_cid("m54.5", politica_extrair_cid=False) is None

    # 3. Profissional de saúde (criação e reuso idempotente)
    id_prof1 = repo.obter_ou_criar_profissional_saude("CRM", "123456", "SP")
    id_prof2 = repo.obter_ou_criar_profissional_saude(
        "crm", "123456", "sp", nome="Dr. João"
    )
    assert id_prof1 == id_prof2
    assert repo.profissionais[id_prof1]["nome"] == "Dr. João"

    # 4. Gravar PENDENTE e checar duplicidade e sobreposição
    pendente = repo.salvar_atestado_pendente(
        id_colaborador=42,
        id_profissional=id_prof1,
        id_municipio=3509502,
        data_emissao=date(2026, 11, 9),
        inicio_afastamento=date(2026, 11, 9),
        fim_afastamento=date(2026, 11, 11),
        arquivo_uri=".storage/atestados/atestado_carla_105.png",
        codigo_cid="M54.5",
        origem_dados="IA",
    )
    assert pendente.id_atestado == 311
    assert pendente.status == "PENDENTE"

    duplicado = repo.verificar_duplicidade_arquivo(
        ".storage/atestados/atestado_carla_105.png", id_colaborador=42
    )
    assert duplicado is not None
    assert duplicado.id_atestado == 311

    sobrepostos = repo.listar_atestados_sobrepostos(
        42, date(2026, 11, 10), date(2026, 11, 12)
    )
    assert len(sobrepostos) == 1
    assert repo.listar_atestados_sobrepostos(
        42, date(2026, 11, 10), date(2026, 11, 12), ignorar_id_atestado=311
    ) == []

    # 5. Confirmar atestado (PENDENTE -> CONFIRMADO + prazo físico)
    confirmado = repo.confirmar_atestado_no_banco(
        311,
        inicio_afastamento=date(2026, 11, 9),
        fim_afastamento=date(2026, 11, 11),
        data_emissao=date(2026, 11, 9),
        origem_dados="IA",
        prazo_entrega_fisica=date(2026, 11, 16),
    )
    assert confirmado.status == "CONFIRMADO"
    assert confirmado.prazo_entrega_fisica == date(2026, 11, 16)
    assert repo.obter_ultimo_atestado_pendente(42) is None

    # Acumulado em 60 dias pelo mesmo CID
    assert (
        repo.calcular_dias_afastamento_ultimos_60_dias(
            42, date(2026, 11, 20), cid="M54.5"
        )
        == 3
    )

    # 6. Baixa física usando fixture oficial (recebido_fisico_por = "rh.ana")
    tarefa_baixa = TarefaA1.model_validate(
        _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
    )
    payload_baixa = tarefa_baixa.validar_payload_acao()
    assert isinstance(payload_baixa, PayloadBaixaFisicaTarefa)

    baixado = repo.registrar_baixa_fisica_no_banco(
        payload_baixa.id_atestado,
        payload_baixa.recebido_fisico_por,
        payload_baixa.recebido_fisico_em,
    )
    assert baixado.recebido_fisico_por == 7
    assert baixado.recebido_fisico_em is not None

    with pytest.raises(ValueError, match="Usuário não encontrado"):
        repo.resolver_id_usuario("usuario.inexistente")


def test_repositorio_postgres_sql_estrito_anexo_c2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garante que o SQL de `RepositorioA1Postgres` nunca acessa `orquestracao.*`
    e usa apenas as tabelas autorizadas do Anexo C.2 (`cadastro.*` e `afastamento.*`).
    """
    queries_executadas: list[str] = []

    class FakeCursor:
        def __init__(self) -> None:
            self._ultima_query = ""
            self._params: Any = None

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            self._ultima_query = sql
            self._params = params
            queries_executadas.append(sql)

        def fetchone(self) -> dict[str, Any] | None:
            sql_low = self._ultima_query.lower()
            if "from cadastro.colaborador c" in sql_low:
                return {
                    "id_colaborador": 42,
                    "matricula": "00123",
                    "nome": "Carla Souza",
                    "id_cargo": 15,
                    "id_setor": 7,
                    "id_area": 2,
                    "id_municipio": 3509502,
                    "municipio_nome": "Campinas",
                    "sigla_uf": "SP",
                    "data_admissao": date(2024, 3, 1),
                    "ativo": True,
                    "cargo_nome": "Analista",
                    "setor_nome": "Logística",
                }
            if "from afastamento.conselho_profissional" in sql_low:
                return {"id_conselho": 1}
            if "into afastamento.profissional_saude" in sql_low:
                return {"id_profissional": 99}
            if "from cadastro.usuario" in sql_low:
                return {"id_usuario": 7}
            if "into afastamento.atestado" in sql_low or "update afastamento.atestado" in sql_low:
                return {
                    "id_atestado": 311,
                    "id_colaborador": 42,
                    "id_profissional": 99,
                    "codigo_cid": "M54.5",
                    "id_municipio": 3509502,
                    "data_emissao": date(2026, 11, 9),
                    "inicio_afastamento": datetime(2026, 11, 9, 0, 0, tzinfo=UTC),
                    "fim_afastamento": datetime(2026, 11, 11, 23, 59, 59, tzinfo=UTC),
                    "arquivo_uri": ".storage/atestados/carla.png",
                    "origem_dados": "IA",
                    "status": "CONFIRMADO",
                    "postado_em": datetime(2026, 11, 9, 12, 0, tzinfo=UTC),
                    "prazo_entrega_fisica": date(2026, 11, 16),
                    "recebido_fisico_em": datetime(2026, 11, 13, 13, 20, tzinfo=UTC),
                    "recebido_fisico_por": 7,
                }
            return None

        def fetchall(self) -> list[dict[str, Any]]:
            return []

    class FakeConn:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    import sinergia.agentes.a1_recepcao.repositorio as repo_mod

    monkeypatch.setattr(
        repo_mod.psycopg,
        "connect",
        lambda *args, **kwargs: FakeConn(),
    )

    repo_pg = RepositorioA1Postgres("postgresql://fake:fake@localhost:5432/sinergia")
    assert isinstance(repo_pg, RepositorioA1Protocol)

    colab = repo_pg.conferir_colaborador_ativo(42)
    assert colab is not None and colab.nome == "Carla Souza"

    id_prof = repo_pg.obter_ou_criar_profissional_saude("CRM", "123456", "SP")
    assert id_prof == 99

    reg = repo_pg.salvar_atestado_pendente(
        id_colaborador=42,
        id_profissional=id_prof,
        id_municipio=3509502,
        data_emissao=date(2026, 11, 9),
        inicio_afastamento=date(2026, 11, 9),
        fim_afastamento=date(2026, 11, 11),
        arquivo_uri=".storage/atestados/carla.png",
    )
    assert reg.id_atestado == 311

    repo_pg.confirmar_atestado_no_banco(
        311,
        inicio_afastamento=date(2026, 11, 9),
        fim_afastamento=date(2026, 11, 11),
        data_emissao=date(2026, 11, 9),
        origem_dados="IA",
        prazo_entrega_fisica=date(2026, 11, 16),
    )
    baixado = repo_pg.registrar_baixa_fisica_no_banco(
        311, "rh.ana", "2026-11-13T10:20:00-03:00"
    )
    assert baixado.recebido_fisico_por == 7

    assert len(queries_executadas) >= 5
    for sql in queries_executadas:
        assert not re.search(r"\borquestracao\.", sql, re.IGNORECASE)
        assert not re.search(r"\bbem_estar\.", sql, re.IGNORECASE)
        assert not re.search(r"\bapoio\.", sql, re.IGNORECASE)
        assert not re.search(r"\brag\.", sql, re.IGNORECASE)


def test_mesclar_atestado_confirmado_e_emissao_futura() -> None:
    extraido_json = _carregar_fixture(
        "a1.processar_documento/carla_divergencia_dias.esperado.json"
    )["payload"]["atestado_extraido"]
    confirmar_json = _carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json")

    base_extraida = AtestadoExtraido.model_validate(extraido_json)
    payload_conf = TarefaA1.model_validate(confirmar_json).validar_payload_acao()
    assert isinstance(payload_conf, PayloadConfirmarTarefa)

    mesclado = mesclar_atestado_confirmado(
        payload_conf.atestado_confirmado, base_extraida
    )
    assert mesclado.nome_no_documento == "Carla Souza"
    assert mesclado.profissional.conselho == "CRM"
    assert mesclado.profissional.numero == "123456"
    assert mesclado.profissional.uf == "SP"
    assert mesclado.municipio == "Campinas"
    assert mesclado.periodo.inicio == date(2026, 11, 9)
    assert mesclado.periodo.fim == date(2026, 11, 11)

    res_mesclado = validar_dados_confirmados(
        mesclado, payload_conf.campos_corrigidos, exigir_profissional=True
    )
    assert res_mesclado.valido is True

    emissao_futura = mesclado.model_copy(update={"data_emissao": date(2026, 11, 12)})
    res_fut = validar_dados_confirmados(emissao_futura)
    assert res_fut.valido is False
    assert res_fut.codigo_evento == "emissao_futura"


def test_registro_atestado_db_invariantes_check() -> None:
    with pytest.raises(ValidationError, match="fim_afastamento"):
        RegistroAtestadoDB(
            id_atestado=1,
            id_colaborador=42,
            id_profissional=1,
            id_municipio=3509502,
            data_emissao=date(2026, 11, 9),
            inicio_afastamento=datetime(2026, 11, 11, 0, 0, tzinfo=UTC),
            fim_afastamento=datetime(2026, 11, 9, 23, 59, 59, tzinfo=UTC),
            arquivo_uri=".storage/atestados/carla.png",
            origem_dados="IA",
            status="PENDENTE",
            postado_em=datetime(2026, 11, 9, 12, 0, tzinfo=UTC),
        )

    with pytest.raises(ValidationError, match="recebido_fisico_em"):
        RegistroAtestadoDB(
            id_atestado=1,
            id_colaborador=42,
            id_profissional=1,
            id_municipio=3509502,
            data_emissao=date(2026, 11, 9),
            inicio_afastamento=datetime(2026, 11, 9, 0, 0, tzinfo=UTC),
            fim_afastamento=datetime(2026, 11, 11, 23, 59, 59, tzinfo=UTC),
            arquivo_uri=".storage/atestados/carla.png",
            origem_dados="IA",
            status="CONFIRMADO",
            postado_em=datetime(2026, 11, 9, 12, 0, tzinfo=UTC),
            prazo_entrega_fisica=date(2026, 11, 16),
            recebido_fisico_em=datetime(2026, 11, 13, 13, 20, tzinfo=UTC),
            recebido_fisico_por=None,
        )


def test_busca_municipio_sem_acento_memoria_e_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_mem = RepositorioA1Memoria()
    assert repo_mem.buscar_id_municipio("Sao Paulo", "SP") == 3550308

    class FakeCursorMunicipio:
        def __init__(self) -> None:
            self._ultima_query = ""

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            self._ultima_query = sql

        def fetchone(self) -> dict[str, Any] | None:
            return None

        def fetchall(self) -> list[dict[str, Any]]:
            if "from cadastro.municipio" in self._ultima_query.lower():
                return [
                    {"id_municipio": 3509502, "nome": "Campinas"},
                    {"id_municipio": 3550308, "nome": "São Paulo"},
                ]
            return []

    class FakeConnMunicipio:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursorMunicipio:
            return FakeCursorMunicipio()

    import sinergia.agentes.a1_recepcao.repositorio as repo_mod

    monkeypatch.setattr(
        repo_mod.psycopg,
        "connect",
        lambda *args, **kwargs: FakeConnMunicipio(),
    )

    repo_pg = RepositorioA1Postgres("postgresql://fake:fake@localhost:5432/sinergia")
    assert repo_pg.buscar_id_municipio("Sao Paulo", "SP") == 3550308
