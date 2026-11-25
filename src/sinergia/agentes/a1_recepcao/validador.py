"""Validador de regras de negócio, divergências e prazos do Agente 1 (A1).

Implementa a Task 1B.1 do plano de desenvolvimento do A1 em conformidade com a
`Documentação - SinergIA v2.2.docx` (Seção 5.3, Anexos C.2, D.2, E.2 e G) e a
fundamentação científica e normativa levantada para o projeto:

- Divergências com o `registro_inicial` (Anexo E.2 / Mahbub et al., 2026):
  divergência nunca rejeita o atestado automaticamente; gera pergunta de
  confirmação humana (`payload.divergencias`).
- Coerência cronológica (`fim >= inicio`) e vínculo ativo (Anexo D.2 e E.2).
- Validação de conselhos profissionais (`CRM`, `CRO`, `CRP` + UF) e formato
  sintático de CID-10 (Res. CFM 1.658/2002, Lei 5.081/1966, Lei 5.766/1971).
- Prazo de entrega física (2 dias úteis após o retorno ao trabalho) e marcos
  legais (16º dia INSS — Lei 8.213/1991 Art. 60 § 3º / Dec. 3.048/1999 Art. 75;
  exame de retorno >= 30 dias — NR-07 subitem 7.5.9).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Sequence
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sinergia.agentes.a1_recepcao.schemas import (
    AtestadoExtraido,
    DivergenciaCampo,
    ItemRevisaoCampo,
    OrigemDadosAtestado,
    PeriodoAfastamento,
    ProfissionalExtraido,
    QuadroRevisaoAtestado,
    RegistroInicialCaso,
    RetornoValidacaoCidFormulario,
    StatusRevisaoCampo,
)

UFS_BRASIL: frozenset[str] = frozenset(
    {
        "AC",
        "AL",
        "AP",
        "AM",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MT",
        "MS",
        "MG",
        "PA",
        "PB",
        "PR",
        "PE",
        "PI",
        "RJ",
        "RN",
        "RO",
        "RR",
        "RS",
        "SC",
        "SP",
        "SE",
        "TO",
    }
)

CONSELHOS_ACEITOS_PADRAO: frozenset[str] = frozenset({"CRM", "CRO", "CRP"})

_CONECTIVOS_NOME: frozenset[str] = frozenset({"de", "da", "do", "dos", "das", "e"})
_PADRAO_CID10 = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")
_PADRAO_CID11 = re.compile(
    r"^[1-9A-HJ-NP-Z][A-HJ-NP-Z0-9]\d[A-HJ-NP-Z0-9](?:\.[A-HJ-NP-Z0-9]{1,2})?$"
)


def _normalizar_tokens_nome(nome: str | None) -> list[str]:
    """Remove acentos, pontuação e conectivos para comparação de nomes."""
    if not nome:
        return []
    sem_acentos = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", nome)
        if not unicodedata.combining(ch)
    )
    limpo = re.sub(r"[.,;:'\"-]+", " ", sem_acentos).lower()
    return [tok for tok in limpo.split() if tok and tok not in _CONECTIVOS_NOME]


def _tokens_sobrenome_compativeis(tok_a: str, tok_b: str) -> bool:
    """Compara sobrenomes aceitando abreviação por inicial (ex.: 's' e 'souza')."""
    if tok_a == tok_b:
        return True
    if len(tok_a) == 1 and tok_b.startswith(tok_a):
        return True
    return bool(len(tok_b) == 1 and tok_a.startswith(tok_b))


def nomes_compativeis(nome_documento: str | None, nome_cadastro: str | None) -> bool:
    """Verifica se o nome no documento é compatível com o nome do cadastro.

    Tolera acentuação diferente, conectivos (`de`/`da`/`dos`), nomes do meio
    omitidos e sobrenomes abreviados por inicial (ex.: ``"Carla S."`` vs
    ``"Carla Souza"``), evitando falsos-positivos de divergência.
    """
    tokens_doc = _normalizar_tokens_nome(nome_documento)
    tokens_cad = _normalizar_tokens_nome(nome_cadastro)
    if not tokens_doc or not tokens_cad:
        return False

    curto, longo = (
        (tokens_doc, tokens_cad)
        if len(tokens_doc) <= len(tokens_cad)
        else (tokens_cad, tokens_doc)
    )

    if len(curto) == 1 and len(longo) > 1:
        return False
    if curto[0] != longo[0]:
        return False

    idx_longo = 1
    for tok_curto in curto[1:]:
        encontrou = False
        while idx_longo < len(longo):
            if _tokens_sobrenome_compativeis(tok_curto, longo[idx_longo]):
                encontrou = True
                idx_longo += 1
                break
            idx_longo += 1
        if not encontrou:
            return False

    return True


def detectar_divergencias(
    extraido: AtestadoExtraido,
    registro_inicial: RegistroInicialCaso | None,
) -> list[DivergenciaCampo]:
    """Confronta os campos extraídos do documento com o `registro_inicial` (Anexo E.2).

    Verifica:
    - ``dias``: quantidade de dias calculada ``(fim - inicio).days + 1`` vs ``dias_informados``.
    - ``inicio``: data de início extraída vs ``inicio_informado``.
    - ``nome``: compatibilidade entre ``nome_no_documento`` e ``nome`` do cadastro.
    """
    if registro_inicial is None:
        return []

    divergencias: list[DivergenciaCampo] = []

    dias_doc = extraido.periodo.dias_calculados
    if (
        dias_doc is not None
        and registro_inicial.dias_informados is not None
        and dias_doc != registro_inicial.dias_informados
    ):
        divergencias.append(
            DivergenciaCampo(
                campo="dias",
                no_documento=dias_doc,
                informado=registro_inicial.dias_informados,
            )
        )

    if (
        extraido.periodo.inicio is not None
        and registro_inicial.inicio_informado is not None
        and extraido.periodo.inicio != registro_inicial.inicio_informado
    ):
        divergencias.append(
            DivergenciaCampo(
                campo="inicio",
                no_documento=extraido.periodo.inicio.isoformat(),
                informado=registro_inicial.inicio_informado.isoformat(),
            )
        )

    if (
        extraido.nome_no_documento
        and registro_inicial.nome
        and not nomes_compativeis(extraido.nome_no_documento, registro_inicial.nome)
    ):
        divergencias.append(
            DivergenciaCampo(
                campo="nome",
                no_documento=extraido.nome_no_documento,
                informado=registro_inicial.nome,
            )
        )

    if (
        extraido.tipo_documento
        and registro_inicial.tipo_solicitacao
        and extraido.tipo_documento != registro_inicial.tipo_solicitacao
    ):
        divergencias.append(
            DivergenciaCampo(
                campo="tipo_documento",
                no_documento=extraido.tipo_documento,
                informado=registro_inicial.tipo_solicitacao,
            )
        )

    return divergencias


def mesclar_atestado_confirmado(
    confirmado: AtestadoExtraido,
    base_extraida: AtestadoExtraido | None,
) -> AtestadoExtraido:
    """Mescla o payload parcial de `a1.confirmar` com a extração prévia (Anexo D.1).

    Preserva campos já extraídos em `a1.processar_documento` (como nome,
    profissional de saúde, município, CID e data de emissão) quando o
    colaborador confirma apenas um subconjunto de campos (ex.: período).
    """
    if base_extraida is None:
        return confirmado.model_copy(deep=True)

    periodo_mesclado = PeriodoAfastamento(
        inicio=(
            confirmado.periodo.inicio
            if confirmado.periodo.inicio is not None
            else base_extraida.periodo.inicio
        ),
        fim=(
            confirmado.periodo.fim
            if confirmado.periodo.fim is not None
            else base_extraida.periodo.fim
        ),
    )

    prof_conf = confirmado.profissional
    prof_base = base_extraida.profissional
    profissional_mesclado = ProfissionalExtraido(
        conselho=prof_conf.conselho if prof_conf.conselho is not None else prof_base.conselho,
        numero=prof_conf.numero if prof_conf.numero is not None else prof_base.numero,
        uf=prof_conf.uf if prof_conf.uf is not None else prof_base.uf,
        nome=prof_conf.nome if prof_conf.nome is not None else prof_base.nome,
    )

    tipo_doc = (
        confirmado.tipo_documento
        if "tipo_documento" in confirmado.model_fields_set
        else base_extraida.tipo_documento
    )
    campos_mesclados = (
        dict(confirmado.campos) if confirmado.campos else dict(base_extraida.campos)
    )

    return AtestadoExtraido(
        tipo_documento=tipo_doc,
        legibilidade=confirmado.legibilidade,
        nome_no_documento=(
            confirmado.nome_no_documento
            if confirmado.nome_no_documento is not None
            else base_extraida.nome_no_documento
        ),
        periodo=periodo_mesclado,
        data_emissao=(
            confirmado.data_emissao
            if confirmado.data_emissao is not None
            else base_extraida.data_emissao
        ),
        profissional=profissional_mesclado,
        cid=confirmado.cid if confirmado.cid is not None else base_extraida.cid,
        municipio=(
            confirmado.municipio
            if confirmado.municipio is not None
            else base_extraida.municipio
        ),
        campos=campos_mesclados,
        confianca_campos=(
            confirmado.confianca_campos
            if confirmado.confianca_campos is not None
            else base_extraida.confianca_campos
        ),
    )


_LABELS_CAMPOS_BAIXA_CONFIANCA: dict[str, str] = {
    "nome_no_documento": "nome no documento",
    "periodo.inicio": "data de início",
    "periodo.fim": "data de término",
    "data_emissao": "data de emissão",
    "profissional.conselho": "conselho profissional",
    "profissional.numero": "número do registro profissional",
    "profissional.uf": "UF do registro profissional",
    "profissional.nome": "nome do profissional",
    "cid": "código CID",
    "municipio": "município",
}

_ROTULOS_TIPO_DOCUMENTO: dict[str, str] = {
    "atestado_medico": "Atestado médico",
    "declaracao_comparecimento": "Declaração de comparecimento",
    "outro": "Outro documento",
}

_CAMPOS_DIVERGENCIA_PARA_ITEM: dict[str, str] = {
    "dias": "periodo.dias",
    "inicio": "periodo.inicio",
    "nome": "nome_no_documento",
    "tipo_documento": "tipo_documento",
}


def _formatar_data_br(data: date | None) -> str:
    return f"{data:%d/%m/%Y}" if data is not None else "Não identificado"


def _obter_confianca_campo(extraido: AtestadoExtraido, campo: str) -> float | None:
    meta = extraido.campos.get(campo)
    bruto: float | None = meta.confianca if meta is not None else None
    if bruto is None and extraido.confianca_campos:
        if campo == "periodo.dias":
            c_ini = extraido.confianca_campos.get("periodo.inicio")
            c_fim = extraido.confianca_campos.get("periodo.fim")
            if isinstance(c_ini, (int, float)) and isinstance(c_fim, (int, float)):
                bruto = min(float(c_ini), float(c_fim))
        else:
            val = extraido.confianca_campos.get(campo)
            if isinstance(val, (int, float)):
                bruto = float(val)
    if bruto is None:
        return None
    return round(max(0.0, min(1.0, float(bruto))), 4)


def _formatar_profissional_resumo(extraido: AtestadoExtraido) -> str:
    prof = extraido.profissional
    conselho = prof.conselho
    uf = prof.uf
    numero = prof.numero
    baixa_prof = any(
        (m := extraido.campos.get(c)) is not None and m.vazio_por_baixa_confianca
        for c in ("profissional.conselho", "profissional.numero", "profissional.uf")
    )
    if conselho and uf and numero:
        return f"{conselho}/{uf} {numero}"
    partes: list[str] = []
    if conselho and uf:
        partes.append(f"{conselho}/{uf}")
    elif conselho:
        partes.append(conselho)
    elif uf:
        partes.append(uf)
    if numero:
        partes.append(numero)
    if partes:
        sufixo = " (registro com baixa confiança)" if baixa_prof else ""
        return f"{' '.join(partes)}{sufixo}"
    return "Baixa confiança (informar)" if baixa_prof else "Não identificado"


def _formatar_valor_resumo(
    valor: Any,
    *,
    extraido: AtestadoExtraido,
    campo: str,
    exibicao_customizada: str | None = None,
) -> str:
    meta = extraido.campos.get(campo)
    if meta is not None and meta.vazio_por_baixa_confianca:
        return "Baixa confiança (informar)"
    if valor is None:
        return "Não identificado"
    return exibicao_customizada if exibicao_customizada is not None else str(valor)


def construir_quadro_revisao_atestado(
    extraido: AtestadoExtraido,
    divergencias: Sequence[DivergenciaCampo],
    *,
    politica_extrair_cid: bool | None = None,
    validacao_cid: RetornoValidacaoCidFormulario | None = None,
) -> QuadroRevisaoAtestado:
    """Constrói o espelho determinístico dos campos lidos pelo LLM para revisão humana."""
    incluir_cid = bool(politica_extrair_cid) or (extraido.cid is not None) or (
        "cid" in extraido.campos and extraido.campos["cid"].vazio_por_baixa_confianca
    )

    if extraido.legibilidade == "ilegivel":
        itens_ileg: list[ItemRevisaoCampo] = []
        for chave, rotulo, obrigatorio in (
            ("nome_no_documento", "Paciente no documento", False),
            ("periodo.inicio", "Início do afastamento", True),
            ("periodo.fim", "Término do afastamento", True),
            ("profissional.conselho", "Conselho profissional", False),
            ("profissional.numero", "Número do registro", False),
            ("profissional.uf", "UF do registro", False),
        ):
            meta = extraido.campos.get(chave)
            eh_baixa = bool(meta and meta.vazio_por_baixa_confianca)
            status_item: StatusRevisaoCampo = (
                "baixa_confianca" if eh_baixa else "nao_identificado"
            )
            itens_ileg.append(
                ItemRevisaoCampo(
                    campo=chave,
                    rotulo=rotulo,
                    valor_lido=None,
                    valor_exibicao=(
                        "Baixa confiança — conferir/preencher"
                        if eh_baixa
                        else "Não identificado"
                    ),
                    confianca=_obter_confianca_campo(extraido, chave),
                    status=status_item,
                    editavel=False,
                    obrigatorio=obrigatorio,
                )
            )
        return QuadroRevisaoAtestado(
            titulo="Documento ilegível — novo envio necessário",
            instrucao_colaborador=(
                "Não foi possível ler os dados essenciais com segurança. "
                "Envie uma foto ou PDF mais nítido e sem cortes."
            ),
            legibilidade="ilegivel",
            resumo_markdown=(
                "**Leitura indisponível:** o documento está ilegível ou incompleto "
                "e nenhum registro foi salvo.\n\n"
                "Envie uma foto ou PDF mais nítido e sem cortes para que eu faça nova leitura."
            ),
            itens=itens_ileg,
            permite_confirmar_direto=False,
            permite_editar=False,
            requer_novo_arquivo=True,
        )

    div_por_campo: dict[str, DivergenciaCampo] = {}
    for div in divergencias:
        chave_item = _CAMPOS_DIVERGENCIA_PARA_ITEM.get(div.campo, div.campo)
        div_por_campo[chave_item] = div

    inicio = extraido.periodo.inicio
    fim = extraido.periodo.fim
    dias = extraido.periodo.dias_calculados
    tipo_exib = _ROTULOS_TIPO_DOCUMENTO.get(
        extraido.tipo_documento, extraido.tipo_documento
    )

    definicoes: list[tuple[str, str, Any, str, bool, bool]] = [
        ("tipo_documento", "Tipo de documento", extraido.tipo_documento, tipo_exib, True, True),
        (
            "nome_no_documento",
            "Paciente no documento",
            extraido.nome_no_documento,
            str(extraido.nome_no_documento or ""),
            True,
            False,
        ),
        (
            "periodo.inicio",
            "Início do afastamento",
            inicio.isoformat() if inicio is not None else None,
            _formatar_data_br(inicio),
            True,
            True,
        ),
        (
            "periodo.fim",
            "Término do afastamento",
            fim.isoformat() if fim is not None else None,
            _formatar_data_br(fim),
            True,
            True,
        ),
        (
            "periodo.dias",
            "Duração",
            dias,
            f"{dias} dia(s)" if dias is not None else "Não identificado",
            False,
            True,
        ),
        (
            "data_emissao",
            "Data de emissão",
            extraido.data_emissao.isoformat()
            if extraido.data_emissao is not None
            else None,
            _formatar_data_br(extraido.data_emissao),
            True,
            False,
        ),
        (
            "profissional.conselho",
            "Conselho profissional",
            extraido.profissional.conselho,
            str(extraido.profissional.conselho or ""),
            True,
            False,
        ),
        (
            "profissional.numero",
            "Número do registro",
            extraido.profissional.numero,
            str(extraido.profissional.numero or ""),
            True,
            False,
        ),
        (
            "profissional.uf",
            "UF do registro",
            extraido.profissional.uf,
            str(extraido.profissional.uf or ""),
            True,
            False,
        ),
    ]

    meta_prof_nome = extraido.campos.get("profissional.nome")
    if extraido.profissional.nome is not None or (
        meta_prof_nome is not None and meta_prof_nome.vazio_por_baixa_confianca
    ):
        definicoes.append(
            (
                "profissional.nome",
                "Nome do profissional",
                extraido.profissional.nome,
                str(extraido.profissional.nome or ""),
                True,
                False,
            )
        )

    definicoes.append(
        (
            "municipio",
            "Município de emissão",
            extraido.municipio,
            str(extraido.municipio or ""),
            True,
            False,
        )
    )
    if incluir_cid:
        definicoes.append(
            (
                "cid",
                "Código CID-10",
                extraido.cid,
                str(extraido.cid or ""),
                True,
                False,
            )
        )

    itens: list[ItemRevisaoCampo] = []
    cid_invalida_4b = (
        validacao_cid is not None and not validacao_cid.cid_valida
    ) or ("cid" in div_por_campo)
    for chave, rotulo, valor_lido, valor_fmt, editavel, obrigatorio in definicoes:
        meta = extraido.campos.get(chave)
        div = div_por_campo.get(chave)
        confianca = _obter_confianca_campo(extraido, chave)

        if div is not None:
            status_campo: StatusRevisaoCampo = "divergente"
            valor_exibicao = valor_fmt if valor_lido is not None else "Não identificado"
            valor_antes = div.informado
        elif chave == "cid" and cid_invalida_4b:
            status_campo = "divergente"
            valor_exibicao = valor_fmt if valor_lido is not None else "Não identificado"
            valor_antes = "CID Não Encontrada / Inválida"
        elif meta is not None and meta.vazio_por_baixa_confianca:
            status_campo = "baixa_confianca"
            valor_exibicao = "Baixa confiança — conferir/preencher"
            valor_antes = None
        elif valor_lido is None:
            status_campo = "nao_identificado"
            valor_exibicao = "Não identificado"
            valor_antes = None
        else:
            status_campo = "lido_ok"
            valor_exibicao = valor_fmt
            valor_antes = None

        itens.append(
            ItemRevisaoCampo(
                campo=chave,
                rotulo=rotulo,
                valor_lido=valor_lido,
                valor_exibicao=valor_exibicao,
                valor_informado_antes=valor_antes,
                confianca=confianca,
                status=status_campo,
                editavel=editavel,
                obrigatorio=obrigatorio,
            )
        )

    if inicio is not None and fim is not None and dias is not None:
        periodo_resumo = f"{inicio:%d/%m/%Y} a {fim:%d/%m/%Y} ({dias} dia(s))"
    elif inicio is not None:
        periodo_resumo = f"Início em {inicio:%d/%m/%Y} (término não identificado)"
    elif fim is not None:
        periodo_resumo = f"Término em {fim:%d/%m/%Y} (início não identificado)"
    else:
        baixa_ini = (m_i := extraido.campos.get("periodo.inicio")) is not None and m_i.vazio_por_baixa_confianca
        baixa_fim = (m_f := extraido.campos.get("periodo.fim")) is not None and m_f.vazio_por_baixa_confianca
        periodo_resumo = (
            "Baixa confiança (informar)"
            if (baixa_ini or baixa_fim)
            else "Não identificado"
        )

    linhas_md = [
        "**Dados lidos no documento:**",
        f"- **Tipo:** {tipo_exib}",
        "- **Paciente:** "
        + _formatar_valor_resumo(
            extraido.nome_no_documento,
            extraido=extraido,
            campo="nome_no_documento",
        ),
        f"- **Período:** {periodo_resumo}",
        "- **Emissão:** "
        + _formatar_valor_resumo(
            extraido.data_emissao,
            extraido=extraido,
            campo="data_emissao",
            exibicao_customizada=_formatar_data_br(extraido.data_emissao),
        ),
        f"- **Profissional:** {_formatar_profissional_resumo(extraido)}",
        "- **Município:** "
        + _formatar_valor_resumo(
            extraido.municipio,
            extraido=extraido,
            campo="municipio",
        ),
    ]
    if incluir_cid:
        linhas_md.append(
            "- **CID-10:** "
            + _formatar_valor_resumo(
                extraido.cid,
                extraido=extraido,
                campo="cid",
            )
        )
        if validacao_cid is not None:
            linhas_md.append(f"- **Status CID:** {validacao_cid.status_validacao}")
            desc_cid = validacao_cid.descricao_oficial or validacao_cid.nome_doenca
            if validacao_cid.cid_valida and desc_cid:
                linhas_md.append(f"- **Descrição oficial CID:** {desc_cid}")

    permite_confirmar = bool(
        inicio is not None and fim is not None and fim >= inicio and not cid_invalida_4b
    )
    if cid_invalida_4b:
        titulo = (
            "Inconsistência no CID — revisão humana necessária"
            if (not divergencias or all(d.campo == "cid" for d in divergencias))
            else "Atenção: divergências encontradas — valide os dados"
        )
        instrucao = (
            "CID Não Encontrada / Inválida. A aprovação automática foi bloqueada: "
            "corrija o código CID ou reenvie o documento."
        )
    else:
        titulo = (
            "Atenção: divergências encontradas — valide os dados"
            if divergencias
            else "Revise e valide os dados lidos no documento"
        )
        instrucao = (
            "Confira abaixo as informações lidas no seu documento. "
            "Confirme se está tudo certo ou informe o que precisa alterar."
        )
    return QuadroRevisaoAtestado(
        titulo=titulo,
        instrucao_colaborador=instrucao,
        legibilidade="ok",
        resumo_markdown="\n".join(linhas_md),
        itens=itens,
        permite_confirmar_direto=permite_confirmar,
        permite_editar=True,
        requer_novo_arquivo=False,
        validacao_cid=validacao_cid,
    )


def gerar_mensagem_confirmacao_extracao(
    extraido: AtestadoExtraido,
    divergencias: Sequence[DivergenciaCampo],
    *,
    politica_extrair_cid: bool | None = None,
    validacao_cid: RetornoValidacaoCidFormulario | None = None,
) -> str:
    """Gera a pergunta de confirmação com todos os dados lidos após `a1.processar_documento`."""
    if extraido.legibilidade == "ilegivel":
        return (
            "Não consegui ler o atestado com clareza. "
            "Pode enviar uma foto ou PDF mais nítido e sem cortes?"
        )

    campos_ileg = [
        _LABELS_CAMPOS_BAIXA_CONFIANCA.get(nome_campo, nome_campo)
        for nome_campo, meta in extraido.campos.items()
        if meta.vazio_por_baixa_confianca
    ]
    sufixo_baixa_conf = (
        f" Também não consegui ler com segurança: {', '.join(campos_ileg)}."
        if campos_ileg
        else ""
    )

    inicio = extraido.periodo.inicio
    fim = extraido.periodo.fim

    if len(divergencias) == 1:
        div = divergencias[0]
        if div.campo == "dias" and inicio is not None and fim is not None:
            frase_inicial = (
                f"Li no atestado {div.no_documento} dias ({inicio:%d/%m} a {fim:%d/%m}), "
                f"mas você informou {div.informado}. Qual está certo?{sufixo_baixa_conf}"
            )
        elif div.campo == "inicio" and inicio is not None:
            data_inf = div.informado
            try:
                data_inf_fmt = f"{date.fromisoformat(str(div.informado)):%d/%m}"
            except ValueError:
                data_inf_fmt = str(data_inf)
            frase_inicial = (
                f"Li no atestado início em {inicio:%d/%m}, mas você informou "
                f"{data_inf_fmt}. Qual data está correta?{sufixo_baixa_conf}"
            )
        elif div.campo == "nome":
            frase_inicial = (
                f"O nome lido no documento foi '{div.no_documento}', diferente do "
                f"seu cadastro ('{div.informado}'). Confirma se este atestado é seu?{sufixo_baixa_conf}"
            )
        elif div.campo == "cid":
            frase_inicial = (
                f"O código CID '{div.no_documento}' foi marcado como 'CID Não Encontrada / Inválida' "
                f"e a aprovação automática foi bloqueada. Corrija o código CID ou reenvie o documento.{sufixo_baixa_conf}"
            )
        else:
            frase_inicial = (
                f"Encontrei diferença em {div.campo} (no documento '{div.no_documento}' vs "
                f"informado '{div.informado}'). Pode revisar e confirmar os dados corretos?{sufixo_baixa_conf}"
            )
    elif divergencias:
        itens = "; ".join(
            f"{d.campo}: no documento '{d.no_documento}' vs informado '{d.informado}'"
            for d in divergencias
        )
        bloqueio_cid_txt = (
            " Aprovação automática bloqueada por CID Não Encontrada / Inválida — corrija o CID ou reenvie o documento."
            if (validacao_cid is not None and not validacao_cid.cid_valida)
            or any(d.campo == "cid" for d in divergencias)
            else " Pode revisar e confirmar os dados corretos?"
        )
        frase_inicial = (
            f"Encontrei diferenças entre o atestado e o informado inicialmente ({itens})."
            f"{bloqueio_cid_txt}{sufixo_baixa_conf}"
        )
    elif inicio is not None and fim is not None:
        dias = extraido.periodo.dias_calculados
        frase_inicial = (
            f"Li seu atestado de {inicio:%d/%m} a {fim:%d/%m} ({dias} dia(s)). "
            f"Os dados extraídos estão corretos para eu registrar?{sufixo_baixa_conf}"
        )
    else:
        frase_inicial = (
            f"Extraí os dados do seu atestado. Pode revisar e confirmar as informações?{sufixo_baixa_conf}"
        )

    quadro = construir_quadro_revisao_atestado(
        extraido,
        divergencias,
        politica_extrair_cid=politica_extrair_cid,
        validacao_cid=validacao_cid,
    )
    return (
        f"{frase_inicial}\n\n"
        f"{quadro.resumo_markdown}\n\n"
        "Confirme se os dados acima estão corretos ou informe o que deseja alterar."
    )


def _eh_dia_util(data: date, feriados: frozenset[date]) -> bool:
    return data.weekday() < 5 and data not in feriados


def calcular_prazo_entrega_fisica(
    fim_afastamento: date,
    dias_uteis: int = 2,
    feriados: Collection[date] | None = None,
) -> date:
    """Calcula a data limite para entrega da via física ao RH (Anexo C.2 / D.2).

    Parte do dia de retorno ao trabalho (``fim_afastamento + 1 dia``, avançando
    para o próximo dia útil se cair em fim de semana ou feriado) e soma
    ``dias_uteis`` dias úteis (padrão: 2 dias úteis).
    """
    if dias_uteis < 0:
        raise ValueError("dias_uteis não pode ser negativo.")

    conjunto_feriados = frozenset(feriados or ())

    data_retorno = fim_afastamento + timedelta(days=1)
    while not _eh_dia_util(data_retorno, conjunto_feriados):
        data_retorno += timedelta(days=1)

    prazo = data_retorno
    restantes = dias_uteis
    while restantes > 0:
        prazo += timedelta(days=1)
        if _eh_dia_util(prazo, conjunto_feriados):
            restantes -= 1

    return prazo


class MarcosLegaisAfastamento(BaseModel):
    """Marcos legais aplicáveis ao afastamento (Anexo E.2, Lei 8.213/91 e NR-07)."""

    model_config = ConfigDict(extra="forbid")

    dias_afastamento: int
    dias_acumulados_60d: int = 0
    dias_totais_60d: int
    exige_encaminhamento_inss: bool
    data_marco_16_dia: date | None = None
    exige_exame_retorno_nr7: bool

    def codigos_prazos_legais(self) -> list[str]:
        """Retorna os códigos de marcos legais padronizados do Guia A1 v0.7."""
        codigos: list[str] = []
        if self.exige_encaminhamento_inss:
            codigos.append("inss_16_dias")
        if self.exige_exame_retorno_nr7:
            codigos.append("exame_retorno")
        return codigos


def calcular_marcos_legais(
    inicio: date,
    fim: date,
    dias_acumulados_60d: int = 0,
) -> MarcosLegaisAfastamento:
    """Calcula o marco do 16º dia (INSS) e a exigência de exame de retorno (NR-07).

    - Lei 8.213/1991 Art. 60 § 3º e Decreto 3.048/1999 Art. 75 §§ 4º-5º:
      afastamentos acima de 15 dias (isolados ou somados em janela de 60 dias)
      geram marco do 16º dia para encaminhamento ao INSS.
    - NR-07 subitem 7.5.9: afastamento igual ou superior a 30 dias exige exame
      clínico de retorno ao trabalho antes da retomada das funções.
    """
    if fim < inicio:
        raise ValueError("Data de fim não pode ser anterior à de início.")
    if dias_acumulados_60d < 0:
        raise ValueError("dias_acumulados_60d não pode ser negativo.")

    dias_afastamento = (fim - inicio).days + 1
    dias_totais_60d = dias_afastamento + dias_acumulados_60d
    exige_inss = dias_afastamento > 15 or dias_totais_60d > 15
    data_marco_16 = inicio + timedelta(days=15) if dias_afastamento > 15 else None

    return MarcosLegaisAfastamento(
        dias_afastamento=dias_afastamento,
        dias_acumulados_60d=dias_acumulados_60d,
        dias_totais_60d=dias_totais_60d,
        exige_encaminhamento_inss=exige_inss,
        data_marco_16_dia=data_marco_16,
        exige_exame_retorno_nr7=dias_afastamento >= 30,
    )


def eh_cid10(codigo_cid: str | None) -> bool:
    """Verifica se o código segue o formato sintático CID-10 (ex.: ``M54.5``, ``J06.9``)."""
    if not codigo_cid:
        return False
    normalizado = codigo_cid.strip().upper()
    return len(normalizado) <= 8 and bool(_PADRAO_CID10.match(normalizado))


def eh_cid11(codigo_cid: str | None) -> bool:
    """Verifica se o código segue o formato sintático CID-11 (ex.: ``ME84.2``, ``1A00``, ``CA07.0``)."""
    if not codigo_cid:
        return False
    normalizado = codigo_cid.strip().upper()
    return len(normalizado) <= 8 and bool(_PADRAO_CID11.match(normalizado))


def cid_sintaxe_valida(codigo_cid: str | None) -> bool:
    """Valida o formato sintático de um código CID-11 ou CID-10."""
    return eh_cid11(codigo_cid) or eh_cid10(codigo_cid)


class ResultadoValidacaoConfirmacao(BaseModel):
    """Resultado consolidado da validação de `a1.confirmar`."""

    model_config = ConfigDict(extra="forbid")

    valido: bool
    motivo: str | None = None
    codigo_evento: str | None = None
    mensagem_usuario: str
    origem: OrigemDadosAtestado | None = None
    dias: int | None = None
    prazo_entrega_fisica: date | None = None
    marcos_legais: MarcosLegaisAfastamento | None = None
    erros_detalhe: list[str] = Field(default_factory=list)


def validar_dados_confirmados(
    atestado: AtestadoExtraido,
    campos_corrigidos: Sequence[str] = (),
    *,
    colaborador_ativo: bool = True,
    conselhos_aceitos: Collection[str] = CONSELHOS_ACEITOS_PADRAO,
    exigir_profissional: bool = False,
    politica_extrair_cid: bool = True,
    cid_existe_catalogo: bool | None = None,
    dias_uteis_entrega_fisica: int = 2,
    feriados: Collection[date] | None = None,
    dias_acumulados_60d: int = 0,
) -> ResultadoValidacaoConfirmacao:
    """Valida as regras de negócio na confirmação/correção do atestado (`a1.confirmar`)."""
    inicio = atestado.periodo.inicio
    fim = atestado.periodo.fim

    if inicio is None or fim is None:
        return ResultadoValidacaoConfirmacao(
            valido=False,
            motivo="período de afastamento incompleto",
            codigo_evento="periodo_incompleto",
            mensagem_usuario=(
                "Preciso das datas de início e fim do afastamento para registrar o atestado. "
                "Pode informá-las?"
            ),
            erros_detalhe=["periodo.inicio e periodo.fim são obrigatórios"],
        )

    if fim < inicio:
        return ResultadoValidacaoConfirmacao(
            valido=False,
            motivo="data de fim anterior à de início",
            codigo_evento="fim < inicio",
            mensagem_usuario=(
                "Acho que há um erro nas datas do atestado. "
                "Pode me confirmar de quando até quando é o afastamento?"
            ),
            erros_detalhe=[f"fim ({fim}) < inicio ({inicio})"],
        )

    if atestado.data_emissao is not None and atestado.data_emissao > fim:
        return ResultadoValidacaoConfirmacao(
            valido=False,
            motivo="data de emissão posterior ao fim do afastamento",
            codigo_evento="emissao_futura",
            mensagem_usuario=(
                "A data de emissão do atestado está posterior ao término do afastamento. "
                "Pode conferir a data de emissão?"
            ),
            erros_detalhe=[f"data_emissao ({atestado.data_emissao}) > fim ({fim})"],
        )

    if not colaborador_ativo:
        return ResultadoValidacaoConfirmacao(
            valido=False,
            motivo="colaborador inativo no período do afastamento",
            codigo_evento="colaborador_inativo",
            mensagem_usuario=(
                "Seu cadastro não aparece como ativo para o período informado. "
                "Encaminhei a solicitação para conferência do RH."
            ),
            erros_detalhe=["cadastro.colaborador.ativo = FALSE"],
        )

    prof = atestado.profissional
    informou_profissional = any(
        valor is not None for valor in (prof.conselho, prof.numero, prof.uf)
    )
    if exigir_profissional or informou_profissional:
        erros_prof: list[str] = []
        conselhos_norm = {c.strip().upper() for c in conselhos_aceitos}
        if not prof.conselho or prof.conselho not in conselhos_norm:
            erros_prof.append(f"conselho inválido ou ausente: {prof.conselho!r}")
        if not prof.numero:
            erros_prof.append("número de registro profissional ausente")
        if not prof.uf or prof.uf not in UFS_BRASIL:
            erros_prof.append(f"UF do registro profissional inválida: {prof.uf!r}")
        if erros_prof:
            return ResultadoValidacaoConfirmacao(
                valido=False,
                motivo="dados do profissional de saúde inválidos",
                codigo_evento="profissional_invalido",
                mensagem_usuario=(
                    "Os dados do profissional de saúde (conselho, número e UF) "
                    "parecem incompletos ou inválidos. Pode revisá-los?"
                ),
                erros_detalhe=erros_prof,
            )

    if politica_extrair_cid and atestado.cid is not None:
        if not cid_sintaxe_valida(atestado.cid):
            return ResultadoValidacaoConfirmacao(
                valido=False,
                motivo="código CID inválido",
                codigo_evento="cid_invalido",
                mensagem_usuario=(
                    f"O código CID '{atestado.cid}' foi marcado como 'CID Não Encontrada / Inválida' "
                    "(formato esperado: ex. ME84.2 ou M54.5) e a aprovação automática foi bloqueada. "
                    "Corrija o código CID ou reenvie o documento."
                ),
                erros_detalhe=[f"CID fora do padrão CID-11/CID-10: {atestado.cid!r}"],
            )
        if cid_existe_catalogo is False:
            return ResultadoValidacaoConfirmacao(
                valido=False,
                motivo="código CID não encontrado na tabela oficial CID",
                codigo_evento="cid_invalido",
                mensagem_usuario=(
                    f"O código CID '{atestado.cid}' foi marcado como 'CID Não Encontrada / Inválida' "
                    "e a aprovação automática foi bloqueada. Corrija o código CID ou reenvie o documento."
                ),
                erros_detalhe=[f"CID inexistente na base/API WHO: {atestado.cid!r}"],
            )

    origem: OrigemDadosAtestado = (
        "MANUAL" if any(campo.strip() for campo in campos_corrigidos) else "IA"
    )
    marcos = calcular_marcos_legais(
        inicio, fim, dias_acumulados_60d=dias_acumulados_60d
    )
    prazo_fisico = calcular_prazo_entrega_fisica(
        fim, dias_uteis=dias_uteis_entrega_fisica, feriados=feriados
    )

    return ResultadoValidacaoConfirmacao(
        valido=True,
        origem=origem,
        dias=marcos.dias_afastamento,
        prazo_entrega_fisica=prazo_fisico,
        marcos_legais=marcos,
        mensagem_usuario=(
            f"Pronto, seu atestado de {inicio:%d/%m} a {fim:%d/%m} foi registrado. "
            f"Entregue o original ao RH até {prazo_fisico:%d/%m}."
        ),
    )
