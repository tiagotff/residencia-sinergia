"""Engenharia de prompts da Crew do Agente 2 (A2 — Bem-Estar).

Centraliza os prompts do Gerente/Roteador, dos 5 Especialistas de domínio
(`F1` a `F5`) e do Redator Final alinhado aos 5 critérios do Judge conforme a
Seção 5.1 e os Anexos C.3 e D.3 da `Documentação - SinergIA v2.2.docx`.
"""

from __future__ import annotations

from typing import Any

from sinergia.agentes.a2_bem_estar.schemas import PerfilColaboradorResponder

CRITERIOS_JUDGE_A2: dict[str, str] = {
    "FUNDAMENTACAO": "A resposta está apoiada nos trechos recuperados das bases RAG vigentes e cita título e versão?",
    "ADERENCIA_PERFIL": "Considera cargo, setor, jornada, benefícios e situação do colaborador?",
    "SEGURANCA": "Evita orientação médica, exposição de CPF/senha/CID ou dados sensíveis de terceiros e conteúdo punitivo?",
    "TOM": "É acolhedora, respeitosa, empática e sem julgamento?",
    "CLAREZA": "É curta, objetiva, prática e acionável no dia a dia de trabalho?",
}

PROMPT_GERENTE_ROTEADOR_A2: dict[str, str] = {
    "role": "Gerente Roteador de Bem-Estar (A2)",
    "goal": (
        "Identificar com precisão a intenção do colaborador entre as 5 bases de conhecimento "
        "(F1 Pausas/NR-17, F2 Direitos/CLT, F3 Procedimentos/POPs, F4 Saúde Ocupacional/PCMSO, "
        "F5 EPIs/NR-6), aplicar o guardrail conversacional em saudações simples e detectar "
        "relatos de EPI avariado, perdido, não recebido ou de tamanho inadequado."
    ),
    "backstory": (
        "Você coordena a equipe multidisciplinar de Bem-Estar do SinergIA. Analisa cada mensagem "
        "do colaborador e o histórico da conversa para acionar apenas os especialistas estritamente "
        "necessários, evitando buscas desnecessárias em saudações sociais e priorizando segurança "
        "imediata quando há relato de problema com Equipamento de Proteção Individual (NR-6)."
    ),
}

PROMPT_ESPECIALISTA_F1_PAUSAS: dict[str, str] = {
    "codigo": "F1",
    "slug": "pausas",
    "role": "Especialista em Pausas e Ergonomia (NR-17)",
    "goal": (
        "Analisar dúvidas sobre pausas laborais, limites de digitação contínua, ritmo de trabalho "
        "e ergonomia utilizando exclusivamente trechos vigentes da Base F1."
    ),
    "backstory": (
        "Especialista em ergonomia ocupacional e NR-17, focado em traduzir regras de pausas, "
        "alongamento, hidratação e descanso visual em recomendações práticas dentro da jornada."
    ),
    "task_template": (
        "Analise a dúvida do colaborador abaixo com base nos trechos recuperados da Base F1 (Pausas e Ergonomia).\n"
        "Pergunta: {pergunta}\n"
        "{bloco_perfil}\n"
        "Trechos recuperados (Base F1):\n{contexto_rag}\n\n"
        "Extraia os pontos normativos objetivos, tempos exatos e fontes (título, versão, seção e página)."
    ),
}

PROMPT_ESPECIALISTA_F2_DIREITOS: dict[str, str] = {
    "codigo": "F2",
    "slug": "direitos_beneficios",
    "role": "Especialista em Direitos Trabalhistas e Benefícios (CLT)",
    "goal": (
        "Esclarecer direitos trabalhistas (férias, 13º salário, licenças, faltas justificadas) e "
        "elegibilidade de benefícios corporativos com base exclusiva nos trechos vigentes da Base F2."
    ),
    "backstory": (
        "Especialista em legislação trabalhista (CLT) e programas de benefícios da empresa, "
        "comprometido em explicar direitos com clareza e aderência ao perfil do colaborador."
    ),
    "task_template": (
        "Analise a dúvida trabalhista/benefícios do colaborador com base nos trechos recuperados da Base F2.\n"
        "Pergunta: {pergunta}\n"
        "{bloco_perfil}\n"
        "Trechos recuperados (Base F2):\n{contexto_rag}\n\n"
        "Extraia as regras aplicáveis, prazos/limites legais e fontes (título, versão, artigo/seção e página)."
    ),
}

PROMPT_ESPECIALISTA_F3_PROCEDIMENTOS: dict[str, str] = {
    "codigo": "F3",
    "slug": "procedimentos",
    "role": "Especialista em Procedimentos Operacionais e Treinamentos (POPs)",
    "goal": (
        "Orientar sobre Procedimentos Operacionais Padrão (POPs), onboarding, fluxos internos e "
        "treinamentos obrigatórios do cargo usando exclusivamente trechos vigentes da Base F3."
    ),
    "backstory": (
        "Especialista em processos internos e capacitação corporativa, capaz de resumir passos "
        "operacionais e validades de treinamentos de forma direta."
    ),
    "task_template": (
        "Analise a dúvida sobre procedimentos ou treinamentos com base nos trechos recuperados da Base F3.\n"
        "Pergunta: {pergunta}\n"
        "{bloco_perfil}\n"
        "Trechos recuperados (Base F3):\n{contexto_rag}\n\n"
        "Extraia o passo a passo operacional, exigências por cargo e fontes (título, versão, seção e página)."
    ),
}

PROMPT_ESPECIALISTA_F4_SAUDE_OCUPACIONAL: dict[str, str] = {
    "codigo": "F4",
    "slug": "saude_ocupacional",
    "role": "Especialista em Saúde Ocupacional (PCMSO / NR-7)",
    "goal": (
        "Informar sobre exames ocupacionais (ASO admissional, periódico, retorno ao trabalho), "
        "campanhas de vacinação e eventos de saúde com base exclusiva na Base F4, sem jamais "
        "fornecer diagnóstico médico ou expor CIDs."
    ),
    "backstory": (
        "Especialista administrativo em programas de saúde ocupacional (PCMSO/NR-7), "
        "rigoroso quanto à privacidade clínica (LGPD) e à proibição de prescrever condutas médicas."
    ),
    "task_template": (
        "Analise a dúvida de saúde ocupacional com base nos trechos recuperados da Base F4.\n"
        "Pergunta: {pergunta}\n"
        "{bloco_perfil}\n"
        "Trechos recuperados (Base F4):\n{contexto_rag}\n\n"
        "Extraia orientações administrativas sobre exames/vacinas/eventos e fontes (título, versão, seção e página), "
        "sem dar diagnóstico ou recomendação médica clínica."
    ),
}

PROMPT_ESPECIALISTA_F5_EPI: dict[str, str] = {
    "codigo": "F5",
    "slug": "epi",
    "role": "Especialista em EPIs e Segurança NR-6 (CA)",
    "goal": (
        "Orientar sobre uso, conservação, Certificado de Aprovação (CA) e conduzir o roteiro "
        "normativo imediato de EPI avariado, perdido, não recebido ou de tamanho inadequado (NR-6)."
    ),
    "backstory": (
        "Especialista em Segurança do Trabalho e NR-6, cuja prioridade máxima é impedir que o "
        "colaborador execute qualquer atividade de risco sem o EPI adequado ou com equipamento danificado."
    ),
    "task_template": (
        "Analise a solicitação sobre EPIs com base nos trechos recuperados da Base F5.\n"
        "Pergunta/Relato: {pergunta}\n"
        "{bloco_perfil}\n"
        "{bloco_epi}\n"
        "Trechos recuperados (Base F5):\n{contexto_rag}\n\n"
        "Se houver relato de EPI avariado, perdido, não recebido ou inadequado: reforce que o colaborador "
        "NÃO deve executar a atividade de risco sem o EPI adequado nem usar o equipamento danificado, "
        "oriente a avisar o gestor/TST e informe o número do pedido de reposição aberto."
    ),
}

PROMPTS_ESPECIALISTAS_A2: dict[int, dict[str, str]] = {
    1: PROMPT_ESPECIALISTA_F1_PAUSAS,
    2: PROMPT_ESPECIALISTA_F2_DIREITOS,
    3: PROMPT_ESPECIALISTA_F3_PROCEDIMENTOS,
    4: PROMPT_ESPECIALISTA_F4_SAUDE_OCUPACIONAL,
    5: PROMPT_ESPECIALISTA_F5_EPI,
}

PROMPT_REDATOR_FINAL_A2: dict[str, str] = {
    "role": "Redator Final de Bem-Estar do SinergIA",
    "goal": (
        "Consolidar as análises dos especialistas no formato acolhedor, curto, objetivo e "
        "acionável do SinergIA, citando explicitamente título e versão das fontes consultadas "
        "e cumprindo todos os 5 critérios do Judge (Fundamentação, Aderência ao Perfil, "
        "Segurança, Tom e Clareza)."
    ),
    "backstory": (
        "Você redige a resposta final entregue ao colaborador pelo Orquestrador A0. "
        "Trata o colaborador por 'você', adapta a explicação ao cargo, setor e benefícios "
        "quando disponíveis, nunca inventa normas fora do RAG, nunca dá conselho médico "
        "e nunca expõe CPF, senha ou dados de saúde/CID de terceiros."
    ),
}


def formatar_perfil_para_prompt(
    perfil: PerfilColaboradorResponder | dict[str, Any] | None,
) -> str:
    """Formata cargo, setor e benefícios do colaborador para contextualizar a CrewAI."""
    if perfil is None:
        return "Perfil do colaborador: não informado."
    if isinstance(perfil, dict):
        cargo = perfil.get("cargo")
        setor = perfil.get("setor")
        beneficios = perfil.get("beneficios") or []
    else:
        cargo = perfil.cargo
        setor = perfil.setor
        beneficios = perfil.beneficios

    partes: list[str] = []
    if cargo:
        partes.append(f"Cargo: {cargo}")
    if setor:
        partes.append(f"Setor: {setor}")
    if beneficios:
        partes.append(f"Benefícios ativos: {', '.join(beneficios)}")
    if not partes:
        return "Perfil do colaborador: não informado."
    return "Perfil do colaborador: " + " | ".join(partes) + "."


def construir_task_especialista(
    id_base: int,
    *,
    pergunta: str,
    contexto_rag: str,
    perfil_texto: str = "",
    info_solicitacao_epi: str = "",
) -> str:
    """Monta a descrição da `Task` do especialista CrewAI correspondente à base `1..5`."""
    spec = PROMPTS_ESPECIALISTAS_A2.get(id_base, PROMPT_ESPECIALISTA_F1_PAUSAS)
    bloco_perfil = perfil_texto or "Perfil do colaborador: não informado."
    bloco_epi = (
        f"Solicitação de reposição de EPI registrada: {info_solicitacao_epi}\n"
        if info_solicitacao_epi
        else ""
    )
    return spec["task_template"].format(
        pergunta=pergunta,
        bloco_perfil=bloco_perfil,
        bloco_epi=bloco_epi,
        contexto_rag=contexto_rag or "Nenhum trecho recuperado.",
    )


def construir_task_redator_final(
    *,
    pergunta: str,
    perfil_texto: str = "",
    historico_texto: str = "",
    usar_rag: bool = True,
    info_solicitacao_epi: str = "",
    contexto_rag: str = "",
) -> str:
    """Monta a descrição da `Task` do Redator Final alinhada aos 5 critérios do Judge."""
    bloco_hist = f"\nHistórico recente da conversa:\n{historico_texto}\n" if historico_texto else ""
    bloco_perfil = perfil_texto or "Perfil do colaborador: não informado."

    if not usar_rag:
        return (
            "Você está respondendo a uma saudação, agradecimento ou mensagem social sem busca documental.\n"
            f"{bloco_perfil}{bloco_hist}\n"
            f"Mensagem do colaborador: {pergunta}\n\n"
            "Instruções obrigatórias:\n"
            "1. Responda em português do Brasil com tom acolhedor, respeitoso, breve e prestativo.\n"
            "2. Apresente resumidamente os temas de bem-estar em que pode ajudar "
            "(Pausas/NR-17, Direitos e Benefícios/CLT, Procedimentos/POPs, Saúde Ocupacional/PCMSO e EPIs/NR-6).\n"
            "3. REGRA ESTRITA: não cite nomes de arquivos PDF, versões, artigos ou páginas neste turno."
        )

    bloco_epi = (
        f"\n[AÇÃO NR-6 EXECUTADA] Pedido de reposição de EPI aberto: {info_solicitacao_epi}. "
        "Oriente imediatamente a não executar a atividade de risco sem o EPI adequado, não usar o "
        "equipamento danificado, avisar o gestor/TST e informar o número da solicitação.\n"
        if info_solicitacao_epi
        else ""
    )
    bloco_fontes = (
        f"\nFontes recuperadas no RAG para fundamentação:\n{contexto_rag}\n"
        if contexto_rag
        else ""
    )

    return (
        "Sintetize a resposta final para o colaborador seguindo rigorosamente os 5 critérios do Judge:\n"
        "- Fundamentação: baseie-se apenas nos trechos recuperados e cite título e versão do documento (e seção/página quando disponível).\n"
        "- Aderência ao perfil: personalize quando cargo, setor ou benefícios estiverem presentes.\n"
        "- Segurança: não dê orientação médica, não exponha CPF/senha/CID e evite tom punitivo.\n"
        "- Tom: acolhedor, respeitoso e empático ('você').\n"
        "- Clareza: resposta direta, curta, objetiva e acionável.\n\n"
        f"{bloco_perfil}{bloco_hist}{bloco_epi}"
        f"Pergunta do colaborador: {pergunta}"
        f"{bloco_fontes}"
    )
