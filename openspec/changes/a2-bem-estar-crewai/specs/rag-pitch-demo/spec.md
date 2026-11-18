# Spec Delta

## MODIFIED Requirements

### Requirement: Interface Conversacional Limpa com Tema Monocromático Light

O sistema SHALL disponibilizar uma aplicação web em `src/sinergia/chat_demo/` intitulada **"SinergIA - RAG Demo"** em **light theme monocromático** (tons de branco, cinza e preto), exibindo um seletor para escolher o modelo da Groq Cloud disponível para a CrewAI do Agente 2, sem exibir filtro manual de base/domínio, ajustes de hiperparâmetros (`top-k`, temperatura, `k` RRF) ou rótulos de infraestrutura no cabeçalho.

#### Scenario: Carregamento inicial da página de demonstração

- **WHEN** o usuário acessa a aplicação em `src/sinergia/chat_demo/`
- **THEN** a página exibe o cabeçalho "SinergIA - RAG Demo" com o seletor de modelo Groq sobre tema claro monocromático, a coluna de chat à esquerda e o painel de métricas globais à direita, sem seletores de base ou hiperparâmetros.

### Requirement: Roteamento Autônomo de Base e Geração Local via Ollama

O sistema SHALL utilizar o **Agente 2 (`src/sinergia/agentes/a2_bem_estar/`)** com **CrewAI** e **Groq Cloud (`GROQ_API_KEY`)** através da ação `a2.responder` (protocolo `sinergia/1`) para decidir autonomamente qual(is) das 5 bases de conhecimento (`F1` Pausas/NR-17, `F2` Direitos/CLT, `F3` Procedimentos/POPs, `F4` Saúde Ocupacional/PCMSO, `F5` EPIs/CA) consultar a partir da pergunta do usuário e gerar a resposta fundamentada com citações.

#### Scenario: Pergunta do usuário respondida com escolha autônoma de base

- **WHEN** o usuário envia uma pergunta no chat em `src/sinergia/chat_demo/` (ex: "Qual a pausa para digitação contínua?")
- **THEN** o `chat_demo` despacha uma `TAREFA` `a2.responder` para o Agente 2 (`a2_bem_estar`), que aciona o especialista correspondente (`F1` - Base 1) via CrewAI + Groq, executa a busca híbrida no RAG e retorna o `RESULTADO` acompanhado das citações de documento, versão, seção e página.
