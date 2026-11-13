# Spec Delta

## Purpose

Fornece a interface web interativa de demonstração (Pitch) intitulada "SinergIA - RAG Demo" em `src/sinergia/rag/pitch/`, integrando um chat assistido por LLM local (Ollama) com roteamento autônomo entre as 5 bases de conhecimento e um painel de telemetria e diagnóstico técnico ("Raio-X Nerd") em tema monocromático claro (light theme).

## ADDED Requirements

### Requirement: Interface Conversacional Limpa com Tema Monocromático Light

O sistema SHALL disponibilizar uma aplicação web em `src/sinergia/rag/pitch/` intitulada **"SinergIA - RAG Demo"** em **light theme monocromático** (tons de branco, cinza e preto), exibindo um seletor para escolher o modelo local do Ollama disponível, sem exibir filtro manual de base/domínio, ajustes de hiperparâmetros (`top-k`, temperatura, `k` RRF) ou rótulos de infraestrutura no cabeçalho.

#### Scenario: Carregamento inicial da página de demonstração

- **WHEN** o usuário acessa a aplicação em `src/sinergia/rag/pitch/`
- **THEN** a página exibe o cabeçalho "SinergIA - RAG Demo" com o seletor de modelo Ollama sobre tema claro monocromático, a coluna de chat à esquerda e o painel de métricas globais à direita, sem seletores de base ou hiperparâmetros.

### Requirement: Roteamento Autônomo de Base e Geração Local via Ollama

O sistema SHALL utilizar uma LLM local via Ollama para decidir autonomamente qual das 5 bases de conhecimento (`F1` Pausas/NR-17, `F2` Direitos/CLT, `F3` Procedimentos/POPs, `F4` Saúde Ocupacional/PCMSO, `F5` EPIs/CA) consultar a partir da pergunta do usuário e gerar a resposta fundamentada com citações.

#### Scenario: Pergunta do usuário respondida com escolha autônoma de base

- **WHEN** o usuário envia uma pergunta no chat (ex: "Qual a pausa para digitação contínua?")
- **THEN** a LLM local seleciona autonomamente a ferramenta/base correspondente (`F1` - Base 1), executa a busca híbrida no RAG e exibe a resposta acompanhada das citações de documento, versão, seção e página.

### Requirement: Indicativo Visual da Base de Conhecimento Consultada

O sistema SHALL exibir, em cada resposta da LLM no chat, um indicativo visual (badge) identificando explicitamente qual base de conhecimento (`F1` a `F5` e seu nome de domínio) foi consultada pela LLM, juntamente com o tempo total da resposta.

#### Scenario: Exibição de badge de domínio na mensagem da assistente

- **WHEN** a resposta da LLM é renderizada no histórico do chat após consultar uma base de conhecimento
- **THEN** o cartão da resposta apresenta o indicativo visual da base acionada (ex: `Base F1 · Pausas e Ergonomia (NR-17)`) e o tempo total decorrido em segundos.

### Requirement: Visão Global de Telemetria por Padrão

Quando nenhuma resposta individual estiver selecionada para inspeção, o painel lateral direito SHALL exibir a **Visão Global da Sessão** com o tempo médio total das respostas, latência média do RAG, latência e velocidade média (tokens/s) da LLM, total de tokens gerados e a distribuição de acionamento por base (`F1` a `F5`).

#### Scenario: Atualização das métricas globais após nova pergunta

- **WHEN** o usuário realiza perguntas no chat sem selecionar uma resposta específica para inspeção
- **THEN** o painel direito exibe a Visão Global da Sessão com as médias de tempo atualizadas, o total de tokens gerados e a contagem/percentual de consultas por base de conhecimento.

### Requirement: Inspeção Técnica por Resposta ("Raio-X Nerd") sob Demanda

O sistema SHALL permitir selecionar qualquer resposta da LLM no chat para alternar o painel direito para o modo **Raio-X Nerd** daquela resposta, exibindo botão de retorno à Visão Global, os tempos desmembrados (`Total`, `Embedding`, `DB + RRF`, `LLM`) e 4 abas técnicas (`Chunks & RRF`, `Embedding (768d)`, `Metadados` e `Tool & Prompt`).

#### Scenario: Seleção de uma resposta para abrir o Raio-X Nerd

- **WHEN** o usuário aciona a inspeção de uma resposta específica da LLM
- **THEN** o painel direito substitui a Visão Global pelo Raio-X Nerd daquela resposta mostrando: (1) Chunks com scores denso, léxico e RRF; (2) Embedding 768d com norma L2, estatísticas e amostra do vetor; (3) Metadados de linhagem e JSONB `extra`; e (4) Tool acionada, métricas do Ollama e prompt formatado.
