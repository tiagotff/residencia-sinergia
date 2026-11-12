# rag-subsystem Specification

## Purpose
Fornece o subsistema de Busca e Geração Aumentada por Recuperação (RAG) para o Agente A2 (Bem-Estar), viabilizando respostas normativas fundamentadas com citação explícita de fonte e versão documental sobre PostgreSQL/pgvector.

## Requirements

### Requirement: Vetorização com Modelo Especializado em Português

O subsistema SHALL gerar representações vetoriais densas de 768 dimensões para textos e consultas utilizando o modelo pré-treinado em português brasileiro `tardellirs/colibri-embed-ptbr`.

#### Scenario: Geração de embeddings de texto

- **WHEN** um texto de trecho ou consulta é submetido para vetorização
- **THEN** o sistema gera um vetor denso em ponto flutuante com exatamente 768 dimensões compatível com o tipo pgvector.

### Requirement: Isolamento e Pré-Filtragem de Bases de Conhecimento

O subsistema SHALL garantir a consulta isolada a 5 domínios de conhecimento autônomos (Pausas, Direitos e Benefícios, Procedimentos/Treinamentos, Saúde Ocupacional e EPI) aplicando pré-filtragem estrita por identificador de base e status de vigência no banco de dados.

#### Scenario: Recuperação restrita à base e vigência

- **WHEN** um especialista do agente solicita recuperação para uma base de conhecimento específica
- **THEN** o sistema filtra estritamente os trechos onde `d.id_base = :id_base` e `d.vigente = true`, excluindo documentos obsoletos ou de outras bases.

### Requirement: Limpeza e Sanitização Textual Multiestágio

O subsistema SHALL sanitizar o conteúdo textual de documentos em 4 estágios antes do chunking: remoção de paginação e rodapés repetitivos, des-hifenização de quebras de linha de termos técnicos, normalização de caracteres de controle e preservação de marcadores de seção jurídica.

#### Scenario: Sanitização de documento PDF

- **WHEN** um documento com cabeçalhos de página, quebras com hífen (ex: "confor-\nme") e caracteres nulos é processado
- **THEN** o texto resultante é unificado semanticamente sem perder marcadores sintáticos como "Art.", "§" e "Item".

### Requirement: Chunking Híbrido com Injeção de Metadados

O subsistema SHALL suportar chunking hierárquico por artigos/itens para textos normativos e chunking estrutural baseado em layout visual (PyMuPDF) para manuais e POPs, injetando no cabeçalho de cada chunk os metadados de contexto (`Documento > Capítulo > Seção/Artigo > Página`).

#### Scenario: Fatiamento de texto normativo

- **WHEN** uma norma legal é submetida ao particionamento
- **THEN** cada chunk preserva a unidade jurídica completa sem fragmentar regras de suas exceções e contém o caminho hierárquico injetado.

### Requirement: Busca Híbrida com Reciprocal Rank Fusion

O subsistema SHALL combinar os resultados de busca vetorial densa (cosseno sobre pgvector) e busca léxica exata (`tsvector` em português) utilizando o algoritmo Reciprocal Rank Fusion (RRF com k=60), retornando os 4 melhores trechos.

#### Scenario: Consulta combinando termos conceituais e códigos exatos

- **WHEN** uma consulta com termos conceituais e códigos alfanuméricos (ex: "CA 12345" ou "Art. 473") é executada
- **THEN** o sistema combina os ranqueamentos denso e léxico pelo cálculo RRF e retorna os 4 trechos com maior pontuação combinada.

### Requirement: Contratos Tipados e Formatação de Prompt

O subsistema SHALL trafegar objetos estritamente tipados com modelos Pydantic (`MetadadosTrecho`, `TrechoRecuperado`, `ContextoRAG`) e fornecer formatação padronizada com citação de título, versão, seção e página para inclusão no prompt dos agentes.

#### Scenario: Formatação de contexto para prompt do agente

- **WHEN** o método `formatar_para_prompt` é invocado no contexto recuperado
- **THEN** os trechos são organizados sequencialmente com blocos de fonte explicitando título, versão, seção e número de página.

### Requirement: Ingestão Idempotente e Versionamento de Documentos

O subsistema SHALL fornecer interface de linha de comando para carga de documentos (suportando fixtures em `src/sinergia/rag/data/mocks/` e documentos operacionais em `src/sinergia/rag/data/`) que invalida versões anteriores (`vigente = false`) do mesmo título e base em transação atômica, inserindo os novos trechos com `vigente = true`.

#### Scenario: Atualização de norma para nova versão

- **WHEN** um documento PDF localizado em `src/sinergia/rag/data/mocks/` ou `src/sinergia/rag/data/` com mesmo título e base é re-ingerido com nova versão
- **THEN** os trechos da versão anterior são marcados como não vigentes e a nova versão torna-se a única ativa para recuperação.

### Requirement: Suíte de Testes Automatizados TDD (Unitários e E2E)
O subsistema SHALL conter uma suíte de testes automatizados dentro de `src/sinergia/rag/tests/` cobrindo testes unitários de cada estágio do pipeline e testes de ponta a ponta (E2E) com os documentos em `data/mocks/`.
#### Scenario: Execução dos testes unitários do RAG
- **WHEN** os testes unitários em `src/sinergia/rag/tests/unit/` são executados com pytest
- **THEN** validam isoladamente os contratos Pydantic, o pipeline de limpeza em 4 estágios, o chunking hierárquico e estrutural e a fusão RRF.

#### Scenario: Execução dos testes ponta a ponta (E2E)
- **WHEN** os testes E2E em `src/sinergia/rag/tests/e2e/` são executados com pytest
- **THEN** executam o fluxo completo de ingestão dos PDFs em `src/sinergia/rag/data/mocks/`, verificação de transação no banco e recuperação das respostas com citações válidas para os agentes.
