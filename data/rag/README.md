# Dados e Documentos do Subsistema RAG

Este diretório contém os documentos PDF das Bases de Conhecimento do subsistema RAG da solução SinergIA.

## Estrutura e Padrão de Layout (`mocks/`)

Os documentos em `mocks/` seguem o layout oficial de **Bases de Conhecimento (KB) SinergIA**:

1. **Cabeçalho e Tabela de Governança (Página 1)**:
   - Contém `Título`, `Código`, `Versão`, `Vigência`, `Tipo de fonte`, `Fonte`, `Base RAG`, `Responsável`, `Aprovação`, `Revisão` e `Convenções`.
   - Extraído estruturalmente por `extrair_metadados_cabecalho_pdf()` sem poluir o índice vetorial.
2. **Seções Temáticas Numeradas (`1. ...`, `2. ...`)**:
   - Definem a hierarquia e o `breadcrumb` dos trechos.
3. **Unidades Q&A (`ID-Qnnn | Pergunta`)**:
   - Linha de título com ID canônico (`RH-ERG-Qnnn`, `RH-BEN-Qnnn`, `SO-Qnnn`, `EPI-Qnnn`),
   - Linha `"Também perguntam:"` com variações separadas por `|`,
   - Corpo da resposta (parágrafos, bullets `•` e passos numerados),
   - Linha `"Fonte:"` distinguindo normas `(OFICIAL)` e regras internas `(INTERNO)`.

### Catálogo Atual (`data/rag/mocks/`)

1. **`rag_ergonomia.pdf`** (2 páginas) — Base 1 (F1 Pausas e Ergonomia): `KB-RH-ERG-001` v2.0 (`RH-ERG-Q001` a `RH-ERG-Q008`).
2. **`rag_trabalho.pdf`** (3 páginas) — Base 2 (F2 Benefícios e Direitos Trabalhistas): `KB-RH-BEN-002` v1.0 (`RH-BEN-Q001` a `RH-BEN-Q011`).
3. **`rag_pcmso_saude_ocupacional.pdf`** (11 páginas) — Base 4 (F4 Saúde Ocupacional / PCMSO): v1.0 (`SO-Q001` a `SO-Q055`).
4. **`rag_epi.pdf`** (12 páginas) — Base 5 (F5/F6 EPIs e Fichas de CA): v1.2 (`EPI-Q001` a `EPI-Q058`).
