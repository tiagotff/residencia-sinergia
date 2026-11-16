# Dados e Documentos do Subsistema RAG

Este diretório contém os artefatos de dados e documentos de suporte para o subsistema RAG da solução SinergIA.

## Estrutura

- `mocks/`: Contém documentos PDF sintéticos gerados para validação de layout, testes unitários, testes de sanitização (`limpeza.py`), chunking híbrido (`chunking.py`) e busca (`retrievers.py`).
  - **Atenção**: Os arquivos neste subdiretório destinam-se exclusivamente a testes e desenvolvimento, não representando documentos oficiais homologados de produção.

### Catálogo de Mocks (`data/mocks/`)

1. **`base1_nr17_ergonomia.pdf`** (3 páginas) — Base 1 (F1 Pausas): Layout normativo (Capítulos, Artigos, Parágrafos com quebras de linha hifenizadas propositais).
2. **`base2_clt_direitos_beneficios.pdf`** (3 páginas) — Base 2 (F2 Direitos): Layout jurídico CLT com artigos e incisos em algarismos romanos (`I` ao `XII`).
3. **`base3_pop_procedimentos_treinamentos.pdf`** (3 páginas) — Base 3 (F3 Procedimentos): Layout corporativo com tabelas de revisão, matriz de capacitação e passos operacionais.
4. **`base4_pcmso_saude_ocupacional.pdf`** (3 páginas) — Base 4 (F4 Saúde Ocupacional): Layout densamente tabular de cronograma de exames clínicos, complementares e campanhas de vacinação.
5. **`base5_epi_fichas_ca.pdf`** (3 páginas) — Base 5 (F5/F6 EPIs): Ficha técnica com números de Certificado de Aprovação (ex: `CA 12345`), vida útil, higienização e fluxo de substituição.

