# Relatório de Extraction Accuracy — Corpus Golden Sintético

## Resumo executivo

- Corpus: `60` amostras sintéticas e 100% anonimizado.
- Execuções avaliadas: `120` (`60 × 2 formatos`).
- Seed determinística: `20261118`.
- Modelo de extração: `MockLLM` offline.
- **Extraction Accuracy (registro completo)**: **55.83%** (67/120).
- Legibilidade correta: **100.00%** (24/24) nos casos ilegíveis.
- OCR com legibilidade `ok`: ver detalhamento por formato.

## Extraction Accuracy por campo

| Campo | Acertos | Total | Taxa |
| :--- | ---: | ---: | ---: |
| `legibilidade` | 120 | 120 | 100.00% |
| `nome` | 115 | 120 | 95.83% |
| `inicio` | 119 | 120 | 99.17% |
| `fim` | 119 | 120 | 99.17% |
| `data_emissao` | 119 | 120 | 99.17% |
| `conselho` | 119 | 120 | 99.17% |
| `numero` | 119 | 120 | 99.17% |
| `uf` | 119 | 120 | 99.17% |
| `cid` | 118 | 120 | 98.33% |
| `municipio` | 70 | 120 | 58.33% |

## Métricas por formato

| Formato | Execuções | Registros exatos | Extraction Accuracy | OCR legível |
| :--- | ---: | ---: | ---: | ---: |
| PDF | 60 | 34 | 56.67% | 60/60 |
| PNG | 60 | 33 | 55.00% | 60/60 |

## Métricas por qualidade da foto

| Qualidade | Execuções | Registros exatos | Extraction Accuracy |
| :--- | ---: | ---: | ---: |
| baixa_luz | 30 | 20 | 66.67% |
| baixa_resolucao | 10 | 2 | 20.00% |
| boa | 20 | 12 | 60.00% |
| borrada | 2 | 2 | 100.00% |
| cortada | 2 | 2 | 100.00% |
| manchado | 10 | 10 | 100.00% |
| ruido | 22 | 9 | 40.91% |
| tremida | 24 | 10 | 41.67% |

## Métricas por layout

| Layout | Execuções | Registros exatos | Extraction Accuracy |
| :--- | ---: | ---: | ---: |
| layout_1 | 30 | 18 | 60.00% |
| layout_2 | 30 | 17 | 56.67% |
| layout_3 | 30 | 16 | 53.33% |
| layout_4 | 30 | 16 | 53.33% |

## Metodologia

- Extraction Accuracy considera o registro completo como correto somente quando todos os 10 campos avaliados coincidem exatamente com o gabarito.
- Os campos avaliados são: `legibilidade`, `nome`, `inicio`, `fim`, `data_emissao`, `conselho`, `numero`, `uf`, `cid` e `municipio`.
- Também são informadas taxas individuais por campo, pois elas identificam falhas localizadas.
- Os PDFs são gerados a partir dos próprios PNG sintéticos.
- O corpus utiliza nomes, clínicas, profissionais, matrículas e CIDs fictícios; não contém documentos reais ou dados pessoais.

## Artefatos

- [Manifesto golden](../.storage/golden-atestados/gabarito_atestados_golden.json)
- [Resultado completo em JSON](../.storage/golden-atestados/relatorio_extraction_accuracy_golden.json)
