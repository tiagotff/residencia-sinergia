# Documentação

| Arquivo | Conteúdo |
| --- | --- |
| `Sinergia_Arquitetura_e_Especificacao.pdf` | Especificação completa: agentes, fluxos, arquitetura técnica, banco de dados, logs e segurança |
| `fundamentacao_cientifica_e_bases_sinergia.md` | Fundamentação científica (artigos 2024–2026), catálogo de bases de dados e novas features do Agente A2 |
| `diagramas/` | Todos os diagramas em PNG, com a fonte Mermaid (`.mmd`) editável ao lado |
| `../db/schema/001_schema_inicial.sql` | Modelo físico do banco (PostgreSQL 16 + pgvector) |

Para editar um diagrama, altere o `.mmd` e gere o PNG de novo em [mermaid.live](https://mermaid.live) ou com `mmdc -i arquivo.mmd -o arquivo.png -s 2 -b white`.
