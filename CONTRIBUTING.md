# Como contribuir

## Branches

A `main` é protegida: só recebe alterações por Pull Request aprovado.

| Branch | Responsável | Escopo |
| --- | --- | --- |
| `feat/a0-orquestrador` | Leo | `orquestrador/`, `api/` |
| `feat/a1-recepcao` | Ítalo e Natália | `agentes/a1_recepcao/` |
| `feat/rede-neural-beneficios` | Ítalo e Natália | `modelos/rede_neural_beneficios/`, `notebooks/` |
| `feat/a2-bem-estar` | Luana e Tiago | `agentes/a2_bem_estar/` |
| `feat/rag` | João | `rag/`, `data/rag/` |
| `feat/a3-apoio` | A definir | `agentes/a3_apoio/` |

## Fluxo de trabalho

```bash
git checkout main
git pull
git checkout -b feat/sua-branch      # só na primeira vez
# ... trabalhe e faça commits ...
git push -u origin feat/sua-branch
```

Abra o Pull Request para a `main` no GitHub. O dono da pasta alterada é chamado como revisor automaticamente.

Para trazer as novidades da `main` para a sua branch:

```bash
git checkout main && git pull
git checkout feat/sua-branch
git merge main
```

## Padrão de commits

| Prefixo | Uso |
| --- | --- |
| `feat:` | Nova funcionalidade |
| `fix:` | Correção de erro |
| `docs:` | Documentação |
| `test:` | Testes |
| `refactor:` | Melhoria de código sem mudar comportamento |
| `chore:` | Configuração, dependências, estrutura |

Exemplo: `feat: adiciona lembrete de pausa no A2`

## Banco de dados

Mudanças no schema entram como um novo arquivo numerado em `db/schema/` (`002_...sql`, `003_...sql`). Arquivos já mergeados não são editados.

## Segredos

Nunca suba chaves, senhas ou o arquivo `.env`. Use `.env.example` para listar as variáveis necessárias.
