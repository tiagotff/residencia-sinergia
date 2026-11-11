# Guia Completo de Estudo: Rede Neural de Recomendação de Benefícios (Agente A2 — SinergIA)

Este documento foi preparado para você dominar **100% da teoria, da prática e do código** da Rede Neural do projeto **SinergIA**, tanto para explicar hoje ao seu grupo quanto para responder às perguntas da banca no **Entregável das Semanas 3 e 4 (13/11/2026)**.

---

## 1. Visão Geral: Onde a Rede Neural entra no SinergIA e por quê?

No ecossistema do **SinergIA**, o **Agente A2 (Bem-Estar)** é o núcleo da solução. Na funcionalidade **F2 (Direitos e Benefícios)**, temos dois componentes de IA trabalhando em conjunto (os dois *Assets* obrigatórios da entrega de 13/11):

1. **Rede Neural (`teste_rede_neural_simples.py`):** Recebe os dados administrativos do colaborador (`idade`, `tempo_empresa`, `dependentes`, `carga_trabalho`) e calcula a probabilidade de adesão para cada benefício, gerando o **Ranking Top 3**.
2. **RAG com Embeddings (frente de RAG):** Recebe os 3 benefícios escolhidos pela Rede Neural, busca nos documentos vigentes da empresa (`pgvector`) as regras e orientações de cada um e monta a explicação final com citação da fonte.

**Pergunta clássica da banca (Material de Apoio):** *"Quando utilizar RAG em vez de treinamento de modelo (Rede Neural), e como os dois se completam?"*
* **Rede Neural (Treinamento de Modelo):** Usada quando queremos **aprender padrões matemáticos e comportamentais** em dados históricos tabulares (ex: descobrir quais benefícios um perfil de trabalhador tem maior probabilidade de aderir: `84.6%`). Ela devolve **números/probabilidades**, não textos normativos.
* **RAG (Recuperação Aumentada por Geração):** Usada quando precisamos consultar **regras, leis (CLT/NRs) e políticas vigentes da empresa** que mudam com o tempo e exigem **citação exata da fonte** sem alucinar.

---

## 2. De Onde Vêm os Dados Reais? (Base Brasileira da UCI no NotebookLM)

Em vez de treinar a rede com sorteios aleatórios (onde a rede tenta adivinhar cara-ou-coroa e não aprende nada), utilizamos a base pública brasileira mapeada no **NotebookLM ("Arquitetura de Dados para Bem-Estar e Saúde Ocupacional")**:
* **Dataset:** `data/uci/Absenteeism_at_work.csv` (*UCI Machine Learning Repository — ID 445*, doada pelos pesquisadores brasileiros Martiniano & Ferreira e utilizada no artigo-base de *Llamas Blázquez, 2025*).
* **Volume:** **740 registros reais** de trabalhadores de uma empresa brasileira, sem nenhum valor faltante.

### Como transformamos o CSV da UCI em Entrada (X) e Saída (Y) respeitando a LGPD?

Na `Documentação - SinergIA v2.1.docx` (Seção 5.1 e Seção 12), estabelecemos uma regra ética fundamental (**LGPD, Art. 11** e princípio *"Cuidado, não vigilância"*):
* **O que NUNCA entra como variável de entrada (X) na rede:** `CID` (diagnóstico médico), `sexo`, `raça/cor` ou `marcações de ponto`.
* **O que entra em X (4 colunas administrativas não-sensíveis da UCI):**
  1. `idade` (`Age` — coluna 8 do CSV: de 27 a 58 anos, média 36,4 anos)
  2. `tempo_empresa` (`Service time` — coluna 7 do CSV: de 1 a 29 anos, média 12,5 anos)
  3. `dependentes` (`Son` — coluna 13 do CSV: de 0 a 4 filhos)
  4. `carga_trabalho` (`Work load Average/day` — coluna 9 do CSV: índice de 205 a 378, calibrado no artigo científico pelo instrumento ergonômico **NASA-TLX**)

* **O que é a Saída Y (Os 5 Benefícios Recomendados):**
  Inspecionando a coluna `Reason for absence` (coluna 1 do CSV da UCI), descobrimos que os motivos reais pelos quais os funcionários brasileiros mais se ausentaram foram exatamente demandas que esses 5 benefícios atendem:
  * **Benefício 0 (`Gympass / Atividade Física`):** Perfis jovens (<= 37 anos) e manutenção preventiva de condicionamento/IMC (<= 28).
  * **Benefício 1 (`Apoio Psicológico Online`):** Colaboradores submetidos a carga diária elevada (>= 265) com histórico de desgaste mental/nervoso/estresse (códigos UCI `5`, `18`, `26`) — também pedido explicitamente na **Pesquisa Forms do grupo (42 respostas)**.
  * **Benefício 2 (`Ergonomia e Fisioterapia`):** Histórico real de **Fisioterapia (código 27 — 69 casos)** ou **Doenças Osteomusculares/Coluna (CID XIII, código 13 — 55 casos)** combinados com tempo de empresa (>= 10 anos) ou idade (>= 40 anos).
  * **Benefício 3 (`Plano Odontológico`):** Histórico real de **Consultas Odontológicas (código 28 — 112 casos)** em colaboradores com dependentes (>= 1) ou maior tempo de casa (>= 12 anos).
  * **Benefício 4 (`Telemedicina Familiar`):** Colaboradores com filhos/dependentes (>= 1) e demanda recorrente de **Consultas Médicas e Acompanhamento (códigos 23 e 22 — 187 casos)**.

---

## 3. Explicação Linha por Linha do Código (`teste_rede_neural_simples.py`)

### Bloco 1 e 2: Leitura, Separação Treino/Teste (Sem Data Leakage) e Normalização
```python
# Divisão Temporal 80/20 ANTES da normalização (evita Data Leakage!): 592 treino / 148 teste
corte = int(0.8 * len(X_bruto))
X_treino_bruto, X_teste_bruto = X_bruto[:corte], X_bruto[corte:]
Y_treino, Y_teste = Y[:corte], Y[corte:]

# Normalização (Z-score) calculada APENAS com a média e desvio do treino
media = X_treino_bruto.mean(dim=0)
desvio = X_treino_bruto.std(dim=0)
X_treino = (X_treino_bruto - media) / desvio
X_teste = (X_teste_bruto - media) / desvio
```
* **Por que dividir em 80% Treino (592 linhas) e 20% Teste (148 linhas)?**
  Se avaliarmos a rede nos mesmos dados em que ela estudou, não saberemos se ela **aprendeu a regra geral** ou se apenas **decorou as respostas** (*Overfitting*). Os 148 casos de teste ficam "escondidos" da rede durante o treinamento e servem como uma prova final.
* **O que é Data Leakage (Vazamento de Dados) e como evitamos aqui?**
  Repare que primeiro cortamos `X_treino_bruto` e `X_teste_bruto`, e calculamos `media` e `desvio` **somente em `X_treino_bruto`**. Se calculássemos a média usando os 740 registros inteiros antes de dividir, informações estatísticas do futuro (os 20% de teste) vazariam para o treinamento!
* **Por que Normalizar (`(X - media) / desvio`)?**
  No CSV bruto, `dependentes` vale de `0 a 4`, enquanto `carga_trabalho` vale de `205 a 378`. Sem normalizar, os números grandes (`378`) explodiriam as multiplicações dos neurônios e dominariam o aprendizado. O *Z-score* deixa todas as 4 variáveis na mesma escala (média `0` e desvio padrão `1`).

---

### Bloco 3: A Arquitetura da Rede Neural (Respondendo às Guiding Questions de 09/11)
```python
modelo = nn.Sequential(
    nn.Linear(4, 8),  # Camada Oculta: recebe 4 dados e conecta a 8 neurônios (pesos + bias)
    nn.ReLU(),        # Ativação: filtra sinais negativos (zera o que for < 0)
    nn.Linear(8, 5),  # Camada de Saída: 8 neurônios decidem sobre os 5 benefícios
    nn.Sigmoid(),     # Converte o resultado final em probabilidade (0% a 100%)
)
```

1. **O que é um Neurônio Artificial, Pesos (Weights) e Bias?**
   Dentro do `nn.Linear(4, 8)`, existem **8 neurônios artificiais**. Cada neurônio recebe as 4 características do colaborador ($x_1, x_2, x_3, x_4$) e faz uma soma ponderada:
   $$z = (w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + w_4 \cdot x_4) + b$$
   * **Pesos ($w_i$):** Representam a **importância** que o neurônio dá para cada variável.
   * **Bias ($b$):** É o **limiar base** (propensão natural de ativação do neurônio quando as entradas são zero).
2. **O que são as Camadas?**
   * **Camada de Entrada (4 valores):** Os dados do colaborador (`idade`, `tempo_empresa`, `dependentes`, `carga_trabalho`).
   * **Camada Oculta (`nn.Linear(4, 8)`):** Os 8 neurônios intermediários que cruzam as informações entre si.
   * **Camada de Saída (`nn.Linear(8, 5)`):** Os 5 neurônios finais — um para cada benefício do catálogo!
3. **O que é e para que serve uma Função de Ativação (`ReLU` e `Sigmoid`)?**
   * **`nn.ReLU()` na camada oculta:** Calcula $\max(0, z)$. Zera valores negativos e permite que a rede aprenda **combinações não-lineares** (ex: idade alta E tempo de empresa alto).
   * **`nn.Sigmoid()` na camada de saída:** Calcula $\frac{1}{1 + e^{-z}}$, convertendo o resultado de cada um dos 5 neurônios finais em uma **probabilidade independente entre 0% e 100%**.

---

### Bloco 4: Como a Rede Aprende? (Forward Pass, Loss Function, Backpropagation e Otimizador)
1. **Forward Pass (`previsao_treino = modelo(X_treino)`):** Os dados atravessam a rede da entrada até a saída gerando as probabilidades previstas.
2. **Loss Function (`erro_treino = criterio(previsao_treino, Y_treino)`):** A função `nn.BCELoss()` (Entropia Cruzada Binária) mede a distância entre as probabilidades que a rede chutou e a realidade (`0` ou `1`).
3. **Backpropagation (`erro_treino.backward()`):** Calcula as derivadas de trás para frente para descobrir a parcela de culpa de cada peso no erro.
4. **Otimizador (`otimizador.step()` com `Adam`):** Ajusta todos os pesos e bias na direção que reduz o erro, usando a taxa de aprendizado `lr=0.03`.

---

## 4. Interpretando os Resultados Experimentais (Rubrica do Slide 3 — 13/11)

```text
--- TREINAMENTO COM BASE REAL UCI (592 treino | 148 teste) ---
Época 01/60 | Loss Treino: 0.7044 | Loss Teste: 0.6853 | Acurácia Teste: 54.5%
Época 15/60 | Loss Treino: 0.4531 | Loss Teste: 0.4895 | Acurácia Teste: 80.0%
Época 30/60 | Loss Treino: 0.3531 | Loss Teste: 0.3828 | Acurácia Teste: 83.9%
Época 45/60 | Loss Treino: 0.2964 | Loss Teste: 0.3418 | Acurácia Teste: 85.4%
Época 60/60 | Loss Treino: 0.2616 | Loss Teste: 0.3224 | Acurácia Teste: 87.4%

[Métrica de Sucesso] Precisão@3 Baseline: 50.2% -> Rede Neural: 66.0%
```

* **Underfitting (Subajuste):** Visível na **Época 01** (Loss ~0.70 e Acurácia de 54.5%, próxima de um chute aleatório).
* **Controle de Overfitting (Sobreajuste):** Como a `Loss Teste` cai junto com a `Loss Treino` até `0.3224` e a `Acurácia Teste` chega a **87.4%**, comprovamos que a rede generalizou para dados inéditos sem decorar o treino.
* **Superação do Baseline (Meta da Documentação v2.1):** Na métrica **Precisão@3**, o Baseline fixo (recomendar os 3 benefícios mais populares sem IA) acerta **50.2%**, enquanto a nossa Rede Neural atinge **66.0%** (+15,8 pontos percentuais de ganho).

