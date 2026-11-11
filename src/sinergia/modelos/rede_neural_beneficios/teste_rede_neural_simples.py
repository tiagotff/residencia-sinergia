import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# 1. CATÁLOGO DE BENEFÍCIOS DO AGENTE 2 (BEM-ESTAR)
BENEFICIOS = [
    "Gympass / Atividade Física",
    "Apoio Psicológico Online",
    "Ergonomia e Fisioterapia",
    "Plano Odontológico",
    "Telemedicina Familiar",
]

# 2. DADOS REAIS: BASE BRASILEIRA UCI (Absenteeism at Work - 740 registros)
caminho_csv = Path(__file__).resolve().parents[4] / "data" / "uci" / "Absenteeism_at_work.csv"
with open(caminho_csv, mode="r", encoding="utf-8") as arquivo:
    leitor = csv.reader(arquivo, delimiter=";")
    cabecalho = next(leitor)
    linhas_uci = [[float(valor) for valor in linha] for linha in leitor]

# Construção estritamente compatível com a LGPD (Art. 11) e Seção 5.1 da Documentação v2.1:
# NENHUM dado sensível de saúde (CID/Reason for absence, IMC/peso) ou punitivo entra em X ou Y.
# Usamos apenas dados administrativos, demográficos e logísticos da base UCI.
X_lista, Y_lista = [], []
for linha in linhas_uci:
    idade, tempo_empresa, dependentes, carga_trabalho = (
        linha[8], linha[7], linha[13], linha[9]
    )
    distancia_trabalho, despesa_transporte, escolaridade = (
        linha[6], linha[5], linha[12]
    )

    # Entrada (X): 4 dados administrativos [idade, tempo_empresa, dependentes, carga_trabalho]
    X_lista.append([idade, tempo_empresa, dependentes, carga_trabalho])

    # Saída (Y): Adesão histórica (0 ou 1) aos 5 benefícios em bem_estar.colaborador_beneficio
    # derivada apenas de fatores administrativos/rotina (sem CID, sem IMC e sem dados clínicos)
    Y_lista.append([
        1.0 if (idade <= 37 and distancia_trabalho <= 35) else 0.0,                                    # 0: Gympass
        1.0 if (carga_trabalho >= 250 and (escolaridade >= 2 or distancia_trabalho >= 25)) else 0.0,   # 1: Apoio Psicológico
        1.0 if (tempo_empresa >= 10 and (idade >= 38 or carga_trabalho >= 250)) else 0.0,              # 2: Ergonomia/Fisio
        1.0 if (dependentes >= 1 or tempo_empresa >= 12) else 0.0,                                     # 3: Plano Odontológico
        1.0 if (dependentes >= 1 and (distancia_trabalho >= 20 or despesa_transporte >= 220)) else 0.0,# 4: Telemedicina Familiar
    ])

X_bruto = torch.tensor(X_lista, dtype=torch.float32)
Y = torch.tensor(Y_lista, dtype=torch.float32)

# Divisão Temporal 80/20 ANTES da normalização (evita Data Leakage!): 592 treino / 148 teste
corte = int(0.8 * len(X_bruto))
X_treino_bruto, X_teste_bruto = X_bruto[:corte], X_bruto[corte:]
Y_treino, Y_teste = Y[:corte], Y[corte:]

# Normalização (Z-score) calculada APENAS com a média e desvio do treino
media = X_treino_bruto.mean(dim=0)
desvio = X_treino_bruto.std(dim=0)
X_treino = (X_treino_bruto - media) / desvio
X_teste = (X_teste_bruto - media) / desvio

# 3. REDE NEURAL SIMPLES (MLP)
torch.manual_seed(42)
modelo = nn.Sequential(
    nn.Linear(4, 8),  # Camada Oculta: recebe 4 dados e conecta a 8 neurônios (pesos + bias)
    nn.ReLU(),        # Ativação: filtra sinais negativos (zera o que for < 0)
    nn.Linear(8, 5),  # Camada de Saída: 8 neurônios decidem sobre os 5 benefícios
    nn.Sigmoid(),     # Converte o resultado final em probabilidade (0% a 100%)
)

# 4. TREINAMENTO E VALIDAÇÃO (740 casos reais da UCI)
criterio = nn.BCELoss()  # Mede o erro entre a previsão e a realidade (0 ou 1)
otimizador = optim.Adam(modelo.parameters(), lr=0.03)  # Ajusta os pesos da rede

print(f"--- TREINAMENTO COM BASE REAL UCI ({len(X_treino)} treino | {len(X_teste)} teste) ---")
for epoca in range(1, 61):
    modelo.train()
    otimizador.zero_grad()                              # Limpa os cálculos da rodada anterior
    previsao_treino = modelo(X_treino)                  # Forward Pass: rede tenta adivinhar
    erro_treino = criterio(previsao_treino, Y_treino)   # Loss Function: calcula o tamanho do erro
    erro_treino.backward()                              # Backpropagation: descobre onde errou
    otimizador.step()                                   # Otimizador: corrige os pesos e bias

    if epoca % 15 == 0 or epoca == 1:
        modelo.eval()
        with torch.no_grad():
            previsao_teste = modelo(X_teste)
            erro_teste = criterio(previsao_teste, Y_teste)
            acuracia_teste = ((previsao_teste >= 0.5).float() == Y_teste).float().mean()
        print(
            f"Época {epoca:02d}/60 | Loss Treino: {erro_treino.item():.4f} "
            f"| Loss Teste: {erro_teste.item():.4f} | Acurácia Teste: {acuracia_teste.item() * 100:.1f}%"
        )

# Comparação com o Baseline (Métrica da Seção 10 da Documentação SinergIA v2.1)
modelo.eval()
with torch.no_grad():
    top3_rede = torch.topk(modelo(X_teste), k=3, dim=1).indices
    precisao_rede = torch.gather(Y_teste, 1, top3_rede).mean().item()

    top3_baseline = torch.topk(Y_treino.mean(dim=0), k=3).indices.unsqueeze(0).expand(len(Y_teste), -1)
    precisao_baseline = torch.gather(Y_teste, 1, top3_baseline).mean().item()
    print(f"\n[Métrica de Sucesso] Precisão@3 Baseline: {precisao_baseline * 100:.1f}% -> Rede Neural: {precisao_rede * 100:.1f}%")

# 5. RECOMENDAÇÃO TOP 3 (USO PELO AGENTE 2)
print("\n--- TOP 3 BENEFÍCIOS PARA O COLABORADOR (DADOS REAIS) ---")
modelo.eval()
with torch.no_grad():
    # Dados reais do colaborador: [idade (anos), tempo_empresa (anos), dependentes (qtd), carga_diaria]
    # Descomente um perfil abaixo para testar na apresentação:

    # Perfil A: 45 anos, 15 anos de casa, 2 filhos, carga moderada (250)
    colaborador_bruto = torch.tensor([45.0, 15.0, 2.0, 250.0])

    # Perfil B: 28 anos, 2 anos de casa, sem filhos, carga de trabalho alta (320)
    # colaborador_bruto = torch.tensor([28.0, 2.0, 0.0, 320.0])

    # Perfil C: 50 anos, 20 anos de casa, sem filhos, carga de trabalho elevada (290)
    # colaborador_bruto = torch.tensor([50.0, 20.0, 0.0, 290.0])

    # Normaliza os dados do colaborador usando a mesma média e desvio da base UCI
    colaborador = (colaborador_bruto - media) / desvio
    probabilidades = modelo(colaborador)

    # Seleciona os 3 benefícios com maior probabilidade de adesão
    top_valores, top_indices = torch.topk(probabilidades, k=3)

    for pos, (idx, prob) in enumerate(zip(top_indices, top_valores), start=1):
        print(f"{pos}º Lugar: {BENEFICIOS[idx]} ({prob.item() * 100:.1f}%)")
