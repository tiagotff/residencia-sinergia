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

# 2. DADOS SIMULADOS (100 colaboradores)
# Entrada (X): 4 características [idade, tempo_empresa, dependentes, engajamento]
# Saída (Y): 0 (não aderiu) ou 1 (aderiu) para cada um dos 5 benefícios
torch.manual_seed(42)
X = torch.randn(100, 4)
Y = torch.randint(0, 2, (100, 5)).float()

# 3. REDE NEURAL SIMPLES (MLP)
modelo = nn.Sequential(
    nn.Linear(4, 8),  # Camada Oculta: recebe 4 dados e conecta a 8 neurônios
    nn.ReLU(),        # Ativação: filtra sinais negativos (zera o que for < 0)
    nn.Linear(8, 5),  # Camada de Saída: 8 neurônios decidem sobre os 5 benefícios
    nn.Sigmoid(),     # Converte o resultado final em probabilidade (0% a 100%)
)

# 4. TREINAMENTO
criterio = nn.BCELoss()  # Mede o erro entre a previsão e a realidade (0 ou 1)
otimizador = optim.Adam(modelo.parameters(), lr=0.05)  # Ajusta os pesos da rede

print("--- TREINAMENTO ---")
for epoca in range(1, 21):
    otimizador.zero_grad()        # Limpa os cálculos da rodada anterior
    previsao = modelo(X)          # Rede tenta adivinhar os benefícios
    erro = criterio(previsao, Y)  # Calcula o tamanho do erro
    erro.backward()               # Descobre onde errou (Backpropagation)
    otimizador.step()             # Corrige os pesos para melhorar

    if epoca % 5 == 0:
        print(f"Época {epoca:02d}/20 | Erro (Loss): {erro.item():.4f}")

# 5. RECOMENDAÇÃO TOP 3 (USO PELO AGENTE 2)
print("\n--- TOP 3 BENEFÍCIOS PARA O COLABORADOR ---")
with torch.no_grad():
    # Dados de 1 colaborador: [idade, tempo_empresa, dependentes, engajamento]
    colaborador = torch.tensor([0.5, 1.2, 1.0, 0.8])
    probabilidades = modelo(colaborador)

    # Seleciona os 3 benefícios com maior probabilidade de adesão
    top_valores, top_indices = torch.topk(probabilidades, k=3)

    for pos, (idx, prob) in enumerate(zip(top_indices, top_valores), start=1):
        print(f"{pos}º Lugar: {BENEFICIOS[idx]} ({prob.item() * 100:.1f}%)")
