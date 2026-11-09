import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RecomendadorBeneficioSimples(nn.Module):

  def __init__(
      self,
      num_cargos=20,
      num_setores=10,
      num_beneficios=8,
      dim_features_num=6,
  ):
    super().__init__()

    # Embeddings para variáveis categóricas
    self.emb_cargo = nn.Embedding(
        num_embeddings=num_cargos, embedding_dim=4
    )  # Vetor dim 4
    self.emb_setor = nn.Embedding(
        num_embeddings=num_setores, embedding_dim=4
    )  # Vetor dim 4
    self.emb_beneficio = nn.Embedding(
        num_embeddings=num_beneficios, embedding_dim=4
    )  # Vetor dim 4

    # Dimensão total de entrada: 4 (cargo) + 4 (setor) + 4 (benefício) + 6 (numéricas) = 18
    dim_entrada = 4 + 4 + 4 + dim_features_num

    # Camadas Ocultas (MLP)
    self.rede = nn.Sequential(
        nn.Linear(dim_entrada, 32),
        nn.ReLU(),
        nn.Dropout(p=0.2),  # Regularização para evitar overfitting
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),  # Saída: Probabilidade de adesão entre 0 e 1
    )

  def forward(self, cargo_id, setor_id, beneficio_id, features_num):
    # Projeção das categorias em vetores densos
    c_emb = self.emb_cargo(cargo_id)
    s_emb = self.emb_setor(setor_id)
    b_emb = self.emb_beneficio(beneficio_id)

    # Concatenação de todas as características
    x = torch.cat([c_emb, s_emb, b_emb, features_num], dim=1)
    return self.rede(x)


# ==============================================================================
# 2. DADOS DE TREINAMENTO SIMULADOS (vw_features_beneficio)
# ==============================================================================
torch.manual_seed(12)
N_AMOSTRAS = 500

# Variáveis Categóricas
cargos = torch.randint(0, 20, (N_AMOSTRAS,))
setores = torch.randint(0, 10, (N_AMOSTRAS,))
beneficios = torch.randint(0, 8, (N_AMOSTRAS,))

# Variáveis Numéricas (normalizadas):
# [tempo_empresa_meses, idade, dependentes, jornada_diaria, outros_beneficios_ativos, taxa_aceite_anterior]
features_num = torch.randn(N_AMOSTRAS, 6)

# Target Binário: Colaborador aderiu (1) ou não (0) ao benefício
targets = torch.randint(0, 2, (N_AMOSTRAS, 1)).float()

# ==============================================================================
# 3. TREINAMENTO DO MODELO (BCE Loss + Otimizador Adam)
# ==============================================================================
modelo = RecomendadorBeneficioSimples()
criterio = nn.BCELoss()  # Binary Cross-Entropy Loss
otimizador = optim.Adam(modelo.parameters(), lr=0.01)

print("Iniciando treinamento da Rede Neural...")
modelo.train()
for epoca in range(1, 11):
  otimizador.zero_grad()

  # Forward pass
  predicoes = modelo(cargos, setores, beneficios, features_num)
  perda = criterio(predicoes, targets)

  # Backpropagation + Gradiente Descendente
  perda.backward()
  otimizador.step()

  if epoca % 2 == 0 or epoca == 1:
    print(f"Época [{epoca:02d}/10] - Perda (Loss): {perda.item():.4f}")

# ==============================================================================
# 4. INFERÊNCIA OPERACIONAL: GERANDO O TOP 3 PARA O AGENTE A2
# ==============================================================================
print("\n" + "=" * 50)
print("INFERÊNCIA: RANKING DE BENEFÍCIOS (FERRAMENTA AGENTE A2)")
print("=" * 50)

# Catálogo oficial de benefícios da empresa
catalogo_beneficios = {
    0: "Gympass / Atividade Física",
    1: "Apoio Psicológico Online",
    2: "Vale Refeição Flexível",
    3: "Programa de Ergonomia e Fisioterapia",
    4: "Plano Odontológico Familiar",
    5: "Auxílio Creche / Educação Infantil",
    6: "Seguro de Vida com Telemedicina",
    7: "Desconto Farmácia e Medicamentos",
}

modelo.eval()
with torch.no_grad():
  # Perfil do colaborador que abriu o chat:
  # Analista de Operações (Cargo=3, Setor=2, 2 anos de casa, 1 dependente)
  colab_cargo = torch.tensor([3] * len(catalogo_beneficios))
  colab_setor = torch.tensor([2] * len(catalogo_beneficios))
  todos_beneficios = torch.tensor(list(catalogo_beneficios.keys()))

  # Features numéricas padronizadas para este colaborador repetidas para cada item candidato
  colab_num = torch.tensor(
      [[0.5, 0.2, 1.0, 0.0, 0.3, 0.8]] * len(catalogo_beneficios)
  )

  # A rede calcula a probabilidade de adesão para todos os benefícios
  probabilidades = (
      modelo(colab_cargo, colab_setor, todos_beneficios, colab_num)
      .squeeze()
      .numpy()
  )

  # Cria o ranking ordenando da maior probabilidade para a menor
  ranking = []
  for b_id, prob in zip(catalogo_beneficios.keys(), probabilidades):
    ranking.append((catalogo_beneficios[b_id], float(prob)))

  ranking.sort(key=lambda x: x[1], reverse=True)

  print("Top 3 Benefícios Recomendados para envio ao Especialista A2:")
  for posicao, (nome, probabilidade) in enumerate(ranking[:3], start=1):
    print(f"  {posicao}º Lugar: {nome} -> Score: {probabilidade*100:.1f}%")