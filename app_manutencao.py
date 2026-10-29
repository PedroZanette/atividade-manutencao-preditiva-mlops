# ==============================================================================
# PROJETO: MANUTENÇÃO PREDITIVA - COMPARATIVO MULTI-MODELOS (MLOps)
# Arquivo: app_manutencao.py
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score
)

# ------------------------------------------------------------------------------
# 1. GERAÇÃO DE DADOS SINTÉTICOS (1 ANO / 8.760 HORAS)
# ------------------------------------------------------------------------------
print("--- [1/6] Gerando base de dados sintética de 1 ano (8.760 horas) ---")
np.random.seed(42)
datas = pd.date_range(start="2025-01-01 00:00:00", end="2025-12-31 23:00:00", freq="h")
n_amostras = len(datas)

# Variáveis de sensores da máquina
temperatura = np.random.normal(loc=65.0, scale=5.0, size=n_amostras)
vibracao = np.random.normal(loc=2.5, scale=0.4, size=n_amostras)
pressao = np.random.normal(loc=101.3, scale=2.0, size=n_amostras)

df_raw = pd.DataFrame({
    "timestamp": datas,
    "temperatura": temperatura,
    "vibracao": vibracao,
    "pressao": pressao
})

# Injeção de regras operacionais de falha e ruído aleatório
condicao_falha = (df_raw["temperatura"] > 75) & (df_raw["vibracao"] > 3.2)
ruido = np.random.choice([0, 1], size=n_amostras, p=[0.98, 0.02])
df_raw["falha"] = (condicao_falha | (ruido == 1)).astype(int)

df_raw.to_csv("telemetria_maquina_2025.csv", index=False)
print("-> Base bruta salva em 'telemetria_maquina_2025.csv'")


# ------------------------------------------------------------------------------
# 2. ENGENHARIA DE FEATURES (JANELAS MÓVEIS)
# ------------------------------------------------------------------------------
print("\n--- [2/6] Executando Engenharia de Features ---")
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

for col in ["temperatura", "vibracao", "pressao"]:
    df_raw[f"{col}_media_6h"] = df_raw[col].rolling(window=6).mean()
    df_raw[f"{col}_std_24h"] = df_raw[col].rolling(window=24).std()

df_raw["diff_temp"] = df_raw["temperatura"].diff()
df_preprocessed = df_raw.dropna().reset_index(drop=True)


# ------------------------------------------------------------------------------
# 3. DIVISÃO TEMPORAL (TREINO VS. TESTE)
# ------------------------------------------------------------------------------
print("\n--- [3/6] Divisão Temporal dos Dados ---")
data_corte = pd.to_datetime("2025-10-01 00:00:00")

df_treino = df_preprocessed[df_preprocessed["timestamp"] < data_corte].copy()
df_teste = df_preprocessed[df_preprocessed["timestamp"] >= data_corte].copy()

df_teste.to_csv("dados_teste_ultimos_3_meses.csv", index=False)
print("-> Base de teste (3 últimos meses) salva em 'dados_teste_ultimos_3_meses.csv'")

features = [col for col in df_preprocessed.columns if col not in ["timestamp", "falha"]]
target = "falha"

X_treino, y_treino = df_treino[features].copy(), df_treino[target].copy()
X_teste, y_teste = df_teste[features].copy(), df_teste[target].copy()

# Normalização de dados (Escalonamento exigido para a Regressão Logística)
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Razão de desbalanço de classe para algoritmos de Boosting
scale_pos_weight = (len(y_treino) - sum(y_treino)) / sum(y_treino)


# ------------------------------------------------------------------------------
# 4. TREINAMENTO DOS MODELOS
# ------------------------------------------------------------------------------
print("\n--- [4/6] Treinando os 5 Modelos de Aprendizado de Máquina ---")

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=4, scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss"
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100, depth=4, scale_pos_weight=scale_pos_weight, random_seed=42,
        verbose=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100, max_depth=4, scale_pos_weight=scale_pos_weight,
        random_state=42, verbosity=-1
    ),
    "Regressão Logística": LogisticRegression(
        class_weight="balanced", random_state=42
    )
}

predicoes = {}

for nome, model in modelos.items():
    if nome == "Regressão Logística":
        model.fit(X_treino_scaled, y_treino)
        predicoes[nome] = model.predict(X_teste_scaled)
    else:
        model.fit(X_treino, y_treino)
        predicoes[nome] = model.predict(X_teste)

    print(f"-> Modelo: {nome} treinado com sucesso!")


# ------------------------------------------------------------------------------
# 5. TABELA COMPARATIVA DE DESEMPENHO
# ------------------------------------------------------------------------------
print("\n--- [5/6] Tabela Comparativa de Métricas (Avaliação nos 3 Últimos Meses) ---")

resumo = []
for nome, y_pred in predicoes.items():
    cm = confusion_matrix(y_teste, y_pred)
    tn, fp, fn, tp = cm.ravel()

    rec = recall_score(y_teste, y_pred)
    prec = precision_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred)
    acc = accuracy_score(y_teste, y_pred)

    resumo.append({
        "Modelo": nome,
        "Recall (Sensibilidade)": f"{rec*100:.2f}%",
        "Precisão": f"{prec*100:.2f}%",
        "F1-Score": f"{f1*100:.2f}%",
        "Acurácia": f"{acc*100:.2f}%",
        "Alarmes Falsos (FP)": fp,
        "FALHAS PERDIDAS (FN)": fn,
        "Falhas Detectadas (TP)": tp
    })

df_resumo = pd.DataFrame(resumo)
print("\n", df_resumo.to_string(index=False))


# ------------------------------------------------------------------------------
# 6. VISUALIZAÇÃO GRÁFICA COMPARATIVA (GRID 2x3 DE MATRIZES)
# ------------------------------------------------------------------------------
print("\n--- [6/6] Exibindo e Salvando Painel Comparativo de Matrizes de Confusão ---")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

for idx, (nome, y_pred) in enumerate(predicoes.items()):
    cm = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Falha (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=axes[idx])
    axes[idx].set_title(f"{nome}", fontsize=12, fontweight="bold")

# Ocultar o 6º submódulo do grid que ficou vazio
fig.delaxes(axes[5])

plt.suptitle("Comparativo de Matrizes de Confusão nos Últimos 3 Meses", fontsize=15,
             fontweight="bold", y=0.98)
plt.tight_layout()

plt.savefig("matriz_comparativa_5_modelos.png", bbox_inches="tight")
print("-> Imagem 'matriz_comparativa_5_modelos.png' salva com sucesso!")
plt.show()
