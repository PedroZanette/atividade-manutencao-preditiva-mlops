# ==============================================================================
# PROJETO: MANUTENÇÃO PREDITIVA - PIPELINE OTIMIZADO (MLOps)
# Arquivo: app_manutencao_otimizado.py
# Atividade Prática: Tarefas 2 (features), 3 (SMOTE) e 4 (custo financeiro)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score
)

# Custos de negócio da TechIndustries
CUSTO_FN = 15_000.00  # parada não planejada (falha perdida)
CUSTO_FP = 500.00     # manutenção preventiva desnecessária (alarme falso)


def formatar_reais(valor):
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------------------
# 1. GERAÇÃO DE DADOS SINTÉTICOS (1 ANO / 8.760 HORAS) - igual ao pipeline base
# ------------------------------------------------------------------------------
print("--- [1/7] Gerando base de dados sintética de 1 ano (8.760 horas) ---")
np.random.seed(42)
datas = pd.date_range(start="2025-01-01 00:00:00", end="2025-12-31 23:00:00", freq="h")
n_amostras = len(datas)

temperatura = np.random.normal(loc=65.0, scale=5.0, size=n_amostras)
vibracao = np.random.normal(loc=2.5, scale=0.4, size=n_amostras)
pressao = np.random.normal(loc=101.3, scale=2.0, size=n_amostras)

df_raw = pd.DataFrame({
    "timestamp": datas,
    "temperatura": temperatura,
    "vibracao": vibracao,
    "pressao": pressao
})

condicao_falha = (df_raw["temperatura"] > 75) & (df_raw["vibracao"] > 3.2)
ruido = np.random.choice([0, 1], size=n_amostras, p=[0.98, 0.02])
df_raw["falha"] = (condicao_falha | (ruido == 1)).astype(int)

df_raw.to_csv("telemetria_maquina_2025.csv", index=False)
print("-> Base bruta salva em 'telemetria_maquina_2025.csv'")


# ------------------------------------------------------------------------------
# 2. ENGENHARIA DE FEATURES (BASE + AVANÇADA - TAREFA 2)
# ------------------------------------------------------------------------------
print("\n--- [2/7] Executando Engenharia de Features (base + avançada) ---")
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

# Features do pipeline base
for col in ["temperatura", "vibracao", "pressao"]:
    df_raw[f"{col}_media_6h"] = df_raw[col].rolling(window=6).mean()
    df_raw[f"{col}_std_24h"] = df_raw[col].rolling(window=24).std()

df_raw["diff_temp"] = df_raw["temperatura"].diff()

features_base = [
    "temperatura", "vibracao", "pressao",
    "temperatura_media_6h", "temperatura_std_24h",
    "vibracao_media_6h", "vibracao_std_24h",
    "pressao_media_6h", "pressao_std_24h",
    "diff_temp"
]

# Tarefa 2.1: taxa de variação (diferença percentual) de vibração e pressão
df_raw["vibracao_var_pct"] = df_raw["vibracao"].pct_change() * 100
df_raw["pressao_var_pct"] = df_raw["pressao"].pct_change() * 100

# Tarefa 2.2: máximo e mínimo em janelas móveis de 12 horas
for col in ["temperatura", "vibracao"]:
    df_raw[f"{col}_max_12h"] = df_raw[col].rolling(window=12).max()
    df_raw[f"{col}_min_12h"] = df_raw[col].rolling(window=12).min()

# Tarefa 2.3: interação entre temperatura e vibração
df_raw["temp_x_vibracao"] = df_raw["temperatura"] * df_raw["vibracao"]

features_novas = [
    "vibracao_var_pct", "pressao_var_pct",
    "temperatura_max_12h", "temperatura_min_12h",
    "vibracao_max_12h", "vibracao_min_12h",
    "temp_x_vibracao"
]
features_otimizadas = features_base + features_novas

df_preprocessed = df_raw.dropna().reset_index(drop=True)
print(f"-> {len(features_base)} features base + {len(features_novas)} novas "
      f"= {len(features_otimizadas)} features")


# ------------------------------------------------------------------------------
# 3. DIVISÃO TEMPORAL (TREINO VS. TESTE)
# ------------------------------------------------------------------------------
print("\n--- [3/7] Divisão Temporal dos Dados ---")
data_corte = pd.to_datetime("2025-10-01 00:00:00")

df_treino = df_preprocessed[df_preprocessed["timestamp"] < data_corte].copy()
df_teste = df_preprocessed[df_preprocessed["timestamp"] >= data_corte].copy()

df_teste.to_csv("dados_teste_ultimos_3_meses.csv", index=False)
print("-> Base de teste (3 últimos meses) salva em 'dados_teste_ultimos_3_meses.csv'")

target = "falha"
y_treino = df_treino[target].copy()
y_teste = df_teste[target].copy()

# Razão de desbalanço de classe (usada apenas nos cenários com pesos)
scale_pos_weight = (len(y_treino) - sum(y_treino)) / sum(y_treino)
print(f"-> Treino: {len(df_treino)} horas ({y_treino.sum()} falhas) | "
      f"Teste: {len(df_teste)} horas ({y_teste.sum()} falhas)")


# ------------------------------------------------------------------------------
# 4. FUNÇÕES DE TREINO E AVALIAÇÃO
# ------------------------------------------------------------------------------
def criar_modelos(usar_pesos):
    """Com pesos: compensa o desbalanço no próprio modelo.
    Sem pesos: usado com SMOTE, que já equilibra as classes no treino."""
    peso_classe = "balanced" if usar_pesos else None
    spw = scale_pos_weight if usar_pesos else 1.0
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight=peso_classe, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=spw,
            random_state=42, eval_metric="logloss"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100, depth=4, scale_pos_weight=spw, random_seed=42,
            verbose=0, allow_writing_files=False
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=spw,
            random_state=42, verbosity=-1
        ),
        "Regressão Logística": LogisticRegression(
            class_weight=peso_classe, random_state=42, max_iter=1000
        )
    }


def treinar_e_prever(features, usar_smote):
    X_treino = df_treino[features].copy()
    X_teste = df_teste[features].copy()
    y_fit = y_treino

    # Tarefa 3: SMOTE aplicado SOMENTE no treino (o teste nunca é reamostrado)
    if usar_smote:
        smote = SMOTE(random_state=42)
        X_treino, y_fit = smote.fit_resample(X_treino, y_treino)

    # Escalonamento para a Regressão Logística (ajustado apenas no treino)
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    predicoes = {}
    for nome, model in criar_modelos(usar_pesos=not usar_smote).items():
        if nome == "Regressão Logística":
            model.fit(X_treino_scaled, y_fit)
            predicoes[nome] = model.predict(X_teste_scaled)
        else:
            model.fit(X_treino, y_fit)
            predicoes[nome] = model.predict(X_teste)
    return predicoes, len(y_fit), int(sum(y_fit))


def avaliar(predicoes, cenario):
    linhas = []
    for nome, y_pred in predicoes.items():
        tn, fp, fn, tp = confusion_matrix(y_teste, y_pred).ravel()
        linhas.append({
            "Cenário": cenario,
            "Modelo": nome,
            "Recall": recall_score(y_teste, y_pred),
            "Precisão": precision_score(y_teste, y_pred, zero_division=0),
            "F1": f1_score(y_teste, y_pred),
            "Acurácia": accuracy_score(y_teste, y_pred),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Custo": fn * CUSTO_FN + fp * CUSTO_FP
        })
    return linhas


# ------------------------------------------------------------------------------
# 5. TREINAMENTO DOS CENÁRIOS
# ------------------------------------------------------------------------------
print("\n--- [4/7] Treinando os 5 modelos em 3 cenários ---")
cenarios = {
    "A - Features base + Pesos": (features_base, False),
    "B - Features otimizadas + Pesos": (features_otimizadas, False),
    "C - Features otimizadas + SMOTE": (features_otimizadas, True),
}

resultados = []
predicoes_por_cenario = {}
for cenario, (features, usar_smote) in cenarios.items():
    predicoes, n_fit, n_falhas_fit = treinar_e_prever(features, usar_smote)
    predicoes_por_cenario[cenario] = predicoes
    resultados += avaliar(predicoes, cenario)
    print(f"-> {cenario}: 5 modelos treinados "
          f"(treino com {n_fit} amostras, {n_falhas_fit} falhas)")

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_cenarios.csv", index=False)
print("-> Métricas de todos os cenários salvas em 'resultados_cenarios.csv'")


def tabela_terminal(df):
    saida = pd.DataFrame({
        "Modelo": df["Modelo"],
        "Recall": (df["Recall"] * 100).map("{:.2f}%".format),
        "Precisão": (df["Precisão"] * 100).map("{:.2f}%".format),
        "Alarmes Falsos (FP)": df["FP"],
        "Falhas Perdidas (FN)": df["FN"],
        "Custo Financeiro Total (R$)": df["Custo"].map(formatar_reais),
    })
    return saida.to_string(index=False)


# ------------------------------------------------------------------------------
# 6. PERGUNTA 1 E TABELA DE CUSTO FINANCEIRO (TAREFA 4)
# ------------------------------------------------------------------------------
print("\n--- [5/7] Pergunta 1: Impacto das novas features no Recall dos Boostings ---")
boostings = ["XGBoost", "CatBoost", "LightGBM"]
recall_a = df_resultados[df_resultados["Cenário"].str.startswith("A")].set_index("Modelo")["Recall"]
recall_b = df_resultados[df_resultados["Cenário"].str.startswith("B")].set_index("Modelo")["Recall"]
df_p1 = pd.DataFrame({
    "Modelo": boostings,
    "Recall base (A)": [f"{recall_a[m]*100:.2f}%" for m in boostings],
    "Recall c/ novas features (B)": [f"{recall_b[m]*100:.2f}%" for m in boostings],
    "Variação (p.p.)": [f"{(recall_b[m]-recall_a[m])*100:+.2f}" for m in boostings],
})
print("\n", df_p1.to_string(index=False))

print("\n--- [6/7] Custo Financeiro Total por Cenário ---")
print(f"Custo = (FN x {formatar_reais(CUSTO_FN)}) + (FP x {formatar_reais(CUSTO_FP)})")
for cenario in cenarios:
    print(f"\n>> {cenario}")
    print(tabela_terminal(df_resultados[df_resultados["Cenário"] == cenario]))

# Referências sem modelo: nunca alarmar x alarmar todas as horas
n_falhas_teste = int(y_teste.sum())
n_normais_teste = len(y_teste) - n_falhas_teste
print("\n>> Referências sem modelo")
print(f"Nunca alarmar  (FN={n_falhas_teste}, FP=0): {formatar_reais(n_falhas_teste * CUSTO_FN)}")
print(f"Sempre alarmar (FN=0, FP={n_normais_teste}): {formatar_reais(n_normais_teste * CUSTO_FP)}")

melhor = df_resultados.loc[df_resultados["Custo"].idxmin()]
print(f"\n-> MENOR CUSTO: {melhor['Modelo']} | {melhor['Cenário']} | "
      f"{formatar_reais(melhor['Custo'])} (Recall {melhor['Recall']*100:.2f}%, "
      f"FN={melhor['FN']}, FP={melhor['FP']})")


# ------------------------------------------------------------------------------
# 7. VISUALIZAÇÃO GRÁFICA (PIPELINE OTIMIZADO: CENÁRIO C)
# ------------------------------------------------------------------------------
print("\n--- [7/7] Salvando Painel de Matrizes de Confusão do Pipeline Otimizado ---")
cenario_final = "C - Features otimizadas + SMOTE"
custos_finais = df_resultados[df_resultados["Cenário"] == cenario_final].set_index("Modelo")["Custo"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

for idx, (nome, y_pred) in enumerate(predicoes_por_cenario[cenario_final].items()):
    cm = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Falha (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=axes[idx])
    axes[idx].set_title(f"{nome}\nCusto: {formatar_reais(custos_finais[nome])}",
                        fontsize=12, fontweight="bold")

fig.delaxes(axes[5])

plt.suptitle("Matrizes de Confusão - Pipeline Otimizado (Features Avançadas + SMOTE)",
             fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout()

plt.savefig("matriz_comparativa_otimizada.png", bbox_inches="tight")
print("-> Imagem 'matriz_comparativa_otimizada.png' salva com sucesso!")
plt.show()
