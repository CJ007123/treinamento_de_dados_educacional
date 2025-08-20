"""
Módulo de Classificação de Situação Acadêmica de Alunos

Este script simula dados de alunos com notas e frequência. A partir disso,
treina tres modelos de Machine Learning (DecisionTree,  RandomForest, KNeighborsClassifier) para prever
se o aluno será aprovado ou reprovado. Também gera gráficos para análise.

"""
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)

n = 100
nota_prova_1 = np.random.randint(20,101, n)
nota_prova_2 = np.random.randint(20,101, n)
frequencia = np.random.randint(40,101, n)


# ========================================================
# 1. Criação dos dados simulados
# ========================================================

dados = pd.DataFrame({
    'nota_prova_1': nota_prova_1,
    'nota_prova_2': nota_prova_2,
    'frequencia': frequencia
})

enconder = LabelEncoder()


#- situação: rótulo final (Aprovado/Reprovado)
dados['situação'] = dados.apply(lambda row: 'Aprovado' if ((row['nota_prova_1'] + row['nota_prova_2']) / 2 >= 60 and row['frequencia'] >= 75) else 'Reprovado', axis=1)
dados['situação_cod'] = enconder.fit_transform(dados["situação"])



# ========================================================
# 3. Preparação dos dados para treinamento
# ========================================================

#X -> features (variáveis explicativas)
#y -> target (situação do aluno)


X = dados[['nota_prova_1', 'nota_prova_2', 'frequencia']]
y = dados['situação_cod']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)





# ========================================================
# 4. Modelos de Machine Learning
# ========================================================

# --- Decision Tree ---
for d in range(1,11):
    modelo_dt = DecisionTreeClassifier()
    modelo_dt.fit(X_treino, y_treino)
    previsao = modelo_dt.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsao)
    print(f'Acurácia DecisionTreeClassifier: {acuracia * 100:.2f} %')

print()

# --- Random Forest ---
for r in range(1,11):
    modelo_rf = RandomForestClassifier()
    modelo_rf.fit(X_treino, y_treino)
    modelo_rf_previsao = modelo_rf.predict(X_teste)
    acuaracia_rf = accuracy_score(y_teste, modelo_rf_previsao)
    print(f'Acurácia RandomForest: {acuaracia_rf * 100:.2f} %')

print()
# --- MinMaxScaler  ---
escaler = MinMaxScaler()
x_escaler_treino = escaler.fit_transform(X_treino)
x_escaler_teste  = escaler.transform(X_teste)

# --- KNeighborsClassifier  ---
for k in range(1,10):
    modelo_knn = KNeighborsClassifier(n_neighbors=k)
    modelo_knn.fit(x_escaler_treino, y_treino)
    previsao_knn = modelo_knn.predict(x_escaler_teste)
    acuracia_knn = accuracy_score(y_teste, previsao_knn)
    print(f'Acurácia KNeighbors: {acuracia_knn * 100:.2f} %')



# modelos = {
#     'Decision': modelo_dt,
#     'Random': modelo_rf,
#     'Knn': modelo_knn
# }

# joblib.dump(modelos, 'modelos.pkl')





# # ========================================================
# # 5. Visualizações
# # ========================================================

# # ------------------ Gráfico 1 ------------------
# """
# Dispersão entre notas das provas e situação do aluno
# """

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="nota_prova_1",
    y="nota_prova_2",
    hue="situação",
    data=dados,
    palette={"Aprovado":"green", "Reprovado":"red"},
    s=120, edgecolor="black"
)
plt.title("📊 Relação entre Notas e Aprovação", fontsize=14, weight="bold")
plt.xlabel("Nota da Prova 1")
plt.ylabel("Nota da Prova 2")
plt.legend(title="Situação")
plt.show()

# # ------------------ Gráfico 2 ------------------
# """
# Matriz de confusão para o modelo Decision Tree
# Mostra erros e acertos nas previsões
# """

matriz = confusion_matrix(y_teste, previsao)
plt.figure(figsize=(6,5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=modelo_dt.classes_, yticklabels=modelo_dt.classes_)
plt.title("📌 Matriz de Confusão - Decision Tree", fontsize=14, weight="bold")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# ------------------ Gráfico 3 ------------------
# """
# Importância das features no modelo Random Forest
# Mostra quais variáveis mais influenciaram a predição

# """

importancias = pd.Series(modelo_rf.feature_importances_, index=[
    "Nota Prova 1", "Nota Prova 2", "Frequência", 
])

plt.figure(figsize=(8,5))
importancias.sort_values().plot(kind="barh", color="royalblue")
plt.title("Importância das Features (Random Forest)", fontsize=14, weight="bold")
plt.xlabel("Impacto na Aprovação")
plt.show()
