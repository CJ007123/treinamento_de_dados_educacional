## 🎓 Classificação de Situação Acadêmica de Alunos

Este projeto tem como objetivo prever a situação acadêmica de alunos (Aprovado/Reprovado) com base em suas notas e frequência.
Para isso, foram gerados dados simulados e treinados três modelos de Machine Learning.
Além disso, o projeto conta com visualizações gráficas e uma API com FastAPI para disponibilizar os modelos.

## 🚀 Funcionalidades

- Simulação de dados de alunos (notas e frequência).

- Treinamento de 3 modelos de ML:

- 🌳 Decision Tree

- 🌲 Random Forest

- 👥 K-Nearest Neighbors (KNN)

- Comparação de desempenho via acurácia.

- Geração de gráficos explicativos:

- Dispersão entre notas e situação do aluno

- Matriz de confusão

- Importância das features no Random Forest

- Exposição dos modelos via API (FastAPI) para uso externo.

## 📊 Tecnologias utilizadas

- Python 3

## 📚 Bibliotecas

- pandas, numpy → manipulação de dados

- scikit-learn → modelos de ML e métricas

- matplotlib, seaborn → visualização de dados

- joblib → salvar/carregar modelos

- fastapi, pydantic → API de previsão



## 🧩 Fluxo do Projeto

- Geração de dados

- 100 alunos com notas (Prova 1 e Prova 2) e frequência.

- Definição da situação:

- ✅ Aprovado → Média ≥ 60 e Frequência ≥ 75

- ❌ Reprovado → Caso contrário

- Treinamento

- Divisão em treino/teste.

- Normalização dos dados para o KNN.

- Avaliação da acurácia de cada modelo.

- Resultados obtidos

- Decision Tree → ~95% acurácia

- Random Forest → ~95% acurácia

- KNN → variando entre 90–95% dependendo do k

- Visualizações

- Gráficos para interpretar desempenho e comportamento dos modelos.

- API com FastAPI

- Endpoint /prever para enviar os dados do aluno e escolher o modelo (Decision, Random, Knn).

- Retorna a previsão e atributos derivados (ex.: se o aluno trabalha, desempenho).



## 📷 Exemplos de Gráficos
- Relação entre notas e aprovação

- Matriz de Confusão (Decision Tree)

- Importância das features (Random Forest)

## ⚡Como executar

Clone o repositório:

git clone https://github.com/seuusuario/seu-repo.git
cd seu-repo


Instale as dependências:

- pip install -r requirements.txt


Execute o script principal (para treinar modelos e gerar gráficos):

- python main.py

Rode a API (FastAPI):

- uvicorn Projeto.app:app --reload


Acesse a documentação interativa da API:
👉 http://127.0.0.1:8000/docs


