import joblib
from fastapi import FastAPI
from pydantic import BaseModel

modelos = joblib.load('modelos.pkl')

app = FastAPI(title="API de Classificação Acadêmica", version="1.0")

class DadosEntrada(BaseModel):
    modelo: str
    nota_prova_1: int
    nota_prova_2: int
    frequencia: int

@app.post("/prever")
def prever(dados: DadosEntrada):
    # validação do modelo
    if dados.modelo not in modelos:
        return {"erro": "Modelo não encontrado. Use: Decision, Random ou Knn."}

    modelo = modelos[dados.modelo]

    # monta os valores exatamente como foram usados no treino
    entrada = [[
        dados.nota_prova_1,
        dados.nota_prova_2,
        dados.frequencia
    ]]

    resultado = modelo.predict(entrada)

    # atributos derivados
    desempenho = 'baixo' if ((dados.nota_prova_1 +  dados.nota_prova_2) / 2 < 60) else 'alto'
    aluno_trabalha = 'sim' if dados.frequencia < 70 else 'não'

    return {
        "modelo": dados.modelo,
        "entrada": {
            'Nota_prova_1': dados.nota_prova_1,
            'Nota_prova_2': dados.nota_prova_2,
            'Frequencia': dados.frequencia,
            'Aluno_trabalha': aluno_trabalha,
            'Desempenho': desempenho,
        },
        "previsao": str(resultado[0])
    }
