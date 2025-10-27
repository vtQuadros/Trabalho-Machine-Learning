"""
API de Previsão de Acidentes Aéreos Fatais — Projeto de Machine Learning & IA

Esta API foi desenvolvida com FastAPI para disponibilizar em produção o modelo de machine learning 
treinado para prever se um acidente aéreo será fatal ou não.

Funcionamento:
- Recebe os dados de um acidente aéreo via requisição POST no endpoint `/prever/`.
- Os dados são automaticamente validados (variáveis numéricas e categóricas relacionadas ao acidente).
- As variáveis categóricas são codificadas automaticamente (one-hot encoding).
- O scaler treinado é aplicado para padronizar os dados de entrada.
- O modelo de Regressão Logística realiza a predição, retornando:
    - `fatal`: booleano indicando se o acidente tem alta probabilidade de ser fatal.
    - `probabilidade`: valor entre 0 e 1 com a confiança do modelo.

Limiar de decisão:
- A predição de fatalidade é feita usando o threshold otimizado durante o treinamento.

Objetivo:
Fornecer uma interface acessível para integrar o modelo a sistemas externos (ex: dashboards de segurança aérea, 
sistemas de análise de risco), possibilitando ações preventivas baseadas em dados históricos.

"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI(
    title="API de Previsão de Acidentes Aéreos Fatais",
    description="API para prever se um acidente aéreo será fatal com base em características do voo e da aeronave",
    version="1.0.0"
)

# Carregar modelo e scaler
try:
    model = joblib.load("modelo_lr.pkl")
    scaler = joblib.load("scaler.pkl")
    # Carregar as colunas do treinamento para garantir consistência
    colunas_treino = joblib.load("colunas_treino.pkl")
except FileNotFoundError as e:
    print(f"⚠ ERRO: Arquivo não encontrado - {e}")
    print("Execute o notebook primeiro para gerar os arquivos .pkl necessários!")
    model = None
    scaler = None
    colunas_treino = None

# Definir threshold otimizado (ajuste conforme resultado do notebook)
THRESHOLD_OTIMIZADO = 0.35  # Ajuste este valor após executar o notebook

# Classe com todos os campos esperados
class DadosAcidente(BaseModel):
    # Variáveis numéricas
    latitude: float
    longitude: float
    peso_max_decolagem: float
    numero_assentos: int
    ano_ocorrencia: int
    mes_ocorrencia: int
    
    # Variáveis categóricas
    fase_operacao: str
    cat_aeronave: str
    regiao: str
    uf: str
    modelo_aeronave: str
    nome_fabricante: str

    class Config:
        schema_extra = {
            "example": {
                "latitude": -23.5505,
                "longitude": -46.6333,
                "peso_max_decolagem": 5700.0,
                "numero_assentos": 9,
                "ano_ocorrencia": 2020,
                "mes_ocorrencia": 6,
                "fase_operacao": "DECOLAGEM",
                "cat_aeronave": "AVIAO",
                "regiao": "SUDESTE",
                "uf": "SP",
                "modelo_aeronave": "EMB-810C",
                "nome_fabricante": "EMBRAER"
            }
        }


@app.get("/")
def root():
    """Endpoint raiz - informações sobre a API"""
    return {
        "mensagem": "API de Previsão de Acidentes Aéreos Fatais",
        "versao": "1.0.0",
        "endpoints": {
            "/prever/": "POST - Realizar predição de fatalidade",
            "/status/": "GET - Verificar status da API",
            "/docs": "Documentação interativa (Swagger UI)"
        }
    }


@app.get("/status/")
def status():
    """Verificar se o modelo está carregado e pronto"""
    if model is None or scaler is None or colunas_treino is None:
        return {
            "status": "erro",
            "mensagem": "Modelo não carregado. Execute o notebook para gerar os arquivos .pkl"
        }
    return {
        "status": "ok",
        "modelo_carregado": True,
        "threshold": THRESHOLD_OTIMIZADO,
        "features_esperadas": len(colunas_treino)
    }


@app.post("/prever/")
def prever(dados: DadosAcidente):
    """
    Realizar predição de fatalidade de acidente aéreo
    
    Retorna:
    - fatal: boolean indicando se o acidente tem alta probabilidade de ser fatal
    - probabilidade: valor entre 0 e 1 indicando a confiança da predição
    - interpretacao: texto explicativo do resultado
    """
    
    # Verificar se o modelo está carregado
    if model is None or scaler is None or colunas_treino is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está disponível. Execute o notebook para gerar os arquivos necessários."
        )
    
    try:
        # Converter dados para DataFrame
        df = pd.DataFrame([dados.dict()])
        
        # Definir colunas numéricas e categóricas
        colunas_categoricas = ['fase_operacao', 'cat_aeronave', 'regiao', 'uf', 
                               'modelo_aeronave', 'nome_fabricante']
        
        # Aplicar one-hot encoding
        df_encoded = pd.get_dummies(df, columns=colunas_categoricas)
        
        # Garantir que todas as colunas do treino estejam presentes
        df_encoded = df_encoded.reindex(columns=colunas_treino, fill_value=0)
        
        # Normalizar os dados
        df_scaled = scaler.transform(df_encoded)
        
        # Realizar predição
        proba = model.predict_proba(df_scaled)[0][1]  # Probabilidade da classe 1 (Fatal)
        pred = int(proba > THRESHOLD_OTIMIZADO)
        
        # Gerar interpretação
        if pred == 1:
            interpretacao = f"ATENÇÃO: Alto risco de fatalidade ({proba*100:.1f}%). Medidas preventivas recomendadas."
        else:
            if proba > 0.25:
                interpretacao = f"Risco moderado de fatalidade ({proba*100:.1f}%). Cautela recomendada."
            else:
                interpretacao = f"Baixo risco de fatalidade ({proba*100:.1f}%)."
        
        return {
            "fatal": bool(pred),
            "probabilidade": round(float(proba), 4),
            "probabilidade_percentual": f"{proba*100:.2f}%",
            "interpretacao": interpretacao,
            "threshold_utilizado": THRESHOLD_OTIMIZADO
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar a predição: {str(e)}"
        )

