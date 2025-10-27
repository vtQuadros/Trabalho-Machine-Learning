from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from datetime import datetime

# ==================== CONFIGURAÇÃO DE LOGGING ====================
logging.basicConfig(
    filename='api_predicoes.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==================== CONFIGURAÇÃO DA API ====================
app = FastAPI(
    title="API Predição de Acidentes Aéreos Fatais",
    description="API para prever fatalidade em acidentes aéreos usando Regressão Logística",
    version="2.0"
)

# ==================== CARREGAR MODELO ====================
MODELO_PATH = "modelo_lr.pkl"
SCALER_PATH = "scaler.pkl"
COLUNAS_PATH = "colunas_treino.pkl"

# 🎯 THRESHOLD OTIMIZADO - Leia do arquivo gerado no notebook
try:
    with open("threshold_otimizado.txt", "r") as f:
        linhas = f.readlines()
        THRESHOLD_OTIMIZADO = float(linhas[0].split("=")[1].strip())
        F1_SCORE_OTIMIZADO = float(linhas[1].split("=")[1].strip())
    print(f"✓ Threshold carregado do arquivo: {THRESHOLD_OTIMIZADO:.4f}")
    print(f"✓ F1-Score associado: {F1_SCORE_OTIMIZADO:.4f}")
except FileNotFoundError:
    print("⚠️ Arquivo threshold_otimizado.txt não encontrado. Usando valor padrão.")
    THRESHOLD_OTIMIZADO = 0.26
    F1_SCORE_OTIMIZADO = None

# Carregar modelo, scaler e colunas
try:
    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    colunas_treino = joblib.load(COLUNAS_PATH)
    print(f"✓ Modelo carregado: {len(colunas_treino)} features")
    print(f"✓ Threshold otimizado: {THRESHOLD_OTIMIZADO}")
    logging.info(f"API iniciada com threshold={THRESHOLD_OTIMIZADO}")
except Exception as e:
    logging.error(f"Erro ao carregar modelo: {e}")
    raise RuntimeError(f"❌ Erro ao carregar modelo: {e}")

# ==================== MODELOS DE DADOS ====================
class AcidenteAereo(BaseModel):
    """Modelo de entrada para predição de acidentes aéreos."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "latitude": -23.5505,
                "longitude": -46.6333,
                "peso_max_decolagem": 5700.0,
                "numero_assentos": 9,
                "fase_operacao": "DECOLAGEM",
                "cat_aeronave": "AVIÃO",
                "regiao": "SUDESTE",
                "uf": "SP",
                "modelo_aeronave": "EMB-110",
                "nome_fabricante": "EMBRAER",
                "ano_ocorrencia": 2023,
                "mes_ocorrencia": 6
            }
        }
    )
    
    latitude: float
    longitude: float
    peso_max_decolagem: float
    numero_assentos: int
    fase_operacao: str
    cat_aeronave: str
    regiao: str
    uf: str
    modelo_aeronave: str
    nome_fabricante: str
    ano_ocorrencia: int
    mes_ocorrencia: int

class RespostaPredicao(BaseModel):
    """Modelo de resposta para predição individual."""
    probabilidade_fatal: float
    predicao: str
    predicao_numerica: int
    threshold_utilizado: float
    nivel_risco: str
    recomendacao: str
    interpretacao_detalhada: str

class RespostaLote(BaseModel):
    """Modelo de resposta para predição em lote."""
    total_acidentes: int
    previstos_fatais: int
    previstos_nao_fatais: int
    taxa_fatalidade_prevista: float
    probabilidade_media: float
    distribuicao_risco: dict
    resultados: List[dict]

# ==================== FUNÇÕES AUXILIARES ====================
def preprocessar_entrada(dados: AcidenteAereo) -> np.ndarray:
    """
    Converte entrada em formato compatível com o modelo.
    
    Aplica:
    1. One-hot encoding nas variáveis categóricas
    2. Alinhamento com as colunas do treino
    3. Normalização usando o scaler treinado
    """
    df = pd.DataFrame([dados.model_dump()])
    
    colunas_categoricas = ['fase_operacao', 'cat_aeronave', 'regiao', 'uf', 
                           'modelo_aeronave', 'nome_fabricante']
    df_encoded = pd.get_dummies(df, columns=colunas_categoricas)
    
    df_encoded = df_encoded.reindex(columns=colunas_treino, fill_value=0)
    
    X_scaled = scaler.transform(df_encoded)
    
    return X_scaled

def interpretar_risco(probabilidade: float) -> str:
    """Classifica o nível de risco baseado na probabilidade."""
    if probabilidade >= 0.70:
        return "CRÍTICO"
    elif probabilidade >= 0.50:
        return "ALTO"
    elif probabilidade >= 0.30:
        return "MODERADO"
    else:
        return "BAIXO"

def gerar_recomendacao(predicao: int, probabilidade: float, nivel_risco: str) -> str:
    """Gera recomendação de ação baseada na predição e nível de risco."""
    if predicao == 1:
        if nivel_risco == "CRÍTICO":
            return "🚨 ALERTA CRÍTICO: Implementar medidas de segurança IMEDIATAS. Investigação prioritária obrigatória."
        elif nivel_risco == "ALTO":
            return "⚠️ ALERTA ALTO: Investigação detalhada recomendada. Reforçar protocolos de segurança."
        else:
            return "⚠️ ATENÇÃO: Monitoramento reforçado necessário. Revisar condições operacionais."
    else:
        if probabilidade >= 0.20:
            return "💡 PRECAUÇÃO: Risco presente mas baixo. Manter vigilância e seguir protocolos padrão."
        else:
            return "✅ SEGURO: Risco muito baixo. Manter procedimentos normais de segurança."

def gerar_interpretacao_detalhada(probabilidade: float, nivel_risco: str) -> str:
    """Gera interpretação detalhada da predição."""
    prob_percentual = probabilidade * 100
    
    interpretacoes = {
        "CRÍTICO": f"Probabilidade MUITO ALTA de fatalidade ({prob_percentual:.1f}%). Situação de risco extremo.",
        "ALTO": f"Probabilidade ELEVADA de fatalidade ({prob_percentual:.1f}%). Situação de alto risco.",
        "MODERADO": f"Probabilidade MODERADA de fatalidade ({prob_percentual:.1f}%). Cautela recomendada.",
        "BAIXO": f"Probabilidade BAIXA de fatalidade ({prob_percentual:.1f}%). Situação relativamente segura."
    }
    
    return interpretacoes.get(nivel_risco, f"Probabilidade: {prob_percentual:.1f}%")

# ==================== ENDPOINTS ====================
@app.get("/")
def root():
    """Página inicial da API com informações básicas."""
    return {
        "message": "🛩️ API de Predição de Acidentes Aéreos Fatais",
        "modelo": "Regressão Logística (Otimizada)",
        "threshold_atual": THRESHOLD_OTIMIZADO,
        "f1_score_otimizado": F1_SCORE_OTIMIZADO,
        "estrategia": "Threshold Otimizado para Máximo F1-Score",
        "versao": "2.0",
        "endpoints": {
            "GET /": "Informações da API",
            "GET /health": "Status de saúde",
            "GET /metricas": "Métricas do modelo",
            "POST /prever": "Predição individual",
            "POST /prever_lote": "Predição em lote",
            "GET /docs": "Documentação interativa"
        }
    }

@app.get("/health")
def health_check():
    """Verifica se a API está operacional."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "modelo_carregado": modelo is not None,
        "scaler_carregado": scaler is not None,
        "features_esperadas": len(colunas_treino),
        "threshold": THRESHOLD_OTIMIZADO
    }

@app.get("/metricas")
def obter_metricas():
    """Retorna métricas do modelo treinado."""
    return {
        "modelo": "Regressão Logística",
        "threshold_otimizado": THRESHOLD_OTIMIZADO,
        "f1_score": F1_SCORE_OTIMIZADO,
        "total_features": len(colunas_treino),
        "estrategia": "Maximização do F1-Score",
        "interpretacao_threshold": f"Predições com probabilidade ≥ {THRESHOLD_OTIMIZADO:.2%} são classificadas como FATAL"
    }

@app.post("/prever", response_model=RespostaPredicao)
def prever_acidente(dados: AcidenteAereo):
    """Prediz se um acidente aéreo será fatal."""
    try:
        X = preprocessar_entrada(dados)
        probabilidade = float(modelo.predict_proba(X)[0, 1])
        predicao = int(probabilidade >= THRESHOLD_OTIMIZADO)
        
        nivel_risco = interpretar_risco(probabilidade)
        recomendacao = gerar_recomendacao(predicao, probabilidade, nivel_risco)
        interpretacao = gerar_interpretacao_detalhada(probabilidade, nivel_risco)
        
        logging.info(
            f"Predição: {dados.uf}/{dados.cat_aeronave} -> "
            f"Prob={probabilidade:.4f}, Fatal={predicao}, Risco={nivel_risco}"
        )
        
        return RespostaPredicao(
            probabilidade_fatal=round(probabilidade, 4),
            predicao="FATAL" if predicao == 1 else "NÃO FATAL",
            predicao_numerica=predicao,
            threshold_utilizado=THRESHOLD_OTIMIZADO,
            nivel_risco=nivel_risco,
            recomendacao=recomendacao,
            interpretacao_detalhada=interpretacao
        )
        
    except Exception as e:
        logging.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/prever_lote", response_model=RespostaLote)
def prever_lote(acidentes: List[AcidenteAereo]):
    """Realiza predições para múltiplos acidentes simultaneamente."""
    try:
        resultados = []
        probabilidades = []
        distribuicao = {"CRÍTICO": 0, "ALTO": 0, "MODERADO": 0, "BAIXO": 0}
        
        for acidente in acidentes:
            X = preprocessar_entrada(acidente)
            probabilidade = float(modelo.predict_proba(X)[0, 1])
            predicao = int(probabilidade >= THRESHOLD_OTIMIZADO)
            nivel_risco = interpretar_risco(probabilidade)
            
            probabilidades.append(probabilidade)
            distribuicao[nivel_risco] += 1
            
            resultados.append({
                "dados_entrada": acidente.model_dump(),
                "probabilidade_fatal": round(probabilidade, 4),
                "predicao": "FATAL" if predicao == 1 else "NÃO FATAL",
                "nivel_risco": nivel_risco
            })
        
        total = len(resultados)
        fatais = sum(1 for r in resultados if r["predicao"] == "FATAL")
        prob_media = sum(probabilidades) / total if total > 0 else 0
        
        logging.info(f"Predição em lote: {total} acidentes, {fatais} fatais previstos, prob_media={prob_media:.4f}")
        
        return RespostaLote(
            total_acidentes=total,
            previstos_fatais=fatais,
            previstos_nao_fatais=total - fatais,
            taxa_fatalidade_prevista=round(fatais / total * 100, 2) if total > 0 else 0,
            probabilidade_media=round(prob_media, 4),
            distribuicao_risco=distribuicao,
            resultados=resultados
        )
        
    except Exception as e:
        logging.error(f"Erro na predição em lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 INICIANDO API DE PREDIÇÃO DE ACIDENTES AÉREOS FATAIS")
    print("="*70)
    print(f"📊 Threshold: {THRESHOLD_OTIMIZADO:.4f}")
    print(f"📈 F1-Score: {F1_SCORE_OTIMIZADO:.4f}" if F1_SCORE_OTIMIZADO else "")
    print(f"🔢 Features: {len(colunas_treino)}")
    print("="*70)
    print("\n🌐 Acesse:")
    print("   • API: http://localhost:8000")
    print("   • Docs: http://localhost:8000/docs")
    print("\n💡 Pressione CTRL+C para parar\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")