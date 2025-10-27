# Projeto Machine Learning - Predição de Acidentes Aéreos Fatais

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

## Visão Geral
Modelo preditivo para identificar se um acidente aéreo será fatal, usando dados geográficos, temporais e operacionais. Inclui pré-processamento, EDA, engenharia de features, modelagem e avaliação de 5 modelos diferentes.

Objetivo: Fornecer insights acionáveis para autoridades de aviação e aumentar a segurança aérea.

## Problema
Acidentes fatais são menos frequentes (~10:1). O desafio é lidar com o desbalanceamento e prever corretamente acidentes graves.

## Dataset
Dados do CENIPA com informações sobre:

- Geográficos: latitude, longitude, UF, região  
- Temporais: data, hora, ano, mês  
- Aeronave: modelo, fabricante, categoria, peso, assentos  
- Operacionais: fase da operação, tipo de operação  

Variável-alvo: `les_fatais_trip` (0 = Não fatal, 1 = Fatal).  

## Metodologia
- Limpeza e pré-processamento: duplicatas, tipos, valores ausentes.  
- Engenharia de features: encoding, normalização, divisão treino/teste.  
- Balanceamento: SMOTE aplicado apenas no treino.  
- Modelagem: Dummy, Regressão Logística, Árvore de Decisão, Random Forest, MLP.  
- Avaliação: Acurácia, Precisão, Recall, F1-Score, AUC-ROC, validação cruzada 5-fold.  
- Otimização de threshold para maximizar F1-Score.

## Resultados
- Modelo vencedor: Regressão Logística com SMOTE e threshold otimizado.  
- F1-Score melhorado, alta interpretabilidade, rápida execução e estabilidade.  
- Principais features: `latitude`, `fase_operacao_Especializada`, `uf_RS`, `ano_ocorrencia`, `nome_fabricante_EMBRAER`.  
- Insights: regiões críticas (RS, SC), operações de risco, tendência de redução de fatalidade ao longo dos anos.

## Tecnologias
- Python 3.x, Pandas, NumPy, Matplotlib, Seaborn  
- Machine Learning: Scikit-learn, imbalanced-learn, Random Forest, MLP  
- Ambiente: Jupyter Notebook / VS Code  

## Como Executar

### 1. Executar o Notebook
```bash
# Clonar o repositório
git clone <url-do-repositorio>

# Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Abrir o notebook
jupyter notebook projeto.ipynb
```

### 2. Usar a API

Após executar o notebook até a seção de exportação:

```bash
# Instalar dependências da API
cd api_predicao_evasao
pip install -r requirements.txt

# Executar a API
uvicorn api_fastapi:app --reload

# Em outro terminal, testar a API
python testar_api.py
```

Acesse a documentação interativa em: `http://127.0.0.1:8000/docs`

Veja mais detalhes no [README da API](api_predicao_evasao/README.md)

## Estrutura do Projeto
```
Trabalho-Machine-Learning/
│
├── projeto.ipynb                    # Notebook principal com toda análise
├── README.md                        # Esta documentação
├── requirements.txt                 # Dependências do projeto
│
├── docs/                           # Dados
│   ├── dicionario.csv
│   ├── treino.csv
│   └── teste.csv
│
└── api_predicao_evasao/            # API FastAPI
    ├── api_fastapi.py              # Código da API
    ├── README.md                   # Documentação da API
    ├── requirements.txt            # Dependências da API
    ├── testar_api.py              # Script de teste
    ├── modelo_lr.pkl              # Modelo exportado (gerado)
    ├── scaler.pkl                 # Scaler exportado (gerado)
    ├── colunas_treino.pkl         # Colunas (gerado)
    └── threshold_otimizado.txt    # Threshold (gerado)
```
4. Executar Jupyter Notebook ou abrir no VS Code  
