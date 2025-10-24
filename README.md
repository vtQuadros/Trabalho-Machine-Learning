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
1. Clonar o repositório  
2. Criar e ativar ambiente virtual  
3. Instalar dependências  
4. Executar Jupyter Notebook ou abrir no VS Code  
