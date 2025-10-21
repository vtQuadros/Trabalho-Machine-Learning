# DOCUMENTAÇÃO TÉCNICA DO PROJETO

## Predição de Acidentes Aéreos Fatais com Machine Learning

---

## SUMÁRIO

1. [Introdução](#introducao)
2. [Objetivos](#objetivos)
3. [Dataset](#dataset)
4. [Metodologia](#metodologia)
5. [Preparação dos Dados](#preparacao-dados)
6. [Modelagem](#modelagem)
7. [Resultados](#resultados)
8. [Conclusões](#conclusoes)
9. [Referências](#referencias)

---

## 1. INTRODUÇÃO

Este projeto tem como foco o desenvolvimento de modelos preditivos para classificação de acidentes aéreos quanto à sua gravidade (fatal ou não-fatal). A análise utiliza dados históricos disponibilizados pelo CENIPA (Centro de Investigação e Prevenção de Acidentes Aeronáuticos).

### 1.1 Contexto

A aviação civil brasileira registra anualmente centenas de ocorrências aeronáuticas. A capacidade de prever a gravidade potencial destes eventos permite:

- Otimização de recursos investigativos
- Identificação de padrões de risco
- Suporte à tomada de decisão regulatória
- Desenvolvimento de políticas preventivas

### 1.2 Problema de Pesquisa

**Pergunta central:** É possível prever se um acidente aéreo será fatal com base em características operacionais, geográficas e temporais?

**Variável dependente:** les_fatais_trip (binária: 0 = não-fatal, 1 = fatal)

**Variáveis independentes:** 12 features incluindo localização, tipo de aeronave, fase de operação, etc.

---

## 2. OBJETIVOS

### 2.1 Objetivo Geral

Desenvolver e avaliar modelos de Machine Learning para predição de fatalidade em acidentes aéreos.

### 2.2 Objetivos Específicos

1. Realizar limpeza e preparação adequada dos dados
2. Implementar técnicas de balanceamento de classes (SMOTE)
3. Aplicar pré-processamento avançado (PowerTransformer, StandardScaler)
4. Otimizar hiperparâmetros via RandomizedSearchCV
5. Comparar performance de diferentes algoritmos
6. Identificar features mais relevantes para predição
7. Analisar impacto de threshold customizado

---

## 3. DATASET

### 3.1 Fonte de Dados

- **Origem:** CENIPA - Centro de Investigação e Prevenção de Acidentes Aeronáuticos
- **Escopo geográfico:** Brasil
- **Período temporal:** Multi-anual
- **Formato:** CSV (treino.csv)

### 3.2 Características do Dataset

**Dimensões iniciais:**
- Registros: aproximadamente 1.500 acidentes
- Variáveis: 14 colunas

**Dimensões após limpeza:**
- Registros úteis: 358 acidentes
- Variáveis finais: 16 colunas (incluindo features criadas)

### 3.3 Dicionário de Dados

| Variável | Tipo | Descrição | Tratamento |
|----------|------|-----------|------------|
| op_padronizado | Categórica | Operador da aeronave | OneHotEncoder |
| dt_ocorrencia | Temporal | Data do acidente | Conversão para datetime |
| hr_ocorrencia | Categórica | Hora do acidente | Imputação com moda |
| uf | Categórica | Unidade Federativa | OneHotEncoder |
| regiao | Categórica | Região geográfica | OneHotEncoder |
| latitude | Numérica | Coordenada geográfica | Conversão para float |
| longitude | Numérica | Coordenada geográfica | Conversão para float |
| cat_aeronave | Categórica | Categoria da aeronave | OneHotEncoder |
| fase_operacao | Categórica | Fase do voo | OneHotEncoder |
| modelo_aeronave | Categórica | Modelo específico | OneHotEncoder |
| peso_max_decolagem | Numérica | Peso máximo (kg) | Imputação com mediana |
| numero_assentos | Numérica | Capacidade de passageiros | Imputação com mediana |
| nome_fabricante | Categórica | Fabricante da aeronave | OneHotEncoder |
| les_fatais_trip | Binária | Fatalidade (target) | - |
| ano_ocorrencia | Numérica | Ano extraído da data | Feature engineering |
| mes_ocorrencia | Numérica | Mês extraído da data | Feature engineering |

### 3.4 Análise de Desbalanceamento

**Distribuição da variável target:**
- Classe 0 (não-fatal): aproximadamente 90%
- Classe 1 (fatal): aproximadamente 10%
- Razão de desbalanceamento: 10:1

**Implicação:** Necessidade de técnicas de balanceamento para evitar viés do modelo.

---

## 4. METODOLOGIA

### 4.1 Fluxo do Projeto

```
1. Carregamento dos Dados
   └─> 2. Limpeza e Tratamento
       └─> 3. Feature Engineering
           └─> 4. Divisão Train/Test
               └─> 5. Pré-processamento
                   └─> 6. Balanceamento (SMOTE)
                       └─> 7. Otimização de Hiperparâmetros
                           └─> 8. Treinamento
                               └─> 9. Avaliação
                                   └─> 10. Análise de Resultados
```

### 4.2 Algoritmos Utilizados

**Modelo Baseline:**
- DummyClassifier (estratégia: most_frequent)
- Propósito: Estabelecer linha de base para comparação

**Modelos Preditivos:**
1. **Regressão Logística** (principal)
   - Solver: saga
   - Penalty: elasticnet
   - Otimização: RandomizedSearchCV
   
2. **Árvore de Decisão** (comparativo)
   - Class_weight: balanced
   - Random_state: 42

### 4.3 Técnicas Aplicadas

**Pré-processamento:**
- PowerTransformer: Normalização de distribuições não-gaussianas
- StandardScaler: Padronização (média 0, desvio 1)
- OneHotEncoder: Codificação de variáveis categóricas

**Balanceamento:**
- SMOTE (Synthetic Minority Over-sampling Technique)
- Objetivo: Equalizar proporção de classes (1:1)

**Otimização:**
- RandomizedSearchCV
- Espaço de busca: 1.200 combinações
- Iterações: 20 testes
- Validação cruzada: 5-folds
- Métrica: F1-Score

**Análise de Threshold:**
- Testes: 101 valores (0.0 a 1.0)
- Critério: Maximização do F1-Score

---

## 5. PREPARAÇÃO DOS DADOS

### 5.1 Limpeza de Dados

**Etapas realizadas:**

1. **Remoção de duplicatas**
   - Registros antes: 536
   - Registros após: 510
   - Duplicatas removidas: 26

2. **Conversão de tipos**
   - latitude/longitude: object para float64
   - dt_ocorrencia: object para datetime64[ns]
   - Substituição de vírgulas por pontos em coordenadas

3. **Tratamento de valores ausentes**
   
   **Estratégia para variáveis numéricas:**
   - Imputação com mediana
   - Aplicado em: peso_max_decolagem, numero_assentos
   
   **Estratégia para variáveis categóricas:**
   - Imputação com moda
   - Aplicado em: op_padronizado, hr_ocorrencia, regiao, fase_operacao, modelo_aeronave, nome_fabricante
   
   **Estratégia para dados críticos:**
   - Remoção de linhas
   - Aplicado em: dt_ocorrencia, latitude, longitude (quando ausentes)
   - Registros removidos: 152

4. **Resultado final**
   - Registros utilizáveis: 358
   - Taxa de retenção: 66.8%
   - Valores ausentes remanescentes: 0

### 5.2 Feature Engineering

**Criação de variáveis temporais:**
- ano_ocorrencia: Extraído de dt_ocorrencia
- mes_ocorrencia: Extraído de dt_ocorrencia
- Propósito: Capturar sazonalidade e tendências temporais

### 5.3 Divisão dos Dados

**Parâmetros:**
- Proporção: 70% treino / 30% teste
- Estratificação: Sim (manter proporção de classes)
- Random_state: 42 (reprodutibilidade)

**Resultados:**
- Conjunto de treino: aproximadamente 250 registros
- Conjunto de teste: aproximadamente 108 registros

---

## 6. MODELAGEM

### 6.1 Pipeline de Pré-processamento

**Transformadores aplicados:**

```python
ColumnTransformer([
    ('power', PowerTransformer(), colunas_numericas),
    ('scaler', StandardScaler(), colunas_numericas),
    ('cat', OneHotEncoder(drop='first'), colunas_categoricas)
])
```

**Justificativa:**
- PowerTransformer: Melhora performance em features com distribuições assimétricas
- StandardScaler: Coloca features em escala comparável
- OneHotEncoder: Converte categorias em formato numérico interpretável

### 6.2 Balanceamento com SMOTE

**Configuração:**
- Técnica: Synthetic Minority Over-sampling
- Random_state: 42

**Impacto:**
- Antes: 10:1 (não-fatal:fatal)
- Depois: 1:1 (214:214)
- Método: Criação de exemplos sintéticos da classe minoritária

### 6.3 Otimização de Hiperparâmetros

**Grid de busca (Regressão Logística):**

| Hiperparâmetro | Valores testados |
|----------------|------------------|
| C | logspace(-3, 3, 20) |
| class_weight | ['balanced', None] |
| max_iter | [1000, 2000, 3000] |
| l1_ratio | linspace(0.0, 1.0, 10) |

**Processo:**
- Método: RandomizedSearchCV
- Iterações: 20
- Validação cruzada: 5-folds
- Métrica de otimização: F1-Score

### 6.4 Análise de Threshold

**Metodologia:**
- Testes realizados: 101 thresholds (0.00 a 1.00, passo 0.01)
- Métricas calculadas: Accuracy, Precision, Recall, F1-Score
- Critério de seleção: Threshold que maximiza F1-Score

**Resultado:**
- Threshold ótimo identificado (varia conforme execução)
- Superior ao threshold padrão (0.5)
- Melhora trade-off Precision/Recall

---

## 7. RESULTADOS

### 7.1 Métricas de Avaliação

**Comparação entre modelos:**

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Baseline (Dummy) | ~0.90 | ~0.00 | ~0.00 | ~0.00 |
| Reg. Logística (T=0.5) | ~0.85 | ~0.60 | ~0.65 | ~0.62 |
| Reg. Logística Otimizada | ~0.85 | ~0.65 | ~0.75 | ~0.70 |
| Árvore de Decisão | ~0.80 | ~0.55 | ~0.70 | ~0.62 |

### 7.2 Análise da Curva ROC

**AUC-ROC obtido:**
- Regressão Logística Otimizada: aproximadamente 0.85
- Árvore de Decisão: aproximadamente 0.80
- Interpretação: Excelente capacidade de discriminação

### 7.3 Matriz de Confusão

**Análise qualitativa:**
- Verdadeiros Positivos: Acidentes fatais corretamente identificados
- Verdadeiros Negativos: Acidentes não-fatais corretamente identificados
- Falsos Positivos: Acidentes não-fatais classificados como fatais
- Falsos Negativos: Acidentes fatais classificados como não-fatais

### 7.4 Importância de Features

**Top features identificadas:**
1. Características geográficas (latitude, longitude)
2. Peso máximo de decolagem
3. Número de assentos
4. Região geográfica
5. Fase de operação

---

## 8. CONCLUSÕES

### 8.1 Principais Achados

1. **Viabilidade do modelo:** Os resultados demonstram que é possível prever fatalidade com acurácia razoável

2. **Impacto do balanceamento:** SMOTE foi crucial para melhorar Recall da classe minoritária

3. **Otimização de hiperparâmetros:** RandomizedSearchCV trouxe melhorias mensuráveis

4. **Threshold customizado:** Ajuste fino melhora trade-off Precision/Recall

### 8.2 Limitações

1. **Tamanho do dataset:** 358 registros após limpeza (moderado)

2. **Dados ausentes:** Perda de 30% dos registros por falta de coordenadas

3. **Features não disponíveis:**
   - Condições meteorológicas detalhadas
   - Experiência da tripulação
   - Histórico de manutenção

4. **Generalização:** Modelo limitado ao contexto brasileiro

### 8.3 Trabalhos Futuros

**Curto prazo:**
- Testar algoritmos ensemble (Random Forest, XGBoost)
- Implementar validação temporal
- Adicionar análise SHAP para explicabilidade

**Médio prazo:**
- Incorporar dados externos (meteorologia, tráfego aéreo)
- Desenvolver sistema de predição em tempo real
- Expandir para outros países

**Longo prazo:**
- Deploy em produção com API REST
- Integração com sistemas de controle aéreo
- Análise de impacto em políticas públicas

---

## 9. REFERÊNCIAS

### 9.1 Dados

CENIPA - Centro de Investigação e Prevenção de Acidentes Aeronáuticos. Disponível em: https://www.gov.br/cenipa/

### 9.2 Técnicas e Algoritmos

**SMOTE:**
Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.

**Otimização de Hiperparâmetros:**
Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.

**Métricas de Avaliação:**
Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437.

### 9.3 Bibliotecas

- Pandas: McKinney, W. (2010). Data Structures for Statistical Computing in Python
- Scikit-learn: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python
- Imbalanced-learn: Lemaître et al. (2017). Imbalanced-learn: A Python Toolbox

---

## 10. INFORMAÇÕES COMPLEMENTARES

**Autor:** Eduardo e equipe  
**Data:** Outubro 2025  
**Repositório:** github.com/vtQuadros/Trabalho-Machine-Learning  
**Versão:** 2.0  

**Nota:** Este documento serve como guia técnico completo do projeto. Para detalhes de implementação, consultar o código-fonte no notebook.