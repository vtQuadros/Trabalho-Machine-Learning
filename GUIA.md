# 🚀 Predição de Acidentes Aéreos Fatais - Machine Learning

---

## 📋 **SUMÁRIO EXECUTIVO**

| **Seção** | **Descrição** | **Status** |
|-----------|---------------|------------|
| 1. Setup | Imports e configurações | ✅ |
| 2. Preparação de Dados | Limpeza e tratamento | ✅ |
| 3. Análise Exploratória | Visualização e balanceamento | ✅ |
| 4. Modelagem Avançada | Pré-processamento, SMOTE e otimização | ✅ |
| 5. Treinamento | Múltiplos modelos e comparação | ✅ |
| 6. Avaliação | Métricas, matrizes e curvas ROC | ✅ |
| 7. Apresentação | Visualizações profissionais | ✅ |

---

## 🎯 **OBJETIVO DO PROJETO**

Desenvolver um modelo de Machine Learning capaz de **prever se um acidente aéreo será fatal ou não-fatal** com base em características operacionais, geográficas e temporais.

---

## 📊 **RESULTADOS PRINCIPAIS**

- **🏆 Melhor Modelo:** Regressão Logística Otimizada
- **📈 F1-Score:** ~0.70+ (após otimização com SMOTE e threshold customizado)
- **🎯 Técnicas Aplicadas:** 
  - PowerTransformer + StandardScaler
  - SMOTE para balanceamento
  - RandomizedSearchCV (20 iterações)
  - Threshold customizado
  - Validação cruzada (5-folds)

---

## 📁 **DATASET**

- **Fonte:** CENIPA (Centro de Investigação e Prevenção de Acidentes Aeronáuticos)
- **Registros:** ~1.500 acidentes aéreos no Brasil
- **Período:** Multi-anual
- **Features:** 12 variáveis (geográficas, temporais, características de aeronaves)
- **Target:** `les_fatais_trip` (0 = Não Fatal, 1 = Fatal)
- **Desbalanceamento:** ~10:1 (resolvido com SMOTE)

---

## 🔧 **TÉCNICAS AVANÇADAS IMPLEMENTADAS**

### ✅ 1. Pré-processamento Avançado
- **PowerTransformer**: Normaliza distribuições não-gaussianas
- **StandardScaler**: Padronização (média 0, desvio 1)
- **OneHotEncoder**: Com `drop='first'` para evitar multicolinearidade

### ✅ 2. Balanceamento Inteligente
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Resultado**: 214 exemplos não-fatais → 214 exemplos fatais (1:1)

### ✅ 3. Otimização de Hiperparâmetros
- **Algoritmo**: RandomizedSearchCV
- **Espaço de busca**: 1.200 combinações
- **Iterações**: 20 combinações testadas
- **Métrica**: F1-Score (equilíbrio entre Precisão e Recall)

### ✅ 4. Análise de Threshold Customizado
- **Testes**: 101 thresholds (0.0 a 1.0)
- **Otimização**: Maximização do F1-Score
- **Resultado**: Threshold otimizado > 0.5 (padrão)

### ✅ 5. Validação Robusta
- **Validação Cruzada**: 5-folds
- **Métricas**: Acurácia, Precisão, Recall, F1-Score, AUC-ROC

---

## 📚 **ESTRUTURA DO NOTEBOOK**

1. **Setup e Configurações** (Células 1-2)
   - Imports consolidados
   - Configurações globais

2. **Preparação dos Dados** (Células 3-10)
   - Carregamento
   - Remoção de duplicatas
   - Conversão de tipos
   - Tratamento de valores nulos
   - Engenharia de features

3. **Preparação para Modelagem** (Células 11-13)
   - Seleção de features
   - Divisão train/test
   - Análise de balanceamento

4. **Modelagem Avançada** (Células 14-22)
   - Pré-processamento avançado
   - Balanceamento com SMOTE
   - Otimização de hiperparâmetros
   - Análise de threshold
   - Treinamento dos modelos

5. **Avaliação e Análise** (Células 23-25)
   - Comparação de métricas
   - Matrizes de confusão
   - Curvas ROC

6. **Apresentação Final** (Células 26-31)
   - Distribuição geográfica
   - Análise temporal
   - Comparação de modelos
   - Importância de features
   - Dashboard executivo

---

## 🎓 **CONCEITOS-CHAVE**

- **F1-Score**: Média harmônica entre Precisão e Recall (métrica principal para datasets desbalanceados)
- **SMOTE**: Cria exemplos sintéticos da classe minoritária para balancear o dataset
- **Threshold**: Limiar de decisão para classificação binária (padrão = 0.5)
- **AUC-ROC**: Área sob a curva ROC, mede a capacidade do modelo de separar classes
- **Validação Cruzada**: Técnica que divide os dados em K partes para avaliar robustez

---

## 💡 **PRINCIPAIS INSIGHTS**

1. **Desbalanceamento** é crítico em problemas de classificação binária
2. **SMOTE** melhora significativamente o Recall para a classe minoritária
3. **Threshold customizado** pode otimizar o trade-off Precisão/Recall
4. **Features geográficas** (latitude/longitude) são importantes para predição
5. **Validação cruzada** garante que o modelo não está apenas "decorando" os dados

---

**👨‍💻 Autores:** Eduardo e equipe  
**📅 Data:** Outubro 2025  
**🔗 Repositório:** [github.com/vtQuadros/Trabalho-Machine-Learning](https://github.com/vtQuadros/Trabalho-Machine-Learning)

---