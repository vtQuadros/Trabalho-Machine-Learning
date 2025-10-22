# Projeto Machine Learning
## Predição de Acidentes Aéreos Fatais

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

## 📖 Visão Geral do Projeto

Este projeto, desenvolvido como trabalho acadêmico de Machine Learning, foca na construção de um modelo preditivo para **identificar se um acidente aéreo será fatal ou não-fatal** com base em características operacionais, geográficas e temporais. A análise abrange desde o pré-processamento dos dados e análise exploratória até a implementação, avaliação e otimização dos modelos preditivos.

O objetivo principal não é apenas alcançar alta acurácia, mas também **entender os padrões e fatores** que contribuem para a gravidade dos acidentes aéreos, possibilitando insights acionáveis para autoridades de aviação civil e companhias aéreas.

## 🎯 Problema de Negócio

A segurança aérea é uma preocupação fundamental no setor de aviação. Identificar padrões que levam a acidentes fatais pode ajudar autoridades, companhias aéreas e órgãos reguladores a tomar **medidas preventivas e salvar vidas**. 

O desafio técnico principal é o **desbalanceamento de classes**, onde acidentes não-fatais são aproximadamente 10 vezes mais frequentes que acidentes fatais, o que pode enviesar os modelos tradicionais.

O projeto busca responder à pergunta: **Com base nas características de um acidente aéreo (localização, tipo de aeronave, fase da operação, etc.), podemos prever se ele será fatal?**

## 📊 O Dataset

O conjunto de dados contém informações sobre acidentes aéreos no Brasil, provenientes do **CENIPA (Centro de Investigação e Prevenção de Acidentes Aeronáuticos)**. As principais features incluem:

* **Dados Geográficos:** `latitude`, `longitude`, `regiao`, `uf`.
* **Dados Temporais:** `dt_ocorrencia`, `hr_ocorrencia`, `ano_ocorrencia`, `mes_ocorrencia`.
* **Características da Aeronave:** `modelo_aeronave`, `nome_fabricante`, `cat_aeronave`, `peso_max_decolagem`, `numero_assentos`.
* **Dados Operacionais:** `fase_operacao`, `op_padronizado`.

### Variável-Alvo

A variável-alvo `les_fatais_trip` classifica os acidentes:
* **0 (Não Fatal)**: Acidente sem vítimas fatais.
* **1 (Fatal)**: Acidente com pelo menos uma vítima fatal.

**Desafio**: Dataset desbalanceado (~10:1 não-fatais vs fatais), resolvido com a técnica SMOTE.

## 🛠️ Metodologia

O projeto seguiu um fluxo estruturado de Ciência de Dados, organizado em 10 seções principais:

### 1. **Limpeza e Pré-Processamento**
* Remoção de duplicatas.
* Conversão de tipos de dados (latitude/longitude para float, datas para datetime).
* Tratamento de valores ausentes:
  * **Mediana** para variáveis numéricas.
  * **Moda** para variáveis categóricas.
  * **Remoção** de linhas com dados essenciais ausentes.

### 2. **Engenharia de Features**
* Criação de features temporais: `ano_ocorrencia` e `mes_ocorrencia`.
* Separação de features numéricas e categóricas.
* Encoding com **get_dummies** (One-Hot Encoding).
* Normalização com **StandardScaler**.

### 3. **Análise Exploratória de Dados (EDA)**
* Visualização da distribuição geográfica dos acidentes.
* Análise de tendências temporais (por ano e mês).
* Identificação do desbalanceamento de classes.
* Análise de correlações e padrões.

### 4. **Balanceamento de Classes**
* Aplicação da técnica **SMOTE (Synthetic Minority Over-sampling Technique)**.
* Resultado: Classes balanceadas 1:1 no conjunto de treino.
* Preservação do conjunto de teste original para avaliação realista.

### 5. **Modelagem e Avaliação**
* Divisão dos dados: **70% treino** e **30% teste** (estratificada).
* Três modelos foram treinados e comparados:

| Modelo | Justificativa | Características |
| :--- | :--- | :--- |
| **Baseline (Dummy)** | Referência simples (estratégia "most_frequent"). | Serve como piso mínimo de performance. |
| **Regressão Logística** | Modelo linear interpretável, treinado com SMOTE. | Bom para entender relações lineares. |
| **Árvore de Decisão** | Modelo não-linear, captura interações complexas. | Útil para identificar regras de decisão. |

* **Treinamento**: Modelos treinados com hiperparâmetros padrão do scikit-learn.
* **Avaliação**: Métricas calculadas no conjunto de teste real.

### 6. **Otimização de Threshold**
* Análise de **101 thresholds** (0.0 a 1.0) na Regressão Logística.
* Otimização do **F1-Score** para equilibrar Precisão e Recall.
* Visualização gráfica do impacto do threshold em todas as métricas.
* Identificação do threshold ótimo que maximiza o F1-Score.
* **Abordagem direta**: Sem GridSearchCV ou RandomizedSearchCV.

### 7. **Avaliação e Comparação**
* **Métricas**: Acurácia, Precisão, Recall, F1-Score, AUC-ROC.
* **Matrizes de Confusão**: Visualização detalhada de VP, VN, FP e FN.
* **Curvas ROC**: Avaliação da capacidade discriminativa com AUC.
* **Comparação Visual**: Gráficos de barras comparando todos os modelos.
* **Sem validação cruzada**: Treinamento direto no conjunto de treino.

## 📈 Resultados e Conclusões

* **Modelo Vencedor:** A **Regressão Logística com SMOTE** apresentou o melhor equilíbrio entre todas as métricas após otimização de threshold.
* **Impacto do SMOTE:** Melhorou significativamente o Recall para acidentes fatais (classe minoritária), permitindo identificar mais casos críticos.
* **Threshold Otimizado:** O ajuste do threshold maximizou o F1-Score, encontrando o ponto ideal entre Precisão e Recall.
* **Features Mais Importantes:**
  * Localização geográfica (`latitude`, `longitude`).
  * Características da aeronave (`peso_max_decolagem`, `numero_assentos`).
  * Fase da operação (`fase_operacao`).

* **Insights Acionáveis:**
  * **Regiões críticas**: Áreas com maior concentração de acidentes fatais identificadas para inspeção prioritária.
  * **Fases de risco**: Operações específicas (pouso, decolagem) requerem protocolos de segurança reforçados.
  * **Aeronaves**: Modelos e fabricantes com maior risco podem ser monitorados proativamente.
  * **Sazonalidade**: Padrões temporais identificados para planejamento de recursos.

* **Abordagem Técnica:**
  * Sem validação cruzada ou otimização de hiperparâmetros (RandomizedSearchCV/GridSearchCV).
  * Foco em **simplicidade, interpretabilidade e eficácia**.
  * Modelos treinados com configurações padrão do scikit-learn.
  * Otimização apenas no threshold de decisão da Regressão Logística.

## 🚀 Como Executar o Projeto

Para replicar esta análise, siga os passos abaixo:

### 1. **Clone o repositório:**
```bash
git clone https://github.com/vtQuadros/Trabalho-Machine-Learning.git
cd Trabalho-Machine-Learning
```

### 2. **Crie e ative um ambiente virtual (recomendado):**
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

### 4. **Execute o Jupyter Notebook:**
Abra o arquivo `projeto.ipynb` em um ambiente Jupyter para explorar a análise completa.
```bash
jupyter notebook projeto.ipynb
```

### 5. **Ou execute localmente em uma IDE de sua escolha**
O notebook é compatível com VS Code, PyCharm, e outras IDEs que suportam Jupyter.

## 📚 Documentação Adicional

**📖 Guia Completo do Projeto:**  
Para instruções detalhadas de utilização e conceitos aprofundados, consulte:
* **GitBook**: https://vitordocs.gitbook.io/machine-learning/
* **GUIA.md**: Documentação técnica completa do projeto

## 📁 Estrutura do Projeto

```
Trabalho-Machine-Learning/
│
├── projeto.ipynb           # Notebook principal com toda a análise
├── requirements.txt        # Dependências do projeto
├── GUIA.md                # Guia completo do projeto
├── README.md              # Este arquivo
│
└── docs/
    ├── treino.csv         # Dataset de treino
    ├── teste.csv          # Dataset de teste
    └── dicionario.csv     # Dicionário de variáveis
```

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Análise de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Balanceamento:** imbalanced-learn (SMOTE)
* **Ambiente:** Jupyter Notebook

## 📊 Métricas de Avaliação

O projeto utilizou múltiplas métricas para avaliação robusta:

* **Acurácia**: Proporção de predições corretas.
* **Precisão**: Proporção de predições positivas que estão corretas.
* **Recall (Sensibilidade)**: Proporção de casos positivos reais que foram identificados.
* **F1-Score**: Média harmônica entre Precisão e Recall (métrica principal).
* **AUC-ROC**: Área sob a curva ROC, mede capacidade discriminativa.

## 🎓 Conceitos Aplicados

* **Desbalanceamento de Classes**: Problema comum em classificação binária, resolvido com SMOTE.
* **One-Hot Encoding**: Conversão de variáveis categóricas em formato numérico.
* **Normalização**: Padronização de features para mesma escala.
* **Trade-off Precisão/Recall**: Equilíbrio entre evitar falsos positivos e capturar todos os positivos.
* **Threshold Customizado**: Ajuste do limiar de decisão para otimizar métricas específicas.
* **Validação Estratificada**: Manutenção da proporção de classes na divisão treino/teste.

## 📚 Referências

* **CENIPA** - Centro de Investigação e Prevenção de Acidentes Aeronáuticos
* Documentação do **Scikit-learn**: https://scikit-learn.org/
* Documentação do **imbalanced-learn**: https://imbalanced-learn.org/

## 👨‍💻 Autor

* **Eduardo**

## 📄 Licença

Este projeto é de código aberto e está disponível para fins educacionais.

## 🔗 Links Úteis

* **Repositório GitHub**: https://github.com/vtQuadros/Trabalho-Machine-Learning
* **Documentação GitBook**: https://vitordocs.gitbook.io/machine-learning/
* **Guia Técnico**: Veja `GUIA.md` para detalhes técnicos aprofundados

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!**

