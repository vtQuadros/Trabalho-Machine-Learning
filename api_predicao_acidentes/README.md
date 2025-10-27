# API de Previsão de Acidentes Aéreos Fatais

## 📋 Descrição

API desenvolvida com FastAPI para disponibilizar em produção o modelo de Machine Learning treinado para prever se um acidente aéreo será fatal ou não, com base em características do voo, da aeronave e do local do acidente.

## 🎯 Características

- **Modelo**: Regressão Logística (otimizada com threshold personalizado)
- **Entrada**: Dados geográficos, temporais e características da aeronave
- **Saída**: Probabilidade de fatalidade e classificação (fatal/não fatal)
- **Framework**: FastAPI com validação automática de dados (Pydantic)

## 🚀 Como Usar

### 1. Pré-requisitos

Execute primeiro o notebook `projeto.ipynb` até a célula de **Exportação do Modelo** para gerar os arquivos necessários:
- `modelo_lr.pkl` - Modelo treinado
- `scaler.pkl` - Scaler para normalização
- `colunas_treino.pkl` - Colunas após encoding
- `threshold_otimizado.txt` - Threshold ideal

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Executar a API

```bash
cd api_predicao_evasao
uvicorn api_fastapi:app --reload
```

A API estará disponível em: `http://127.0.0.1:8000`

### 4. Documentação Interativa

Acesse a documentação automática (Swagger UI):
```
http://127.0.0.1:8000/docs
```

## 📡 Endpoints

### `GET /`
Informações básicas sobre a API

### `GET /status/`
Verificar status e disponibilidade do modelo

**Resposta:**
```json
{
  "status": "ok",
  "modelo_carregado": true,
  "threshold": 0.35,
  "features_esperadas": 145
}
```

### `POST /prever/`
Realizar predição de fatalidade

**Exemplo de requisição:**
```json
{
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
```

**Resposta:**
```json
{
  "fatal": false,
  "probabilidade": 0.2145,
  "probabilidade_percentual": "21.45%",
  "interpretacao": "Baixo risco de fatalidade (21.4%).",
  "threshold_utilizado": 0.35
}
```

## 🔍 Variáveis de Entrada

### Variáveis Numéricas:
- `latitude`: Latitude do local do acidente
- `longitude`: Longitude do local do acidente
- `peso_max_decolagem`: Peso máximo de decolagem da aeronave (kg)
- `numero_assentos`: Número de assentos da aeronave
- `ano_ocorrencia`: Ano do acidente
- `mes_ocorrencia`: Mês do acidente (1-12)

### Variáveis Categóricas:
- `fase_operacao`: Fase do voo (ex: "DECOLAGEM", "POUSO", "CRUZEIRO")
- `cat_aeronave`: Categoria da aeronave (ex: "AVIAO", "HELICOPTERO")
- `regiao`: Região do Brasil (ex: "SUDESTE", "SUL", "NORTE")
- `uf`: Unidade Federativa (sigla do estado)
- `modelo_aeronave`: Modelo da aeronave
- `nome_fabricante`: Nome do fabricante

## 🧪 Testando a API

### Via cURL:
```bash
curl -X POST "http://127.0.0.1:8000/prever/" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Via Python:
```python
import requests

url = "http://127.0.0.1:8000/prever/"
dados = {
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

response = requests.post(url, json=dados)
print(response.json())
```

## 📊 Interpretação dos Resultados

A API retorna:
- **fatal**: `true` se a probabilidade for maior que o threshold otimizado
- **probabilidade**: Valor entre 0 e 1 (probabilidade de fatalidade)
- **interpretacao**: Mensagem explicativa baseada no risco:
  - **Alto risco** (> threshold): Requer medidas preventivas
  - **Risco moderado** (0.25 - threshold): Cautela recomendada
  - **Baixo risco** (< 0.25): Situação mais segura

## ⚙️ Configuração do Threshold

O threshold otimizado é definido no arquivo `api_fastapi.py`:

```python
THRESHOLD_OTIMIZADO = 0.35  # Ajustar conforme resultado do notebook
```

Este valor deve ser atualizado após executar a otimização no notebook.

## 🛠️ Estrutura de Arquivos

```
api_predicao_evasao/
│
├── api_fastapi.py          # Código principal da API
├── modelo_lr.pkl           # Modelo treinado (gerado pelo notebook)
├── scaler.pkl              # Scaler (gerado pelo notebook)
├── colunas_treino.pkl      # Colunas após encoding (gerado pelo notebook)
├── threshold_otimizado.txt # Threshold ideal (gerado pelo notebook)
├── requirements.txt        # Dependências
└── README.md              # Esta documentação
```

## 📝 Observações Importantes

1. **Execute o notebook primeiro**: Os arquivos `.pkl` são gerados durante o treinamento
2. **Atualize o threshold**: Após otimização no notebook, atualize o valor na API
3. **Encoding automático**: A API trata automaticamente o one-hot encoding das variáveis categóricas
4. **Validação de entrada**: Pydantic valida automaticamente os tipos de dados

## 👥 Equipe do Projeto

| RA      | Nome                 |
|---------|----------------------|
| 1134868 | Ábner Panazollo      |
| 1134433 | Ariel Diefenthaeler  |
| 1134933 | Eduardo Sichelero    |
| 1134890 | Gabriel Duarte       |
| 1135384 | Gabriel Onofre       |
| 1134821 | Vitor Quadros        |

## 📄 Licença

Projeto acadêmico - Machine Learning & IA
