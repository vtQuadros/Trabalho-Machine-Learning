"""
Script para gerar o modelo de produção (pipeline completo)

Este script:
1. Carrega os dados de treino (`treino.csv`)
2. Aplica o pré-processamento correto (Imputers, Encoding, Scaling)
3. Aplica SMOTE para balancear 100% dos dados de treino
4. Treina o modelo de Regressão Logística com dados balanceados
5. Exporta todos os artefatos (modelo, imputers, scaler, colunas) para a API
6. Valida o pipeline final usando a base `teste.csv`

Autor: (Seu Nome / Projeto)
Data: 28/10/2025
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # <--- IMPORTANTE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
# O código corrigido
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    roc_auc_score,
    precision_score,  # <-- ADICIONE ESTA LINHA
    recall_score      # <-- E ESTA LINHA
)

print("=" * 80)
print("GERADOR DE MODELO DE PRODUÇÃO (COM SMOTE e PIPELINE COMPLETO)")
print("=" * 80)

# ============================================================================
# 1. CARREGAR DADOS DE TREINO
# ============================================================================
print("\n📁 1. CARREGANDO DADOS DE TREINO (`treino.csv`)...")
print("-" * 80)

try:
    df_treino = pd.read_csv('docs/treino.csv', encoding='latin-1', delimiter=',')
    print(f"✓ Treino: {df_treino.shape[0]} linhas, {df_treino.shape[1]} colunas")
except FileNotFoundError:
    print("❌ ERRO: `docs/treino.csv` não encontrado. Abortando.")
    exit()

# ============================================================================
# 2. PRÉ-PROCESSAMENTO (Feature Engineering e Imputação)
# ============================================================================
print("\n🔧 2. PRÉ-PROCESSAMENTO...")
print("-" * 80)

# --- Conversão de Tipos ---
df_treino['latitude'] = pd.to_numeric(
    df_treino['latitude'].astype(str).str.replace(',', '.'), 
    errors='coerce'
)
df_treino['longitude'] = pd.to_numeric(
    df_treino['longitude'].astype(str).str.replace(',', '.'), 
    errors='coerce'
)
df_treino['dt_ocorrencia'] = pd.to_datetime(df_treino['dt_ocorrencia'], format='%d/%m/%Y', errors='coerce')

# --- Feature Engineering ---
df_treino['ano_ocorrencia'] = df_treino['dt_ocorrencia'].dt.year
df_treino['mes_ocorrencia'] = df_treino['dt_ocorrencia'].dt.month
print("✓ Features de data criadas ('ano_ocorrencia', 'mes_ocorrencia')")

# --- Tratamento de Nulos (com Imputers) ---
# Remover linhas onde a data (essencial) é nula
linhas_antes = len(df_treino)
df_treino = df_treino.dropna(subset=['dt_ocorrencia', 'ano_ocorrencia', 'mes_ocorrencia'])
print(f"✓ {linhas_antes - len(df_treino)} linhas removidas por falta de data")

# Definir colunas para imputação
colunas_numericas_nan = ['peso_max_decolagem', 'numero_assentos', 'latitude', 'longitude', 'ano_ocorrencia', 'mes_ocorrencia']
colunas_categoricas_nan = ['op_padronizado', 'hr_ocorrencia', 'regiao', 'fase_operacao', 
                           'modelo_aeronave', 'nome_fabricante', 'pais_fabricante', 
                           'tipo_motor', 'espectro_dano', 'tipo_operacao']

# Garantir que apenas colunas existentes sejam processadas
colunas_numericas_nan = [col for col in colunas_numericas_nan if col in df_treino.columns]
colunas_categoricas_nan = [col for col in colunas_categoricas_nan if col in df_treino.columns]

# Criar e treinar Imputers
imputer_mediana = SimpleImputer(strategy='median')
imputer_moda = SimpleImputer(strategy='most_frequent')

df_treino[colunas_numericas_nan] = imputer_mediana.fit_transform(df_treino[colunas_numericas_nan])
df_treino[colunas_categoricas_nan] = imputer_moda.fit_transform(df_treino[colunas_categoricas_nan])

print("✓ Imputers (Mediana e Moda) treinados e aplicados.")

# ============================================================================
# 3. SEPARAR FEATURES E TARGET
# ============================================================================
print("\n📊 3. SEPARANDO FEATURES E TARGET...")
print("-" * 80)

features = ['latitude', 'longitude', 'peso_max_decolagem', 'numero_assentos',
            'fase_operacao', 'cat_aeronave', 'regiao', 'uf', 'modelo_aeronave', 
            'nome_fabricante', 'ano_ocorrencia', 'mes_ocorrencia', 
            'op_padronizado', 'hr_ocorrencia', 'pais_fabricante', 'tipo_motor', 
            'espectro_dano', 'tipo_operacao']

# Garantir que apenas features existentes sejam usadas
features_finais = [col for col in features if col in df_treino.columns]
print(f"Features selecionadas: {len(features_finais)}")

X = df_treino[features_finais]
y = df_treino['les_fatais_trip']

print(f"\nDistribuição original:")
print(f"  • Não Fatal (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
print(f" • Fatal (1):     {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")

# ============================================================================
# 4. ENCODING (One-Hot)
# ============================================================================
print("\n🔢 4. ENCODING DE VARIÁVEIS CATEGÓRICAS...")
print("-" * 80)

colunas_categoricas = ['fase_operacao', 'cat_aeronave', 'regiao', 'uf', 
                       'modelo_aeronave', 'nome_fabricante', 'op_padronizado', 
                       'hr_ocorrencia', 'pais_fabricante', 'tipo_motor', 
                       'espectro_dano', 'tipo_operacao']

colunas_categoricas_finais = [col for col in colunas_categoricas if col in X.columns]

X_encoded = pd.get_dummies(X, columns=colunas_categoricas_finais)
print(f"✓ Features após encoding: {X_encoded.shape[1]}")

# Salvar a lista de colunas do treino (ESSENCIAL PARA A API)
colunas_treino = X_encoded.columns.tolist()

# ============================================================================
# 5. NORMALIZAÇÃO (Scaling)
# ============================================================================
print("\n⚖️ 5. NORMALIZAÇÃO...")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
print("✓ Dados normalizados com StandardScaler (treinado)")

# ============================================================================
# 6. SMOTE - BALANCEAMENTO DE CLASSES
# ============================================================================
print("\n🎯 6. APLICANDO SMOTE (em 100% do treino)...")
print("-" * 80)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

print("Depois do SMOTE:")
print(f" • Não Fatal (0): {sum(y_balanced == 0)}")
print(f" • Fatal (1):     {sum(y_balanced == 1)}")
print(f"✓ Classes perfeitamente balanceadas 1:1")

# ============================================================================
# 7. TREINAMENTO DO MODELO FINAL
# ============================================================================
print("\n🤖 7. TREINAMENTO DO MODELO FINAL (COM SMOTE)...")
print("-" * 80)

modelo_logistica = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    C=1.0 # Valor padrão C=1.0 (ou o valor otimizado do notebook)
)

modelo_logistica.fit(X_balanced, y_balanced)
print("✅ Modelo final treinado com 100% dos dados de treino balanceados")

# ============================================================================
# 8. EXPORTAÇÃO DOS ARTEFATOS PARA API
# ============================================================================
print("\n💾 8. EXPORTANDO ARTEFATOS PARA A API...")
print("-" * 80)

api_dir = "api_predicao_acidentes"
if not os.path.exists(api_dir):
    os.makedirs(api_dir)

# Salvar Modelo
model_path = os.path.join(api_dir, "modelo_lr.pkl")
joblib.dump(modelo_logistica, model_path)
print(f"✅ Modelo salvo: {model_path}")

# Salvar Scaler
scaler_path = os.path.join(api_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler salvo: {scaler_path}")

# --- SALVAR OS IMPUTERS (NOVO E CRÍTICO) ---
imputer_med_path = os.path.join(api_dir, "imputer_mediana.pkl")
joblib.dump(imputer_mediana, imputer_med_path)
print(f"✅ Imputer (Mediana) salvo: {imputer_med_path}")

imputer_moda_path = os.path.join(api_dir, "imputer_moda.pkl")
joblib.dump(imputer_moda, imputer_moda_path)
print(f"✅ Imputer (Moda) salvo: {imputer_moda_path}")

# Salvar Colunas
colunas_path = os.path.join(api_dir, "colunas_treino.pkl")
joblib.dump(colunas_treino, colunas_path)
print(f"✅ Colunas salvas: {colunas_path}")

# Salvar Threshold Otimizado (do notebook)
threshold_path = os.path.join(api_dir, "threshold_otimizado.txt")
threshold_otimizado = 0.2600 # Valor da Célula 377 do notebook
f1_otimizado = 0.3673       # Valor da Célula 377 do notebook
with open(threshold_path, 'w') as f:
    f.write(f"THRESHOLD_OTIMIZADO = {threshold_otimizado}\n")
    f.write(f"F1-SCORE = {f1_otimizado}\n")
print(f"✅ Threshold otimizado salvo: {threshold_path} (Valor: {threshold_otimizado})")


# ============================================================================
# 9. VALIDAÇÃO FINAL (no `df_teste`)
# ============================================================================
print("\n📈 9. VALIDAÇÃO FINAL (usando `teste.csv`)...")
print("-" * 80)
print("Aplicando o pipeline completo nos dados de teste...")

try:
    df_teste = pd.read_csv('docs/teste.csv', encoding='latin-1', delimiter=',')
    print(f"✓ Teste: {df_teste.shape[0]} linhas, {df_teste.shape[1]} colunas")
except FileNotFoundError:
    print("❌ ERRO: `docs/teste.csv` não encontrado. Abortando validação.")
    exit()

# --- Aplicar o mesmo pipeline ---
df_teste_proc = df_teste.copy()

# 1. Conversão e Feature Eng.
df_teste_proc['latitude'] = pd.to_numeric(df_teste_proc['latitude'].astype(str).str.replace(',', '.'), errors='coerce')
df_teste_proc['longitude'] = pd.to_numeric(df_teste_proc['longitude'].astype(str).str.replace(',', '.'), errors='coerce')
df_teste_proc['dt_ocorrencia'] = pd.to_datetime(df_teste_proc['dt_ocorrencia'], format='%d/%m/%Y', errors='coerce')
df_teste_proc['ano_ocorrencia'] = df_teste_proc['dt_ocorrencia'].dt.year
df_teste_proc['mes_ocorrencia'] = df_teste_proc['dt_ocorrencia'].dt.month
df_teste_proc = df_teste_proc.dropna(subset=['dt_ocorrencia']) # Remover nulos de data

# 2. Imputação (APENAS .transform()!)
df_teste_proc[colunas_numericas_nan] = imputer_mediana.transform(df_teste_proc[colunas_numericas_nan])
df_teste_proc[colunas_categoricas_nan] = imputer_moda.transform(df_teste_proc[colunas_categoricas_nan])
print("✓ Imputers aplicados (transform)")

# 3. Features e Encoding
X_teste = df_teste_proc[features_finais]
X_teste_encoded = pd.get_dummies(X_teste, columns=colunas_categoricas_finais)

# 4. Alinhar Colunas (CRÍTICO!)
X_teste_aligned = X_teste_encoded.reindex(columns=colunas_treino, fill_value=0)
print("✓ Colunas alinhadas com o treino")

# 5. Scaling (APENAS .transform()!)
X_teste_scaled = scaler.transform(X_teste_aligned)
print("✓ Scaler aplicado (transform)")

# --- Fazer Previsões ---
y_teste_real = df_teste_proc['les_fatais_trip']
y_teste_proba = modelo_logistica.predict_proba(X_teste_scaled)[:, 1]

# Aplicar threshold otimizado
y_teste_pred = (y_teste_proba >= threshold_otimizado).astype(int)
print(f"✓ Previsões feitas usando THRESHOLD = {threshold_otimizado}")

# --- Métricas Finais ---
print("\n" + "="*60)
print(f"MÉTRICAS FINAIS NO CONJUNTO DE TESTE (Threshold={threshold_otimizado})")
print("="*60)
print(f"⚡ F1-Score (Fatal): {f1_score(y_teste_real, y_teste_pred, zero_division=0):.4f}")
print(f"🎯 Precisão (Fatal): {precision_score(y_teste_real, y_teste_pred, zero_division=0):.4f}")
print(f"🔍 Recall (Fatal):   {recall_score(y_teste_real, y_teste_pred, zero_division=0):.4f}")
print(f"📈 AUC-ROC:          {roc_auc_score(y_teste_real, y_teste_proba):.4f}")

print("\n📋 Relatório de Classificação (Teste Real):")
print(classification_report(y_teste_real, y_teste_pred, target_names=['Não Fatal (0)', 'Fatal (1)']))

print("\n🔢 Matriz de Confusão (Teste Real):")
cm = confusion_matrix(y_teste_real, y_teste_pred)
print(cm)
print(f" • Verdadeiros Positivos (TP): {cm[1,1]} (Fatais previstos corretamente)")
print(f" • Falsos Negativos (FN):    {cm[1,0]} (Fatais NÃO detectados)")


print("\n" + "=" * 80)
print("✅ SCRIPT DE GERAÇÃO DE MODELO CONCLUÍDO COM SUCESSO!")
print("Todos os artefatos estão em `api_predicao_acidentes/`")
print("=" * 80)