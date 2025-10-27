import requests
import json
import pandas as pd
from typing import List, Dict
import time

# ==================== CONFIGURAÇÕES ====================
API_URL = "http://localhost:8000"
DATASET_PATH = "../docs/teste.csv"

# ==================== FUNÇÕES AUXILIARES ====================

def testar_conexao():
    """Testa se a API está rodando."""
    print("=" * 80)
    print("🔍 TESTANDO CONEXÃO COM A API")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API está ONLINE e respondendo!")
            print(f"📊 Status: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ API retornou status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ERRO: Não foi possível conectar à API.")
        print("💡 Certifique-se de que a API está rodando em http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def carregar_dados_reais(n_amostras: int = 10) -> pd.DataFrame:
    """Carrega dados reais do arquivo de teste."""
    print("\n" + "=" * 80)
    print(f"📂 CARREGANDO DADOS REAIS DE: {DATASET_PATH}")
    print("=" * 80)
    
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Dataset carregado: {len(df)} registros totais")
        
        # ✅ CORREÇÃO: Converter data e extrair ano/mês
        print("🔄 Processando datas...")
        df['dt_ocorrencia'] = pd.to_datetime(df['dt_ocorrencia'], format='%d/%m/%Y', errors='coerce')
        df['ano_ocorrencia'] = df['dt_ocorrencia'].dt.year
        df['mes_ocorrencia'] = df['dt_ocorrencia'].dt.month
        
        # ✅ CORREÇÃO: Converter latitude e longitude
        print("🔄 Convertendo coordenadas...")
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        
        # ✅ Tratar valores nulos
        print("🔄 Tratando valores nulos...")
        df['peso_max_decolagem'] = df['peso_max_decolagem'].fillna(df['peso_max_decolagem'].median())
        df['numero_assentos'] = df['numero_assentos'].fillna(df['numero_assentos'].median())
        df['fase_operacao'] = df['fase_operacao'].fillna(df['fase_operacao'].mode()[0] if not df['fase_operacao'].mode().empty else 'DESCONHECIDO')
        df['cat_aeronave'] = df['cat_aeronave'].fillna(df['cat_aeronave'].mode()[0] if not df['cat_aeronave'].mode().empty else 'AVIÃO')
        df['regiao'] = df['regiao'].fillna(df['regiao'].mode()[0] if not df['regiao'].mode().empty else 'SUDESTE')
        df['modelo_aeronave'] = df['modelo_aeronave'].fillna('DESCONHECIDO')
        df['nome_fabricante'] = df['nome_fabricante'].fillna('DESCONHECIDO')
        
        # Remover linhas com dados essenciais faltando
        print("🔄 Removendo linhas inválidas...")
        df = df.dropna(subset=['latitude', 'longitude', 'les_fatais_trip', 'ano_ocorrencia', 'mes_ocorrencia'])
        
        print(f"✅ Dados processados: {len(df)} registros válidos")
        
        # Selecionar amostra
        df_amostra = df.sample(n=min(n_amostras, len(df)), random_state=42)
        
        print(f"✅ Amostra selecionada: {len(df_amostra)} registros")
        print(f"   • Fatais reais: {int(df_amostra['les_fatais_trip'].sum())}")
        print(f"   • Não fatais reais: {len(df_amostra) - int(df_amostra['les_fatais_trip'].sum())}")
        
        return df_amostra
        
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo não encontrado: {DATASET_PATH}")
        print("💡 Verifique o caminho do arquivo.")
        return None
    except Exception as e:
        print(f"❌ ERRO ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None

def preparar_acidente_para_api(row: pd.Series) -> Dict:
    """Converte uma linha do DataFrame em formato de requisição da API."""
    return {
        "latitude": float(row['latitude']),
        "longitude": float(row['longitude']),
        "peso_max_decolagem": float(row['peso_max_decolagem']),
        "numero_assentos": int(row['numero_assentos']),
        "fase_operacao": str(row['fase_operacao']),
        "cat_aeronave": str(row['cat_aeronave']),
        "regiao": str(row['regiao']),
        "uf": str(row['uf']),
        "modelo_aeronave": str(row['modelo_aeronave']),
        "nome_fabricante": str(row['nome_fabricante']),
        "ano_ocorrencia": int(row['ano_ocorrencia']),
        "mes_ocorrencia": int(row['mes_ocorrencia'])
    }

def testar_predicao_individual(df: pd.DataFrame):
    """Testa predição individual com um caso real."""
    print("\n" + "=" * 80)
    print("🧪 TESTE 1: PREDIÇÃO INDIVIDUAL")
    print("=" * 80)
    
    # Selecionar um acidente fatal e um não fatal
    df_fatal = df[df['les_fatais_trip'] == 1]
    df_nao_fatal = df[df['les_fatais_trip'] == 0]
    
    casos = []
    if len(df_fatal) > 0:
        casos.append((df_fatal.iloc[0], "FATAL"))
    if len(df_nao_fatal) > 0:
        casos.append((df_nao_fatal.iloc[0], "NÃO FATAL"))
    
    if not casos:
        print("⚠️ Nenhum caso disponível para teste individual")
        return
    
    for i, (caso, label) in enumerate(casos, 1):
        print(f"\n--- Caso {i}: Acidente REAL {label} ---")
        
        acidente = preparar_acidente_para_api(caso)
        
        print(f"📍 Local: {caso['uf']} ({caso['regiao']})")
        print(f"✈️ Aeronave: {caso['modelo_aeronave']} - {caso['nome_fabricante']}")
        print(f"📅 Data: {int(caso['mes_ocorrencia']):02d}/{int(caso['ano_ocorrencia'])}")
        print(f"🎯 Fase: {caso['fase_operacao']}")
        print(f"🏷️ Classe Real: {label}")
        
        try:
            response = requests.post(f"{API_URL}/prever", json=acidente, timeout=10)
            
            if response.status_code == 200:
                resultado = response.json()
                
                print(f"\n🤖 PREDIÇÃO DA API:")
                print(f"   • Probabilidade: {resultado['probabilidade_fatal']:.2%}")
                print(f"   • Predição: {resultado['predicao']}")
                print(f"   • Nível de Risco: {resultado['nivel_risco']}")
                print(f"   • {resultado['recomendacao']}")
                
                # Verificar acerto
                predicao_correta = (resultado['predicao'] == label)
                emoji = "✅" if predicao_correta else "❌"
                print(f"\n{emoji} Predição {'CORRETA' if predicao_correta else 'INCORRETA'}")
            else:
                print(f"❌ Erro na requisição: Status {response.status_code}")
                print(f"Resposta: {response.text}")
                
        except Exception as e:
            print(f"❌ ERRO: {e}")

def testar_predicao_lote(df: pd.DataFrame):
    """Testa predição em lote com múltiplos casos reais."""
    print("\n" + "=" * 80)
    print("🧪 TESTE 2: PREDIÇÃO EM LOTE")
    print("=" * 80)
    
    # Preparar lista de acidentes
    acidentes = [preparar_acidente_para_api(row) for _, row in df.iterrows()]
    
    print(f"📦 Enviando {len(acidentes)} acidentes para predição em lote...")
    
    try:
        inicio = time.time()
        response = requests.post(f"{API_URL}/prever_lote", json=acidentes, timeout=30)
        tempo_decorrido = time.time() - inicio
        
        if response.status_code == 200:
            resultado = response.json()
            
            print(f"\n✅ Predição em lote concluída em {tempo_decorrido:.2f} segundos!")
            print(f"   • Tempo médio por predição: {tempo_decorrido/len(acidentes):.3f}s")
            
            print(f"\n📊 ESTATÍSTICAS DO LOTE:")
            print(f"   • Total de acidentes: {resultado['total_acidentes']}")
            print(f"   • Previstos como FATAIS: {resultado['previstos_fatais']}")
            print(f"   • Previstos como NÃO FATAIS: {resultado['previstos_nao_fatais']}")
            print(f"   • Taxa de fatalidade prevista: {resultado['taxa_fatalidade_prevista']:.2f}%")
            print(f"   • Probabilidade média: {resultado['probabilidade_media']:.4f}")
            
            print(f"\n🎯 DISTRIBUIÇÃO DE RISCO:")
            for nivel, qtd in resultado['distribuicao_risco'].items():
                porcentagem = (qtd / resultado['total_acidentes'] * 100)
                print(f"   • {nivel}: {qtd} ({porcentagem:.1f}%)")
            
            # Calcular acurácia
            acertos = 0
            for i, pred in enumerate(resultado['resultados']):
                classe_real = "FATAL" if df.iloc[i]['les_fatais_trip'] == 1 else "NÃO FATAL"
                if pred['predicao'] == classe_real:
                    acertos += 1
            
            acuracia = (acertos / len(resultado['resultados'])) * 100
            print(f"\n🎯 ACURÁCIA NESTA AMOSTRA: {acuracia:.2f}%")
            print(f"   • Acertos: {acertos}/{len(resultado['resultados'])}")
            
        else:
            print(f"❌ Erro na requisição: Status {response.status_code}")
            print(f"Resposta: {response.text}")
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

def testar_metricas():
    """Testa endpoint de métricas."""
    print("\n" + "=" * 80)
    print("🧪 TESTE 3: MÉTRICAS DO MODELO")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_URL}/metricas", timeout=5)
        
        if response.status_code == 200:
            metricas = response.json()
            print(f"\n📊 MÉTRICAS DO MODELO:")
            for chave, valor in metricas.items():
                print(f"   • {chave}: {valor}")
        else:
            print(f"❌ Erro: Status {response.status_code}")
            
    except Exception as e:
        print(f"❌ ERRO: {e}")

def gerar_relatorio_completo(df: pd.DataFrame):
    """Gera relatório completo de testes."""
    print("\n" + "=" * 80)
    print("📋 RELATÓRIO COMPLETO DE TESTES")
    print("=" * 80)
    
    resultados = []
    
    print(f"🔄 Testando {len(df)} acidentes...")
    
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        acidente = preparar_acidente_para_api(row)
        
        try:
            response = requests.post(f"{API_URL}/prever", json=acidente, timeout=10)
            
            if response.status_code == 200:
                pred = response.json()
                
                resultados.append({
                    'classe_real': 'FATAL' if row['les_fatais_trip'] == 1 else 'NÃO FATAL',
                    'classe_prevista': pred['predicao'],
                    'probabilidade': pred['probabilidade_fatal'],
                    'nivel_risco': pred['nivel_risco'],
                    'acerto': pred['predicao'] == ('FATAL' if row['les_fatais_trip'] == 1 else 'NÃO FATAL')
                })
            
            # Exibir progresso
            if idx % 5 == 0:
                print(f"   Progresso: {idx}/{len(df)} ({idx/len(df)*100:.0f}%)")
                
        except Exception as e:
            print(f"   ⚠️ Erro no acidente {idx}: {e}")
            continue
    
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        
        print(f"\n🎯 DESEMPENHO GERAL:")
        print(f"   • Acurácia: {df_resultados['acerto'].mean()*100:.2f}%")
        print(f"   • Total de testes: {len(df_resultados)}")
        print(f"   • Acertos: {df_resultados['acerto'].sum()}")
        print(f"   • Erros: {len(df_resultados) - df_resultados['acerto'].sum()}")
        
        print(f"\n📊 MATRIZ DE CONFUSÃO:")
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            
            y_true = [1 if r == 'FATAL' else 0 for r in df_resultados['classe_real']]
            y_pred = [1 if r == 'FATAL' else 0 for r in df_resultados['classe_prevista']]
            
            cm = confusion_matrix(y_true, y_pred)
            print(f"\n   Verdadeiros Negativos: {cm[0,0]}")
            print(f"   Falsos Positivos: {cm[0,1]}")
            print(f"   Falsos Negativos: {cm[1,0]}")
            print(f"   Verdadeiros Positivos: {cm[1,1]}")
            
            print(f"\n📈 RELATÓRIO DE CLASSIFICAÇÃO:")
            print(classification_report(y_true, y_pred, target_names=['NÃO FATAL', 'FATAL']))
        except ImportError:
            print("   ⚠️ scikit-learn não disponível para métricas detalhadas")
    else:
        print("\n⚠️ Nenhum resultado válido para gerar relatório")

# ==================== MENU PRINCIPAL ====================

def main():
    """Função principal para executar os testes."""
    print("\n" + "=" * 80)
    print("🚀 TESTES AUTOMATIZADOS - API DE PREDIÇÃO DE ACIDENTES AÉREOS FATAIS")
    print("=" * 80)
    
    # 1. Testar conexão
    if not testar_conexao():
        print("\n💡 Inicie a API primeiro com: python api_fastapi.py")
        return
    
    # 2. Carregar dados reais
    df = carregar_dados_reais(n_amostras=20)
    
    if df is None or len(df) == 0:
        print("\n❌ Não foi possível carregar dados reais para teste.")
        return
    
    # 3. Executar testes
    testar_metricas()
    testar_predicao_individual(df)
    testar_predicao_lote(df)
    gerar_relatorio_completo(df)
    
    print("\n" + "=" * 80)
    print("✅ TESTES CONCLUÍDOS COM SUCESSO!")
    print("=" * 80)

if __name__ == "__main__":
    main()