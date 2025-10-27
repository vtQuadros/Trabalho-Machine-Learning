"""
Script de Teste para API de Previsão de Acidentes Aéreos Fatais

Este script testa a API localmente com diferentes cenários de acidentes.
"""

import requests
import json

# URL da API (ajuste se necessário)
BASE_URL = "http://127.0.0.1:8000"

def testar_status():
    """Testar endpoint de status"""
    print("=" * 70)
    print("TESTANDO STATUS DA API")
    print("=" * 70)
    
    try:
        response = requests.get(f"{BASE_URL}/status/")
        print(f"Status Code: {response.status_code}")
        print(f"Resposta: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        return False

def testar_predicao(nome_teste, dados):
    """Testar endpoint de predição"""
    print("\n" + "=" * 70)
    print(f"TESTE: {nome_teste}")
    print("=" * 70)
    print(f"Dados enviados:")
    print(json.dumps(dados, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(f"{BASE_URL}/prever/", json=dados)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            resultado = response.json()
            print(f"\n✅ RESULTADO DA PREDIÇÃO:")
            print(f"   • Fatal: {resultado['fatal']}")
            print(f"   • Probabilidade: {resultado['probabilidade']} ({resultado['probabilidade_percentual']})")
            print(f"   • Interpretação: {resultado['interpretacao']}")
            print(f"   • Threshold usado: {resultado['threshold_utilizado']}")
        else:
            print(f"❌ Erro: {response.json()}")
            
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")

def main():
    """Executar todos os testes"""
    
    print("\n🚀 INICIANDO TESTES DA API DE PREVISÃO DE ACIDENTES AÉREOS\n")
    
    # Teste 1: Status da API
    if not testar_status():
        print("\n❌ API não está disponível. Certifique-se de que está rodando com:")
        print("   uvicorn api_fastapi:app --reload")
        return
    
    # Teste 2: Cenário de baixo risco
    testar_predicao(
        "Cenário 1 - Baixo Risco (Aeronave moderna, região desenvolvida)",
        {
            "latitude": -23.5505,
            "longitude": -46.6333,
            "peso_max_decolagem": 5700.0,
            "numero_assentos": 9,
            "ano_ocorrencia": 2020,
            "mes_ocorrencia": 6,
            "fase_operacao": "CRUZEIRO",
            "cat_aeronave": "AVIAO",
            "regiao": "SUDESTE",
            "uf": "SP",
            "modelo_aeronave": "EMB-810C",
            "nome_fabricante": "EMBRAER"
        }
    )
    
    # Teste 3: Cenário de risco moderado
    testar_predicao(
        "Cenário 2 - Risco Moderado (Fase crítica, aeronave menor)",
        {
            "latitude": -15.7801,
            "longitude": -47.9292,
            "peso_max_decolagem": 3500.0,
            "numero_assentos": 4,
            "ano_ocorrencia": 2015,
            "mes_ocorrencia": 12,
            "fase_operacao": "DECOLAGEM",
            "cat_aeronave": "AVIAO",
            "regiao": "CENTRO-OESTE",
            "uf": "DF",
            "modelo_aeronave": "C-152",
            "nome_fabricante": "CESSNA"
        }
    )
    
    # Teste 4: Cenário de alto risco
    testar_predicao(
        "Cenário 3 - Alto Risco (Fase perigosa, helicóptero, região isolada)",
        {
            "latitude": -3.7327,
            "longitude": -38.5270,
            "peso_max_decolagem": 2200.0,
            "numero_assentos": 5,
            "ano_ocorrencia": 2010,
            "mes_ocorrencia": 1,
            "fase_operacao": "POUSO",
            "cat_aeronave": "HELICOPTERO",
            "regiao": "NORDESTE",
            "uf": "CE",
            "modelo_aeronave": "R22",
            "nome_fabricante": "ROBINSON"
        }
    )
    
    # Teste 5: Cenário com dados incomuns
    testar_predicao(
        "Cenário 4 - Dados Variados (Aeronave grande, região sul)",
        {
            "latitude": -30.0346,
            "longitude": -51.2177,
            "peso_max_decolagem": 15000.0,
            "numero_assentos": 50,
            "ano_ocorrencia": 2022,
            "mes_ocorrencia": 3,
            "fase_operacao": "TAXI",
            "cat_aeronave": "AVIAO",
            "regiao": "SUL",
            "uf": "RS",
            "modelo_aeronave": "ERJ-145",
            "nome_fabricante": "EMBRAER"
        }
    )
    
    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS!")
    print("=" * 70)
    print("\n💡 Dicas:")
    print("   • Valores de probabilidade próximos ao threshold são mais incertos")
    print("   • A interpretação ajuda a entender o nível de risco")
    print("   • Use a documentação interativa em http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
