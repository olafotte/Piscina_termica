# -*- coding: utf-8 -*-
import math

# ==============================================================================
# CONSTANTES FÍSICAS
# ==============================================================================
# Densidade da água em kg/m³
DENSIDADE_AGUA = 1000.0
# Calor específico da água em J/(kg·°C)
CALOR_ESPECIFICO_AGUA = 4186.0
# Fator de absorção da água para radiação solar (adimensional)
FATOR_ABSORCAO_SOLAR = 0.90

# ==============================================================================
# FUNÇÕES DE CÁLCULO DE PERDA DE CALOR
# ==============================================================================

def calcular_pressao_saturacao_vapor(T):
    """Calcula a pressão de saturação do vapor de água."""
    return 610.94 * math.exp((17.625 * T) / (243.04 + T))

def calcular_perda_calor_evaporacao(area_piscina, temp_agua, temp_ar, umidade_relativa, vel_vento):
    """Calcula a perda de calor por evaporação em Watts para uma PISCINA ABERTA."""
    h_fg = 2.4 * 1e6
    a = 3.8e-9
    b = 3.1e-9

    Pw = calcular_pressao_saturacao_vapor(temp_agua)
    Ps_ar = calcular_pressao_saturacao_vapor(temp_ar)
    Pa = Ps_ar * (umidade_relativa / 100.0)

    if Pw > Pa:
        q_evap = area_piscina * (a + b * vel_vento) * (Pw - Pa) * h_fg
    else:
        q_evap = 0.0
    return q_evap

def calcular_perda_calor_conveccao(area_piscina, temp_agua, temp_ar, vel_vento):
    """Calcula a perda de calor por convecção em Watts para uma PISCINA ABERTA."""
    hc = 5.7 + 3.8 * vel_vento
    
    if temp_agua > temp_ar:
        q_conv = hc * area_piscina * (temp_agua - temp_ar)
    else:
        q_conv = 0.0
    return q_conv

# ==============================================================================
# FUNÇÃO PRINCIPAL E INTERAÇÃO COM O USUÁRIO
# ==============================================================================

def obter_input_numerico(prompt):
    """Função auxiliar para garantir que o usuário digite um número positivo."""
    while True:
        try:
            valor = float(input(prompt))
            if valor >= 0:
                return valor
            else:
                print("Por favor, insira um valor positivo.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def main():
    """Função principal para executar o programa."""
    print("=====================================================")
    print("   Calculadora de Balanço Energético de Piscinas")
    print("=====================================================")
    print("\nPor favor, insira os dados abaixo:")

    # --- Coleta de Dados da Piscina ---
    comprimento = obter_input_numerico("Comprimento da piscina (metros): ")
    largura = obter_input_numerico("Largura da piscina (metros): ")
    profundidade = obter_input_numerico("Profundidade MÉDIA da piscina (metros): ")
    
    area_superficie = comprimento * largura
    volume_agua = area_superficie * profundidade
    massa_agua = volume_agua * DENSIDADE_AGUA
    
    print(f"-> Área da superfície calculada: {area_superficie:.2f} m²")
    print(f"-> Volume de água calculado: {volume_agua:.2f} m³ ({volume_agua*1000:,.0f} litros)")

    # --- Coleta de Condições da Cobertura ---
    fator_cobertura = -1.0
    while not 0.0 <= fator_cobertura <= 1.0:
        fator_cobertura = obter_input_numerico(
            "Fator de cobertura (1.0 = sem cobertura, 0.1-0.3 = com cobertura): "
        )
        if not 0.0 <= fator_cobertura <= 1.0:
            print("Valor inválido. O fator deve ser entre 0.0 e 1.0.")

    # --- Coleta de Condições Climáticas e da Água ---
    temp_agua_inicial = obter_input_numerico("Temperatura ATUAL da água (°C): ")
    temp_ar = obter_input_numerico("Temperatura média do ar ambiente (°C): ")
    
    umidade_do_ar = -1
    while not 0 <= umidade_do_ar <= 100:
        umidade_do_ar = obter_input_numerico("Umidade relativa do ar (%): ")
        if not 0 <= umidade_do_ar <= 100:
            print("A umidade deve ser um valor entre 0 e 100.")

    vel_vento_kmh = obter_input_numerico("Velocidade média do vento (km/h): ")
    vel_vento_ms = vel_vento_kmh / 3.6

    irradiancia_solar = obter_input_numerico(
        "Irradiância solar média na superfície (W/m²)\n"
        "(Ex: 0 noite, 300-600 nublado, 700-1000+ céu limpo): "
    )

    # --- Cálculos ---
    # 1. Ganho de calor
    ganho_solar_watts = irradiancia_solar * area_superficie * FATOR_ABSORCAO_SOLAR

    # 2. Perdas de calor (calculadas como se estivesse aberta e depois ajustadas)
    perda_evap_base = calcular_perda_calor_evaporacao(
        area_piscina=area_superficie, temp_agua=temp_agua_inicial, temp_ar=temp_ar,
        umidade_relativa=umidade_do_ar, vel_vento=vel_vento_ms
    )
    perda_conv_base = calcular_perda_calor_conveccao(
        area_piscina=area_superficie, temp_agua=temp_agua_inicial,
        temp_ar=temp_ar, vel_vento=vel_vento_ms
    )
    
    # 3. Ajuste das perdas com o fator de cobertura
    perda_evaporacao_final = perda_evap_base * fator_cobertura
    perda_conveccao_final = perda_conv_base * fator_cobertura
    perda_total_watts = perda_evaporacao_final + perda_conveccao_final

    # 4. Balanço energético
    potencia_liquida_watts = ganho_solar_watts - perda_total_watts
    
    # --- Apresentação dos Resultados ---
    print("\n==============================================")
    print("           BALANÇO ENERGÉTICO ATUAL          ")
    print("==============================================")
    print(f"Condição da Cobertura: Fator {fator_cobertura:.2f} ({((1-fator_cobertura)*100):.0f}% de redução de perdas)")
    print("----------------------------------------------")
    print(f"(+) Ganho de Calor Solar:  {ganho_solar_watts:,.2f} Watts")
    print(f"(-) Perda por Evaporação:  {perda_evaporacao_final:,.2f} Watts")
    print(f"(-) Perda por Convecção:   {perda_conveccao_final:,.2f} Watts")
    print("----------------------------------------------")
    
    if potencia_liquida_watts >= 0:
        print(f"BALANÇO LÍQUIDO:           GANHO de {potencia_liquida_watts:,.2f} Watts")
    else:
        print(f"BALANÇO LÍQUIDO:           PERDA de {-potencia_liquida_watts:,.2f} Watts")
    
    # --- Cálculo da Projeção de Temperatura ---
    print("\n==============================================")
    print("      PROJEÇÃO DE TEMPERATURA APÓS 1 HORA     ")
    print("==============================================")
    
    energia_liquida_joules = potencia_liquida_watts * 3600
    
    if massa_agua > 0:
        variacao_temperatura = energia_liquida_joules / (massa_agua * CALOR_ESPECIFICO_AGUA)
        temp_final = temp_agua_inicial + variacao_temperatura
        
        resultado_variacao = f"Aumento de {variacao_temperatura:.2f} °C" if variacao_temperatura >= 0 else f"Queda de {-variacao_temperatura:.2f} °C"
        
        print(f"Balanço de energia em 1 hora: {energia_liquida_joules / 1e6:,.2f} Megajoules")
        print(f"Variação de temp. estimada:   {resultado_variacao}")
        print("----------------------------------------------")
        print(f"Temperatura final após 1 hora:  {temp_final:.2f} °C")
        
    else:
        print("Massa de água é zero. Não é possível calcular a variação de temperatura.")

    print("\nAVISOS IMPORTANTES:")
    print("1. Este cálculo assume que as condições (vento, sol, etc.) permanecem constantes.")
    print("2. O fator de cobertura aqui reduz apenas as PERDAS. Uma cobertura real também")
    print("   reduz o GANHO solar. Para simular isso, use um valor menor de irradiância.")
    print("==============================================")


# Executa a função principal quando o script é iniciado
if __name__ == "__main__":
    main()