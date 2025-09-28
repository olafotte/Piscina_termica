# PRD.md

## Nome do Projeto
Simulador de Piscina Térmica

## Objetivo
Desenvolver um sistema web para simulação térmica de piscinas residenciais e coletivas, permitindo estimar o consumo energético, custos, e dimensionamento de sistemas de aquecimento e energia solar, considerando variáveis ambientais reais e físicas.

## Problema a Ser Resolvido
- Proprietários e síndicos de piscinas não possuem ferramentas acessíveis para prever custos de aquecimento, consumo energético e impacto de fatores ambientais (chuva, vento, sombreamento, etc.).
- Falta de integração entre dados meteorológicos reais, modelagem física e visualização dos resultados.

## Público-Alvo
- Síndicos de condomínios
- Proprietários de piscinas residenciais
- Engenheiros e consultores de eficiência energética
- Empresas de aquecimento e energia solar

## Funcionalidades Principais
1. **Simulação Horária de Temperatura da Piscina**
   - Modelagem física detalhada: perdas por evaporação, convecção, radiação, condução.
   - Ganho solar dinâmico com sombreamento geométrico 3D.
   - Consideração de chuva, vento, umidade e temperatura real.
2. **Configuração Personalizada**
   - Parâmetros da piscina: dimensões, volume, capa térmica, localização.
   - Parâmetros de aquecimento: COP, programação de funcionamento, custo da energia.
3. **Visualização dos Resultados**
   - Gráficos interativos: temperatura, consumo, custos, potência requerida, balanço energético.
   - Relatório detalhado em texto.
4. **Dimensionamento de Sistema Solar**
   - Estimativa de potência de sistema fotovoltaico para compensação do consumo.
   - Cálculo de payback e custos de instalação.
5. **Exportação de Dados**
   - Exportação dos resultados em CSV.

## Requisitos Funcionais
- O usuário deve poder carregar arquivos de configuração XML e dados meteorológicos (CSV/XLSX).
- O sistema deve simular o balanço térmico hora a hora por pelo menos um ano.
- O usuário deve poder visualizar gráficos e relatórios detalhados.
- O sistema deve permitir exportar os resultados.

## Requisitos Não Funcionais
- Interface web responsiva (Flask, HTML, JS, Plotly).
- Processamento eficiente de grandes volumes de dados meteorológicos.
- Código modular e documentado.
- Suporte a múltiplos cenários/configurações.

## Critérios de Aceite
- Simulação física validada com base em literatura técnica.
- Resultados coerentes com dados reais de consumo e temperatura.
- Interface intuitiva e responsiva.
- Exportação de dados funcionando corretamente.

## Restrições
- O sistema depende de dados meteorológicos reais para maior precisão.
- O cálculo de sombreamento exige configuração geométrica detalhada dos blocos/edifícios.

## Referências
- pvlib (modelagem solar)
- Normas técnicas de piscinas e aquecimento
- Artigos científicos sobre balanço térmico de piscinas

## Roadmap (Resumo)
- MVP: Simulação física + gráficos + relatório
- V2: Exportação CSV, múltiplos cenários, integração com API meteorológica
- V3: Visualização 3D, integração com sistemas de automação

---
Este documento descreve os requisitos e visão do projeto Simulador de Piscina Térmica. Atualize conforme novas funcionalidades forem implementadas.