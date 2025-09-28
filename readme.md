# Simulador de Piscina Térmica

Este projeto é um sistema web para simulação térmica de piscinas, permitindo estimar consumo energético, custos, e dimensionamento de sistemas de aquecimento e energia solar, considerando variáveis ambientais reais e físicas.

## Funcionalidades
- Simulação horária do balanço térmico da piscina (evaporação, convecção, radiação, condução)
- Ganho solar dinâmico com sombreamento geométrico 3D
- Consideração de chuva, vento, umidade e temperatura real
- Configuração personalizada da piscina e do sistema de aquecimento
- Visualização de gráficos interativos e relatório detalhado
- Estimativa de sistema solar fotovoltaico para compensação do consumo
- Exportação dos resultados em CSV

## Como Usar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o servidor Flask:
   ```bash
   python pool_simulator_app.py
   ```
3. Acesse o sistema via navegador em `http://localhost:5001`
4. Carregue um arquivo de configuração XML e os dados meteorológicos (CSV/XLSX)
5. Configure os parâmetros desejados e execute a simulação
6. Visualize os gráficos, relatório e exporte os resultados

## Estrutura do Projeto
- `pool_simulator_app.py`: Backend principal (Flask, simulação física)
- `templates/`: Templates HTML para interface web
- `static/`: Arquivos estáticos (JS, CSS)
- `requirements.txt`: Dependências do projeto
- `PRD.md`: Documento de requisitos do projeto

## Requisitos
- Python 3.8+
- Bibliotecas: Flask, pandas, numpy, pvlib, plotly, tqdm, shapely

## Contribuição
Pull requests são bem-vindos! Para sugestões, abra uma issue.

## Licença
Este projeto está sob a licença MIT.

---
Para dúvidas ou suporte, entre em contato com o mantenedor do projeto.