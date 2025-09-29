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
2. Edite um arquivo `config.xml` para definir a geometria da sua piscina, os parâmetros de simulação e os caminhos para os arquivos de dados (`chuva.xlsx` e `Outdoor Thermometer_export_*.csv`).
3. Execute o servidor Flask:
   ```bash
   python pool_simulator_app.py
   ```
4. Acesse o sistema via navegador em `http://localhost:5001`
5. Selecione o arquivo de configuração na interface.
6. Configure os parâmetros desejados e execute a simulação.
7. Visualize os gráficos, relatório e exporte os resultados.

## Arquivos auxiliares

- `visualizador.py`: Arquivo auxiliar para visualizar o esquema 3D de sol e sombras.
- `calculo_piscina.py`: Rotina simples para balanço energético.
- `config.xml`: Arquivo de configuração da piscina, aquecimento e localização dos arquivos de dados.
- `Outdoor Thermometer_export_202508100845.csv`: Exemplo de arquivo de dados meteorológicos.
- `chuva.xlsx`: Exemplo de arquivo de dados de chuva. Dados da defesa civil de Blumenau.

## Estrutura do Projeto
- `pool_simulator_app.py`: Backend principal (Flask, simulação física)
- `templates/`: Templates HTML para interface web
- `static/`: Arquivos estáticos (JS, CSS)
- `requirements.txt`: Dependências do projeto
- `PRD.md`: Documento de requisitos do projeto

## Requisitos
- Python 3.8+
- Bibliotecas: Flask, pandas, numpy, pvlib, plotly, tqdm, shapely

## Histórico de Alterações

### 28/09/2025 - Configuração Dinâmica de Arquivos de Dados
- **Refatoração**: O caminho para os arquivos de dados de chuva (`chuva.xlsx`) e de temperatura (`Outdoor Thermometer_export_*.csv`) foi removido do código fonte (`pool_simulator_app.py`).
- **Nova Funcionalidade**: Os caminhos para esses arquivos agora são especificados no arquivo `config.xml`, dentro da tag `<data_files>`. Isso permite que o usuário utilize diferentes arquivos de dados sem precisar alterar o código.

Exemplo da nova seção em `config.xml`:
```xml
<data_files>
    <chuva_file>chuva.xlsx</chuva_file>
    <thermometer_file>Outdoor Thermometer_export_202508100845.csv</thermometer_file>
</data_files>
```

## Contribuição
Pull requests são bem-vindos! Para sugestões, abra uma issue.

## Licença
Este projeto está sob a licença MIT.

---
Para dúvidas ou suporte, entre em contato com o mantenedor do projeto.