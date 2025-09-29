# pool_simulator_app.py

import pandas as pd
import numpy as np
import math
import itertools
import os
import glob
from collections import Counter
from flask import Flask, render_template, request, jsonify, flash, session
import plotly.graph_objects as go
import plotly.io as pio
import pvlib
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
import xml.etree.ElementTree as ET

# --- CONFIGURATION LOADERS ----------------------------------------------------------

def get_xml_metadata(file_path):
    """Parses just the metadata from a config XML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        description_node = root.find('.//description')
        date_node = root.find('.//creation_date')
        description = description_node.text if description_node is not None else "N/A"
        creation_date = date_node.text if date_node is not None else "N/A"
        return {"description": description, "creation_date": creation_date}
    except (ET.ParseError, FileNotFoundError):
        return {"description": "Erro ao ler", "creation_date": ""}

def load_config_from_xml(file_path):
    """Parses a config XML file and returns a dictionary of configuration variables."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        config = {}

        def to_num(s):
            try: return int(s)
            except (ValueError, TypeError):
                try: return float(s)
                except (ValueError, TypeError): return s

        # Metadata
        meta_root = root.find('metadata')
        config['metadata'] = {
            'description': meta_root.find('description').text,
            'creation_date': meta_root.find('creation_date').text
        } if meta_root is not None else {'description': 'N/A', 'creation_date': 'N/A'}

        # Location
        loc_root = root.find('location')
        config['location'] = {
            'latitude': to_num(loc_root.find('latitude').text),
            'longitude': to_num(loc_root.find('longitude').text),
            'altitude': to_num(loc_root.find('altitude').text),
            'timezone': loc_root.find('timezone').text
        }

        # Scene Config
        scene_root = root.find('scene')
        pool_attrs = {k: to_num(v) for k, v in scene_root.find('pool').attrib.items()}
        blocks = [{k: to_num(v) for k, v in block_node.attrib.items()} for block_node in scene_root.find('blocks').findall('block')]
        config['scene_config'] = {'pool': pool_attrs, 'blocks': blocks}

        # Visualization Config
        vis_root = root.find('visualization')
        vis_scene = {k: to_num(v) for k, v in vis_root.find('scene').attrib.items()}
        vis_ground = {k: to_num(v) for k, v in vis_root.find('ground').attrib.items()}
        vis_pool = {k: to_num(v) for k, v in vis_root.find('pool_visual').attrib.items()}
        cam_root = vis_root.find('camera')
        cam_pos = {k: to_num(v) for k, v in cam_root.find('initialPosition').attrib.items()}
        vis_camera = {k: to_num(v) for k, v in cam_root.attrib.items()}
        vis_camera['initialPosition'] = cam_pos
        sun_root = vis_root.find('sun')
        sun_cam = {k: to_num(v) for k, v in sun_root.find('shadowCamera').attrib.items()}
        vis_sun = {k: to_num(v) for k, v in sun_root.attrib.items()}
        vis_sun['shadowCamera'] = sun_cam
        config['visualization_config'] = {
            'scene': vis_scene, 'ground': vis_ground, 'camera': vis_camera,
            'sun': vis_sun, 'pool_visual': vis_pool
        }

        # Simulation Defaults
        sim_defaults_root = root.find('simulation_defaults')
        config['default_params'] = {child.tag: to_num(child.text) for child in sim_defaults_root}

        # Data Files
        data_files_root = root.find('data_files')
        if data_files_root is not None:
            config['data_files'] = {
                'chuva_file': data_files_root.find('chuva_file').text,
                'thermometer_file': data_files_root.find('thermometer_file').text
            }
        else:
            # Fallback to hardcoded values if not in config
            config['data_files'] = {
                'chuva_file': 'chuva.xlsx',
                'thermometer_file': 'Outdoor Thermometer_export_202508100845.csv'
            }

        return config
    except (ET.ParseError, FileNotFoundError, AttributeError) as e:
        flash(f"Erro crítico ao carregar o arquivo de configuração '{file_path}': {e}", "error")
        return None

# --- CORE LOGIC CLASSES -------------------------------------------------------------

class ShadingCalculator:
    """Encapsulates the geometric shadow calculation logic."""
    def __init__(self, scene_config):
        self.scene_config = scene_config
        self.pool_polygon = self._create_pool_polygon()

    def _create_pool_polygon(self):
        pool_config = self.scene_config['pool']
        pool_rot_rad = np.deg2rad(pool_config['rotation'])
        c, s = np.cos(pool_rot_rad), np.sin(pool_rot_rad)
        rot_matrix = np.array([[c, -s], [s, c]])
        half_l, half_w = pool_config['length'] / 2, pool_config['width'] / 2
        corners = [(-half_w, half_l), (half_w, half_l), (half_w, -half_l), (-half_w, -half_l)]
        world_corners = [rot_matrix @ p for p in corners]
        mirrored_corners = [(-corner[0], corner[1]) for corner in world_corners]
        return Polygon(mirrored_corners)

    def _get_shadow_polygon(self, block_config, sun_azimuth, sun_elevation):
        if sun_elevation <= 0: return None
        w, h, l = block_config['width'], block_config['height'], block_config['length']
        x_c, z_c, angle = -block_config['x'], -block_config['z'], block_config['angle']
        
        angle_rad = np.deg2rad(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[c, -s], [s, c]])
        half_w, half_l = w / 2, l / 2
        local_verts = [np.array([-half_w, half_l]), np.array([half_w, half_l]), np.array([half_w, -half_l]), np.array([-half_w, -half_l])]
        
        world_verts_2d = []
        for p in local_verts:
            rotated_p = rot_matrix @ p
            mirrored_world_p = np.array([-rotated_p[0] + x_c, rotated_p[1] + z_c])
            world_verts_2d.append(mirrored_world_p)

        world_verts_3d = [np.array([v[0], 0, v[1]]) for v in world_verts_2d] + [np.array([v[0], h, v[1]]) for v in world_verts_2d]

        sun_elev_rad, sun_azim_rad = np.deg2rad(sun_elevation), np.deg2rad(sun_azimuth)
        if np.tan(sun_elev_rad) < 1e-6: return None
        cot_elev = 1.0 / np.tan(sun_elev_rad)
        shadow_dx = cot_elev * np.sin(sun_azim_rad)
        shadow_dz = -cot_elev * np.cos(sun_azim_rad)

        shadow_points = [(v[0] + v[1] * shadow_dx, v[2] + v[1] * shadow_dz) for v in world_verts_3d]
        return MultiPoint(shadow_points).convex_hull

    def calculate_hourly_shading(self, solar_position):
        if self.pool_polygon.area == 0: return pd.Series([1.0] * len(solar_position), index=solar_position.index)
        
        sun_factors = []
        for _, sun in tqdm(solar_position.iterrows(), total=len(solar_position), desc="Calculando Sombreamento Geométrico"):
            if sun['elevation'] <= 0:
                sun_factors.append(0.0)
                continue

            shadows = [self._get_shadow_polygon(b, sun['azimuth'], sun['elevation']) for b in self.scene_config['blocks']]
            valid_shadows = [s for s in shadows if s and s.is_valid]

            if not valid_shadows:
                sun_factors.append(1.0)
                continue

            total_shadow = unary_union(valid_shadows)
            shaded_intersection = self.pool_polygon.intersection(total_shadow)
            sun_factor = 1.0 - (shaded_intersection.area / self.pool_polygon.area)
            sun_factors.append(sun_factor)

        return pd.Series(sun_factors, index=solar_position.index)

class PoolSimulator:
    """Handles the core thermal simulation for the pool."""
    def __init__(self, params, weather_data):
        self.params = params
        self.weather_data = weather_data
        self.pool_area = params['pool_length'] * params['pool_width']
        self.sim_data = None

    def _calculate_cop(self, t):
        p = self.params
        if not (10 < t < 30): return p['cop_min'] if t <= 10 else p['cop_max']
        return p['cop_min'] + (t - 10) * (p['cop_max'] - p['cop_min']) / 20.0

    def _pressao_saturacao_vapor(self, T):
        # T em °C, retorna Pa
        return 610.94 * np.exp((17.625 * T) / (243.04 + T))
    
    def _perda_calor_conducao(self,comprimento, largura, profundidade, temp_agua_c, temp_solo_c):
        
        """
        Calcula a perda de calor de uma piscina por condução para o solo.

        Args:
            comprimento (float): Comprimento da piscina em metros.
            largura (float): Largura da piscina em metros.
            profundidade (float): Profundidade média da piscina em metros.
            temp_agua_c (float): Temperatura da água em graus Celsius (°C).
            temp_solo_c (float): Temperatura média do solo em graus Celsius (°C).

        Returns:
            float: Perda de calor por condução em Watts (W).
        """
        # U-valor típico para piscina sem isolamento (Watts por m² por grau Celsius)
        U_VALOR_SOLO = 3.0

        # 1. Calcular a área total de contato com o solo (fundo + paredes)
        area_fundo = comprimento * largura
        area_paredes = 2 * (comprimento + largura) * profundidade
        area_total_submersa = area_fundo + area_paredes

        # 2. Calcular a perda de calor por condução
        # A perda só ocorre se a água for mais quente que o solo

        q_cond = U_VALOR_SOLO * area_total_submersa * (temp_agua_c - temp_solo_c)
     
        return q_cond        
    

    def _perda_calor_evaporacao(self, area, temp_agua, temp_ar, umidade_relativa, vel_vento, fator_cobertura):
        # h_fg = calor latente de vaporização (J/kg)
        h_fg = 2.4e6
        a = 3.8e-9
        b = 3.1e-9
        Pw = self._pressao_saturacao_vapor(temp_agua)
        Ps_ar = self._pressao_saturacao_vapor(temp_ar)
        Pa = Ps_ar * (umidade_relativa / 100.0)
        if Pw > Pa:
            q_evap = area * (a + b * vel_vento) * (Pw - Pa) * h_fg
        else:
            q_evap = 0.0
        return q_evap * fator_cobertura  # já ajusta pelo fator de cobertura 
    
    def _perda_calor_radiacao(self,area, temp_agua_c, temp_ar_c,fator_cobertura):
        
        SIGMA = 5.67e-8  # Constante de Stefan-Boltzmann em W/m^2K^4
        """
        Calcula a perda de calor de uma piscina por radiação para um CÉU LIMPO.

        Args:
            area_piscina (float): Área da superfície da piscina em metros quadrados (m²).
            temp_agua_c (float): Temperatura da água da piscina em graus Celsius (°C).
            temp_ar_c (float): Temperatura do ar ambiente em graus Celsius (°C).

        Returns:
            float: Perda de calor por radiação em Watts (W).
        """
        EMISSIVIDADE_AGUA = 0.95

        # 1. Converter todas as temperaturas de Celsius para Kelvin
        temp_agua_k = temp_agua_c + 273.15
        temp_ar_k = temp_ar_c + 273.15

        # 2. Estimar a temperatura efetiva do céu para uma noite limpa
        temp_ceu_k = 0.0552 * (temp_ar_k ** 1.5)

        # 3. Calcular a perda de calor líquida por radiação
        # A perda só ocorre se a água for mais quente que a temperatura radiante do céu
    
        q_rad = EMISSIVIDADE_AGUA * SIGMA * area * (temp_agua_k**4 - temp_ceu_k**4)


        return q_rad * fator_cobertura
    
    

    def _perda_calor_conveccao(self, area, temp_agua, temp_ar, vel_vento, fator_cobertura):
        hc = 5.7 + 3.8 * vel_vento

        q_conv = hc * area * (temp_agua - temp_ar)

        return q_conv * fator_cobertura

    def _get_energy_balance(self, water_t, air_t, humid, solar_gain, vel_vento, fator_cobertura,comprimento, largura, profundidade, temp_solo_c):
        # Todas as entradas em unidades físicas corretas
        area = self.pool_area
        perda_evap = self._perda_calor_evaporacao(area, water_t, air_t, humid, vel_vento, fator_cobertura) #valor em Watts
        perda_conv = self._perda_calor_conveccao(area, water_t, air_t, vel_vento, fator_cobertura) #valor em Watts
        perda_rad = self._perda_calor_radiacao(area, water_t, air_t, fator_cobertura) #valor em Watts
        perda_cond = self._perda_calor_conducao(comprimento, largura, profundidade, water_t, temp_solo_c) #valor em Watts
        return solar_gain - (perda_evap + perda_conv + perda_rad+ perda_cond), perda_evap, perda_conv, perda_rad, perda_cond

    def run(self, hourly_shading):
        p = self.params
        loc = pvlib.location.Location(p['latitude'], p['longitude'])
        cs = loc.get_clearsky(self.weather_data.index)
        solar_pos = loc.get_solarposition(self.weather_data.index)
        
        indice_absorcao = 0.85  # Fator de absorção da água para radiação solar (adimensional)

        solar_gain_series = (cs['ghi'] * self.pool_area * hourly_shading).where(solar_pos['elevation'] > 0, 0) * indice_absorcao #va
        self.sim_data = self.weather_data.copy()
        
        #Valores em Watts serão considerados Watt-horas

        self.sim_data['solar_gain_Wh'] = solar_gain_series
        self.sim_data['hourly_shading'] = hourly_shading
        if p['ignore_solar_on_rain']:
            # Zera o ganho solar durante a chuva e nas 4 horas após o fim da chuva
            is_raining = self.sim_data['is_raining'].astype(bool).values
            zero_solar = np.zeros(len(self.sim_data), dtype=bool)
            rain_indices = np.where(is_raining)[0]
            for idx in rain_indices:
                zero_solar[idx] = True
                # Zera as 4 horas seguintes ao fim da chuva
                # Só marca se a próxima hora não for chuva
                if idx+1 < len(is_raining) and not is_raining[idx+1]:
                    for j in range(1, 5):
                        if idx+j < len(is_raining):
                            zero_solar[idx+j] = True
                        else:
                            break
            self.sim_data.loc[zero_solar, 'solar_gain_Wh'] = 0
  
        # Inicialização dos arrays de saída
        temp_inicio = []
        temp_fim = []
        heater_gain_Wh_list = []
        evaporation_loss_Wh_list = []
        convection_loss_Wh_list = []
        radiation_loss_Wh_list = []
        conduction_loss_Wh_list = []
        balance_list = []
        dynamic_cop_list = []
        consumed_kwh_list = []
        cost_hour_list = []

        # Parâmetros físicos
        volume = p['pool_length'] * p['pool_width'] * p['pool_depth']
        massa_agua = volume * 1000  # kg
        c_agua = 4186  # J/kg.K
        temp_ini = p['desired_temp'] # Initialize with first ambient temperature

        temp_solo_c = self.weather_data['temp_air'].mean()   

        for idx, row in self.sim_data.iterrows():
            temp_inicio.append(temp_ini)
            air_t = row['temp_air']
            humid = row['humidity']
            solar_gain_Wh = row['solar_gain_Wh'] 
            vel_vento = row['wind_speed'] if 'wind_speed' in row and not pd.isnull(row['wind_speed']) else 1.0
            fator_cobertura = p['cover_factor'] if 'cover_factor' in p else 1.0

            # Calcula balanço energético e perdas (tudo em Watts)
            balance_Wh, perda_evap_Wh, perda_conv_Wh, perda_rad_Wh, perda_cond_Wh = self._get_energy_balance(temp_ini, air_t, humid, solar_gain_Wh , vel_vento, fator_cobertura,p['pool_length'], p['pool_width'], p['pool_depth'], temp_solo_c)
            evaporation_loss_Wh = perda_evap_Wh 
            convection_loss_Wh = perda_conv_Wh 
            radiation_loss_Wh = perda_rad_Wh
            conduction_loss_Wh = perda_cond_Wh
            evaporation_loss_Wh_list.append(evaporation_loss_Wh)
            convection_loss_Wh_list.append(convection_loss_Wh)
            radiation_loss_Wh_list.append(radiation_loss_Wh)
            conduction_loss_Wh_list.append(conduction_loss_Wh)
            

            # Balanço horário (Wh)
            balance = balance_Wh 
            balance_list.append(balance)

            # Potência da bomba (W)
            heater_gain_Wh = 0
            if temp_ini < p['desired_temp']:               
                heater_gain_Wh = np.maximum(0, -balance_Wh)
                if p.get('use_heater_schedule'):
                    try:
                        off_day, off_month = map(int, p['heater_off_date'].split('/'))
                        on_day, on_month = map(int, p['heater_on_date'].split('/'))
                        off_date, on_date = pd.to_datetime(f'2024-{off_month}-{off_day}'), pd.to_datetime(f'2024-{on_month}-{on_day}')
                        sim_dayofyear = row.name.dayofyear
                        off_dayofyear, on_dayofyear = off_date.dayofyear, on_date.dayofyear
                        heater_off = (sim_dayofyear >= off_dayofyear) & (sim_dayofyear <= on_dayofyear) if off_dayofyear <= on_dayofyear else (sim_dayofyear >= off_dayofyear) | (sim_dayofyear <= on_dayofyear)
                        if heater_off:                     
                            heater_gain_Wh = 0
                    except (ValueError, KeyError):
                        pass
            heater_gain_Wh_list.append(heater_gain_Wh)

            # COP dinâmico
            dynamic_cop = self._calculate_cop(air_t)
            dynamic_cop_list.append(dynamic_cop)
            consumed_kwh = (heater_gain_Wh / 1000) / dynamic_cop if dynamic_cop > 0 else 0 #Electrical energy consumed in kWh
            consumed_kwh_list.append(consumed_kwh)
            cost_hour = consumed_kwh * p['electricity_cost']
            cost_hour_list.append(cost_hour)

            # Atualizar energia líquida (J)
            energia_liquida = (solar_gain_Wh + heater_gain_Wh - evaporation_loss_Wh - convection_loss_Wh - radiation_loss_Wh - conduction_loss_Wh) * 3600
            delta_T = energia_liquida / (massa_agua * c_agua)
            temp_fim_hora = temp_ini + delta_T
            temp_fim.append(temp_fim_hora)
            temp_ini = temp_fim_hora

        self.sim_data['evaporation_loss_Wh'] = evaporation_loss_Wh_list
        self.sim_data['convection_loss_Wh'] = convection_loss_Wh_list
        self.sim_data['radiation_loss_Wh'] = radiation_loss_Wh_list
        self.sim_data['conduction_loss_Wh'] = conduction_loss_Wh_list
        self.sim_data['balance'] = balance_list
        self.sim_data['heater_power_w'] = heater_gain_Wh_list
        self.sim_data['heater_gain_Wh'] = heater_gain_Wh_list
        self.sim_data['dynamic_cop'] = dynamic_cop_list
        self.sim_data['consumed_kwh'] = consumed_kwh_list
        self.sim_data['cost_hour'] = cost_hour_list
        self.sim_data['water_temp_start'] = temp_inicio
        self.sim_data['water_temp_end'] = temp_fim

        return self.sim_data

class FinancialAnalyzer:
    """Handles financial calculations based on simulation results."""
    def find_best_heat_pump(self, required_power_kw):
        bombas = [
            {'modelo': 'Trocador de Calor TOP+7 Full Inverter - ASTRALPOOL', 'potencia_kW': 7.0, 'preco': 15699.00},
            {'modelo': 'Trocador de Calor TOP+9 Full Inverter - ASTRALPOOL', 'potencia_kW': 9.1, 'preco': 17141.00},
            {'modelo': 'Trocador de Calor TOP+14 Full Inverter - ASTRALPOOL', 'potencia_kW': 13.8, 'preco': 22310.00},
            {'modelo': 'Trocador de Calor TOP+19 Full Inverter - ASTRALPOOL', 'potencia_kW': 18.7, 'preco': 26566.08},
            {'modelo': 'Trocador de Calor TOP+24 Full Inverter - ASTRALPOOL', 'potencia_kW': 23.8, 'preco': 38545.10},
            {'modelo': 'Trocador de Calor TOP+34 Full Inverter - ASTRALPOOL', 'potencia_kW': 34.2, 'preco': 52388.79},
            {'modelo': 'Trocador de Calor TOP+41 Full Inverter - ASTRALPOOL', 'potencia_kW': 41.2, 'preco': 59766.24}
        ]
        best = {'modelos': [], 'preco_total': float('inf'), 'potencia_total_kw': 0}

        for i in range(1, 4):
            for combo in itertools.combinations_with_replacement(bombas, i):
                potencia_total = sum(b['potencia_kW'] for b in combo)
                preco_total = sum(b['preco'] for b in combo)
                if potencia_total >= required_power_kw and preco_total < best['preco_total']:
                    best = {'modelos': [b['modelo'] for b in combo], 'preco_total': preco_total, 'potencia_total_kw': potencia_total}
        
        if not best['modelos']: return f"Nenhuma combinação de até 3 bombas encontrada para {required_power_kw:.1f} kW."
        counts = Counter(best['modelos'])
        modelos_str = ", ".join([f"{c}x {m}" if c > 1 else m for m, c in counts.items()])
        return f"Combinação mais econômica: {modelos_str} (Total: {best['potencia_total_kw']:.1f} kW) - Custo: {format_price(best['preco_total'])}."

    def estimate_solar_system(self, total_kwh, days_of_heating):
        if days_of_heating <= 0 or total_kwh <= 0: return {'potencia_de_pico_kWp': 0, 'custo_mais_baixo_reais': 0, 'custo_mais_alto_reais': 0}
        consumo_diario = total_kwh / days_of_heating
        potencia_pico = round(consumo_diario / 4.5, 1)
        return {'potencia_de_pico_kWp': potencia_pico, 'custo_mais_baixo_reais': potencia_pico * 4500, 'custo_mais_alto_reais': potencia_pico * 6000}

# --- DATA LOADING, REPORTING, UTILS -------------------------------------------------

def format_price(price):
    return f"R$ {price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def load_and_merge_data(timezone, thermometer_file, chuva_file):
    """Loads and merges temperature and rain data, using the provided timezone."""
    try:
        df_temp = pd.read_csv(thermometer_file, usecols=[0,1,2], names=['timestamp', 'temp_air', 'humidity'], header=0, encoding='utf-8')
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], errors='coerce').dt.tz_localize(timezone, ambiguous='infer')
        df_temp = df_temp.dropna().set_index('timestamp').resample('h').mean().interpolate()
    except Exception as e:
        return None, f"Erro ao processar arquivo de temperatura: {e}"

    try:
        df_rain = pd.read_excel(chuva_file, usecols=[0,1], names=['timestamp', 'rain'], header=0)
        df_rain['timestamp'] = pd.to_datetime(df_rain['timestamp'], errors='coerce').dt.tz_localize(timezone, ambiguous='infer')
        df_rain['is_raining'] = (df_rain['rain'] > 0)
        df_rain = df_rain.dropna().set_index('timestamp')[['is_raining']].resample('h').max()
    except FileNotFoundError:
        df_rain = pd.DataFrame(columns=['is_raining'])
    except Exception as e:
        return None, f"Erro ao processar arquivo de chuva: {e}"

    df_combined = df_temp.join(df_rain, how='left').fillna({'is_raining': False})
    if df_combined.empty or df_combined['temp_air'].isnull().all():
        return None, "Erro: A combinação dos dados resultou em um conjunto vazio."
    return df_combined, None

def find_natural_temp(air_t, humid, daily_solar_gain_wh, params, area):
    test_temp = air_t
    avg_solar_gain_w = daily_solar_gain_wh / 24.0
    for _ in range(30):
        temp_diff = test_temp - air_t
        convection_loss = np.maximum(0, 15.0 * area * temp_diff)
        p_water = 0.61094 * np.exp((17.625 * test_temp) / (test_temp + 243.04))
        p_air = 0.61094 * np.exp((17.625 * air_t) / (air_t + 243.04)) * (humid / 100)
        evaporation_loss = np.maximum(0, 25 * (p_water - p_air) * area) * params['cover_factor']
        balance = avg_solar_gain_w - (convection_loss + evaporation_loss)
        if abs(balance) < 50: break
        test_temp += 0.1 if balance > 0 else -0.1
    return test_temp

def create_charts(df_daily, params, sim_data):
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df_daily.index, y=df_daily['temp_air'], mode='lines', name='Temp. do Ar', line=dict(color='royalblue')))
    fig_temp.add_trace(go.Scatter(x=df_daily.index, y=df_daily['natural_water_temp'], mode='lines', name='Temp. Água (Natural)', line=dict(color='firebrick')))
    fig_temp.add_trace(go.Scatter(x=df_daily.index, y=[params['desired_temp']]*len(df_daily), mode='lines', name='Temp. Desejada', line=dict(color='black', dash='dash')))
    fig_temp.update_layout(title='Temperaturas Médias Diárias', xaxis_title='Data', yaxis_title='Temperatura (°C)')

    df_rainy_days = df_daily[df_daily['is_raining'] > 0]
    fig_solar = go.Figure()
    fig_solar.add_trace(go.Bar(x=df_daily.index, y=df_daily['solar_gain_Wh'] / 1000, name='Ganho Solar (kWh/dia)'))
    if not df_rainy_days.empty:
        fig_solar.add_trace(go.Scatter(x=df_rainy_days.index, y=[(df_daily['solar_gain_Wh'].max()/1000)*1.1]*len(df_rainy_days), mode='markers', name='Dias com Chuva', marker=dict(symbol='triangle-down', color='blue', size=8)))
    fig_solar.update_layout(title='Ganho Solar Diário e Ocorrência de Chuva', xaxis_title='Data', yaxis_title='Energia (kWh)')

    fig_heater = go.Figure()
    fig_heater.add_trace(go.Bar(x=df_daily.index, y=df_daily['heater_power_w'] / 1000, name='Energia Adicionada (kWh/dia)'))
    fig_heater.update_layout(title='Energia Adicionada (Aquecedor)', xaxis_title='Data', yaxis_title='Energia (kWh)')

    fig_power_hist = go.Figure()
    heater_power_filtered = sim_data[sim_data['heater_power_w'] > 0]['heater_power_w'] / 1000 # in kW
    fig_power_hist.add_trace(go.Histogram(x=heater_power_filtered, nbinsx=30, name='Ocorrências'))
    fig_power_hist.update_layout(title='Histograma de Potência do Aquecedor (quando ligado)', xaxis_title='Potência Requerida (kW)', yaxis_title='Número de Horas')

    fig_hourly_solar = go.Figure()
    fig_hourly_solar.add_trace(go.Scatter(x=sim_data.index, y=sim_data['solar_gain_Wh'] / 1000, mode='lines', name='Ganho Solar Horário (kWh)'))
    fig_hourly_solar.update_layout(title='Ganho Solar Horário', xaxis_title='Data/Hora', yaxis_title='Energia (kWh)')

    return pio.to_html(fig_temp, full_html=False), pio.to_html(fig_solar, full_html=False), pio.to_html(fig_heater, full_html=False), pio.to_html(fig_power_hist, full_html=False), pio.to_html(fig_hourly_solar, full_html=False)

def gerar_relatorio(params, results):
    report_template = '''
# Relatório de Simulação de Custos de Aquecimento de Piscina

## 1. Dados de Entrada

**Cenário de Simulação:**
- Arquivo de Configuração: {config_file}
- Descrição: {config_description}

**Parâmetros da Piscina:**
- Dimensões: {pool_length}m (comprimento) x {pool_width}m (largura) x {pool_depth}m (profundidade)
- Volume: {pool_volume:.2f} m³
- Litros: {pool_liters:.0f} l
- Temperatura Desejada: {desired_temp} °C

**Fatores Ambientais e de Uso:**
- Fator de Sombreamento: Dinâmico (calculado hora a hora com base na geometria 3D)
- Fator de Cobertura: {cover_factor:.0f}% (100% = sem capa, 10% = com capa)
- Localização: Latitude {latitude}, Longitude {longitude}
- Ignorar Ganho Solar com Chuva: {ignore_solar_on_rain}

**Parâmetros Econômicos e Técnicos:**
- Custo da Eletricidade: R$ {electricity_cost:.2f} / kWh
- COP Mínimo da Bomba: {cop_min}
- COP Máximo da Bomba: {cop_max}
- Número de Apartamentos (para rateio): {num_apartments}
- Programação do Aquecedor Ativa: {use_heater_schedule}
{heater_schedule_info}

## 2. Metodologia de Cálculo

O cálculo foi realizado através de uma simulação horária para um período de um ano, utilizando dados históricos de temperatura, umidade e chuva para a localização especificada.

- **Perdas Térmicas:** As perdas de calor da piscina para o ambiente foram calculadas considerando a convecção (troca de calor com o ar) e a evaporação (perda de calor latente).
- **Ganhos Solares:** O ganho de energia solar foi estimado utilizando a biblioteca `pvlib`, que calcula a irradiância solar para a localidade com base em um modelo de céu claro (`clearsky`).
- **Sombreamento:** O sombreamento foi calculado dinamicamente para cada hora do ano, com base na posição do sol e na geometria 3D dos edifícios vizinhos, determinando a porcentagem de sol direto que atinge a piscina.
- **Balanço Energético:** A cada hora, o balanço entre perdas e ganhos foi calculado. Se o balanço fosse negativo (perda líquida de calor), o sistema de aquecimento seria acionado para suprir a diferença e manter a temperatura desejada.
- **Potência Necessária:** A potência da bomba de calor foi dimensionada para atender a {power_percentile:.0f}% da demanda de aquecimento ao longo do ano (desconsiderando os picos mais extremos).
- **Consumo e Custo:** O consumo de energia da bomba de calor foi calculado dividindo a energia fornecida à piscina pelo Coeficiente de Performance (COP) dinâmico da bomba (que varia com a temperatura do ar). O custo foi então calculado multiplicando o consumo pelo preço da eletricidade.

## 3. Resultados da Simulação

- **Consumo Anual de Energia:** {total_kwh:.0f} kWh
- **Custo Anual Total:** R$ {total_cost:.2f}
- **Custo Mensal Médio por Apartamento:** R$ {monthly_cost_per_apartment:.2f}
- **Potência Dimensionada da Bomba ({power_percentile:.0f}%):** {required_power_kw:.1f} kW

## 4. Conclusão e Recomendação

Com base na simulação, o custo anual estimado para manter a piscina na temperatura desejada é de **R$ {total_cost:.2f}**.

Para atender à demanda de aquecimento, é necessário um sistema com capacidade de **{required_power_kw:.1f} kW**.

**Recomendação de Equipamento:**
{resultado_bomba}

## 5. Sistema Solar Fotovoltaico (Estimativa)

Para compensar o consumo elétrico do sistema de aquecimento, seria necessário um sistema solar fotovoltaico com as seguintes características:
- **Potência de Pico do Sistema:** {solar_potencia_pico_kWp:.1f} kWp
- **Custo de Instalação Estimado:** Entre R$ {solar_custo_minimo:,.2f} e R$ {solar_custo_maximo:,.2f}
- **Tempo de Payback Simples:** {solar_payback_years:.1f} anos (considerando o custo médio do sistema e a economia anual de energia).

*Observação: Estes valores são estimativas e podem variar conforme o local, fornecedor e equipamentos escolhidos.*
'''

    heater_schedule_info = ""
    if params.get('use_heater_schedule'):
        heater_schedule_info = f"- Período Desligado: de {params.get('heater_off_date', 'N/A')} a {params.get('heater_on_date', 'N/A')}"

    report = report_template.format(
        config_file=params.get('config_file', 'N/A'),
        config_description=params.get('config_description', 'N/A'),
        pool_length=params['pool_length'],
        pool_width=params['pool_width'],
        pool_depth=params['pool_depth'],
        pool_volume=params['pool_length'] * params['pool_width'] * params['pool_depth'],
        pool_liters=params['pool_length'] * params['pool_width'] * params['pool_depth']*1000,
        desired_temp=params['desired_temp'],
        cover_factor=params['cover_factor'] * 100,
        latitude=params['latitude'],
        longitude=params['longitude'],
        ignore_solar_on_rain='Sim' if params['ignore_solar_on_rain'] else 'Não',
        electricity_cost=params['electricity_cost'],
        cop_min=params['cop_min'],
        cop_max=params['cop_max'],
        num_apartments=params['num_apartments'],
        use_heater_schedule='Sim' if params.get('use_heater_schedule') else 'Não',
        heater_schedule_info=heater_schedule_info,
        power_percentile=params['power_percentile'],
        total_kwh=results['total_kwh'],
        total_cost=results['total_cost'],
        monthly_cost_per_apartment=results['monthly_cost_per_apartment'],
        required_power_kw=results['required_power_kw'],
        resultado_bomba=results['resultado_bomba'],
        solar_potencia_pico_kWp=results['solar_heating_system']['potencia_de_pico_kWp'],
        solar_custo_minimo=results['solar_heating_system']['custo_mais_baixo_reais'],
        solar_custo_maximo=results['solar_heating_system']['custo_mais_alto_reais'],
        solar_payback_years=results['solar_payback_years']
    )

    return report

# --- FLASK APPLICATION --------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'uma-chave-secreta-muito-segura-e-dificil-de-adivinhar'

# --- ROUTES -------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def simulator_route():
    xml_files = glob.glob('*.xml')
    if not xml_files:
        return "<h1>Erro</h1><p>Nenhum arquivo de configuração .xml encontrado.</p>", 500

    xml_files_metadata = []
    for f in sorted(xml_files):
        meta = get_xml_metadata(f)
        xml_files_metadata.append({
            'filename': f,
            'description': meta['description'],
            'creation_date': meta['creation_date']
        })

    config_file_to_load = request.values.get('config_file', session.get('config_file', xml_files[0]))
    if config_file_to_load not in xml_files:
        flash(f"Arquivo de configuração '{config_file_to_load}' não encontrado. Usando '{xml_files[0]}'.", "warning")
        config_file_to_load = xml_files[0]
    
    session['config_file'] = config_file_to_load

    config = load_config_from_xml(config_file_to_load)
    if not config:
        return render_template('simulator_tabs.html', p={}, xml_files=xml_files_metadata, selected_config=config_file_to_load, results=None)

    LOCATION = config['location']
    SCENE_CONFIG = config['scene_config']
    DEFAULT_PARAMS = config['default_params']
    METADATA = config['metadata']

    DATA_FILES = config.get('data_files', {})
    thermometer_file = DATA_FILES.get('thermometer_file', 'Outdoor Thermometer_export_202508100845.csv')
    chuva_file = DATA_FILES.get('chuva_file', 'chuva.xlsx')

    all_data, startup_error = load_and_merge_data(LOCATION['timezone'], thermometer_file, chuva_file)
    if startup_error:
        flash(f"Erro na Carga de Dados: {startup_error}", "error")
        return render_template('simulator_tabs.html', p={}, xml_files=xml_files_metadata, selected_config=config_file_to_load, results=None)

    params_for_template = DEFAULT_PARAMS.copy()
    params_for_template.update({
        'latitude': LOCATION['latitude'],
        'longitude': LOCATION['longitude'],
        'pool_length': SCENE_CONFIG['pool']['length'],
        'pool_width': SCENE_CONFIG['pool']['width'],
        'pool_depth': SCENE_CONFIG['pool']['depth'],
    })

    if request.method == 'POST':
        form_params = {k: request.form.get(k, default, type=type(default)) for k, default in params_for_template.items()}
        form_params.update({
            'ignore_solar_on_rain': 1 if 'ignore_solar_on_rain' in request.form else 0,
            'use_heater_schedule': 1 if 'use_heater_schedule' in request.form else 0,
            'config_file': config_file_to_load,
            'config_description': METADATA['description']
        })

        shading_calculator = ShadingCalculator(SCENE_CONFIG)
        solar_pos = pvlib.location.Location(form_params['latitude'], form_params['longitude']).get_solarposition(all_data.index)
        
        cache_filename = f"shading_cache_{os.path.splitext(os.path.basename(config_file_to_load))[0]}.csv"
        if os.path.exists(cache_filename):
            hourly_shading = pd.read_csv(cache_filename, index_col=0, parse_dates=True).squeeze("columns")
        else:
            hourly_shading = shading_calculator.calculate_hourly_shading(solar_pos)
            hourly_shading.to_csv(cache_filename)

        simulation = PoolSimulator(form_params, all_data)
        sim_data = simulation.run(hourly_shading)

        sim_data.to_csv(f"simulation_output_{os.path.splitext(os.path.basename(config_file_to_load))[0]}.csv")

        heater_on_demand = sim_data['heater_power_w'][sim_data['heater_power_w'] > 0]
        required_power_kw = heater_on_demand.quantile(form_params['power_percentile'] / 100.0) / 1000 if not heater_on_demand.empty else 0

        analyzer = FinancialAnalyzer()
        bomba_result = analyzer.find_best_heat_pump(required_power_kw)
        
        total_kwh = sim_data['consumed_kwh'].sum()
        dias_de_aquecimento = 365
        if form_params.get('use_heater_schedule'):
            try:
                off_day, off_month = map(int, form_params['heater_off_date'].split('/'))
                on_day, on_month = map(int, form_params['heater_on_date'].split('/'))
                off_date = pd.to_datetime(f"2024-{off_month}-{off_day}")
                on_date = pd.to_datetime(f"2024-{on_month}-{on_day}")
                if off_date <= on_date:
                    dias_de_aquecimento = 365 - (on_date - off_date).days
                else: # Wraps around the new year
                    dias_de_aquecimento = (pd.to_datetime("2024-12-31") - off_date).days + (on_date - pd.to_datetime("2024-01-01")).days
            except (ValueError, KeyError): pass

        solar_system_estimate = analyzer.estimate_solar_system(total_kwh, dias_de_aquecimento)

        total_cost = sim_data['cost_hour'].sum()
        solar_cost = (solar_system_estimate['custo_mais_baixo_reais'] + solar_system_estimate['custo_mais_alto_reais']) / 2
        payback = solar_cost / total_cost if total_cost > 0 else float('inf')


        df_results = sim_data[[
            'temp_air', 'humidity', 'is_raining',
            'evaporation_loss_Wh',
            'heater_power_w', 'heater_gain_Wh',
            'consumed_kwh', 'cost_hour',
            'hourly_shading',
            'water_temp_start', 'water_temp_end','balance', 'dynamic_cop', 'solar_gain_Wh','convection_loss_Wh', 'radiation_loss_Wh','conduction_loss_Wh'
        ]]

        df_daily = df_results.resample('D').agg({
            'temp_air': 'mean',
            'humidity': 'mean',
            'is_raining': 'max',
            'solar_gain_Wh': 'sum',
            'consumed_kwh': 'sum',
            'cost_hour': 'sum',
            'heater_power_w': 'sum',
            'hourly_shading': 'mean'
        })
        tqdm.pandas(desc="Calculando Temp. Natural")
        df_daily['natural_water_temp'] = df_results['water_temp_end']#df_daily.progress_apply(
            #lambda row: find_natural_temp(row['temp_air'], row['humidity'], row['solar_gain_Wh'], form_params, simulation.pool_area), axis=1)
        df_monthly = df_results.resample('ME').sum()

        charts = create_charts(df_daily, form_params, sim_data)

        results = {
            'hourly': df_results.reset_index().to_dict('records'),
            'daily': df_daily.reset_index().to_dict('records'),
            'monthly': df_monthly.reset_index().to_dict('records'),
            'total_cost': total_cost, 'total_kwh': total_kwh,
            'required_power_kw': required_power_kw,
            'resultado_bomba': bomba_result,
            'monthly_cost_per_apartment': (total_cost / 12) / form_params['num_apartments'] if form_params['num_apartments'] > 0 else 0,
            'solar_heating_system': solar_system_estimate,
            'solar_payback_years': payback,
            'temp_chart_html': charts[0], 'solar_chart_html': charts[1], 'heater_chart_html': charts[2],
            'power_hist_chart_html': charts[3], 'hourly_solar_chart_html': charts[4]
        }
        results['report_text'] = gerar_relatorio(form_params, results)

        return render_template('simulator_tabs.html', results=results, p=form_params, xml_files=xml_files_metadata, selected_config=config_file_to_load)

    return render_template('simulator_tabs.html', p=params_for_template, xml_files=xml_files_metadata, selected_config=config_file_to_load, results=None)

@app.route('/api/scene-config')
def get_scene_config():
    """Combines simulation and visualization configs for the frontend."""
    config_file = session.get('config_file', glob.glob('*.xml')[0])
    config = load_config_from_xml(config_file)
    if not config: return jsonify({"error": "Config file not found or invalid"}), 500

    VISUALIZATION_CONFIG = config['visualization_config']
    SCENE_CONFIG = config['scene_config']
    
    full_config = {
        **VISUALIZATION_CONFIG,
        'pool': {
            **SCENE_CONFIG['pool'],
            **VISUALIZATION_CONFIG['pool_visual']
        },
        'blocks': SCENE_CONFIG['blocks']
    }
    full_config.pop('pool_visual', None)
    return jsonify(full_config)

@app.route('/api/solar-data')
def get_solar_data():
    config_file = session.get('config_file', glob.glob('*.xml')[0])
    config = load_config_from_xml(config_file)
    if not config: return jsonify({"error": "Config file not found or invalid"}), 500
    
    LOCATION = config['location']

    datetime_str = request.args.get('datetime')
    if not datetime_str:
        return jsonify({'error': 'Datetime parameter is required'}), 400
    
    try:
        timestamp = pd.to_datetime(datetime_str).tz_localize(LOCATION['timezone'])
        location = pvlib.location.Location(LOCATION['latitude'], LOCATION['longitude'], tz=LOCATION['timezone'])
        solar_pos = location.get_solarposition(timestamp)
        
        return jsonify({
            'azimuth': solar_pos['azimuth'].iloc[0],
            'elevation': solar_pos['elevation'].iloc[0]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
