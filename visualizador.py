# visualize_shading.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pvlib
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate as shapely_triangulate
from scipy.spatial import Delaunay
from typing import Dict, Any
import xml.etree.ElementTree as ET

# --- CONFIGURATION LOADER ----------------------------------------------------------

def load_config_from_xml(file_path):
    """Parses the config.xml file and returns configuration variables."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    def to_num(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            try:
                return float(s)
            except (ValueError, TypeError):
                return s

    loc_root = root.find('location')
    LATITUDE = to_num(loc_root.find('latitude').text)
    LONGITUDE = to_num(loc_root.find('longitude').text)
    ALTITUDE = to_num(loc_root.find('altitude').text)
    TIMEZONE = loc_root.find('timezone').text

    scene_root = root.find('scene')
    pool_attrs = {k: to_num(v) for k, v in scene_root.find('pool').attrib.items()}
    
    blocks = []
    for block_node in scene_root.find('blocks').findall('block'):
        blocks.append({k: to_num(v) for k, v in block_node.attrib.items()})

    SCENE_CONFIG = {
        'pool': pool_attrs,
        'blocks': blocks
    }

    vis_root = root.find('visualization')
    vis_pool = {k: to_num(v) for k, v in vis_root.find('pool_visual').attrib.items()}
    
    # Combine the simulation pool config with the visual pool config
    SCENE_CONFIG['pool'].update(vis_pool)

    return LATITUDE, LONGITUDE, ALTITUDE, TIMEZONE, SCENE_CONFIG

# Load configuration from XML
LATITUDE, LONGITUDE, ALTITUDE, TIMEZONE, SCENE_CONFIG = load_config_from_xml('config.xml')


class ShadingVisualizer:
    """Encapsulates the logic for creating a 3D shading visualization with a time slider."""

    def __init__(self, scene_config, lat, lon, alt, timezone, date, hours, freq):
        self.scene_config = scene_config
        self.location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt, tz=timezone)
        self.time_steps = pd.date_range(start=f"{date} {hours.start:02d}:00", end=f"{date} {hours.stop-1:02d}:59", freq=freq, tz=timezone)
        
        self.fig = go.Figure()
        self.pool_polygon = None
        self.traces_by_time = {}
        self.num_static_traces = 0
        self.MAX_BLOCKS = len(self.scene_config['blocks'])

    def _get_shadow_polygon(self, block_config, sun_azimuth_deg, sun_elevation_deg):
        if sun_elevation_deg <= 0: return None
        w, h, l = block_config['width'], block_config['height'], block_config['length']
        x_c, z_c, angle_deg = -block_config['x'], -block_config['z'], block_config['angle']
        
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])

        half_w, half_l = w / 2, l / 2
        local_vertices_2d = [np.array([-half_w, half_l]), np.array([half_w, half_l]), np.array([half_w, -half_l]), np.array([-half_w, -half_l])]
        
        world_vertices_2d = []
        for p in local_vertices_2d:
            rotated_p = rotation_matrix @ p
            mirrored_world_p = np.array([-rotated_p[0] + x_c, rotated_p[1] + z_c])
            world_vertices_2d.append(mirrored_world_p)
        
        world_vertices_3d = []
        for x, z in world_vertices_2d:
            world_vertices_3d.append(np.array([x, 0, z]))
            world_vertices_3d.append(np.array([x, h, z]))

        sun_elevation_rad = np.deg2rad(sun_elevation_deg)
        sun_azimuth_rad = np.deg2rad(sun_azimuth_deg)

        if np.tan(sun_elevation_rad) < 1e-6: return None
        cot_elevation = 1.0 / np.tan(sun_elevation_rad)
        shadow_dx = cot_elevation * np.sin(sun_azimuth_rad)
        shadow_dz = -cot_elevation * np.cos(sun_azimuth_rad)

        shadow_points = []
        for vx, vy, vz in world_vertices_3d:
            shadow_x = vx + vy * shadow_dx
            shadow_z = vz + vy * shadow_dz
            shadow_points.append((shadow_x, shadow_z))
        
        return MultiPoint(shadow_points).convex_hull

    def _create_cuboid_trace(self, block_config, name):
        w, h, l = block_config['width'], block_config['height'], block_config['length']
        x_c, z_c, angle_deg = -block_config['x'], -block_config['z'], block_config['angle']

        x = np.array([-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2])
        y = np.array([0, 0, 0, 0, h, h, h, h])
        z = np.array([-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2])

        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        x_rot, z_rot = x * c - z * s, x * s + z * c
        x_final, z_final = -x_rot + x_c, z_rot + z_c
        
        return go.Mesh3d(
            x=x_final, y=y, z=z_final,
            i=[7, 0, 0, 0, 4, 4, 1, 1, 3, 3, 4, 5],
            j=[3, 4, 1, 2, 5, 6, 2, 6, 2, 7, 5, 6],
            k=[0, 7, 2, 3, 6, 7, 6, 5, 6, 6, 7, 7],
            color=block_config['color'], opacity=0.7, name=name
        )

    def _triangulate_simple_polygon(self, polygon):
        if polygon.is_empty: return np.array([]), np.array([]), np.array([])
        if isinstance(polygon, MultiPolygon):
            all_x, all_z, all_tris, point_offset = [], [], [], 0
            for poly in polygon.geoms:
                points = np.array(poly.exterior.coords)
                if len(points) < 3: continue
                tri = Delaunay(points[:, :2])
                all_x.extend(points[:, 0]); all_z.extend(points[:, 1])
                all_tris.extend(tri.simplices + point_offset)
                point_offset += len(points)
            return np.array(all_x), np.array(all_z), np.array(all_tris)
        
        points = np.array(polygon.exterior.coords)
        if len(points) < 3: return np.array([]), np.array([]), np.array([])
        tri = Delaunay(points[:, :2])
        return points[:, 0], points[:, 1], tri.simplices

    def _triangulate_complex_polygon(self, polygon):
        if polygon.is_empty: return np.array([]), np.array([]), np.array([])
        triangles = shapely_triangulate(polygon)
        all_x, all_z, all_tris, point_offset = [], [], [], 0
        for t in triangles:
            p = np.array(t.exterior.coords)
            all_x.extend(p[:3, 0]); all_z.extend(p[:3, 1])
            all_tris.append([point_offset, point_offset + 1, point_offset + 2])
            point_offset += 3
        return np.array(all_x), np.array(all_z), np.array(all_tris)

    def _create_polygon_trace(self, polygon, name, color, y_level):
        if not polygon or polygon.is_empty: return None
        x, z, tri = self._triangulate_simple_polygon(polygon)
        if len(x) == 0: return None
        opacity = 0.5 if name == 'Sombra no Chão' else 0.9
        return go.Mesh3d(x=x, y=np.full_like(x, y_level), z=z, i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], color=color, opacity=opacity, name=name)

    def calculate_shading_data(self, solar_position: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, Any]]:
        """Calculates shadow geometries and factors for a given set of solar positions."""
        shading_data = {}
        if self.pool_polygon is None: self._prepare_pool_polygon()

        for timestamp, sun in solar_position.iterrows():
            sun_elevation, sun_azimuth = sun['elevation'], sun['azimuth']

            all_shadows = [self._get_shadow_polygon(b, sun_azimuth, sun_elevation) for b in self.scene_config['blocks'] if sun_elevation > 0]
            all_shadows = [s for s in all_shadows if s and s.is_valid]

            total_shadow_area = unary_union(all_shadows)
            shaded_intersection = self.pool_polygon.intersection(total_shadow_area)
            shading_factor = (shaded_intersection.area / self.pool_polygon.area) if self.pool_polygon.area > 0 else 0

            shading_data[timestamp] = {
                "all_shadows": all_shadows,
                "shaded_intersection": shaded_intersection,
                "shading_factor": shading_factor,
                "solar_position": sun
            }
        return shading_data

    def _prepare_pool_polygon(self):
        pool_config = self.scene_config['pool']
        pool_rot_rad = np.deg2rad(pool_config['rotation'])
        c, s = np.cos(pool_rot_rad), np.sin(pool_rot_rad)
        rot_matrix = np.array([[c, -s], [s, c]])
        half_l, half_w = pool_config['length'] / 2, pool_config['width'] / 2
        pool_corners_local = [(-half_w, half_l), (half_w, half_l), (half_w, -half_l), (-half_w, -half_l)]
        pool_corners_world = [rot_matrix @ p for p in pool_corners_local]
        self.pool_polygon = Polygon([(-c[0], c[1]) for c in pool_corners_world])

    def _prepare_static_traces(self):
        for i, block_conf in enumerate(self.scene_config['blocks']):
            self.fig.add_trace(self._create_cuboid_trace(block_conf, f'Edifício {i+1}'))
        
        if self.pool_polygon is None: self._prepare_pool_polygon()
        pool_x, pool_z, pool_tris = self._triangulate_complex_polygon(self.pool_polygon)
        self.fig.add_trace(go.Mesh3d(x=pool_x, y=np.full_like(pool_x, 0.01), z=pool_z, i=pool_tris[:,0], j=pool_tris[:,1], k=pool_tris[:,2], color=self.scene_config['pool']['color'], opacity=0.9, name='Piscina'))
        self.num_static_traces = len(self.fig.data)

    def _prepare_dynamic_traces(self):
        solar_position = self.location.get_solarposition(self.time_steps)
        shading_data = self.calculate_shading_data(solar_position)

        for timestamp, data in shading_data.items():
            shadow_traces = [self._create_polygon_trace(s, 'Sombra no Chão', 'grey', 0.0) for s in data["all_shadows"]]
            while len(shadow_traces) < self.MAX_BLOCKS: shadow_traces.append(go.Mesh3d())

            inter_x, inter_z, inter_tris = self._triangulate_complex_polygon(data["shaded_intersection"])
            intersection_trace = go.Mesh3d(x=inter_x, y=np.full_like(inter_x, 0.02), z=inter_z, i=inter_tris[:,0], j=inter_tris[:,1], k=inter_tris[:,2], color='darkblue', opacity=0.9, name='Área Sombreada') if len(inter_x) > 0 else go.Mesh3d()

            sun_pos = data["solar_position"]
            sun_dist = 150
            sun_x = sun_dist * np.sin(np.deg2rad(sun_pos['azimuth'])) * np.cos(np.deg2rad(sun_pos['elevation']))
            sun_y = sun_dist * np.sin(np.deg2rad(sun_pos['elevation']))
            sun_z = sun_dist * np.cos(np.deg2rad(sun_pos['azimuth'])) * np.cos(np.deg2rad(sun_pos['elevation']))
            sun_trace = go.Scatter3d(x=[-sun_x], y=[sun_y], z=[sun_z], mode='markers', marker=dict(size=10, color='yellow', symbol='circle'), name='Sol') if sun_pos['elevation'] > 0 else go.Scatter3d()

            self.traces_by_time[timestamp] = {"shadows": shadow_traces, "intersection": intersection_trace, "sun": sun_trace, "shading_factor": data["shading_factor"]}

    def _build_figure_layout(self):
        if not self.time_steps.empty:
            num_dynamic_per_step = self.MAX_BLOCKS + 2
            sliders = [dict(active=0, currentvalue={"prefix": "Hora: "}, pad={"t": 50}, steps=[])]
            num_dynamic_total = len(self.time_steps) * num_dynamic_per_step

            for i, timestamp in enumerate(self.time_steps):
                visibility = [True] * self.num_static_traces + [False] * num_dynamic_total
                step_start_index = self.num_static_traces + i * num_dynamic_per_step
                for j in range(num_dynamic_per_step): visibility[step_start_index + j] = True
                
                shading_factor = self.traces_by_time[timestamp]["shading_factor"]
                datetime_str = timestamp.strftime('%Y-%m-%d %H:%M')
                
                annotation_text = f"Fator de Sombreamento: {shading_factor:.2%}"
                annotations = [dict(text=annotation_text, showarrow=False, xref="paper", yref="paper", x=0.05, y=0.95, align="left")]

                step = dict(method="update", args=[{"visible": visibility}, {"title": f'Visualização de Sombreamento - {datetime_str}', "annotations": annotations}], label=timestamp.strftime('%H:%M'))
                sliders[0]['steps'].append(step)

            initial_timestamp = self.time_steps[0]
            initial_shading_factor = self.traces_by_time[initial_timestamp]["shading_factor"]
            initial_datetime_str = initial_timestamp.strftime('%Y-%m-%d %H:%M')
            initial_annotation_text = f"Fator de Sombreamento: {initial_shading_factor:.2%}"
            
            self.fig.update_layout(
                title=f'Visualização de Sombreamento - {initial_datetime_str}',
                annotations=[dict(text=initial_annotation_text, showarrow=False, xref="paper", yref="paper", x=0.05, y=0.95, align="left")],
                scene=dict(xaxis_title='Eixo X (m)', yaxis_title='Altura Y (m)', zaxis_title='Eixo Z (m)', aspectmode='data'),
                margin=dict(r=10, l=10, b=10, t=60),
                sliders=sliders
            )

    def _add_traces_to_figure(self):
        if not self.time_steps.empty:
            num_dynamic_per_step = self.MAX_BLOCKS + 2
            for i, timestamp in enumerate(self.time_steps):
                is_visible = (i == 0)
                traces = self.traces_by_time[timestamp]
                
                for k, shadow_trace in enumerate(traces["shadows"]):
                    shadow_trace.visible = is_visible
                    shadow_trace.legendgroup = 'Sombra no Chão'
                    shadow_trace.showlegend = is_visible and k == 0
                    self.fig.add_trace(shadow_trace)

                traces["intersection"].visible = is_visible
                traces["intersection"].legendgroup = 'Área Sombreada'
                traces["intersection"].showlegend = is_visible
                self.fig.add_trace(traces["intersection"])

                traces["sun"].visible = is_visible
                traces["sun"].legendgroup = 'Sol'
                traces["sun"].showlegend = is_visible
                self.fig.add_trace(traces["sun"])

    def run(self):
        print("Preparando traços estáticos...")
        self._prepare_static_traces()
        print("Calculando dados de sombreamento...")
        self._prepare_dynamic_traces()
        print("Adicionando traços dinâmicos à figura...")
        self._add_traces_to_figure()
        print("Construindo o layout da figura final com slider...")
        self._build_figure_layout()
        print("Exibindo visualização.")
        self.fig.show()

if __name__ == '__main__':
    # Configurações específicas para esta execução do visualizador
    VISUALIZATION_DATE = '2025-09-01'
    HOURS_RANGE = range(7, 20)
    TIME_FREQUENCY = '15min'
    
    try:
        print(f"Tentando gerar visualização para a data: {VISUALIZATION_DATE}")
        visualizer = ShadingVisualizer(
            scene_config=SCENE_CONFIG,
            lat=LATITUDE,
            lon=LONGITUDE,
            alt=ALTITUDE,
            timezone=TIMEZONE,
            date=VISUALIZATION_DATE,
            hours=HOURS_RANGE,
            freq=TIME_FREQUENCY
        )
        visualizer.run()
    except Exception as e:
        print(f"\n--- ERRO AO GERAR VISUALIZAÇÃO ---")
        print(f"Ocorreu um erro: {e}")
        print("Gerando visualização com uma data de fallback segura (2025-12-21).")
        print("-------------------------------------\n")
        try:
            SAFE_DATE = '2025-12-21'
            visualizer_fallback = ShadingVisualizer(
                scene_config=SCENE_CONFIG,
                lat=LATITUDE,
                lon=LONGITUDE,
                alt=ALTITUDE,
                timezone=TIMEZONE,
                date=SAFE_DATE,
                hours=HOURS_RANGE,
                freq=TIME_FREQUENCY
            )
            visualizer_fallback.run()
        except Exception as fallback_e:
            print(f"\n--- ERRO FATAL ---")
            print(f"A visualização de fallback também falhou: {fallback_e}")
            print("Verifique as configurações da cena (config.xml) e as coordenadas.")
            print("------------------\n")