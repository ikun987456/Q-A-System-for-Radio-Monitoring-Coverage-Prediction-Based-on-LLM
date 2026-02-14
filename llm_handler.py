import math
import streamlit as st
from openai import OpenAI
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
from map.show_map import show_baidu_map
from utils.calculate_function import calculate_path_loss, calculate_field_strength
import pandas as pd
import numpy as np
from coord_convert import transform
from plotly.subplots import make_subplots
import rasterio
from utils.baidu_profile import show_enhanced_baidu_map11

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def calculate_distance(lat1, lon1, lat2, lon2):
    """计算两点间距离（公里）"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_coordinates_parameters(query, calculation_type="path_loss"):
    """
    通用AI参数解析函数

    Args:
        query: 用户输入的查询文本
        calculation_type: 计算类型 ("path_loss" 或 "field_strength")

    Returns:
        dict: 解析后的参数字典
    """

    # 初始化客户端
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )

    # 🎯 根据计算类型构建不同的提示词
    if calculation_type == "path_loss":
        prompt = f"""
你是专业的无线电工程师AI助手。请从用户指令中提取**路径损耗计算**的参数。

用户指令: {query}

**路径损耗计算需要的参数：**

必需参数：
- tx_lat: 发射点纬度（数字，必须提取）
- tx_lon: 发射点经度（数字，必须提取）  
- rx_lat: 接收点纬度（数字，必须提取）
- rx_lon: 接收点经度（数字，必须提取）
- frequency: 频率(MHz，数字，必须提取)

可选参数（括号内为默认值）：
- gap: 计算间距(米，默认200)
- Pt: 发射功率(kW，默认1)
- Gt: 发射天线增益(dB，默认0)
- tx_antenna_height: 发射天线高度(米，默认10)
- rx_antenna_height: 接收天线高度(米，默认5)
- time_percentage: 时间概率(%，默认50)
- signal_polarization: 信号极化(1=水平，2=垂直，默认1)

解析示例：
输入："从(26.128,103.147)到(26.042,103.216)，340MHz，发射高度30米,接收高度1.5米，采样间隔50.54米"
输出：
{{
    "tx_lat": 26.128,
    "tx_lon": 103.147,
    "rx_lat": 26.042,
    "rx_lon": 103.216,
    "frequency": 340,
    "tx_antenna_height": 30,
    "rx_antenna_height": 5,
    "gap": 50.54,
    "Pt": 1,
    "Gt": 0,
    "time_percentage": 50,
    "signal_polarization": 1
}}
        """

    elif calculation_type == "field_strength":
        prompt = f"""
你是专业的无线电工程师AI助手。请从用户指令中提取**区域场强计算**的参数。

用户指令: {query}

**区域场强计算需要的参数：**

必需参数：
- lat: 发射点纬度（数字，必须提取）
- lon: 发射点经度（数字，必须提取）
- frequency: 频率(MHz，数字，必须提取)

可选参数（括号内为默认值）：
- rad: 计算半径(公里，默认1)
- gap: 计算间距(米，默认10)  
- Pt: 发射功率(kW，默认1)
- Gt: 发射天线增益(dB，默认0)
- tx_antenna_height: 发射天线高度(米，默认10)
- rx_antenna_height: 接收天线高度(米，默认5)
- time_percentage: 时间概率(%，默认50)
- signal_pol: 信号极化(1=水平，2=垂直，默认1)

解析示例：
输入："计算坐标(26.128,103.147)周围1公里范围的340MHz场强分布，采样间隔200米"
输出：
{{
    "lat": 26.128,
    "lon": 103.147,
    "frequency": 340,
    "rad": 2,
    "Pt": 5,
    "gap": 10,
    "Gt": 0,
    "tx_antenna_height": 10,
    "rx_antenna_height": 5,
    "time_percentage": 50,
    "signal_pol": 1
}}
        """

    else:
        st.error(f"❌ 不支持的计算类型: {calculation_type}")
        return None

    # 添加通用规则
    prompt += """

解析规则：
1. 坐标格式：支持 (纬度,经度) 或 纬度,经度 或 lat,lon
2. 频率格式：支持 XXXMHz 或 频率XXX 或 XX兆赫
3. 功率格式：支持 XXkW 或 功率XX千瓦 或 XXW
4. 高度格式：支持 XX米 或 XXm 或 高度XX
5. 距离格式：支持 XX公里 或 XXkm 或 半径XX
6. 如果缺少必需参数，返回 null
7. 数值参数必须是数字类型，不要字符串
8. 只返回纯JSON，不要任何解释

现在请解析：
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 低温度确保一致性
            max_tokens=800  # 限制token数量提高速度
        )

        content = response.choices[0].message.content.strip()

        # 🔧 清理JSON格式
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # 解析JSON
        parsed_data = json.loads(content.strip())

        # ✅ 根据计算类型验证必需参数
        if calculation_type == "path_loss":
            required_params = ['tx_lat', 'tx_lon', 'rx_lat', 'rx_lon', 'frequency']
            defaults = {
                'gap': 200, 'Pt': 1, 'Gt': 0, 'tx_antenna_height': 10,
                'rx_antenna_height': 5, 'time_percentage': 50, 'signal_polarization': 1
            }
        elif calculation_type == "field_strength":
            required_params = ['lat', 'lon', 'frequency']
            defaults = {
                'rad': 1, 'gap': 10, 'Pt': 1, 'Gt': 0, 'tx_antenna_height': 10,
                'rx_antenna_height': 5, 'time_percentage': 50, 'signal_pol': 1
            }

        # 检查必需参数
        missing_params = []
        for param in required_params:
            if param not in parsed_data or parsed_data[param] is None:
                missing_params.append(param)

        if missing_params:
            st.error(f"❌ 缺少必需参数: {missing_params}")
            return None

        # 🔧 设置默认值
        for key, default_value in defaults.items():
            if key not in parsed_data or parsed_data[key] is None:
                parsed_data[key] = default_value

        # 🔍 参数类型转换和验证
        try:
            for key, value in parsed_data.items():
                if key in ['signal_polarization', 'signal_pol', 'gap']:
                    parsed_data[key] = int(float(value))
                else:
                    parsed_data[key] = float(value)
        except (ValueError, TypeError) as e:
            st.error(f"❌ 参数类型转换失败: {e}")
            return None

        return parsed_data

    except json.JSONDecodeError as e:
        st.error(f"❌ JSON解析失败: {str(e)}")
        st.error(f"AI返回内容: {content}")
        return None
    except Exception as e:
        st.error(f"❌ 参数解析异常: {str(e)}")
        return None


def process_smart_query(query, calculation_type="path_loss"):
    """处理智能查询 - 支持两种计算类型"""

    # 显示解析状态
    calc_name = "路径损耗" if calculation_type == "path_loss" else "区域场强"

    with st.spinner(f"🤖 AI正在解析{calc_name}计算需求..."):
        try:
            # 🚀 使用通用AI智能解析
            params = parse_coordinates_parameters(query, calculation_type)

            if not params:
                st.error("❌ 参数解析失败，请检查输入格式")
                return None

            # ✅ 显示解析结果
            st.success("✅ 参数解析成功！")
            with st.expander("📋 解析到的参数"):
                st.json(params)

            # ⚡ 根据类型执行不同计算
            with st.spinner(f"⚡ 正在计算{calc_name}..."):
                result = None

                if calculation_type == "path_loss":
                    result = calculate_path_loss(**params)
                    show_path_loss_results(result)
                    show_path_visualization(result)
                    show_loss_analysis(result)

                elif calculation_type == "field_strength":
                    result = calculate_field_strength(**params)
                    show_field_strength_results(result)
                    show_enhanced_baidu_map11(result)

                # 🆕 在计算成功后保存到历史记录
                if result:
                    import datetime
                    result['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    result['smart_query'] = query  # 保存原始查询
                    result['source'] = 'smart_query'
                    result['params'] = params

                    if calculation_type == "path_loss":
                        st.session_state.path_loss_history.append(result)
                    elif calculation_type == "field_strength":
                        st.session_state.field_strength_history.append(result)

                st.success(f"🎉 {calc_name}计算完成！结果已保存到历史记录。")
                return result

            # 在计算成功后添加：
        except Exception as e:
            st.error(f"❌ {calc_name}智能分析失败: {str(e)}")
            return None



def show_path_loss_results(data):

    # 🆕 主要结果展示 - 改为损耗统计信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📈 最大路径损耗",
            f"{data.get('max_loss', 0):.2f} dB",
            help="传播路径上的最大路径损耗值"
        )

    with col2:
        st.metric(
            "📉 最小路径损耗",
            f"{data.get('min_loss', 0):.2f} dB",
            help="传播路径上的最小路径损耗值"
        )

    with col3:
        st.metric(
            "📏 传播距离",
            f"{data.get('distance', 0):.2f} km",
            help="发射点到接收点的直线距离"
        )

    with col4:
        st.metric(
            "📊 平均路径损耗",
            f"{data.get('avg_loss', 0):.2f} dB",
            help="传播路径上所有计算点的平均路径损耗"
        )

    # 🆕 损耗统计分析
    st.subheader("📈 路径损耗统计分析")

    if data.get('max_loss', 0) > 0:
        # ✅ 修正这里的语法错误
        points = data.get('points', [])  # 修正：get方法的语法
        losses = [p.get('count', 0) for p in points]

        if len(losses) > 1:
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=losses,
                    nbinsx=min(20, max(1, len(losses) // 2)),
                    marker_color='skyblue',
                    opacity=0.7
                )
            ])

            fig_hist.update_layout(
                title="路径损耗分布直方图",
                xaxis_title="路径损耗 (dB)",
                yaxis_title="频次",
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("数据点不足，无法显示分布图")


def show_field_strength_results(data):
    """显示区域场强计算结果"""

    # 🆕 主要结果展示 - 场强统计信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🔥 最大场强",
            f"{data.get('max_field_strength', 0):.2f} dBμV/m",
            help="计算区域内的最大场强值"
        )

    with col2:
        st.metric(
            "❄️ 最小场强",
            f"{data.get('min_field_strength', 0):.2f} dBμV/m",
            help="计算区域内的最小场强值"
        )

    with col3:
        st.metric(
            "📐 计算半径",
            f"{data.get('rad', 0):.2f} km",
            help="场强计算的覆盖半径"
        )

    with col4:
        st.metric(
            "📊 平均场强",
            f"{data.get('avg_field_strength', 0):.2f} dBμV/m",
            help="计算区域内所有点的平均场强"
        )

    # 🆕 损耗统计分析
    st.subheader("📈 场强值统计分析")

    points = data.get('actual_points', [])
    field_strengths = [p.get('count', 0) for p in points]

    if len(field_strengths) > 1:
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=field_strengths,
                nbinsx=min(15, max(1, len(field_strengths) // 3)),
                marker_color='skyblue',
                opacity=0.7
            )
        ])

        fig_hist.update_layout(
            title="场强值分布直方图",
            xaxis_title="场强 (dBμV/m)",
            yaxis_title="频次",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("数据点不足，无法显示分布图")



def show_path_visualization(data):
    """显示路径可视化"""
    st.header("🗺️ 传播路径可视化")

    try:
        show_baidu_map(data)
    except Exception as e:
        st.error(f"❌ 百度地图加载失败: {str(e)}")
        st.info("💡 请检查百度地图API密钥和网络连接")

def generate_terrain_profile_from_csv(tx_data, rx_data, csv_path="P1812_line/output/combined_results.csv"):
    """
    直接从CSV文件读取损耗数据，结合地形和建筑物数据进行可视化
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            st.error(f"❌ 文件不存在: {csv_path}")
            return []

        # 读取CSV数据
        data = pd.read_csv(csv_path)

        # 提取有效的损耗数据点（过滤负值）
        valid_data = data[data['PredictedPL'] >= 0]

        if len(valid_data) == 0:
            st.warning("⚠️ 没有有效的损耗数据")
            return []

        # 构建地形剖面数据
        terrain_profile = []

        for idx, row in valid_data.iterrows():
            # WGS84坐标转换为BD09
            bd_lon, bd_lat = transform.wgs2bd(row["Rx_lon"], row["Rx_lat"])

            # 计算距离
            distance = calculate_distance(tx_data['lat'], tx_data['lon'], bd_lat, bd_lon)

            # 读取地形数据
            elevation, building_height = read_terrain_data_at_point(row["Rx_lat"], row["Rx_lon"])

            terrain_profile.append({
                'distance': distance,
                'lat': bd_lat,
                'lon': bd_lon,
                'wgs_lat': row["Rx_lat"],  # 保存原始WGS84坐标
                'wgs_lon': row["Rx_lon"],
                'elevation': float(elevation),
                'building_height': float(building_height),
                'total_height': float(elevation + building_height),
                'path_loss': float(row["PredictedPL"]),  # 直接从CSV读取的损耗值
                'field_strength': float(row["Predicted"]) if 'Predicted' in row else None
            })

        # 按距离排序
        terrain_profile.sort(key=lambda x: x['distance'])

        return terrain_profile

    except Exception as e:
        st.error(f"❌ 读取CSV文件时出错: {str(e)}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        return []


def read_terrain_data_at_point(wgs_lat, wgs_lon):
    """
    根据WGS84坐标读取单点的地形和建筑物高度
    """
    try:
        # 读取DEM数据
        with rasterio.open("E:/pycharm/radioMitoringA/china_dem_tif/yun_yuenan1.tif") as dem_ds:
            # 使用WGS84坐标采样
            coords = [(wgs_lon, wgs_lat)]
            elevation_vals = list(dem_ds.sample(coords))
            elevation = elevation_vals[0][0] if elevation_vals else 1500

            # 处理无效值
            if np.isnan(elevation) or elevation < -1000:
                elevation = 1500

        # 读取建筑物数据
        with rasterio.open("E:/pycharm/radioMitoringA/china_dem_tif/building_yun.tif") as building_ds:
            building_vals = list(building_ds.sample(coords))
            building_height = building_vals[0][0] if building_vals else 0

            # 处理无效值
            if np.isnan(building_height) or building_height < 0:
                building_height = 0

        return elevation, building_height

    except Exception as e:
        st.warning(f"⚠️ 读取地形数据失败: {str(e)}")
        return 1500, 0  # 返回默认值


def show_csv_based_terrain_loss_analysis(tx_data, rx_data, csv_path="P1812_line/output/combined_results.csv"):
    """
    基于CSV文件的地形损耗综合分析
    """
    st.header("📈 基于CSV的地形损耗分析")

    # 从CSV读取数据生成地形剖面
    terrain_profile = generate_terrain_profile_from_csv(tx_data, rx_data, csv_path)

    if not terrain_profile:
        st.warning("⚠️ 无法生成地形剖面数据")
        return

    # 创建综合图表
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('地形剖面', '路径损耗'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )

    distances = [p['distance'] for p in terrain_profile]
    elevations = [p['elevation'] for p in terrain_profile]
    building_heights = [p['building_height'] for p in terrain_profile]
    total_heights = [p['total_height'] for p in terrain_profile]
    path_losses = [p['path_loss'] for p in terrain_profile]

    # 🔧 计算y轴合适的显示范围
    min_elevation = min(elevations)
    max_total_height = max(total_heights)

    # 为天线高度预留空间
    antenna_space = 100  # 预留100米显示天线

    # y轴范围：从最低地形往下一点开始，到最高建筑+天线空间
    y_min = min_elevation - 50  # 往下留50米边距
    y_max = max_total_height + antenna_space

    # 第一行：地形剖面
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=elevations,
            mode='lines',
            name='地形高度',
            line=dict(color='brown', width=2),
            fill='tozeroy',  # 填充到y=0，但显示范围不包含0
            fillcolor='rgba(139, 69, 19, 0.3)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=distances,
            y=total_heights,
            mode='lines',
            name='地形+建筑',
            line=dict(color='gray', width=1),
            fill='tonexty',  # 填充到上一条线
            fillcolor='rgba(128, 128, 128, 0.3)'
        ),
        row=1, col=1
    )

    # 发射点和接收点标记
    fig.add_trace(
        go.Scatter(
            x=[distances[0], distances[-1]],
            y=[total_heights[0] + 30, total_heights[-1] + 30],  # 减少天线高度显示
            mode='markers+text',
            name='收发点',
            marker=dict(size=12, color=['red', 'blue'], symbol=['triangle-up', 'triangle-down']),
            text=['发射点', '接收点'],
            textposition=['top center', 'top center']
        ),
        row=1, col=1
    )

    # 第二行：路径损耗
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=path_losses,
            mode='lines+markers',
            name='路径损耗',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )

    # 🔧 设置y轴范围
    fig.update_yaxes(
        range=[y_min, y_max],  # 设置地形剖面的y轴范围
        title_text="高度 (m)",
        row=1, col=1
    )

    # 更新布局
    fig.update_layout(
        title="地形剖面与路径损耗分析（基于CSV数据）",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="距离 (km)", row=2, col=1)
    fig.update_yaxes(title_text="路径损耗 (dB)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 显示y轴范围信息
    st.info(f"📏 地形剖面显示范围: {y_min:.0f}m ~ {y_max:.0f}m")

    return terrain_profile

def show_loss_analysis(data):
    """
    基于CSV的损耗分析主函数
    """
    has_tx_rx_data = ('tx' in data and 'rx' in data)

    if has_tx_rx_data:
        tx_data = data['tx']
        rx_data = data['rx']

        # 检查CSV文件路径
        csv_path = "P1812_line/output/combined_results.csv"

        if os.path.exists(csv_path):
            terrain_profile = show_csv_based_terrain_loss_analysis(tx_data, rx_data, csv_path)

            if terrain_profile:
                st.success(f"✅ 成功生成基于数据的地形损耗分析")
            else:
                st.error("❌ 无法生成分析结果")
        else:
            st.error(f"❌ CSV文件不存在: {csv_path}")
            st.info("💡 请先运行路径损耗计算以生成CSV数据")
    else:
        st.warning("⚠️ 缺少发射点和接收点数据")
