import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import os


def show_calculation_results(data):
    """显示计算结果"""
    st.header("📊 路径损耗计算结果")

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


def show_field_strength_results(result, params):
    """显示场强覆盖计算结果 - 从CSV文件读取数据"""

    # 场强覆盖CSV文件路径
    csv_path = "P1812_area/output/combined_results.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        st.error(f"❌ 找不到场强覆盖结果文件: {csv_path}")
        return

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        st.info(f"✅ 成功读取场强覆盖数据，共 {len(df)} 条记录")
    except Exception as e:
        st.error(f"❌ 读取CSV文件失败: {str(e)}")
        return

    # 检查必要的列
    if 'Predicted' not in df.columns:
        st.error(f"❌ CSV文件缺少场强数据列 'Predicted'")
        st.info(f"📋 CSV文件现有列: {list(df.columns)}")
        return

    st.success("🗺️ 场强覆盖计算结果")

    # 提取场强数据
    field_strengths = df['Predicted'].values

    # 计算统计值
    max_field = field_strengths.max()
    min_field = field_strengths.min()
    avg_field = field_strengths.mean()
    std_field = field_strengths.std()
    coverage_radius = result.get('rad', 0)
    coverage_area = np.pi * coverage_radius ** 2

    # 关键指标显示
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📶 最大场强",
            value=f"{max_field:.2f} dBμV/m"
        )

    with col2:
        st.metric(
            label="📶 最小场强",
            value=f"{min_field:.2f} dBμV/m"
        )

    with col3:
        st.metric(
            label="📏 覆盖半径",
            value=f"{coverage_radius} km"
        )

    with col4:
        st.metric(
            label="📍 覆盖面积",
            value=f"{coverage_area:.1f} km²"
        )

    # 图表展示
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # 场强分布直方图
        st.markdown("**场强分布直方图**")

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=field_strengths,
            nbinsx=20,
            marker_color='lightgreen',
            marker_line_color='darkgreen',
            marker_line_width=1.2,
            opacity=0.8
        ))

        fig1.update_layout(
            xaxis_title="场强 (dBμV/m)",
            yaxis_title="频次",
            height=400,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        # 场强统计表格
        st.markdown("**场强统计**")

        stats_data = {
            '统计指标': ['场强范围', '最大场强', '最小场强', '平均场强', '标准偏差'],
            '数值': [
                f"{max_field - min_field:.2f} dBμV/m",
                f"{max_field:.2f} dBμV/m",
                f"{min_field:.2f} dBμV/m",
                f"{avg_field:.2f} dBμV/m",
                f"{std_field:.2f} dBμV/m"
            ]
        }

        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        st.info(f"📍 **计算点数**: {len(df)} 个覆盖点")
        st.info(f"📁 **数据文件**: {csv_path}")

    # 场强覆盖热力图
    if 'Rx_lat' in df.columns and 'Rx_lon' in df.columns:
        st.markdown("**场强覆盖热力图**")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df['Rx_lon'],
            y=df['Rx_lat'],
            mode='markers',
            marker=dict(
                size=8,
                color=field_strengths,
                colorscale='Viridis',
                colorbar=dict(title="场强 (dBμV/m)"),
                showscale=True
            ),
            text=[f"场强: {fs:.1f} dBμV/m" for fs in field_strengths],
            hovertemplate="经度: %{x}<br>纬度: %{y}<br>%{text}<extra></extra>",
            name="覆盖点"
        ))

        # 标记发射机位置
        tx_lat = result.get('lat', df['Tx_lat'].iloc[0] if 'Tx_lat' in df.columns else 0)
        tx_lon = result.get('lon', df['Tx_lon'].iloc[0] if 'Tx_lon' in df.columns else 0)
        fig2.add_trace(go.Scatter(
            x=[tx_lon],
            y=[tx_lat],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='发射机',
            hovertemplate="发射机位置<br>经度: %{x}<br>纬度: %{y}<extra></extra>"
        ))

        fig2.update_layout(
            xaxis_title="经度",
            yaxis_title="纬度",
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # CSV数据预览和下载
    with st.expander("📊 CSV数据预览 (前10行)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("**数据概况:**")
        st.write(f"- 总记录数: {len(df)}")
        st.write(f"- 数据列: {', '.join(df.columns)}")

        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 下载完整CSV数据",
            data=csv_data,
            file_name=f"场强覆盖计算结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # 参数详情
    with st.expander("📋 计算参数详情", expanded=False):
        param_data = {
            '参数名称': ['发射机纬度', '发射机经度', '计算半径', '覆盖面积', '计算点数'],
            '参数值': [
                f"{result.get('lat', 'N/A')}°",
                f"{result.get('lon', 'N/A')}°",
                f"{result.get('rad', 'N/A')} km",
                f"{coverage_area:.1f} km²",
                f"{len(df)} 个"
            ]
        }

        df_params = pd.DataFrame(param_data)
        st.dataframe(df_params, use_container_width=True, hide_index=True)


        with col2:
            # 损耗变化范围
            loss_range = data.get('max_loss', 0) - data.get('min_loss', 0)

            metrics_data = [
                ["损耗范围", f"{loss_range:.2f} dB"],
                ["最大损耗", f"{data.get('max_loss', 0):.2f} dB"],
                ["最小损耗", f"{data.get('min_loss', 0):.2f} dB"],
                ["平均损耗", f"{data.get('avg_loss', 0):.2f} dB"],
            ]

            metrics_df = pd.DataFrame(metrics_data, columns=["统计指标", "数值"])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
