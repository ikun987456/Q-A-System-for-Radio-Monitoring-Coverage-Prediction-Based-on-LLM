import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


def show_baidu_map(data, ak="INfiyL0BrLG3avXpKSu6RdlTOHxEeKDj"):
    if not all(k in data for k in ["tx", "rx", "points"]):
        st.error("数据格式错误：缺少tx或rx坐标或points")
        return

    tx = data["tx"]
    rx = data["rx"]
    points = data["points"]

    if not all(k in tx for k in ["lat", "lon"]) or not all(k in rx for k in ["lat", "lon"]):
        st.error("坐标格式错误：每个点需要包含lat和lon")
        return

    if not points:
        st.error("没有数据点")
        return

    count_values = [p["count"] for p in points]
    min_count = min(count_values)
    max_count = max(count_values)

    # 先分析数据分布
    count_array = np.array(count_values)
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    percentile_values = np.percentile(count_array, percentiles)


    # 改进的分档方法 - 使用百分位数确保每个颜色区间都有数据
    result_json_lists = [[] for _ in range(60)]

    # 使用更智能的分档策略
    for point in points:
        loss_value = point["count"]

        # 使用百分位数进行分档，确保分布更均匀
        if loss_value <= percentile_values[1]:  # 0-10%
            index = int((loss_value - min_count) / (percentile_values[1] - min_count + 0.001) * 10)
            index = max(0, min(9, index))
        elif loss_value <= percentile_values[2]:  # 10-25%
            index = int(
                (loss_value - percentile_values[1]) / (percentile_values[2] - percentile_values[1] + 0.001) * 10) + 10
            index = max(10, min(19, index))
        elif loss_value <= percentile_values[4]:  # 25-75%
            index = int(
                (loss_value - percentile_values[2]) / (percentile_values[4] - percentile_values[2] + 0.001) * 20) + 20
            index = max(20, min(39, index))
        elif loss_value <= percentile_values[5]:  # 75-90%
            index = int(
                (loss_value - percentile_values[4]) / (percentile_values[5] - percentile_values[4] + 0.001) * 10) + 40
            index = max(40, min(49, index))
        else:  # 90-100%
            index = int((loss_value - percentile_values[5]) / (max_count - percentile_values[5] + 0.001) * 10) + 50
            index = max(50, min(59, index))

        result_json_lists[index].append({
            "lng": point["lon"],
            "lat": point["lat"],
            "count": point["count"]
        })

    # 显示分档后的统计
    non_empty_groups = [i for i, group in enumerate(result_json_lists) if len(group) > 0]
    st.write(f"**实际使用的颜色档次：** {len(non_empty_groups)}/60")

    # 确保红色区间有数据的颜色方案
    colors_js = json.dumps([
        # 深蓝到浅蓝 (0-9)
        '#000080', '#000099', '#0000B3', '#0000CC', '#0000E6',
        '#0000FF', '#1919FF', '#3333FF', '#4D4DFF', '#6666FF',
        # 蓝绿过渡 (10-19)
        '#6666FF', '#4D79FF', '#338CFF', '#199FFF', '#00B3FF',
        '#00C6FF', '#00D9FF', '#00ECFF', '#00FFFF', '#00FFEC',
        # 绿色系 (20-39)
        '#00FFD9', '#00FFC6', '#00FFB3', '#00FF9F', '#00FF8C',
        '#00FF79', '#00FF66', '#00FF4D', '#00FF33', '#00FF19',
        '#00FF00', '#19FF00', '#33FF00', '#4DFF00', '#66FF00',
        '#79FF00', '#8CFF00', '#9FFF00', '#B3FF00', '#C6FF00',
        # 黄色系 (40-49)
        '#D9FF00', '#ECFF00', '#FFFF00', '#FFEC00', '#FFD900',
        '#FFC600', '#FFB300', '#FF9F00', '#FF8C00', '#FF7900',
        # 红色系 (50-59) - 确保高损耗显示为红色
        '#FF6600', '#FF5200', '#FF3F00', '#FF2C00', '#FF1900',
        '#FF0600', '#FF0000', '#F20000', '#E60000', '#D90000'
    ])

    # 修改图例显示，使用实际的百分位数值
    legend_info = {
        'low': f"{min_count:.1f}-{percentile_values[2]:.1f}",
        'mid_low': f"{percentile_values[2]:.1f}-{percentile_values[4]:.1f}",
        'mid_high': f"{percentile_values[4]:.1f}-{percentile_values[5]:.1f}",
        'high': f"{percentile_values[5]:.1f}-{max_count:.1f}"
    }

    # 其余HTML代码保持不变，只修改图例部分
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>路径损耗地图</title>
    <style>
        body {{ margin: 0; padding: 0; }}
        #container {{ position: relative; width: 100%; height: 600px; }}
        #map {{ width: 100%; height: 100%; }}
        #legend {{ 
            position: absolute; 
            top: 10px; 
            right: 10px; 
            background: rgba(255,255,255,0.95); 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            font-size: 12px;
            max-width: 220px;
            z-index: 1000;
        }}
        #legend h4 {{ margin: 0 0 10px 0; color: #333; }}
        .legend-item {{ 
            display: flex; 
            align-items: center; 
            margin: 4px 0; 
        }}
        .legend-color {{ 
            width: 16px; 
            height: 16px; 
            margin-right: 8px; 
            border: 1px solid #999;
            border-radius: 2px;
        }}
        .legend-gradient {{
            height: 8px;
            background: linear-gradient(to right, #000080, #00FF00, #FFFF00, #FF0000);
            margin: 8px 0;
            border: 1px solid #ccc;
        }}
        #status {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
        }}
        .map-controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
        }}
        .control-btn {{
            background: rgba(255,255,255,0.9);
            border: 1px solid #ddd;
            padding: 6px 10px;
            margin-right: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }}
        .control-btn:hover {{
            background: #007cff;
            color: white;
        }}
    </style>
</head>
<body>
<div id="container">
    <div id="map"></div>

    <!-- 改进的图例 -->
    <div id="legend">
        <h4>📡 路径损耗图例</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF0000; border-radius: 50%;"></div>
            <span>发射点 (Tx)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #0066FF; border-radius: 50%;"></div>
            <span>接收点 (Rx)</span>
        </div>
        <div style="margin: 12px 0; border-top: 1px solid #ddd; padding-top: 10px;">
            <div><strong>损耗分布:</strong></div>
            <div class="legend-gradient"></div>
            <div class="legend-item">
                <div class="legend-color" style="background: #000080;"></div>
                <span>低损耗: {legend_info['low']} dB</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #00FF00;"></div>
                <span>中低损耗: {legend_info['mid_low']} dB</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FFFF00;"></div>
                <span>中高损耗: {legend_info['mid_high']} dB</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF0000;"></div>
                <span>高损耗: {legend_info['high']} dB</span>
            </div>
        </div>
        <div style="font-size: 10px; color: #666; margin-top: 8px; border-top: 1px solid #eee; padding-top: 6px;">
            总点数: {len(points)}<br>
            使用档次: {len(non_empty_groups)}/60<br>
            频率: {data.get('frequency', 'N/A')} MHz
        </div>
    </div>

    <!-- 状态显示 -->
    <div id="status">正在加载地图...</div>
</div>

<script>
    // 地图初始化代码保持不变...
    function initMap() {{
        try {{
            document.getElementById('status').innerHTML = '正在初始化地图...';

            if (typeof BMap === 'undefined') {{
                throw new Error('百度地图API加载失败');
            }}

            var map = new BMap.Map("map");
            window.currentMap = map;

            // 发射接收点代码保持不变...
            var tx = {json.dumps(tx)};
            var rx = {json.dumps(rx)};
            var txPt = new BMap.Point(tx.lon, tx.lat);
            var rxPt = new BMap.Point(rx.lon, rx.lat);

            // 自定义图标
            var txIcon = new BMap.Icon(
                'data:image/svg+xml;base64,' + btoa(`
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                        <circle cx="10" cy="10" r="8" fill="#FF0000" stroke="white" stroke-width="2"/>
                        <text x="10" y="14" text-anchor="middle" fill="white" font-size="10" font-weight="bold">T</text>
                    </svg>
                `),
                new BMap.Size(20, 20),
                {{ anchor: new BMap.Size(10, 10) }}
            );

            var rxIcon = new BMap.Icon(
                'data:image/svg+xml;base64,' + btoa(`
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                        <rect x="2" y="2" width="16" height="16" fill="#0066FF" stroke="white" stroke-width="2"/>
                        <text x="10" y="14" text-anchor="middle" fill="white" font-size="10" font-weight="bold">R</text>
                    </svg>
                `),
                new BMap.Size(20, 20),
                {{ anchor: new BMap.Size(10, 10) }}
            );

            var txMarker = new BMap.Marker(txPt, {{ icon: txIcon }});
            var rxMarker = new BMap.Marker(rxPt, {{ icon: rxIcon }});

            map.addOverlay(txMarker);
            map.addOverlay(rxMarker);

            // 连接线
            var line = new BMap.Polyline([txPt, rxPt], {{
                strokeColor: "#FF6B6B",
                strokeWeight: 3,
                strokeOpacity: 0.8
            }});
            map.addOverlay(line);

            // 热力点数据
            var colors = {colors_js};
            var result_json_lists = {json.dumps(result_json_lists)};
            window.pointCollections = [];
            window.highLossCollections = [];  // 存储高损耗点集合

            var totalPoints = 0;
            var highLossPoints = 0;

            for (var i = 0; i < result_json_lists.length; i++) {{
                var pts = result_json_lists[i];
                if (pts.length === 0) continue;

                totalPoints += pts.length;

                // 标记高损耗点 (索引50-59为红色区间)
                var isHighLoss = i >= 50;
                if (isHighLoss) {{
                    highLossPoints += pts.length;
                }}

                try {{
                    var pointCollection = new BMap.PointCollection(
                        pts.map(function(p) {{ 
                            return new BMap.Point(p.lng, p.lat); 
                        }}),
                        {{
                            shape: BMAP_POINT_SHAPE_CIRCLE, 
                            color: colors[i], 
                            size: isHighLoss ? 4 : 4  // 高损耗点稍大一些
                        }}
                    );

                    map.addOverlay(pointCollection);
                    window.pointCollections.push(pointCollection);

                    if (isHighLoss) {{
                        window.highLossCollections.push(pointCollection);
                    }}

                }} catch (e) {{
                    console.warn('添加点集合失败:', e);
                }}
            }}

            // 设置视角
            try {{
                var allPoints = [txPt, rxPt];
                for (var i = 0; i < result_json_lists.length && allPoints.length < 20; i++) {{
                    for (var j = 0; j < result_json_lists[i].length && allPoints.length < 20; j++) {{
                        var pt = result_json_lists[i][j];
                        allPoints.push(new BMap.Point(pt.lng, pt.lat));
                    }}
                }}

                if (allPoints.length > 2) {{
                    var view = map.getViewport(allPoints);
                    map.centerAndZoom(view.center, Math.max(view.zoom - 1, 10));
                }} else {{
                    map.centerAndZoom(txPt, 13);
                }}
            }} catch (e) {{
                map.centerAndZoom(txPt, 13);
            }}

            map.enableScrollWheelZoom(true);
            map.addControl(new BMap.NavigationControl());
            map.addControl(new BMap.ScaleControl());

            document.getElementById('status').innerHTML = 
                `✅ 地图加载完成！总点数: ${{totalPoints}}, 高损耗点: ${{highLossPoints}}`;

            setTimeout(function() {{
                document.getElementById('status').style.display = 'none';
            }}, 4000);

        }} catch (error) {{
            console.error('地图初始化错误:', error);
            document.getElementById('status').innerHTML = '❌ 错误: ' + error.message;
        }}
    }}

    // 控制函数
    var heatPointsVisible = true;
    var showingHighLossOnly = false;

    function toggleHeatPoints() {{
        if (window.pointCollections) {{
            heatPointsVisible = !heatPointsVisible;
            for (var i = 0; i < window.pointCollections.length; i++) {{
                if (heatPointsVisible) {{
                    window.currentMap.addOverlay(window.pointCollections[i]);
                }} else {{
                    window.currentMap.removeOverlay(window.pointCollections[i]);
                }}
            }}
        }}
    }}

    function showHighLossOnly() {{
        if (window.pointCollections && window.highLossCollections) {{
            showingHighLossOnly = !showingHighLossOnly;

            if (showingHighLossOnly) {{
                // 隐藏所有点，只显示高损耗点
                for (var i = 0; i < window.pointCollections.length; i++) {{
                    window.currentMap.removeOverlay(window.pointCollections[i]);
                }}
                for (var i = 0; i < window.highLossCollections.length; i++) {{
                    window.currentMap.addOverlay(window.highLossCollections[i]);
                }}
                document.querySelector('.control-btn:last-child').innerHTML = '显示所有点';
            }} else {{
                // 显示所有点
                for (var i = 0; i < window.pointCollections.length; i++) {{
                    window.currentMap.addOverlay(window.pointCollections[i]);
                }}
                document.querySelector('.control-btn:last-child').innerHTML = '仅显示高损耗';
            }}
        }}
    }}

    function resetView() {{
        if (window.currentMap) {{
            var tx = {json.dumps(tx)};
            var rx = {json.dumps(rx)};
            var txPt = new BMap.Point(tx.lon, tx.lat);
            var rxPt = new BMap.Point(rx.lon, rx.lat);
            var view = window.currentMap.getViewport([txPt, rxPt]);
            window.currentMap.centerAndZoom(view.center, Math.max(view.zoom - 1, 10));
        }}
    }}

    if (typeof BMap !== 'undefined') {{
        initMap();
    }} else {{
        window.addEventListener('load', function() {{
            setTimeout(initMap, 1000);
        }});
    }}
</script>

<script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak={ak}&callback=initMap"></script>

</body>
</html>
    """

    st.components.v1.html(html, height=620)
