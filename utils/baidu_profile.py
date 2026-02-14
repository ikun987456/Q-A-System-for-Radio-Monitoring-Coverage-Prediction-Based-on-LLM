import streamlit as st
import streamlit.components.v1 as components
import json


def show_enhanced_baidu_map11(data, ak="百度地图api"):
    """增强版百度地图显示（使用海量点技术）"""
    if not data or "points" not in data or len(data["points"]) == 0:
        st.error("❌ 没有可显示的数据点")
        return

    tx_lat = data["lat"]
    tx_lon = data["lon"]
    rad = data.get("rad", 1)

    # 数据分类处理
    valid_points = []
    for p in data["points"]:
        if (isinstance(p.get("lat"), (int, float)) and
                isinstance(p.get("lon"), (int, float)) and
                isinstance(p.get("count"), (int, float))):
            point = {
                "lat": float(p["lat"]),
                "lon": float(p["lon"]),
                "count": float(p["count"])
            }
            valid_points.append(point)

    if len(valid_points) == 0:
        st.error("❌ 没有有效的数据点")
        return

    calc_points = [p for p in valid_points if p["count"] > 0]
    zero_points = [p for p in valid_points if p["count"] == 0]

    if len(calc_points) == 0:
        st.error("❌ 没有可计算的数据点（所有点count都为0）")
        return

    # 统计信息
    calc_counts = [p["count"] for p in calc_points]
    min_count = min(calc_counts)
    max_count = max(calc_counts)
    avg_count = sum(calc_counts) / len(calc_counts)

    # 将数据按场强值分层（60层对应60种颜色）
    layers_count = 59
    step = (max_count - min_count) / layers_count if max_count > min_count else 1

    # 🆕 创建 60 层：59层有值 + 1层无值
    point_layers = [[] for _ in range(60)]
    count_levels = []

    for i in range(layers_count):
        level_value = max_count - (i * step)
        count_levels.append(level_value)

    count_levels.append(0)  # 无值点的值为0

    # 将点分配到不同层级
    for point in calc_points:
        layer_index = min(int((max_count - point["count"]) / step), layers_count - 1)
        if layer_index < 0:
            layer_index = 0
        point_layers[layer_index].append({
            "lng": point["lon"],
            "lat": point["lat"],
            "count": point["count"]
        })

    for point in zero_points:
        point_layers[59].append({
            "lng": point["lon"],
            "lat": point["lat"],
            "count": 0
        })

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak={ak}"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        #container {{
            position: relative;
            height: 100%;
            width: 100%;
        }}

        #map {{
            height: 100%;
            width: 100%;
        }}
        
        #controls-toggle-btn {{
            position: absolute;
            top: 12px;
            right: 12px;
            z-index: 1100;
            background: #2196F3;
            color: white;
            border-radius: 6px;
            padding: 8px 14px;
            font-weight: 600;
            font-size: 12px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(33,150,243,0.13);
            transition: background 0.25s;
            user-select: none;
        }}
        
        #controls-toggle-btn:hover {{
            background: #1976D2;
        }}


        #controls {{
            position: absolute;
            top: 55px;
            right: 10px;
            background: rgba(255, 255, 255, 0.92);
            padding: 8px 10px;
            border-radius: 6px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            backdrop-filter: blur(8px);
            min-width: 160px;
            max-width: 200px;
            max-height: 65vh;
            overflow-y: auto;
            font-size: 11px;
        }}

        #controls button {{
            margin: 1px;
            padding: 4px 8px;
            border: none;
            border-radius: 3px;
            background: #2196F3;
            color: white;
            cursor: pointer;
            font-size: 10px;
            transition: background 0.3s;
            width: 100%;
            box-sizing: border-box;
        }}
        
        #controls button:hover {{
            background: #1976D2;
        }}
        
        #controls button.active {{
            background: #4CAF50;
        }}

        #controls input {{
            font-size: 10px;
            width: 50px;
            padding: 2px 4px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}

        .color-legend {{
            display: none;
            margin-top: 8px;
        }}

        .color-legend.show {{
            display: block;
        }}

        .color-columns {{
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
        }}

        .color-column {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1px;
        }}

        .color-item {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 8px;
        }}

        .color-box {{
            width: 12px;
            height: 10px;
            border: 1px solid #ddd;
            border-radius: 1px;
        }}

        .threshold-controls {{
            margin-top: 8px;
            padding: 6px;
            background: rgba(255,193,7,0.1);
            border-radius: 4px;
        }}

        .threshold-input-group {{
            display: flex;
            align-items: center;
            gap: 3px;
            margin-bottom: 4px;
        }}

        .stats-info {{
            margin-top: 8px;
            padding: 6px;
            background: rgba(76, 175, 80, 0.1);
            border-radius: 4px;
            font-size: 9px;
            line_height:1.2;
        }}

        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            z-index: 2000;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading">
            <div style="text-align: center;">
                <div style="margin-bottom: 10px;">🗺️ 地图加载中...</div>
                <div style="font-size: 12px; color: #666;">正在渲染 {len(valid_points):,} 个数据点</div>
            </div>
        </div>
        
        <div id="controls-toggle-btn" onclick="toggleControlsPanel()">
          <span id="controls-btn-icon" style="font-size:13px;">显示控制 ⬇️</span>
        </div>

        <div id="map"></div>
        
        <div id="controls" style="display:none;">
            <div style="font-weight: 600; margin-bottom: 10px; color: #333;">🎛️ 显示控制</div>
            <button id="showPointsBtn" onclick="showHeatMap()">显示热力图</button>
            <button id="hidePointsBtn" onclick="hideHeatMap()">隐藏热力图</button>
            <button id="showLegendBtn" onclick="toggleColorLegend()">颜色对照表</button>
            <button onclick="zoomIn()">放大 +</button>
            <button onclick="zoomOut()">缩小 -</button>
            <button onclick="resetView()">重置视图</button>

            <div class="threshold-controls">
                <div style="font-weight: 600; margin-bottom: 8px; color: #333;">🎯 阈值筛选</div>
                <div class="threshold-input-group">
                    <label style="font-size: 11px;">阈值:</label>
                    <input type="number" id="thresholdInput" value="50" min="0" max="200">
                    <span style="font-size: 10px; color: #666;">dBμV/m</span>
                </div>
                <button onclick="showStrongSignals()">强信号区域</button>
                <button onclick="showWeakSignals()">弱信号区域</button>
                <button onclick="showAllSignals()">显示全部</button>
            </div>

            <div class="color-legend" id="colorLegend">
                <div style="font-weight: 600; margin-bottom: 10px; color: #333;">颜色-场强（dBμV/m）对照</div>
                <div class="color-columns" id="colorColumns">
                    <!-- 颜色对照表将在这里动态生成 -->
                </div>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        var map;
        var pointCollections = [];
        var circleOverlay;
        var txMarker;
        var isHeatMapVisible = false;
        var isLegendVisible = false;

        // 60种颜色数组（从深红到蓝色渐变）
        var colors = [
            'rgb(90, 0, 0)', 'rgb(100, 0, 0)', 'rgb(110, 0, 0)', 'rgb(120, 0, 0)', 'rgb(130, 0, 0)',
            'rgb(140, 0, 0)', 'rgb(150, 0, 0)', 'rgb(170, 0, 0)', 'rgb(190, 0, 0)', 'rgb(210, 0, 0)',
            'rgb(230, 0, 0)', 'rgb(255, 0, 0)', 'rgb(255, 30, 0)', 'rgb(255, 60, 0)', 'rgb(255, 90, 0)',
            'rgb(255, 120, 0)', 'rgb(255, 150, 0)', 'rgb(255, 170, 0)', 'rgb(255, 190, 0)', 'rgb(255, 210, 0)',
            'rgb(255, 230, 0)', 'rgb(255, 240, 0)', 'rgb(255, 255, 0)', 'rgb(240, 255, 0)', 'rgb(215, 255, 0)',
            'rgb(190, 255, 0)', 'rgb(165, 255, 0)', 'rgb(140, 255, 0)', 'rgb(115, 255, 0)', 'rgb(90, 255, 0)',
            'rgb(65, 255, 0)', 'rgb(40, 255, 0)', 'rgb(15, 255, 0)', 'rgb(0, 255, 0)', 'rgb(0, 255, 30)',
            'rgb(0, 255, 60)', 'rgb(0, 255, 90)', 'rgb(0, 255, 120)', 'rgb(0, 255, 150)', 'rgb(0, 255, 180)',
            'rgb(0, 255, 210)', 'rgb(0, 255, 240)', 'rgb(0, 255, 270)', 'rgb(0, 240, 255)', 'rgb(0, 210, 255)',
            'rgb(0, 180, 255)', 'rgb(0, 150, 255)', 'rgb(0, 120, 255)', 'rgb(0, 90, 255)', 'rgb(0, 60, 255)',
            'rgb(0, 30, 255)', 'rgb(0, 0, 255)', 'rgb(0, 0, 270)', 'rgb(0, 0, 240)', 'rgb(0, 0, 210)',
            'rgb(0, 0, 180)', 'rgb(0, 0, 150)', 'rgb(0, 0, 120)', 'rgb(0, 0, 90)', 'rgb(0,0,0,0)',

            // 无值点灰色
           
        ];

        var pointLayers = {json.dumps(point_layers)};
        var countLevels = {json.dumps(count_levels)};
        var minCount = {min_count};
        var maxCount = {max_count};
        var radius = {rad} * 1000 + 150;

        function initMap() {{
            try {{
                setTimeout(() => document.getElementById('loading').style.display = 'none', 1000);

                map = new BMap.Map("map", {{ mapType: BMAP_HYBRID_MAP }});
                var txPt = new BMap.Point({tx_lon}, {tx_lat});

                // 启用地图拖动和其他交互功能
                map.enableDragging();
                map.enableScrollWheelZoom();
                map.enableDoubleClickZoom();
                map.enableKeyboard();

                // 🔧 发射点标记 - 使用Circle代替Marker
                txMarker = new BMap.Circle(txPt, 50, {{  // 50米半径的圆圈
                    strokeColor: "#FF4444",     // 边框红色
                    strokeWeight: 3,            // 边框宽度
                    strokeOpacity: 1,           // 边框透明度
                    fillColor: "#FF4444",       // 填充红色
                    fillOpacity: 0.8            // 填充透明度
                }});

                // 发射点信息窗口
                txMarker.addEventListener('click', function(e) {{
                    var infoContent = `
                        <div style="padding: 15px; min-width: 250px;">
                            <h4 style="margin: 0 0 10px 0; color: #FF4444;">📡 发射点信息</h4>
                            <p><strong>经纬度:</strong> {tx_lat:.6f}, {tx_lon:.6f}</p>
                            <p><strong>覆盖半径:</strong> {rad} km</p>
                            <p><strong>有效覆盖率:</strong> {len(calc_points) / len(valid_points) * 100:.1f}%</p>
                            <div style="margin-top: 10px; padding: 8px; background: rgba(255,68,68,0.1); border-radius: 4px; font-size: 11px;">
                                💡 使用右上角控制面板可以筛选不同强度的信号区域
                            </div>
                        </div>
                    `;
                    var infoWindow = new BMap.InfoWindow(infoContent, {{ width: 300, height: 180 }});
                    map.openInfoWindow(infoWindow, txPt);
                }});

                map.addOverlay(txMarker);

                // 创建覆盖范围圆圈
                circleOverlay = new BMap.Circle(txPt, radius, {{
                    strokeColor: "#CCCCCC",
                    strokeWeight: 2,
                    strokeOpacity: 1,
                    fillColor: "#CCCCCC",
                    fillOpacity: 0.3
                }});

                // 创建海量点图层
                for (var i = 0; i < pointLayers.length; i++) {{
                    if (pointLayers[i].length > 0) {{
                        var colorIndex = i;
                        if (colorIndex >= colors.length) colorIndex = colors.length - 1;

                        var color = colors[colorIndex];

                        var points = pointLayers[i].map(function(item) {{
                            return new BMap.Point(item.lng, item.lat);
                        }});

                        var pointCollection = new BMap.PointCollection(points, {{
                            shape: BMAP_POINT_SHAPE_CIRCLE,
                            color: color,
                            size: 3
                        }});

                        // 保存层索引和数据
                        pointCollection.layerIndex = i;
                        pointCollection.pointData = pointLayers[i];

                        map.addOverlay(pointCollection);
                        pointCollections.push(pointCollection);
                    }}
                }}

                // 生成颜色对照表
                generateColorLegend();

                // 强制以发射点为中心显示
                map.centerAndZoom(txPt, 15);

                // 确保发射点在视野中心
                setTimeout(function() {{
                    map.panTo(txPt);
                }}, 500);

                // 默认显示热力图
                isHeatMapVisible = true;
                document.getElementById('showPointsBtn').classList.add('active');

                // 智能设置默认阈值
                setSmartThreshold();

                console.log('✅ 地图初始化完成，共', pointCollections.length, '个点集层');

            }} catch (error) {{
                console.error('地图初始化错误:', error);
                document.getElementById('loading').innerHTML = 
                    '<div style="color: red; text-align: center;">❌ 地图初始化失败<br><small>' + error.message + '</small></div>';
            }}
        }}
        
        var controlsOpen = false;
        function toggleControlsPanel() {{
            var controlsPanel = document.getElementById('controls');
            var btnIcon = document.getElementById('controls-btn-icon');
            if (controlsOpen) {{
                controlsPanel.style.display = 'none';
                btnIcon.textContent = '显示控制 ⬇️';
            }} else {{
                controlsPanel.style.display = 'block';
                btnIcon.textContent = '收起控制 ⬆️';
            }}
            controlsOpen = !controlsOpen;
        }}
        
        function generateColorLegend() {{
            var container = document.getElementById('colorColumns');
            var itemsPerColumn = Math.ceil(colors.length / 3);

            for (var col = 0; col < 3; col++) {{
                var columnDiv = document.createElement('div');
                columnDiv.className = 'color-column';

                var startIdx = col * itemsPerColumn;
                var endIdx = Math.min(startIdx + itemsPerColumn, colors.length);

                for (var i = startIdx; i < endIdx; i++) {{
                    var itemDiv = document.createElement('div');
                    itemDiv.className = 'color-item';

                    var colorBox = document.createElement('div');
                    colorBox.className = 'color-box';
                    colorBox.style.backgroundColor = colors[i];

                    var label = document.createElement('span');
                    var value = countLevels[i];
                    label.textContent = (value >= 0 ? value.toFixed(1) : '-' + (-value).toFixed(1));

                    itemDiv.appendChild(colorBox);
                    itemDiv.appendChild(label);
                    columnDiv.appendChild(itemDiv);
                }}

                container.appendChild(columnDiv);
            }}
        }}

        function setSmartThreshold() {{
            var thresholdInput = document.getElementById('thresholdInput');
            var smartThreshold = 50;

            // 如果所有数据都大于50，设置为中位数
            if (minCount > 50) {{
                smartThreshold = Math.round((minCount + maxCount) / 2);
            }}
            // 如果所有数据都小于50，设置为平均值
            else if (maxCount < 50) {{
                smartThreshold = Math.round((minCount + maxCount) / 2);
            }}

            thresholdInput.value = smartThreshold;
        }}

        function getThreshold() {{
            var thresholdInput = document.getElementById('thresholdInput');
            var threshold = parseFloat(thresholdInput.value);

            // 如果用户输入的阈值超出数据范围，自动调整
            if (threshold < minCount) {{
                threshold = minCount;
                thresholdInput.value = threshold.toFixed(1);
            }} else if (threshold > maxCount) {{
                threshold = maxCount;
                thresholdInput.value = threshold.toFixed(1);
            }}

            return threshold;
        }}

        function showHeatMap() {{
            map.removeOverlay(circleOverlay);
            pointCollections.forEach(function(pointCollection) {{
                pointCollection.show();
            }});
            isHeatMapVisible = true;

            document.getElementById('showPointsBtn').classList.add('active');
            document.getElementById('hidePointsBtn').classList.remove('active');
        }}

        function hideHeatMap() {{
            pointCollections.forEach(function(pointCollection) {{
                pointCollection.hide();
            }});
            map.addOverlay(circleOverlay);
            isHeatMapVisible = false;

            document.getElementById('hidePointsBtn').classList.add('active');
            document.getElementById('showPointsBtn').classList.remove('active');
        }}

        function toggleColorLegend() {{
            var legend = document.getElementById('colorLegend');
            var btn = document.getElementById('showLegendBtn');

            if (isLegendVisible) {{
                legend.classList.remove('show');
                btn.classList.remove('active');
                btn.textContent = '颜色对照表';
                isLegendVisible = false;
            }} else {{
                legend.classList.add('show');
                btn.classList.add('active');
                btn.textContent = '隐藏对照表';
                isLegendVisible = true;
            }}
        }}

        function zoomIn() {{
            var currentZoom = map.getZoom();
            map.setZoom(currentZoom + 1);
        }}

        function zoomOut() {{
            var currentZoom = map.getZoom();
            map.setZoom(currentZoom - 1);
        }}

        function showStrongSignals() {{
            var threshold = getThreshold();

            pointCollections.forEach(function(pointCollection) {{
                var layerValue = countLevels[pointCollection.layerIndex];
                if (layerValue >= threshold) {{
                    pointCollection.show();
                }} else {{
                    pointCollection.hide();
                }}
            }});

            map.removeOverlay(circleOverlay);
            console.log('显示强于', threshold, 'dBμV/m的信号区域');
        }}

        function showWeakSignals() {{
            var threshold = getThreshold();

            pointCollections.forEach(function(pointCollection) {{
                var layerValue = countLevels[pointCollection.layerIndex];
                if (layerValue < threshold) {{
                    pointCollection.show();
                }} else {{
                    pointCollection.hide();
                }}
            }});

            map.removeOverlay(circleOverlay);
            console.log('显示弱于', threshold, 'dBμV/m的信号区域');
        }}

        function showAllSignals() {{
            pointCollections.forEach(function(pointCollection) {{
                pointCollection.show();
            }});
            map.removeOverlay(circleOverlay);
            console.log('显示所有信号区域');
        }}

        function resetView() {{
            var txPt = new BMap.Point({tx_lon}, {tx_lat});
            map.centerAndZoom(txPt, 15);

            setTimeout(function() {{
                map.panTo(txPt);
            }}, 300);

            showAllSignals();
        }}

        // 检查API并初始化
        if (typeof BMap === 'undefined') {{
            document.getElementById('loading').innerHTML = 
                '<div style="color: red; text-align: center;">❌ 百度地图API加载失败</div>';
        }} else {{
            initMap();
        }}
    </script>
</body>
</html>
"""

    components.html(html, height=750)

