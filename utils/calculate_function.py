import re
import os
import csv
import math
from utils import models
from utils import P1812
import pandas as pd
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from coord_convert import transform
from decimal import Decimal
import matplotlib.pyplot as pl
import uuid
import shutil
import streamlit as st

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


def calculate_path_loss(tx_lat=26.128111, tx_lon=103.147275,
                       rx_lat=26.041924, rx_lon=103.215690,
                       gap=200, Pt=1, Gt=0, frequency=340,
                       tx_antenna_height=30, rx_antenna_height=1.5,
                       time_percentage=50, signal_polarization=1):
    cache = st.session_state.get('cache_system')
    if cache and cache.is_available():
        # 构建缓存参数（包含所有影响计算结果的参数）
        cache_params = {
            'tx_lat': float(tx_lat),
            'tx_lon': float(tx_lon),
            'rx_lat': float(rx_lat),
            'rx_lon': float(rx_lon),
            'gap': float(gap),
            'Pt': float(Pt),
            'Gt': float(Gt),
            'frequency': float(frequency),
            'tx_antenna_height': float(tx_antenna_height),
            'rx_antenna_height': float(rx_antenna_height),
            'time_percentage': float(time_percentage),
            'signal_polarization': int(signal_polarization)
        }
        print(f"🔍 尝试从缓存获取路径损耗结果...")
        cached_result = cache.get_cached_path_loss(cache_params)

        if cached_result:
            print("✅ 从缓存中获取路径损耗结果！")
            if 'st' in globals():  # 检查是否在Streamlit环境中
                st.success("⚡ 从缓存中获取结果，计算速度更快！")
            return cached_result

    print("🧮 执行实际路径损耗计算...")
    if 'st' in globals():
        st.info("🧮 正在计算路径损耗...")

    if not (30 <= frequency <= 6000):
        raise ValueError("频率范围必须在 30 MHz 到 6000 MHz 之间")
    if not (1 <= tx_antenna_height <= 3000):
        raise ValueError("发射机天线高度范围必须在1到3000米之间")
    if not (1 <= rx_antenna_height <= 3000):
        raise ValueError("接收机天线高度范围必须在1到3000米之间")
    if not (1 <= time_percentage <= 50):
        raise ValueError("时间概率必须在 1% 到 50% 之间")

    # coverage_code = form.cleaned_data['Coverage_Code']
    # 只考虑陆地区域，故radio=4
    radio_met_code = 3

    # 转换为 WGS84 坐标系
    wgstx_lon, wgstx_lat = transform.bd2wgs(tx_lon, tx_lat)
    wgsrx_lon, wgsrx_lat = transform.bd2wgs(rx_lon, rx_lat)

    wgs_distance = models.getDistance(wgstx_lon, wgstx_lat, wgsrx_lon, wgsrx_lat)
    distance = models.getDistance(tx_lon, tx_lat, rx_lon, rx_lat)

    containBoth = True
    gap = round(gap / 1000, 3)  # # 将间距转换为千米

    wgsallPoints = models.getFillPoints(wgstx_lon, wgstx_lat, wgsrx_lon, wgsrx_lat, wgs_distance, gap,
                                   containBoth)  # 补点数据
    bdallPoints = models.getFillPoints(tx_lon, tx_lat, rx_lon, rx_lat, wgs_distance, gap,
                                  containBoth)

    # 打开地形高度tif数据并根据WGS坐标进行索引
    gdal.UseExceptions()
    dem = gdal.Open("E:/pycharm/radioMitoringA/china_dem_tif/yun_yuenan1.tif")
    band = dem.GetRasterBand(1)
    elevation = band.ReadAsArray()  # 数组(29093,31199)

    x0, dx, dxdy, y0, dydx, dy = dem.GetGeoTransform()
    high_list = []
    prev_b = None
    for point in wgsallPoints:
        if dx == dy:
            new_ncols, new_nrows = int((y0 - point['lat']) / dx), int((point['lon'] - x0) / dx)
        else:
            new_ncols, new_nrows = int((y0 - point['lat']) / -dy), int((point['lon'] - x0) / dx)

        b = elevation[new_ncols][new_nrows]
        if b == 0:
            if prev_b is not None:
                b = prev_b
        point['high'] = round(b, 2)
        high_list.append(round(b, 2))
        prev_b = b

    # 在高程列表 high_list 中检查 NaN 值并替换为前一个元素的值
    for i in range(1, len(high_list)):
        if np.isnan(high_list[i]):
            high_list[i] = high_list[i - 1]
    dem = None

    # 打开建筑高度tif数据并根据WGS坐标进行索引
    gdal.UseExceptions()
    ds = gdal.Open("E:/pycharm/radioMitoringA/china_dem_tif//building_yun.tif")
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    building_hight_list = []
    for geojwd in wgsallPoints:
        if dx == dy:
            new_ncols, new_nrows = int((y0 - geojwd['lat']) / dx), int((geojwd['lon'] - x0) / dx)
        else:
            new_ncols, new_nrows = int((y0 - geojwd['lat']) / -dy), int((geojwd['lon'] - x0) / dx)
        building_hight_list.append(round(elevation[new_ncols][new_nrows], 2))
    building_hight_list = [0 if math.isnan(x) else x for x in building_hight_list]
    ds = None

    # 求解建筑高度平均数，以此判断区域类型
    # 0~5:农村/开阔地（2）；5~10：郊外（3）；10~15：城市/森林（4）；15之上：密集城市（5）
    average_buil = round(sum(building_hight_list) / len(building_hight_list), 2)
    if average_buil > 15:
        coverage_code = 5
    elif average_buil > 10:
        coverage_code = 4
    elif average_buil > 5:
        coverage_code = 3
    else:
        coverage_code = 2

    # 构造间距（序号）列表
    gap_list = []
    current_value = Decimal(0.0)
    while current_value <= distance:
        gap_list.append(round(current_value, 3))
        current_value += Decimal(gap)
    gap_float_list = [float(str(Decimal(decimal_str))) for decimal_str in gap_list]

    # 结合序号与lon_lat_high数据组合列表
    combined_list = []
    for item1, item2 in zip(gap_float_list, wgsallPoints):
        combined_item = {'number': item1, 'lon_lat_high': item2}
        combined_list.append(combined_item)

    template_csv_path = 'P1812_line/input_template/template.csv'
    input_folder = "P1812_line/input"
    result_folder = "P1812_line/output"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    # 实验前先将文件夹清空
    for filename in os.listdir(input_folder):
        file_path1 = os.path.join(input_folder, filename)
        if os.path.isfile(file_path1):
            os.remove(file_path1)
    for filename in os.listdir(result_folder):
        file_path2 = os.path.join(result_folder, filename)
        if os.path.isfile(file_path2):
            os.remove(file_path2)

    csv_len = len(combined_list)  # 59
    # print(f"Total items to process: {csv_len}")
    # print(f"dbfg_list length: {len(dbfg_list)}")  # 61
    # print(f"building_hight_list length: {len(building_hight_list)}")  # 61
    # print(f"high_list length: {len(high_list)}")  # 61
    # print(f"gap_list length: {len(gap_list)}")  # 59

    # 为所有海拔点都产生一个对应的data_i.csv文件
    for i in range(1, csv_len):

        chunk = combined_list[i]
        output_csv_path = os.path.join(input_folder, f'date_{i}.csv')
        extend_csv_path = os.path.join(input_folder, f'date_{i - 1}.csv')
        if i == 1:
            with open(template_csv_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                lines = list(csv_reader)  # 读取所有行

            required_lines = 45 + i  # 至少需要这么多行
            while len(lines) < required_lines:
                lines.append([''] * len(lines[0]))  # 添加空行，列数与已有行保持一致

            lines[1][1] = round(wgstx_lat, 6)
            lines[2][1] = round(wgstx_lon, 6)
            lines[3][1] = round(chunk['lon_lat_high']['lat'], 6)
            lines[4][1] = round(chunk['lon_lat_high']['lon'], 6)
            lines[9][1] = chunk['number']
            lines[36][0] = frequency
            lines[36][1] = tx_antenna_height
            lines[36][3] = rx_antenna_height
            lines[36][4] = signal_polarization
            lines[36][14] = time_percentage

            lines[43][1] = i + 1
            lines[44][0] = gap_list[0]
            lines[44][1] = high_list[0]
            lines[44][2] = coverage_code
            lines[44][3] = building_hight_list[0]
            lines[44][4] = radio_met_code
            lines[44 + i][0] = chunk['number']
            lines[44 + i][1] = chunk['lon_lat_high']['high']
            lines[44 + i][2] = coverage_code
            lines[44 + i][3] = building_hight_list[i]
            lines[44 + i][4] = radio_met_code

            # 将更新后的内容写回.csv文件并保存
            with open(output_csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(lines)

        else:
            with open(extend_csv_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                lines = list(csv_reader)

            required_lines = 45 + i
            while len(lines) < required_lines:
                lines.append([''] * len(lines[0]))

            lines[3][1] = round(chunk['lon_lat_high']['lat'], 6)
            lines[4][1] = round(chunk['lon_lat_high']['lon'], 6)
            if i == len(combined_list) - 1:
                lines[9][1] = distance
            else:
                lines[9][1] = chunk['number']

            lines[43][1] = i + 1
            lines[44 + i][0] = chunk['number']
            lines[44 + i][1] = chunk['lon_lat_high']['high']
            lines[44 + i][2] = coverage_code
            lines[44 + i][3] = building_hight_list[i]
            lines[44 + i][4] = radio_met_code

            with open(output_csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(lines)

    # 获取文件夹中所有文件
    # 选取前三个 CSV 文件进行删除
    for i in range(1, 4):
        file_delete = os.path.join(input_folder, 'date_{}.csv'.format(i))
        os.remove(file_delete)

    tol = 1e-8
    hit = 0
    total = 0

    # path to the folder containing test profiles
    pathname = 'P1812_line/input/'

    # path to the folder where the resulting log files will be saved
    out_dir = 'P1812_line/output/'

    # format of the test profile (measurement) files
    fileformat = 'Fryderyk_csv'

    # Clutter code type
    ClutterCode = 'DNR1812'

    # set to 1 if the csv log files need to be produced (together with stdout)
    flag_debug = 1

    # set to 1 if the plots of the height profile are to be shown
    flag_plot = 0

    # pathprofile is available (=1), not available (=0)
    flag_path = 1

    # set to 1 if Attachment 4 to Annex 1 is to be used for computation of
    # the spherical earth diffraction Lbs w/o terrain profile analysis

    flag4 = 0

    # set variabilities to zero and location percentage to 50
    pL = 50
    sigmaL = 0

    # begin code
    # Collect all the filenames .csv in the folder pathname that contain the profile data
    try:
        filenames = [f for f in os.listdir(pathname) if f.endswith('.csv')]
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=False)
    except:
        print("The system cannot find the given folder " + pathname)

    # create the output directory

    try:
        os.makedirs(out_dir)
    except OSError:
        if not os.path.isdir(out_dir):
            raise

    if flag_debug == 1:
        fid_all = open(out_dir + 'combined_results.csv', 'w')
        if fid_all == -1:
            raise IOError('The file combined_results.csv could not be opened')

        fid_all.write(' %s,%s,%s,%s,%s,%s\n' % (
            'Tx_lat', 'Tx_lon', 'Rx_lat', 'Rx_lon', 'Predicted', 'PredictedPL'))
    if len(filenames) < 1:
        raise IOError('There are no .csv files in the test profile folder ' + pathname)

    # figure counter
    fig_cnt = 0

    for filename1 in filenames:

        print('***********************************************\n')
        print('Processing file ' + pathname + filename1 + '\n')
        print('***********************************************\n')

        # read the file and populate sg3db input data structure

        sg3db = P1812.read_sg3_measurements2(pathname + filename1, fileformat)

        # collect intermediate results in log files (=1), or not (=0)
        sg3db.debug = flag_debug

        # pathprofile is available (=1), not available (=0)
        sg3db.pathinfo = flag_path

        # update the data structure with the Tx Power (kW)
        # 这里设置发射功率默认位1kw
        for kindex in range(0, sg3db.Ndata):
            PERP = sg3db.ERPMaxTotal[kindex]
            HRED = sg3db.HRPred[kindex]
            PkW = 10.0 ** (PERP / 10.0) * 1e-3  # kW

            if (np.isnan(PkW)):
                # use complementary information from Basic Transmission Loss and
                # received measured strength to compute the transmitter power + gain
                # E = sg3db.MeasuredFieldStrength[kindex]
                # PL = sg3db.BasicTransmissionLoss[kindex]
                # f = sg3db.frequency[kindex]
                # PdBkW = -137.2217 + E - 20 * np.log10(f) + PL
                # PkW = 10 ** (PdBkW / 10.0)
                # 修改！！！！！！
                PkW = 1

            sg3db.TransmittedPower = np.append(sg3db.TransmittedPower, PkW)

        sg3db.ClutterCode = []

        x = sg3db.x
        h_gamsl = sg3db.h_gamsl

        # # plot the profile
        if (flag_plot):
            fig_cnt = fig_cnt + 1
            newfig = plt.figure(fig_cnt)
            h_plot = plt.plot(x, h_gamsl, linewidth=2, color='k')
            plt.xlim(np.min(x), np.max(x))
            hTx = sg3db.hTx
            hRx = sg3db.hRx

            plt.title(
                'Tx: ' + sg3db.TxSiteName + ', Rx: ' + sg3db.RxSiteName + ', ' + sg3db.TxCountry + sg3db.MeasurementFileName)
            plt.grid(True)
            plt.xlabel('distance [km]')
            plt.ylabel('height [m]')

        # # plot the position of transmitter/receiver

        hTx = sg3db.hTx
        hRx = sg3db.hRx

        if (flag_plot):
            ax = plt.gca()

        for measID in range(0, len(hRx)):
            if (measID != []):
                if (measID > len(hRx) or measID < 0):
                    raise ValueError('The chosen dataset does not exist.')
                # print('Computing the fields for Dataset #%d\n', %(dataset))
                sg3db.userChoiceInt = measID
                hhRx = hRx[measID]
                hhTx = hTx[0]
                # this will be a separate function
                # Transmitter
                if (flag_plot):
                    if (sg3db.first_point_transmitter == 1):
                        plt.plot(np.array([x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0] + hhTx]), linewidth=2,
                                color='b')
                        plt.plot(x[0], h_gamsl[0] + hTx[0], marker='v', color='b')
                        plt.plot(np.array([x[-1], x[-1]]), np.array([h_gamsl[-1], h_gamsl[-1] + hhRx]), linewidth=2,
                                color='r')
                        plt.plot(x[-1], h_gamsl[-1] + hhRx, marker='v', color='r')
                    else:
                        plt.plot(np.array([x[-1], x[-1]]), np.array([h_gamsl[-1], h_gamsl[-1] + hhTx]), linewidth=2,
                                color='b')
                        plt.plot(x[-1], h_gamsl[0] + hTx[0], marker='v', color='b')
                        plt.plot(np.array([x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0] + hhRx]), linewidth=2,
                                color='r')
                        plt.plot(x[0], h_gamsl[0] + hhRx, marker='v', color='r')

                    ax = plt.gca()

            if not P1812.isempty(sg3db.coveragecode):

                # fill in the  missing fields in Rx clutter
                i = sg3db.coveragecode[-1]
                RxClutterCode, RxP1546Clutter, R2external = P1812.clutter(i, ClutterCode)
                i = sg3db.coveragecode[0]
                TxClutterCode, TxP1546Clutter, R1external = P1812.clutter(i, ClutterCode)

                sg3db.RxClutterCodeP1546 = RxP1546Clutter

                if not P1812.isempty(sg3db.h_ground_cover):
                    if not np.isnan(sg3db.h_ground_cover[-1]):
                        if (sg3db.h_ground_cover[-1] > 3):
                            sg3db.RxClutterHeight = sg3db.h_ground_cover[-1]
                        else:
                            sg3db.RxClutterHeight = R2external

                    else:
                        sg3db.RxClutterHeight = R2external

                    if not np.isnan(sg3db.h_ground_cover[0]):
                        sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                        if (sg3db.h_ground_cover[0] > 3):
                            sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                        else:
                            sg3db.TxClutterHeight = R1external

                    else:
                        sg3db.TxClutterHeight = R1external

                else:
                    sg3db.RxClutterHeight = R2external
                    sg3db.TxClutterHeight = R1external

            # Execute P.1812
            fid_log = -1
            if (flag_debug == 1):

                filename2 = out_dir + filename1[0:-4] + '_' + 'log.csv'
                fid_log = open(filename2, 'w')
                if (fid_log == -1):
                    error_str = filename2 + ' cannot be opened.'
                    raise IOError(error_str)

            sg3db.fid_log = fid_log

            sg3db.dct = 500
            sg3db.dcr = 500

            if sg3db.radio_met_code[0] == 1:  # Tx at sea
                sg3db.dct = 0

            if sg3db.radio_met_code[-1] == 1:  # Rx at sea
                sg3db.dcr = 0

            sg3db.Lb, sg3db.PredictedFieldStrength = P1812.bt_loss(sg3db.frequency[measID] / 1e3,
                                                                   sg3db.TimePercent[measID],
                                                                   sg3db.x,
                                                                   sg3db.h_gamsl,
                                                                   sg3db.h_ground_cover,
                                                                   sg3db.coveragecode,
                                                                   sg3db.radio_met_code,
                                                                   sg3db.hTx[measID],
                                                                   sg3db.hRx[measID],
                                                                   sg3db.polHVC[measID],
                                                                   sg3db.TxLAT,
                                                                   sg3db.RxLAT,
                                                                   sg3db.TxLON,
                                                                   sg3db.RxLON,
                                                                   pL=pL,
                                                                   sigmaL=sigmaL,
                                                                   Ptx=sg3db.TransmittedPower[measID],
                                                                   DN=sg3db.DN,
                                                                   N0=sg3db.N0,
                                                                   dct=sg3db.dct,
                                                                   dcr=sg3db.dcr,
                                                                   flag4=flag4,
                                                                   debug=flag_debug,
                                                                   fid_log=sg3db.fid_log)

            delta = sg3db.PredictedFieldStrength - sg3db.MeasuredFieldStrength[measID]

            if (abs(delta) <= tol):
                hit = hit + 1

            total = total + 1

            # 根据实际发射功率对场强和损耗值进行修正
            if sg3db.PredictedFieldStrength != 1:
                sg3db.PredictedFieldStrength = sg3db.PredictedFieldStrength + 10 * math.log10(Pt) + Gt
            sg3db.Lb = 139.3 - sg3db.PredictedFieldStrength + 20 * np.log10(sg3db.frequency)

            if (flag_debug):
                fid_log.close()
                fid_all.write(' %.6f,%.6f,%.6f,%.6f,%.8f,%.8f\n' % (
                    sg3db.TxLAT, sg3db.TxLON, sg3db.RxLAT, sg3db.RxLON, sg3db.PredictedFieldStrength,
                    sg3db.Lb))

    if flag_debug == 1:
        fid_all.close()
    print('Validation results: %d out of %d tests passed successfully.\n' % (hit, total))
    if hit == total:
        print('The deviation from the reference results is smaller than %g.\n' % tol)

    # 使用正则表达式提取文件名中的数字部分
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return -1  # 如果没有找到数字，返回一个负数   102.7121,25.0467   102.7619,24.9772

    # 取出output_csv中最后一个点的csv
    result_list = []
    file_list = os.listdir(result_folder)
    file_list = sorted(file_list, key=extract_number)
    file_list_str = str(file_list[-1])
    last_file = os.path.join(result_folder, file_list_str)

    with open(last_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            result_list.append(row)


    # 保存历史数据
    combined_result_path = "P1812_line/output/combined_results.csv"
    data = pd.read_csv(combined_result_path)

    # 处理实际数据点
    actual_points = []
    for _, row in data.iterrows():
        path_loss = float(row["PredictedPL"])
        # 跳过负值点
        if path_loss < 0:
            continue

        lon, lat = transform.wgs2bd(row["Rx_lon"], row["Rx_lat"])
        actual_points.append({
            "lon": float(lon),
            "lat": float(lat),
            "count": float(row["PredictedPL"])
        })

    losses = [p["count"] for p in actual_points]
    max_loss = max(losses) if losses else 0
    min_loss = min(losses) if losses else 0
    avg_loss = np.mean(losses) if losses else 0
    distance = calculate_distance(tx_lat, tx_lon, rx_lat, rx_lon)
    points = []

    if actual_points:
        first_actual_point = actual_points[0]

        # 在发射点和第一个实际点之间插入3个估计点
        for i in range(1, 4):  # 1, 2, 3
            factor = i / 4  # 0.25, 0.5, 0.75

            # 坐标线性插值
            est_lon = tx_lon + (first_actual_point["lon"] - tx_lon) * factor
            est_lat = tx_lat + (first_actual_point["lat"] - tx_lat) * factor

            est_path_loss = first_actual_point["count"] * factor

            points.append({
                "lon": est_lon,
                "lat": est_lat,
                "count": est_path_loss
            })

        # 添加所有实际点
        points.extend(actual_points)

    safe_points = [
        {"lon": float(p["lon"]), "lat": float(p["lat"]), "count": float(p["count"])}
        for p in points
    ]

    result =  {
        "max_loss": float(max_loss),
        "min_loss": float(min_loss),
        "avg_loss": float(avg_loss),
        "distance": float(distance),
        "tx": {"lat": float(tx_lat), "lon": float(tx_lon)},
        "rx": {"lat": float(rx_lat), "lon": float(rx_lon)},
        "points": safe_points,
    }
    # 🔍 调试：打印返回值类型
    # print("\n=== DEBUG: calculate_path_loss 返回值数据类型 ===")
    # for k, v in result.items():
    #     if isinstance(v, dict):
    #         for kk, vv in v.items():
    #             print(f"{k}.{kk} -> {type(vv)} -> {vv}")
    #     elif isinstance(v, list):
    #         for i, item in enumerate(v[:5]):  # 只打印前 5 个
    #             print(f"{k}[{i}] type: {type(item)}")
    #             if isinstance(item, dict):
    #                 for kk, vv in item.items():
    #                     print(f"  {kk} -> {type(vv)} -> {vv}")
    #     else:
    #         print(f"{k} -> {type(v)} -> {v}")
    # print("============================================\n")
    # 💾 保存到缓存
    if cache and cache.is_available():
        print("💾 保存路径损耗结果到缓存...")
        save_success = cache.cache_path_loss_result(cache_params, result)
        if save_success:
            print("✅ 路径损耗结果已保存到缓存")
            if 'st' in globals():
                st.success("💾 结果已保存到缓存，下次计算更快！")
        else:
            print("⚠️ 路径损耗结果保存到缓存失败")

    return result


def calculate_field_strength(lat=26.128111, lon=103.147275, rad=1, gap=10, Pt=1, Gt=0,
               frequency=340, tx_antenna_height=30, rx_antenna_height=1.5, time_percentage=50, signal_pol=1):
    cache = st.session_state.get('cache_system')
    if cache and cache.is_available():
        # 构建缓存参数（包含所有影响计算结果的参数）
        cache_params = {
            'tx_lat': float(lat),
            'tx_lon': float(lon),
            'rad': float(rad),
            'gap': float(gap),
            'Pt': float(Pt),
            'Gt': float(Gt),
            'frequency': float(frequency),
            'tx_antenna_height': float(tx_antenna_height),
            'rx_antenna_height': float(rx_antenna_height),
            'time_percentage': float(time_percentage),
            'signal_polarization': int(signal_pol)
        }
        # 尝试从缓存获取结果
        print(f"🔍 尝试从缓存获取区域场强结果...")
        cached_result = cache.get_cached_field_strength(cache_params)

        if cached_result:
            print("✅ 从缓存中获取区域场强结果！")
            if 'st' in globals():
                st.success("⚡ 从缓存中获取区域场强结果！")
            return cached_result

        # 🧮 执行实际计算
    print("🧮 执行实际区域场强计算...")
    if 'st' in globals():
        st.info("🧮 正在计算区域场强...")

    tx_lat = lat
    tx_lon = lon

    radio_met_code = 4

    # 将输入的百度坐标系 (lon, lat) 转换为 WGS-84 坐标系 (
    wgstx_lon, wgstx_lat = transform.bd2wgs(lon, lat)

    # 以 (wgstx_lat, wgstx_lon) 为中心、rad 为半径，生成圆形边界上的点集合
    circle_points = models.calculate_circle_points(wgstx_lat, wgstx_lon, rad)

    containBoth = True
    gap = round(gap / 1000, 3)
    allpoints = []
    for circle_point in circle_points:
        points = models.getFillPoints(wgstx_lon, wgstx_lat, circle_point[1], circle_point[0], rad, gap, containBoth)
        allpoints.append(points)
    # [[{'lon': 102.853745, 'lat': 24.83487}, {'lon': 102.853745, 'lat': 24.839825}]]
    points_list = [point for sublist in allpoints for point in sublist]
    # [{'lon': 102.85365, 'lat': 24.839824}, {'lon': 102.853554, 'lat': 24.844778}]

    # 获取无法预测的前三个点经纬度集合
    threepoints_list = [[sublist[1], sublist[2], sublist[3]] for sublist in allpoints]

    # 该数据用于填补绘图，故将其转换为百度坐标
    for sublist in threepoints_list:
        for point in sublist:
            point['lon'], point['lat'] = transform.wgs2bd(point['lon'], point['lat'])  # 使用wgs2bd函数直接转换

    # 处理地形高度tif数据
    gdal.UseExceptions()
    dem = gdal.Open("E:/pycharm/radioMitoringA/china_dem_tif/yun_yuenan1.tif")
    band = dem.GetRasterBand(1)
    elevation = band.ReadAsArray()
    nrows, ncols = elevation.shape
    x0, dx, dxdy, y0, dydx, dy = dem.GetGeoTransform()
    high_list = []
    prev_b = None
    for point in points_list:
        if dx == dy:
            new_ncols, new_nrows = int((y0 - point['lat']) / dx), int((point['lon'] - x0) / dx)
        else:
            new_ncols, new_nrows = int((y0 - point['lat']) / -dy), int((point['lon'] - x0) / dx)

        b = elevation[new_ncols][new_nrows]

        if b == 0:
            if prev_b is not None:
                b = prev_b
        point['high'] = b
        high_list.append(b)
        prev_b = b

    # 在高程列表 high_list 中检查 NaN 值并替换为前一个元素的值
    for i in range(1, len(high_list)):
        if np.isnan(high_list[i]):
            high_list[i] = high_list[i - 1]

    dem = None

    # 打开地表覆盖tif数据
    gdal.UseExceptions()
    ds = gdal.Open("E:/pycharm/radioMitoringA/china_dem_tif/yun_yuenan_fg3.tif")
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    dbfg_list = []
    for geojwd in points_list:
        if dx == dy:
            new_ncols, new_nrows = int((y0 - geojwd['lat']) / dx), int((geojwd['lon'] - x0) / dx)
        else:
            new_ncols, new_nrows = int((y0 - geojwd['lat']) / -dy), int((geojwd['lon'] - x0) / dx)
        if elevation[new_ncols][new_nrows] == 0:
            print("输入坐标超出系统计算范围！！！")
        dbfg_list.append(elevation[new_ncols][new_nrows])

    # 将地表覆盖类型转换为ITU输入参数
    mapping = {
        10: 2, 30: 3, 50: 2, 70: 2, 90: 2, 100: 1,
        20: 4, 80: 4,
        40: 3,
        60: 1
    }
    # 使用列表推导式进行转换
    dbfg_list = [mapping[item] for item in dbfg_list]
    # 释放生成的数组以释放服务器的运行内存
    ds = None
    elevation = None
    band = None

    # 打开建筑高度tif数据并根据WGS坐标进行索引
    gdal.UseExceptions()
    ds = gdal.Open("E:/pycharm/radioMitoringA/china_dem_tif/building_yun.tif")
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()
    nrows, ncols = elevation.shape
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    building_hight_list = []

    for geojwd in points_list:
        if dx == dy:
            new_ncols, new_nrows = int((y0 - geojwd['lat']) / dx), int((geojwd['lon'] - x0) / dx)
        else:
            new_ncols, new_nrows = int((y0 - geojwd['lat']) / -dy), int((geojwd['lon'] - x0) / dx)
        building_hight_list.append(round(elevation[new_ncols][new_nrows], 2))

    building_hight_list = [0 if math.isnan(x) else x for x in building_hight_list]
    ds = None

    # 求解建筑高度平均数，以此判断区域类型
    # 0~5:农村/开阔地（2）；5~10：郊外（3）；10~15：城市/森林（4）；15之上：密集城市（5）
    # 1. 把列表转成 float64 数组，避免 float32 极限
    building_hight_arr = np.array(building_hight_list, dtype=np.float64)
    # 2. 用 nanmean 自动跳过 NaN，且内部累加是 64 位，避免溢出
    average_buil = round(float(np.nanmean(building_hight_arr)), 2)
    # average_buil = round(sum(building_hight_list) / len(building_hight_list), 2)
    if average_buil > 15:
        coverage_code = 5
    elif average_buil > 10:
        coverage_code = 4
    elif average_buil > 5:
        coverage_code = 3
    else:
        coverage_code = 2

    # 分割出各条路径的建筑高度子列表
    split_blhigh_list = []
    split_size = len(building_hight_list) // 360
    for i in range(0, len(building_hight_list), split_size):
        chunk = building_hight_list[i:i + split_size]
        split_blhigh_list.append(chunk)
    # print(split_blhigh_list)

    # 分割出各条路径的高度子列表
    split_high_list = []
    split_size = len(high_list) // 360
    for i in range(0, len(high_list), split_size):
        chunk = high_list[i:i + split_size]
        split_high_list.append(chunk)
    # print(split_high_list)

    # 构造序号间距列表
    gap_list = []
    current_value = Decimal(0.0)
    while current_value < rad:
        gap_list.append(round(current_value, 3))
        current_value += Decimal(gap)
    gap_list.append(round(rad, 3))
    gap_float_list = [float(str(Decimal(decimal_str))) for decimal_str in gap_list]
    # print(gap_float_list)

    combined_list = []
    for i in range(len(allpoints)):
        temp_list = []
        # 使用zip函数将对应位置的元素一一配对
        for point, h, g, bh in zip(allpoints[i], split_high_list[i], gap_float_list, split_blhigh_list[i]):
            # 创建一个新的字典，包含原始point字典的所有键值对
            new_point = point.copy()
            # 添加新的键值对到new_point字典中
            new_point['high'] = h
            new_point['gap'] = g
            new_point['buil_high'] = bh
            # 将new_point添加到temp_list中
            temp_list.append(new_point)
        # 将temp_list添加到combined_list中
        combined_list.append(temp_list)

    template_csv_path = 'P1812_area/input_template/template.csv'
    input_folder = "P1812_area/input"
    output_folder = "P1812_area/output"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 实验前先将validation_profiles和validation_results文件夹清空
    for filename in os.listdir(input_folder):
        file_path1 = os.path.join(input_folder, filename)
        if os.path.isfile(file_path1):
            os.remove(file_path1)
    for filename in os.listdir(output_folder):
        file_path2 = os.path.join(output_folder, filename)
        if os.path.isfile(file_path2):
            os.remove(file_path2)

    # 为所有海拔点都产生一个对应的data_i.csv文件
    files_to_delete = []
    files_to_check = []
    for path in combined_list:
        for i in range(len(path)):
            chunk = path[i]

            # 使用uuid生成唯一标识符作为文件名的一部分
            unique_id = str(uuid.uuid4().hex)

            output_csv_path = os.path.join(input_folder, f'date_{i}.csv')
            extend_csv_path = os.path.join(input_folder, f'date_{i - 1}.csv')

            if i == 0:
                with open(template_csv_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    lines = list(csv_reader)

                required_lines = 45 + i
                while len(lines) < required_lines:
                    lines.append([''] * len(lines[0]))

                lines[1][1] = round(wgstx_lat, 6)
                lines[2][1] = round(wgstx_lon, 6)
                lines[3][1] = round(chunk['lat'], 6)
                lines[4][1] = round(chunk['lon'], 6)
                lines[9][1] = chunk['gap']
                lines[36][0] = frequency
                lines[36][1] = tx_antenna_height
                lines[36][3] = rx_antenna_height
                lines[36][4] = signal_pol
                lines[36][14] = time_percentage

                lines[43][1] = i + 1
                lines[44][0] = chunk['gap']
                lines[44][1] = chunk['high']
                lines[44][2] = coverage_code
                lines[44][3] = chunk['buil_high']
                lines[44][4] = radio_met_code

                # 将更新后的内容写回.csv文件并保存
                with open(output_csv_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(lines)

                # 删除之前的未使用唯一标识符命名的文件
                files_to_delete.append(output_csv_path)

            else:
                with open(extend_csv_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    lines = list(csv_reader)

                required_lines = 45 + i
                while len(lines) < required_lines:
                    lines.append([''] * len(lines[0]))

                lines[3][1] = round(chunk['lat'], 6)
                lines[4][1] = round(chunk['lon'], 6)
                if i == len(path) - 1:
                    lines[9][1] = rad
                else:
                    lines[9][1] = chunk['gap']

                lines[43][1] = i + 1
                lines[44 + i][0] = chunk['gap']
                lines[44 + i][1] = chunk['high']
                lines[44 + i][2] = coverage_code
                lines[44 + i][3] = chunk['buil_high']
                lines[44 + i][4] = radio_met_code

                with open(output_csv_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(lines)

                # 删除之前的未使用唯一标识符命名的文件
                files_to_delete.append(output_csv_path)

                # 复制文件并设置不同的文件名
                new_file_name = f'date_{unique_id}_{i}.csv'
                target_path = os.path.join(input_folder, new_file_name)
                shutil.copyfile(output_csv_path, target_path)
                # 将输出文件路径添加到检查列表中
                files_to_check.append(target_path)

    # 在整个循环结束后删除文件date_0.csv、date_1.csv...
    for file_to_delete in files_to_delete:
        if os.path.exists(file_to_delete):  # 检查文件是否存在
            os.remove(file_to_delete)

    # 循环结束后检查并删除少于四个点的文件文件
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                lines = list(csv_reader)
            if int(lines[43][1]) < 5:
                os.remove(file_path)

    tol = 1e-8
    hit = 0
    total = 0

    # path to the folder containing test profiles
    pathname = 'P1812_area/input/'

    # path to the folder where the resulting log files will be saved
    out_dir = 'P1812_area/output/'

    # format of the test profile (measurement) files
    fileformat = 'Fryderyk_csv'

    # Clutter code type
    ClutterCode = 'DNR1812'

    # set to 1 if the csv log files need to be produced (together with stdout)
    flag_debug = 1

    # set to 1 if the plots of the height profile are to be shown
    flag_plot = 0

    # pathprofile is available (=1), not available (=0)
    flag_path = 1

    # set to 1 if Attachment 4 to Annex 1 is to be used for computation of
    # the spherical earth diffraction Lbs w/o terrain profile analysis

    flag4 = 0

    # set variabilities to zero and location percentage to 50
    pL = 50
    sigmaL = 0

    # begin code
    # Collect all the filenames .csv in the folder pathname that contain the profile data
    try:
        filenames = [f for f in os.listdir(pathname) if f.endswith('.csv')]
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=False)
    except:
        print("The system cannot find the given folder " + pathname)

    # create the output directory

    try:
        os.makedirs(out_dir)
    except OSError:
        if not os.path.isdir(out_dir):
            raise

    if (flag_debug == 1):
        fid_all = open(out_dir + 'combined_results.csv', 'w')
        if (fid_all == -1):
            raise IOError('The file combined_results.csv could not be opened')

        fid_all.write('# # %s,%s,%s,%s,%s,%s\n' % (
            'Tx_lat', 'Tx_lon', 'Rx_lat', 'Rx_lon', 'Predicted', 'PredictedPL'))
    if (len(filenames) < 1):
        raise IOError('There are no .csv files in the test profile folder ' + pathname)

    # figure counter
    fig_cnt = 0

    for filename1 in filenames:

        print('***********************************************\n')
        print('Processing file ' + pathname + filename1 + '\n')
        print('***********************************************\n')

        # read the file and populate sg3db input data structure

        sg3db = P1812.read_sg3_measurements2(pathname + filename1, fileformat)

        # collect intermediate results in log files (=1), or not (=0)
        sg3db.debug = flag_debug

        # pathprofile is available (=1), not available (=0)
        sg3db.pathinfo = flag_path

        # update the data structure with the Tx Power (kW)
        for kindex in range(0, sg3db.Ndata):
            PERP = sg3db.ERPMaxTotal[kindex]
            HRED = sg3db.HRPred[kindex]
            PkW = 10.0 ** (PERP / 10.0) * 1e-3  # kW

            if (np.isnan(PkW)):
                # use complementary information from Basic Transmission Loss and
                # received measured strength to compute the transmitter power + gain
                # E = sg3db.MeasuredFieldStrength[kindex]
                # PL = sg3db.BasicTransmissionLoss[kindex]
                # f = sg3db.frequency[kindex]
                # PdBkW = -137.2217 + E - 20 * np.log10(f) + PL
                # PkW = 10 ** (PdBkW / 10.0)
                # 修改！！！！！！
                # f = sg3db.frequency[kindex]
                # E = 106.9 - 20 * math.log(rad)
                # PL = 139.3 - E + 20 * math.log(f)
                # PdBkW = -137.2217 + E - 20 * np.log10(f) + PL
                # PkW = 10 ** (PdBkW / 10.0)
                PkW = 1

            sg3db.TransmittedPower = np.append(sg3db.TransmittedPower, PkW)

        sg3db.ClutterCode = []

        x = sg3db.x
        h_gamsl = sg3db.h_gamsl

        # # plot the profile
        if (flag_plot):
            fig_cnt = fig_cnt + 1
            newfig = pl.figure(fig_cnt)
            h_plot = pl.plot(x, h_gamsl, linewidth=2, color='k')
            pl.xlim(np.min(x), np.max(x))
            hTx = sg3db.hTx
            hRx = sg3db.hRx

            pl.title(
                'Tx: ' + sg3db.TxSiteName + ', Rx: ' + sg3db.RxSiteName + ', ' + sg3db.TxCountry + sg3db.MeasurementFileName)
            pl.grid(True)
            pl.xlabel('distance [km]')
            pl.ylabel('height [m]')

        # # plot the position of transmitter/receiver

        hTx = sg3db.hTx
        hRx = sg3db.hRx

        if (flag_plot):
            ax = pl.gca()

        for measID in range(0, len(hRx)):
            if (measID != []):
                if (measID > len(hRx) or measID < 0):
                    raise ValueError('The chosen dataset does not exist.')
                # print('Computing the fields for Dataset #%d\n', %(dataset))
                sg3db.userChoiceInt = measID
                hhRx = hRx[measID]
                hhTx = hTx[0]
                # this will be a separate function
                # Transmitter
                if (flag_plot):
                    if (sg3db.first_point_transmitter == 1):
                        pl.plot(np.array([x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0] + hhTx]), linewidth=2,
                                color='b')
                        pl.plot(x[0], h_gamsl[0] + hTx[0], marker='v', color='b')
                        pl.plot(np.array([x[-1], x[-1]]), np.array([h_gamsl[-1], h_gamsl[-1] + hhRx]), linewidth=2,
                                color='r')
                        pl.plot(x[-1], h_gamsl[-1] + hhRx, marker='v', color='r')
                    else:
                        pl.plot(np.array([x[-1], x[-1]]), np.array([h_gamsl[-1], h_gamsl[-1] + hhTx]), linewidth=2,
                                color='b')
                        pl.plot(x[-1], h_gamsl[0] + hTx[0], marker='v', color='b')
                        pl.plot(np.array([x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0] + hhRx]), linewidth=2,
                                color='r')
                        pl.plot(x[0], h_gamsl[0] + hhRx, marker='v', color='r')

                    ax = pl.gca()

            if not P1812.isempty(sg3db.coveragecode):

                # fill in the  missing fields in Rx clutter
                i = sg3db.coveragecode[-1]
                RxClutterCode, RxP1546Clutter, R2external = P1812.clutter(i, ClutterCode)
                i = sg3db.coveragecode[0]
                TxClutterCode, TxP1546Clutter, R1external = P1812.clutter(i, ClutterCode)

                sg3db.RxClutterCodeP1546 = RxP1546Clutter

                if not P1812.isempty(sg3db.h_ground_cover):
                    if not np.isnan(sg3db.h_ground_cover[-1]):
                        if (sg3db.h_ground_cover[-1] > 3):
                            sg3db.RxClutterHeight = sg3db.h_ground_cover[-1]
                        else:
                            sg3db.RxClutterHeight = R2external

                    else:
                        sg3db.RxClutterHeight = R2external

                    if not np.isnan(sg3db.h_ground_cover[0]):
                        sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                        if (sg3db.h_ground_cover[0] > 3):
                            sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                        else:
                            sg3db.TxClutterHeight = R1external

                    else:
                        sg3db.TxClutterHeight = R1external

                else:
                    sg3db.RxClutterHeight = R2external
                    sg3db.TxClutterHeight = R1external

            # Execute P.1812
            fid_log = -1
            if (flag_debug == 1):

                filename2 = out_dir + filename1[0:-4] + '_' + 'log.csv'
                fid_log = open(filename2, 'w')
                if (fid_log == -1):
                    error_str = filename2 + ' cannot be opened.'
                    raise IOError(error_str)

            sg3db.fid_log = fid_log

            sg3db.dct = 500
            sg3db.dcr = 500

            if sg3db.radio_met_code[0] == 1:  # Tx at sea
                sg3db.dct = 0

            if sg3db.radio_met_code[-1] == 1:  # Rx at sea
                sg3db.dcr = 0

            sg3db.Lb, sg3db.PredictedFieldStrength = P1812.bt_loss(sg3db.frequency[measID] / 1e3,
                                                                   sg3db.TimePercent[measID],
                                                                   sg3db.x,
                                                                   sg3db.h_gamsl,
                                                                   sg3db.h_ground_cover,
                                                                   sg3db.coveragecode,
                                                                   sg3db.radio_met_code,
                                                                   sg3db.hTx[measID],
                                                                   sg3db.hRx[measID],
                                                                   sg3db.polHVC[measID],
                                                                   sg3db.TxLAT,
                                                                   sg3db.RxLAT,
                                                                   sg3db.TxLON,
                                                                   sg3db.RxLON,
                                                                   pL=pL,
                                                                   sigmaL=sigmaL,
                                                                   Ptx=sg3db.TransmittedPower[measID],
                                                                   DN=sg3db.DN,
                                                                   N0=sg3db.N0,
                                                                   dct=sg3db.dct,
                                                                   dcr=sg3db.dcr,
                                                                   flag4=flag4,
                                                                   debug=flag_debug,
                                                                   fid_log=sg3db.fid_log)

            delta = sg3db.PredictedFieldStrength - sg3db.MeasuredFieldStrength[measID]

            if (abs(delta) <= tol):
                hit = hit + 1

            total = total + 1

            # 根据实际发射功率对场强和损耗值进行修正,并将结果保存在combined文档
            # if sg3db.PredictedFieldStrength != 1:
            #     sg3db.PredictedFieldStrength = sg3db.PredictedFieldStrength + 10 * math.log10(Pt) + Gt
            # sg3db.Lb = 139.3 - sg3db.PredictedFieldStrength + 20 * math.log10(sg3db.frequency)

            # 先把 PredictedFieldStrength 转成标量
            fs = float(sg3db.PredictedFieldStrength)

            if fs != 1:
                fs = fs + 10 * math.log10(float(Pt)) + float(Gt)

            sg3db.Lb = 139.3 - fs + 20 * math.log10(float(sg3db.frequency))

            if (flag_debug):
                fid_log.close()
                fid_all.write(' %.6f,%.6f,%.6f,%.6f,%.8f,%.8f\n' % (
                    sg3db.TxLAT, sg3db.TxLON, sg3db.RxLAT, sg3db.RxLON, sg3db.PredictedFieldStrength,
                    sg3db.Lb))

    if (flag_debug == 1):
        fid_all.close()

    print('Validation results: %d out of %d tests passed successfully.\n' % (hit, total))
    if (hit == total):
        print('The deviation from the reference results is smaller than %g.\n' % (tol))

    # 读取CSV文件,生成预测区域所需列表文件
    csv_file_path = "P1812_area/output/combined_results.csv"
    df = pd.read_csv(csv_file_path)

    actual_points = []
    for _, row in df.iterrows():
        lon, lat = transform.wgs2bd(row["Rx_lon"], row["Rx_lat"])
        actual_points.append({
            "lon": float(lon),
            "lat": float(lat),
            "count": float(row["Predicted"])
        })

    field_strength = [p["count"] for p in actual_points]
    max_field_strength = max(field_strength) if field_strength else 0
    min_field_strength = min(field_strength) if field_strength else 0
    avg_field_strength = np.mean(field_strength) if field_strength else 0

    points = []

    for sublist in threepoints_list:
        for point in sublist:
            points.append({
                "lon": float(point['lon']),
                "lat": float(point['lat']),
                "count": 0.0
            })

    points.extend(actual_points)

    result = {
        "actual_points":actual_points,
        "points": points,
        "lat": float(tx_lat),
        "lon": float(tx_lon),
        "rad": float(rad),
        "max_field_strength": max_field_strength,
        "min_field_strength": min_field_strength,
        "avg_field_strength": float(avg_field_strength)
    }
    # 🔍 调试：打印返回值类型
    # print("\n=== DEBUG: calculate 返回值数据类型 ===")
    # for k, v in result.items():
    #     if isinstance(v, dict):
    #         for kk, vv in v.items():
    #             print(f"{k}.{kk} -> {type(vv)} -> {vv}")
    #     elif isinstance(v, list):
    #         for i, item in enumerate(v[:5]):  # 只打印前 5 个
    #             print(f"{k}[{i}] type: {type(item)}")
    #             if isinstance(item, dict):
    #                 for kk, vv in item.items():
    #                     print(f"  {kk} -> {type(vv)} -> {vv}")
    #     else:
    #         print(f"{k} -> {type(v)} -> {v}")
    # print("============================================\n")
    # 💾 保存到缓存
    if cache and cache.is_available():
        print("💾 保存区域场强结果到缓存...")
        save_success = cache.cache_field_strength_result(cache_params, result)
        if save_success:
            print("✅ 区域场强结果已保存到缓存")
            if 'st' in globals():
                st.success("💾 区域场强结果已保存到缓存")
    return result
