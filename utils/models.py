import numpy as np
from decimal import Decimal
import os
os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'
from osgeo import _gdal
import math
import shutil
from scipy import interpolate
import time
import uuid
import pyproj
from coord_convert.transform import wgs2gcj, wgs2bd, gcj2wgs, gcj2bd, bd2wgs, bd2gcj



# 计算自由空间信号路径损耗
def free_path_loss(d, f):
    pl = 32.4 + 20 * math.log(d / 1000) + 20 * math.log(f)
    return pl


# 将腾讯坐标转换成百度坐标
def qqMapTransBMap(lon, lat):
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    x = lon
    y = lat
    z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * x_pi)
    lons = z * math.cos(theta) + 0.0065  # 转换后的坐标
    lats = z * math.sin(theta) + 0.006
    return (lons, lats)


def bd09_to_wgs84(bd_lon, bd_lat):
    # 定义百度坐标系和WGS 84坐标系的转换器
    bd_to_wgs84 = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    # 将百度坐标转换为WGS 84坐标
    wgs_lon, wgs_lat = bd_to_wgs84.transform(bd_lon, bd_lat)

    return wgs_lon, wgs_lat


# 将经纬度转换成度分形式
def getRad(input):
    out = (input * math.pi) / 180.0
    return out


# 计算两点间的距离返回的距离是千米
def getDistance(lon1, lat1, lon2, lat2):
    radLat1 = getRad(lat1)
    radLat2 = getRad(lat2)
    a = radLat1 - radLat2
    b = getRad(lon1) - getRad(lon2)
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * 6378137  # 地球半径(米);
    return round(s / 1000, 3)


# 计算360°上圆周的点经纬度
# [(25.195581, 102.650614)]
def calculate_circle_points(center_lat, center_lon, R):
    # 地球半径（单位：千米）
    earth_radius = 6371.0
    # 初始化结果列表
    points = []

    # 遍历360个角度（每度一个点）
    for angle in range(360):
        # 将角度转换为弧度
        angle_rad = math.radians(angle)

        # 计算经度偏移
        lon_offset = R / (earth_radius * math.cos(math.radians(center_lat)))

        # 计算新的经纬度坐标
        lat = round(math.degrees(math.asin(math.sin(math.radians(center_lat)) * math.cos(lon_offset) +
                                           math.cos(math.radians(center_lat)) * math.sin(lon_offset) * math.cos(
            angle_rad))), 6)
        lon = round(math.degrees(math.radians(center_lon) + math.atan2(
            math.sin(angle_rad) * math.sin(lon_offset) * math.cos(math.radians(center_lat)),
            math.cos(lon_offset) - math.sin(math.radians(center_lat)) * math.sin(math.radians(lat)))), 6)

        points.append((lat, lon))

    return points


# 计算各补点
# 起点：sp;  终点：ep;  总距离(qian米)：d;  步长(米)：r;  是否包含起始点(Ture/False)：c;
def getFillPoints(lon1, lat1, lon2, lat2, d, r, c):
    lngDiff = lon2 - lon1  # 起点与终端经度差
    latDiff = lat2 - lat1  # 起点与终端纬度差
    n = math.ceil(d / r)  # 补点的总数
    a = lngDiff / n  # 每步的经度差
    b = latDiff / n  # 每步的纬度差
    points = []  # 坐标数组

    i = 1
    while i < n:
        lon = round(lon1 + a * i, 6)
        lat = round(lat1 + b * i, 6)
        points.append({'lon': lon, 'lat': lat})
        i += 1

    if c:
        points.insert(0, {'lon': lon1, 'lat': lat1})  # 添加起点
        points.append({'lon': lon2, 'lat': lat2})  # 添加终点

    return points


# 将经纬度转换成为像素坐标
def geo2imagexy(trans, x, y):
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解




# 坐标转换：百度坐标 (BD-09) 转换为 WGS 1984
def bd09_to_wgs84(lon, lat):
    # 创建坐标转换器
    bd09 = pyproj.Proj(init='epsg:4490')  # 百度坐标系 (BD-09)
    wgs84 = pyproj.Proj(init='epsg:4326')  # WGS 1984

    # 执行坐标转换
    wgs_lon, wgs_lat = pyproj.transform(bd09, wgs84, lon, lat)
    return wgs_lon, wgs_lat




def latlon_to_pixel(file_path, lat, lon):
    dataset = _gdal.Open(file_path)
    if dataset is None:
        print("Failed to open the file.")
        return None

    geo_transform = dataset.GetGeoTransform()
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]

    x = int((lon - x_origin) / pixel_width)
    y = int((lat - y_origin) / pixel_height)

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    if x < 0 or x >= cols or y < 0 or y >= rows:
        print("Coordinates are out of the raster bounds.")
        return 0

    band = dataset.GetRasterBand(1)
    pixel_value = band.ReadAsArray(x, y, 1, 1)[0, 0]
    pixel_value = np.where(np.isnan(pixel_value), 0.0, pixel_value)

    dataset = None

    return pixel_value
