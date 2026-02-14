## 一、环境配置

### 1. 创建 Python 3.9 虚拟环境

推荐使用 Conda：

```bash
conda create -n dem_system python=3.9 -y
conda activate dem_system
或者使用：
python3.9 -m venv dem_system

# Linux / Mac
source dem_system/bin/activate

# Windows
dem_system\Scripts\activate
### 2. 安装依赖
pip install -r requirements.txt
懂了 👍

你要的是：

✅ 中文版本

✅ 直接复制到 README.md 就能用

✅ 不要外层包裹 ```markdown

✅ 不要被当成代码块

✅ GitHub 可正常渲染

下面内容 从“#”开始，到最后一行结束，直接复制进 README.md 文件即可。

（不要包含我这句话）

DEM与地表覆盖分析系统

本系统基于 Streamlit 开发，用于处理与分析：

Copernicus DEM 地形高程数据

GlobeLand30 地表覆盖类型数据

BGE-M3 文本向量模型

一、环境配置
1. 创建 Python 3.9 虚拟环境

推荐使用 Conda：

conda create -n dem_system python=3.9 -y
conda activate dem_system


或使用 venv：

python3.9 -m venv dem_system

# Linux / Mac
source dem_system/bin/activate

# Windows
dem_system\Scripts\activate

2. 安装依赖库

在项目根目录执行：

pip install -r requirements.txt

二、数据准备

根据 china_dem_tif 文件夹内提供的下载链接下载 数据。

下载完成后：

解压数据

将所有 .tif 文件放入：

china_dem_tif/

三、下载 BGE-M3 向量模型

安装下载工具：

pip install huggingface_hub


下载模型到 document_vector 文件夹：

huggingface-cli download BAAI/bge-m3 --local-dir document_vector/bge-m3


下载完成后目录结构应为：

document_vector/
└── bge-m3/

四、启动系统

激活虚拟环境后运行：

conda activate dem_system
streamlit run app.py


浏览器将自动打开系统页面
