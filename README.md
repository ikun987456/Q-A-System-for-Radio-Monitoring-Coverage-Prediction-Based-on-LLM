## 一、环境配置

### 1. 创建 Python 3.9 虚拟环境

推荐使用 Conda：

conda create -n dem_system python=3.9 -y
conda activate dem_system
或者使用：
python3.9 -m venv dem_system

# Linux / Mac
source dem_system/bin/activate

# Windows
dem_system\Scripts\activate

### 2. 安装依赖
在项目根目录执行：

pip install -r requirements.txt

### 二、数据准备

根据 china_dem_tif 文件夹内提供的下载链接下载 数据。

下载完成后：

解压数据

将所有 .tif 文件放入：

china_dem_tif/

### 三、下载 BGE-M3 向量模型

安装下载工具：

pip install huggingface_hub


下载模型到 document_vector 文件夹：

huggingface-cli download BAAI/bge-m3 --local-dir document_vector/bge-m3


下载完成后目录结构应为：

document_vector/
└── bge-m3/

### 四、启动系统

激活虚拟环境后运行：

conda activate dem_system
streamlit run app.py


浏览器将自动打开系统页面
