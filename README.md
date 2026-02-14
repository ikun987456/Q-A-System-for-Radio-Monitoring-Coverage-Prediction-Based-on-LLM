🚀 1️⃣ 环境配置
1. 创建 Python 3.9 虚拟环境（推荐使用 Conda）
conda create -n dem_system python=3.9 -y
conda activate dem_system


或使用 venv：

python3.9 -m venv dem_system
source dem_system/bin/activate  # Linux / Mac
dem_system\Scripts\activate     # Windows

2. 安装依赖库

确保项目根目录下已有 requirements.txt，然后执行：

pip install -r requirements.txt

📂 2️⃣ 数据准备
🗻 2.1 Copernicus DEM 地形高程数据

请根据 china_dem_tif/ 文件夹内提供的下载链接：

下载 Copernicus DEM 数据

解压

将 .tif 文件放入：

china_dem_tif/

🌍 2.2 GlobeLand30 地表覆盖数据

请从 GlobeLand30 官方网站下载对应区域数据：

👉 http://www.globeland30.org/

下载完成后：

解压文件

将数据放入项目指定数据目录

🤖 3️⃣ 下载 BGE-M3 向量模型

在 document_vector/ 目录下下载 bge-m3 模型

推荐使用 HuggingFace 下载：

pip install huggingface_hub


然后运行：

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="document_vector/bge-m3"
)


下载完成后目录结构应为：

document_vector/
└── bge-m3/

▶ 4️⃣ 启动系统

确保已激活环境：

conda activate dem_system


运行：

streamlit run app.py


浏览器将自动打开系统界面。
