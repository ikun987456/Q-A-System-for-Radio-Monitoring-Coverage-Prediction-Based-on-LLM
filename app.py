import os
import sys
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from llm_handler import process_smart_query,show_path_loss_results,show_field_strength_results,show_path_visualization,show_loss_analysis
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from utils.calculate_function import calculate_path_loss, calculate_field_strength
from document_vector.faiss_vectorstore import FaissVectorStore, BGEEmbedder, DeepSeekLLM, RAGChain
from new_build_vector import incremental_update_knowledge_base
from datetime import datetime
from database import CalculationCache
from utils.baidu_profile import show_enhanced_baidu_map11

# ===== 初始化 =====
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 知识库配置
INDEX_PATH = "document_vector/faiss_index"

# 🔧 初始化缓存系统
@st.cache_resource
def init_cache_system():
    """初始化缓存系统"""
    return CalculationCache()

# 知识库初始化函数
@st.cache_resource
def initialize_rag_components():
    """初始化RAG组件并缓存"""
    try:
        print("正在初始化知识库组件...")
        embedder = BGEEmbedder()
        vectorstore = FaissVectorStore(embedder, INDEX_PATH)
        vectorstore.load()

        llm = DeepSeekLLM(api_key=API_KEY, base_url=BASE_URL)
        rag_chain = RAGChain(vectorstore, llm)
        return rag_chain
    except Exception as e:
        st.error(f"知识库初始化失败: {str(e)}")
        return None


def show_main_page():
    """显示居中的主页面"""

    # 🎨 修复按钮居中的CSS
    # 🎨 修复页面布局和输入框显示问题
    st.markdown("""
     <style>
     /* 🔧 整体页面布局 - 确保有足够空间 */
     .main .block-container {
         display: flex !important;
         flex-direction: column !important;
         justify-content: center !important;
         align-items: center !important;
         min-height: 85vh !important;  /* 🔧 减少高度，留出空间 */
         padding: 2rem 1rem 4rem 1rem !important;  /* 🔧 增加底部padding */
         max-width: 900px !important;
         margin: 0 auto !important;
     }

     /* 🔧 修复主标题 - 确保一行显示 */
     .main-title {
         font-size: 2.2rem !important;
         font-weight: 700;
         color: #1f1f1f;
         text-align: center;
         margin-bottom: 2rem;
         white-space: nowrap !important;
         overflow: visible !important;  /* 🔧 允许显示完整 */
         width: 100% !important;
         max-width: 800px !important;
     }

     /* 🔧 响应式标题 */
     @media (max-width: 768px) {
         .main-title {
             font-size: 1.8rem !important;
         }
     }

     @media (max-width: 480px) {
         .main-title {
             font-size: 1.5rem !important;
             white-space: normal !important;  /* 小屏幕允许换行 */
         }
     }

     /* 🔵 强力修复 chat_input 位置和显示 */
     [data-testid="stChatInput"] {
         position: relative !important;
         bottom: auto !important;
         left: auto !important;
         right: auto !important;
         width: 100% !important;
         max-width: 500px !important;
         margin: 2rem auto 2rem auto !important;  /* 🔧 增加上下边距 */
         z-index: 10 !important;
         transform: none !important;
         height: auto !important;  /* 🔧 确保高度自适应 */
     }

     [data-testid="stChatInput"] > div {
         position: relative !important;
         bottom: auto !important;
         width: 100% !important;
         margin: 0 auto !important;
         height: auto !important;  /* 🔧 确保容器高度正确 */
     }

     /* 🔵 修复输入框本身的显示 - 改为蓝色边框 */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        height: 60px !important;
        min-height: 60px !important;
        border: 3px solid #2196F3 !important;  /* 🔵 蓝色边框，与按钮一致 */
        border-radius: 25px !important;
        background-color: #f8fafe !important;  /* 🔵 淡蓝色背景，与按钮一致 */
        font-size: 1.1rem !important;
        padding: 18px 25px !important;
        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.2) !important;  /* 🔵 蓝色阴影 */
        transition: all 0.3s ease !important;
        resize: none !important;
        width: 100% !important;
        box-sizing: border-box !important;
        line-height: 1.4 !important;
        color: #1976D2 !important;  /* 🔵 蓝色文字 */
    }

    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] input:focus {
        border-color: #1976D2 !important;  /* 🔵 聚焦时深蓝色边框 */
        background-color: #ffffff !important;
        box-shadow: 0 0 0 4px rgba(33, 150, 243, 0.3) !important;  /* 🔵 蓝色光晕 */
        outline: none !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #64B5F6 !important;  /* 🔵 蓝色占位符文字 */
        font-style: italic !important;
    }


     /* 🔧 确保按钮容器有足够空间 */
     .element-container:has(.stButton) {
         display: flex !important;
         justify-content: center !important;
         width: 100% !important;
         margin: 0.5rem 0 !important;
     }

     /* 🔧 按钮样式 */
     .stButton {
         display: flex !important;
         justify-content: center !important;
         width: 100% !important;
         margin: 0.5rem 0 !important;
     }

     .stButton > button {
         width: 300px !important;
         height: 60px !important;
         font-size: 1.1rem !important;
         font-weight: 600 !important;
         border-radius: 15px !important;
         border: 2px solid #2196F3 !important;
         background: linear-gradient(135deg, #f8fafe 0%, #ffffff 100%) !important;
         color: #1976D2 !important;
         transition: all 0.3s ease !important;
         box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1) !important;
         display: block !important;
         margin: 0 auto !important;
     }

     .stButton > button:hover {
         background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%) !important;
         color: white !important;
         transform: translateY(-3px) !important;
         box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3) !important;
     }

     /* 🔧 确保内容容器有足够空间 */
     .centered-content {
         text-align: center;
         width: 100%;
         max-width: 600px;
         margin: 1rem auto;
         padding-bottom: 2rem;  /* 🔧 增加底部空间 */
     }

     /* 🔧 按钮容器居中 */
     .button-container {
         display: flex;
         flex-direction: column;
         align-items: center;
         gap: 1rem;
         margin: 2rem 0;
         width: 100%;
     }

     /* 🔧 确保页面底部有足够空间 */
     .main {
         padding-bottom: 5rem !important;
     }
     </style>
     """, unsafe_allow_html=True)

    # 🎯 居中布局
    st.markdown('<h1 class="main-title">LLM-based Radio Monitoring Coverage System</h1>', unsafe_allow_html=True)

    # 🔧 使用容器强制居中
    with st.container():
        # 创建三列，中间列放按钮
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("🛠️ Path Loss Prediction ", key="btn_path_loss", use_container_width=True):
                st.session_state.page_mode = "path_loss"
                st.rerun()

            if st.button("📊 Regional Coverage Prediction ", key="btn_field_strength", use_container_width=True):
                st.session_state.page_mode = "field_strength"
                st.rerun()

            if st.button("📚 Knowledge Q&A ", key="btn_knowledge", use_container_width=True):
                st.session_state.page_mode = "knowledge"
                st.rerun()

    # 🔧 添加一些底部空间，确保输入框完全可见
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
def ensure_knowledge_base():
    """
    确保知识库存在并是最新的
    返回: (是否成功, 是否进行了更新)
    """
    # 检查索引文件是否存在
    index_files = ["document_vector/faiss_index/index.faiss", "document_vector/faiss_index/index.pkl"]

    all_exist = all(os.path.exists(file) for file in index_files)

    if not all_exist:
        st.info("🔨 知识库不存在，正在构建...")

        # 执行完整构建
        try:
            from build_vector_first import build_knowledge_base
            success = build_knowledge_base()

            if success:
                st.success("✅ 知识库构建完成！")
                return True, True
            else:
                st.error("❌ 知识库构建失败")
                return False, False
        except Exception as e:
            st.error(f"❌ 知识库构建出错: {str(e)}")
            return False, False
    else:
        # 尝试增量更新
        try:
            with st.spinner("🔍 检查知识库更新..."):
                success, update_count = incremental_update_knowledge_base()

                if success and update_count > 0:
                    st.success(f"✅ 知识库已更新，新增 {update_count} 个文档")
                    return True, True
                elif success:
                    # 知识库已是最新，没有更新
                    return True, False
                else:
                    st.warning("⚠️ 知识库更新检查失败，使用现有版本")
                    return True, False

        except Exception as e:
            st.warning(f"⚠️ 知识库更新检查出错: {str(e)}，使用现有版本")
            return True, False
def show_path_loss_page():
    """路径损耗计算页面"""
    st.subheader("📈 路径损耗计算")

    # 🔧 获取缓存实例
    cache = st.session_state.get('cache_system')

    # 返回按钮
    if st.button("← 返回智能问答", type="secondary"):
        st.session_state.page_mode = "chat"
        st.rerun()

    st.markdown("---")

    # 输入方式选择
    input_method = st.radio(
        "选择输入方式：",
        ["🤖 智能解析（推荐）", "📝 手动输入"],
        horizontal=True,
        key="path_input_method"
    )

    if input_method == "🤖 智能解析（推荐）":
        st.markdown("### 💡 智能输入示例")
        st.markdown("**用自然语言描述您的需求，AI会自动提取参数：**")

        example = "计算从坐标(24.887731，102.840305)到(25.020263，102.790709)，频率340MHz,发射天线高度30m,接收天线1.5m的路径损耗"
        st.code(example, language=None)

        # 智能输入框
        if smart_input := st.chat_input("描述路径损耗计算需求..."):
            with st.spinner("🤖 正在解析您的需求..."):
                try:
                    process_smart_query(smart_input, calculation_type="path_loss")
                except Exception as e:
                    st.error(f"智能解析失败: {str(e)}")

    else:
        # 手动输入表单
        st.markdown("### 📝 手动参数输入")

        with st.form("path_loss_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**发射点坐标**")
                tx_lat = st.number_input("发射点纬度", value=26.128111, format="%.6f")
                tx_lon = st.number_input("发射点经度", value=103.147275, format="%.6f")
                tx_height = st.number_input("发射天线高度 (m)", value=10.0, min_value=0.1)

            with col2:
                st.write("**接收点坐标**")
                rx_lat = st.number_input("接收点纬度", value=26.041924, format="%.6f")
                rx_lon = st.number_input("接收点经度", value=103.215690, format="%.6f")
                rx_height = st.number_input("接收天线高度 (m)", value=5.0, min_value=0.1)

            col3, col4 = st.columns(2)
            with col3:
                frequency = st.number_input("频率 (MHz)", value=340.0, min_value=0.1)
                gap = st.number_input("计算间距 (m)", value=200, min_value=1)
                time_percentage = st.number_input("时间概率 (%)", value=50.0, min_value=1.0, max_value=50.0)


            with col4:
                Pt = st.number_input("发射功率 (kW)", value=1.0, min_value=0.001)
                Gt = st.number_input("发射天线增益 (dB)", value=0.0)
                signal_pol = st.selectbox("信号极化", options=[1, 2],
                                          format_func=lambda x: "水平" if x == 1 else "垂直")

            submitted = st.form_submit_button("🚀 开始计算", use_container_width=True)

            if submitted:
                params = {
                    'tx_lat': tx_lat, 'tx_lon': tx_lon,
                    'rx_lat': rx_lat, 'rx_lon': rx_lon,
                    'frequency': frequency, 'gap': gap, 'Pt': Pt, 'Gt': Gt,
                    'tx_height': tx_height, 'rx_height': rx_height,
                    'time_percentage': time_percentage, 'signal_pol': signal_pol
                }

                # 缓存未命中，计算
                with st.spinner("🔄 首次计算此配置，正在计算并缓存结果..."):
                    try:
                        result = calculate_path_loss(
                            tx_lat=tx_lat, tx_lon=tx_lon, rx_lat=rx_lat, rx_lon=rx_lon,
                            frequency=frequency, gap=gap, Pt=Pt, Gt=Gt,
                            tx_antenna_height=tx_height, rx_antenna_height=rx_height,
                            time_percentage=time_percentage, signal_polarization=signal_pol
                        )

                        # 保存到历史记录
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result['params'] = params
                        show_path_loss_results(result)
                        show_path_visualization(result)
                        show_loss_analysis(result)

                        # 保存到历史记录
                        st.session_state.path_loss_history.append(result)

                    except Exception as e:
                        st.error(f"❌ 计算出错：{str(e)}")


def show_field_strength_page():
    """区域场强计算页面"""
    st.subheader("📊 区域场强计算")
    # 🔧 获取缓存实例
    cache = st.session_state.get('cache_system')

    # 返回按钮
    if st.button("← 返回智能问答", type="secondary"):
        st.session_state.page_mode = "chat"
        st.rerun()

    st.markdown("---")

    # 输入方式选择
    input_method = st.radio(
        "选择输入方式：",
        ["🤖 智能解析（推荐）", "📝 手动输入"],
        horizontal=True,
        key="field_input_method"
    )

    if input_method == "🤖 智能解析（推荐）":
        st.markdown("### 💡 智能输入示例")
        st.markdown("**用自然语言描述您的需求，AI会自动提取参数：**")

        example = "计算坐标(25.053859，102.726936)周围1km范围，频率为340MHz，采样间隔10m的场强分布"
        st.code(example, language=None)

        # 智能输入框
        if smart_input := st.chat_input("描述场强计算需求..."):
            with st.spinner("🤖 正在解析您的需求..."):
                try:
                    process_smart_query(smart_input, calculation_type="field_strength")
                except Exception as e:
                    st.error(f"智能解析失败: {str(e)}")

    else:
        # 手动输入表单
        st.markdown("### 📝 手动参数输入")

        with st.form("field_strength_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**中心点坐标**")
                center_lat = st.number_input("中心点纬度", value=26.085, format="%.6f")
                center_lon = st.number_input("中心点经度", value=103.181, format="%.6f")
                tx_height = st.number_input("发射天线高度 (m)", value=10.0, min_value=0.1)
                rx_height = st.number_input("接收天线高度 (m)", value=5.0, min_value=0.1)

            with col2:
                st.write("**计算参数**")
                frequency = st.number_input("频率 (MHz)", value=340.0, min_value=0.1, key="fs_freq")
                radius = st.number_input("覆盖半径 (km)", value=1.0, min_value=0.1)
                gap = st.number_input("计算间距 (m)", value=10, min_value=1)

            col3, col4 = st.columns(2)
            with col3:
                Pt = st.number_input("发射功率 (kW)", value=1.0, min_value=0.001, key="fs_power")
                Gt = st.number_input("发射天线增益 (dB)", value=0.0, key="fs_gain")

            with col4:
                time_percentage = st.number_input("时间概率 (%)", value=50.0, min_value=1.0, max_value=50.0)
                signal_pol = st.selectbox("信号极化", options=[1, 2],
                                          format_func=lambda x: "水平" if x == 1 else "垂直")

            submitted = st.form_submit_button("🚀 开始计算", use_container_width=True)

            if submitted:
                # 🔧 构建参数字典
                params = {
                    'center_lat': center_lat, 'center_lon': center_lon,
                    'radius': radius, 'gap': gap, 'frequency': frequency,
                    'Pt': Pt, 'Gt': Gt, 'tx_height': tx_height, 'rx_height': rx_height,
                    'time_percentage': time_percentage, 'signal_pol': signal_pol
                }

                with st.spinner("🔄 正在计算区域场强分布..."):
                    try:
                        result = calculate_field_strength(
                            lat=center_lat,
                            lon=center_lon,
                            rad=radius,
                            gap=gap,
                            Pt=Pt,
                            Gt=Gt,
                            frequency=frequency,
                            tx_antenna_height=tx_height,
                            rx_antenna_height=rx_height,
                            time_percentage=time_percentage,
                            signal_pol=signal_pol
                        )

                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result['params'] = params

                        st.session_state.field_strength_history.append(result)
                        st.success("✅ 计算完成！")
                        show_field_strength_results(result)
                        show_enhanced_baidu_map11(result)

                    except Exception as e:
                        st.error(f"❌ 计算出错：{str(e)}")


def show_knowledge_page():
    """知识库问答页面"""
    st.subheader("📚 知识库问答")

    # 确保知识库存在并检查更新
    success, updated = ensure_knowledge_base()

    if not success:
        st.error("❌ 知识库初始化失败，无法进行问答")
        return

    # 显示更新状态
    if updated:
        st.info("🔄 知识库已更新，可以开始问答")
    else:
        st.info("✅ 知识库已是最新，可以开始问答")

    # 返回按钮
    if st.button("← 返回智能问答", type="secondary"):
        st.session_state.page_mode = "chat"
        st.rerun()

    st.markdown("---")

    # 手动更新按钮
    with st.expander("🔧 知识库管理"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 手动检查更新", use_container_width=True):
                with st.spinner("检查更新中..."):
                    try:
                        success, update_count = incremental_update_knowledge_base()
                        if success and update_count > 0:
                            st.success(f"✅ 更新完成，新增 {update_count} 个文档")
                            st.rerun()
                        elif success:
                            st.info("✅ 知识库已是最新，没有新内容")
                        else:
                            st.error("❌ 更新失败")
                    except Exception as e:
                        st.error(f"❌ 更新出错: {str(e)}")

        with col2:
            if st.button("🔨 重新构建知识库", use_container_width=True, type="secondary"):
                st.warning("⚠️ 这将重新构建整个知识库，可能会花费较长时间")
                if st.button("确认重新构建", type="primary"):
                    with st.spinner("正在重新构建..."):
                        try:
                            from build_vector_first import build_knowledge_base
                            success = build_knowledge_base()
                            if success:
                                st.success("✅ 知识库重新构建完成！")
                                st.rerun()
                            else:
                                st.error("❌ 重新构建失败")
                        except Exception as e:
                            st.error(f"❌ 重新构建出错: {str(e)}")

    # 问答模式选择
    st.subheader("🎯 选择问答模式")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 文本问答", use_container_width=True, type="primary"):
            st.session_state.qa_mode = "text"
            st.rerun()
    with col2:
        if st.button("🔬 混合问答", use_container_width=True, type="secondary"):
            st.session_state.qa_mode = "hybrid"
            st.rerun()

    # 初始化问答模式
    if 'qa_mode' not in st.session_state:
        st.session_state.qa_mode = "text"

    # 根据模式显示不同的问答界面
    if st.session_state.qa_mode == "text":
        show_text_qa()
    else:
        show_hybrid_qa()


def show_text_qa():
    """文本问答模式 - 排除MongoDB数据库内容"""
    st.markdown("### 📝 文本问答模式")
    st.info("此模式下，系统仅基于知识库中的技术文档进行问答，不会计算和可视化，也不会使用数据库缓存数据。")

    # 检查知识库
    if not os.path.exists(INDEX_PATH):
        st.warning("⚠️ 知识库索引文件不存在，请先构建知识库")
        st.info("请确保 faiss_index 文件夹存在且包含必要的索引文件")
        st.stop()

    # 初始化RAG组件
    rag_chain = initialize_rag_components()
    if not rag_chain:
        st.error("❌ 错误：知识库初始化失败！")
        st.stop()

    # 高级选项
    with st.expander("🔧 高级选项"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("检索文档数量", 1, 10, 3, key="text_top_k")
        with col2:
            temperature = st.slider("回答创造性", 0.0, 1.0, 0.3, key="text_temperature")

    # 自定义RAG查询函数 - 只检索技术文档
    def text_only_retrieve(query, top_k=3):
        """只检索技术文档，排除MongoDB数据，并加入去重和多样性控制"""
        if hasattr(rag_chain.retriever, 'improved_retrieve'):
            try:
                # 1. 获取更多结果以便筛选
                all_results = rag_chain.retriever.improved_retrieve(query, top_k=top_k * 10)  # 获取更多结果

                # 2. 按相似度分组和去重
                filtered_results = []
                seen_sources = set()  # 用于文档去重
                source_counts = {}  # 统计每个来源的文档数量

                for doc, score in all_results:
                    # 检查是否是技术文档（排除MongoDB和缓存数据）
                    is_technical_doc = True

                    if hasattr(doc, 'metadata') and doc.metadata:
                        metadata = doc.metadata

                        # 获取文档来源信息
                        source = metadata.get('source', '')

                        # 判断是否是数据库/缓存数据
                        is_cache_source = any(keyword in str(source).lower() for keyword in [
                            'mongodb', 'cache', 'geospatial', 'field_strength', 'path_loss'
                        ])

                        if is_cache_source:
                            is_technical_doc = False

                    # 如果是技术文档，进行去重处理
                    if is_technical_doc:
                        # 获取文档标识符（用于去重）
                        doc_identifier = None
                        if hasattr(doc, 'metadata') and doc.metadata:
                            # 使用文件路径作为标识符
                            source = doc.metadata.get('source', '')
                            if source:
                                # 提取文件名（去掉路径）
                                import os
                                doc_identifier = os.path.basename(source)

                        # 如果无法获取标识符，使用内容摘要
                        if not doc_identifier:
                            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                            # 取前100字符作为标识符
                            doc_identifier = content[:100]

                        # 3. 多样性控制策略
                        # a) 确保同一来源的文档不超过1个（避免重复）
                        if doc_identifier in seen_sources:
                            continue  # 跳过重复文档

                        # b) 统计同一文档的不同片段（相似内容）
                        similar_content_found = False
                        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                        for existing_doc, _ in filtered_results:
                            existing_content = existing_doc.page_content if hasattr(existing_doc,
                                                                                    "page_content") else str(
                                existing_doc)
                            # 检查内容重叠率（简单方法）
                            overlap_ratio = calculate_content_overlap(content, existing_content)
                            if overlap_ratio > 0.5:  # 如果内容重叠超过50%，视为相似
                                similar_content_found = True
                                break

                        if similar_content_found:
                            continue  # 跳过高度相似的内容

                        # 4. 记录已处理的文档
                        seen_sources.add(doc_identifier)

                        # 5. 添加到结果列表
                        filtered_results.append((doc, score))

                        # 6. 达到所需数量后停止
                        if len(filtered_results) >= top_k:
                            break

                # 7. 如果过滤后数量不足，补充一些结果
                if len(filtered_results) < top_k and len(filtered_results) < len(all_results):
                    # 从未选择的文档中选择一些补充
                    for doc, score in all_results:
                        if (doc, score) not in [(d, s) for d, s in filtered_results]:
                            # 再次检查是否是技术文档
                            is_technical_doc = True
                            if hasattr(doc, 'metadata') and doc.metadata:
                                source = doc.metadata.get('source', '')
                                is_cache_source = any(keyword in str(source).lower() for keyword in [
                                    'mongodb', 'cache', 'geospatial', 'field_strength', 'path_loss'
                                ])
                                if is_cache_source:
                                    is_technical_doc = False

                            if is_technical_doc:
                                filtered_results.append((doc, score))
                                if len(filtered_results) >= top_k:
                                    break

                return filtered_results[:top_k]

            except Exception as e:
                st.error(f"检索过程中出错: {str(e)}")
                # 返回空结果
                return []
        else:
            try:
                docs = rag_chain.retriever.get_relevant_documents(query)
                # 简单去重
                unique_docs = []
                seen_contents = set()
                for doc in docs:
                    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    content_hash = hash(content[:200])  # 取前200字符的哈希值
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        unique_docs.append((doc, 0.5))
                    if len(unique_docs) >= top_k:
                        break
                return unique_docs
            except:
                return []

    # 辅助函数：计算内容重叠率
    def calculate_content_overlap(text1, text2):
        """计算两个文本内容的重叠率"""
        if not text1 or not text2:
            return 0.0

        # 取较短的文本长度
        min_len = min(len(text1), len(text2))
        if min_len == 0:
            return 0.0

        # 计算重叠字符数
        overlap_chars = 0
        for i in range(min_len):
            if text1[i] == text2[i]:
                overlap_chars += 1

        return overlap_chars / min_len

    # 显示文本问答历史
    if 'text_messages' not in st.session_state:
        st.session_state.text_messages = [
            {"role": "assistant",
             "content": "您好！我是纯文本问答助手，我仅基于技术文档回答您的技术问题，不会使用数据库缓存数据。"}
        ]

    for msg in st.session_state.text_messages[1:]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "docs" in msg:
                # 显示回答
                st.markdown(msg["content"])
                # 显示参考文档
                with st.expander("📄 查看参考技术文档"):
                    for i, (doc_content, score, doc_type) in enumerate(msg["docs"]):
                        st.write(f"**技术文档 {i + 1}** (相似度: {score:.3f}, 类型: {doc_type}):")
                        st.text(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
                        st.markdown("---")
            else:
                st.markdown(msg["content"])

    # 文本问答输入框
    if text_input := st.chat_input("请输入您想查询的文本问题..."):
        # 用户消息
        st.session_state.text_messages.append({"role": "user", "content": text_input})
        with st.chat_message("user"):
            st.markdown(text_input)

        # 获取知识库回复
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在搜索技术文档..."):
                try:
                    # 使用自定义检索函数
                    docs = text_only_retrieve(text_input, top_k=top_k)

                    if not docs:
                        st.warning("⚠️ 未找到相关的技术文档")
                        result = "抱歉，我没有在技术文档中找到相关信息。请尝试其他问题或确保技术文档已正确加载。"
                        st.markdown(result)

                        # 保存到历史记录
                        st.session_state.text_messages.append({
                            "role": "assistant",
                            "content": result,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        return

                    # 显示检索到的文档
                    with st.expander("📄 查看检索到的技术文档", expanded=True):
                        for i, (doc, score) in enumerate(docs):
                            # 确定文档类型
                            doc_type = "技术文档"
                            if hasattr(doc, 'metadata') and doc.metadata:
                                source = doc.metadata.get('source', '')
                                collection = doc.metadata.get('collection', '')
                                doc_type_meta = doc.metadata.get('type', '')

                                # 更精确的类型判断
                                if 'mongodb' in str(source).lower():
                                    doc_type = "MongoDB数据"
                                elif 'cache' in str(source).lower():
                                    doc_type = "缓存数据"
                                elif collection in ['field_strength_cache', 'path_loss_cache']:
                                    doc_type = f"缓存集合: {collection}"
                                elif doc_type_meta:
                                    doc_type = doc_type_meta
                                elif source:
                                    # 尝试从source中提取文件名
                                    if os.path.sep in source:
                                        filename = os.path.basename(source)
                                        doc_type = f"文件: {filename}"

                            st.write(f"**文档 {i + 1}** (相似度: {score:.3f}, 类型: {doc_type}):")

                            # 显示文档内容
                            if hasattr(doc, 'page_content'):
                                content = doc.page_content
                                # 显示适当长度的内容
                                display_content = content[:800] + "..." if len(content) > 800 else content
                                st.text(display_content)
                            else:
                                content = str(doc)
                                display_content = content[:800] + "..." if len(content) > 800 else content
                                st.text(display_content)

                            # 显示元数据（用于调试）
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.caption(f"元数据: {str(doc.metadata)[:200]}...")

                            st.markdown("---")

                    with st.spinner("🤖 正在生成专业回答..."):
                        # 构建上下文
                        context_parts = []
                        for doc, score in docs:
                            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                            context_parts.append(f"[相似度: {score:.3f}] {content}")

                        context = "\n\n".join(context_parts)

                        if context:
                            prompt = f"""请基于以下资料，简洁回答无线电管理问题：

                        相关资料（供参考）：
                        {context}

                        问题：{text_input}

                        **要求**：
                        1. **简洁明了**：直接回答问题，不超过150字
                        2. **重点突出**：只说核心要点，不要详细展开
                        3. **结构简单**：输出必须为纯文本。禁止使用任何格式标记，包括但不限于加粗（**）、标题（#）、列表符号（-/*）、代码块等。
                        4. **专业准确**：确保信息准确，术语正确

                        现在请简洁回答：
                        """
                        else:
                            prompt = f"""请简洁回答以下无线电管理问题：

                        问题：{text_input}

                        **要求**：
                        1. **直接回答**：不超过150字
                        2. **重点明确**：只说核心内容
                        3. **避免冗长**：不要详细解释每个细节

                        现在请简洁回答：
                        """

                        # 调用LLM生成回答 - 根据您的RAGChain类调整调用方式
                        try:
                            # 方法1: 直接调用RAGChain
                            if hasattr(rag_chain, 'invoke'):
                                result = rag_chain.invoke({
                                    "question": text_input,
                                    "context": context
                                })
                            # 方法2: 使用llm属性
                            elif hasattr(rag_chain, 'llm'):
                                if hasattr(rag_chain.llm, 'generate'):
                                    result = rag_chain.llm.generate(prompt)
                                elif hasattr(rag_chain.llm, 'invoke'):
                                    result = rag_chain.llm.invoke(prompt)
                                else:
                                    # 默认方法
                                    result = "生成回答时出错：无法确定LLM调用方法"
                            else:
                                result = "RAG链配置错误：没有找到可用的LLM"

                            # 提取内容
                            if hasattr(result, 'content'):
                                result = result.content
                            elif isinstance(result, dict) and 'answer' in result:
                                result = result['answer']
                            elif isinstance(result, dict) and 'result' in result:
                                result = result['result']
                            elif not isinstance(result, str):
                                result = str(result)

                        except Exception as e:
                            result = f"生成回答时出错: {str(e)}"
                            st.error(f"LLM调用错误: {str(e)}")

                    st.markdown(result)

                    # 保存到历史记录
                    docs_info = []
                    for doc, score in docs:
                        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                        doc_type = "技术文档"
                        if hasattr(doc, 'metadata') and doc.metadata:
                            source = doc.metadata.get('source', '')
                            if 'mongodb' in str(source).lower():
                                doc_type = "数据库数据"
                            elif 'cache' in str(source).lower():
                                doc_type = "缓存数据"

                        # 截取内容用于显示
                        display_content = content[:500] + "..." if len(content) > 500 else content
                        docs_info.append((display_content, score, doc_type))

                    st.session_state.text_messages.append({
                        "role": "assistant",
                        "content": result,
                        "docs": docs_info,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                except Exception as e:
                    error_msg = f"❌ 知识库查询失败: {str(e)}"
                    st.error(error_msg)
                    # 显示详细错误信息
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.text_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # 文本问答统计信息
    if st.session_state.text_messages:
        with st.expander("📊 本次会话统计"):
            user_questions = len([msg for msg in st.session_state.text_messages if msg["role"] == "user"])
            assistant_responses = len([msg for msg in st.session_state.text_messages if msg["role"] == "assistant"])

            # 统计使用的文档类型
            doc_types = {}
            for msg in st.session_state.text_messages:
                if msg.get("role") == "assistant" and "docs" in msg:
                    for _, _, doc_type in msg["docs"]:
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            st.metric("提问次数", user_questions)
            st.metric("回答次数", assistant_responses)

            if doc_types:
                st.write("📋 使用的文档类型统计:")
                for doc_type, count in doc_types.items():
                    st.write(f"- {doc_type}: {count} 次")

            if st.button("🗑️ 清空文本对话", key="clear_text"):
                st.session_state.text_messages = [st.session_state.text_messages[0]]
                st.rerun()

def parse_question_parameters(question):
    """
    解析问题中的参数
    返回: (parsed_params, calculation_type)
    """
    # 默认参数值
    default_params = {
        # 公共参数
        'tx_lat': 26.128111, 'tx_lon': 103.147275,
        'rx_lat': 26.041924, 'rx_lon': 103.215690,
        'lat': 26.128111, 'lon': 103.147275,  # 场强计算用
        'rad': 1, 'gap': 200,
        'Pt': 1, 'Gt': 0, 'frequency': 340,
        'tx_antenna_height': 30, 'rx_antenna_height': 1.5,
        'time_percentage': 50, 'signal_polarization': 1
    }

    # 初始化解析参数
    parsed_params = {}

    # 确定计算类型
    calculation_type = "path_loss"  # 默认路径损耗

    # 关键词判断计算类型
    if any(keyword in question for keyword in ['场强', 'field_strength', '区域场强', '覆盖', '场强计算']):
        calculation_type = "field_strength"
    elif any(keyword in question for keyword in ['路径损耗', 'path_loss', '损耗', '传播损耗', '路径损耗计算']):
        calculation_type = "path_loss"

    # 简单正则提取数值
    import re

    # 查找坐标信息 - 改进的正则表达式
    coord_patterns = [
        r'([\d\.]+)[度°]?\s*[,，]\s*([\d\.]+)[度°]?',  # 26.128111, 103.147275
        r'纬度\s*[:：]?\s*([\d\.]+)[,，\s]+经度\s*[:：]?\s*([\d\.]+)',  # 纬度26.128111, 经度103.147275
        r'lat\s*[:=]?\s*([\d\.]+)[,，\s]+lon\s*[:=]?\s*([\d\.]+)',  # lat:26.128111, lon:103.147275
        r'([\d\.]+)\s*°?\s*N\s*[,，]\s*([\d\.]+)\s*°?\s*E',  # 26.128111°N, 103.147275°E
        r'\(([\d\.]+)[，,]\s*([\d\.]+)\)',  # (24.887731，102.840305)
        r'到\s*\(([\d\.]+)[，,]\s*([\d\.]+)\)',  # 到(25.020263，102.790709)
    ]

    # 查找所有坐标
    all_coords = []
    for pattern in coord_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                try:
                    lat = float(match[0])
                    lon = float(match[1])
                    all_coords.append((lat, lon))
                except:
                    pass

    # 根据计算类型分配坐标
    if calculation_type == "field_strength":
        # 场强计算：只需要一个坐标（发射点）
        if all_coords:
            parsed_params['lat'] = all_coords[0][0]
            parsed_params['lon'] = all_coords[0][1]
    else:
        # 路径损耗计算：需要两个坐标（发射点和接收点）
        if len(all_coords) >= 2:
            parsed_params['tx_lat'] = all_coords[0][0]
            parsed_params['tx_lon'] = all_coords[0][1]
            parsed_params['rx_lat'] = all_coords[1][0]
            parsed_params['rx_lon'] = all_coords[1][1]
        elif len(all_coords) == 1:
            # 只有一个坐标，使用默认的接收点
            parsed_params['tx_lat'] = all_coords[0][0]
            parsed_params['tx_lon'] = all_coords[0][1]

    # 首先提取天线高度信息 - 更精确的匹配
    # 匹配发射天线高度
    tx_height_patterns = [
        r'发射天线高度\s*[:：]?\s*([\d\.]+)\s*m',
        r'发射天线\s*[:：]?\s*([\d\.]+)\s*m',
        r'tx[_\s]*height\s*[:：]?\s*([\d\.]+)',
        r'发射[^\d]*([\d\.]+)\s*m',
    ]

    for pattern in tx_height_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            try:
                parsed_params['tx_antenna_height'] = float(matches[0])
                break
            except:
                pass

    # 匹配接收天线高度
    rx_height_patterns = [
        r'接收天线\s*([\d\.]+)\s*m',
        r'接收天线高度\s*[:：]?\s*([\d\.]+)\s*m',
        r'rx[_\s]*height\s*[:：]?\s*([\d\.]+)',
        r'接收[^\d]*([\d\.]+)\s*m',
    ]

    for pattern in rx_height_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            try:
                parsed_params['rx_antenna_height'] = float(matches[0])
                break
            except:
                pass

    # 查找其他数值参数
    number_patterns = {
        'radius|半径': r'(?:半径|radius)[\s:：]*([\d\.]+)',
        'gap|间距|采样密度': r'(?:间距|采样密度|gap)[\s:：]*([\d\.]+)',
        'power|功率': r'(?:功率|power|Pt)[\s:：]*([\d\.]+)',
        '增益|gain|Gt': r'(?:增益|gain|Gt)[\s:：]*([\d\.]+)',
        'frequency|频率': r'(?:频率|frequency)[\s:：]*([\d\.]+)',
        '高度|height|天线高度': r'(?:天线高度|height)[\s:：]*([\d\.]+)',  # 通用匹配，但前面已经处理过
        '距离|distance': r'(?:距离|distance)[\s:：]*([\d\.]+)',
        '时间|time|概率': r'(?:时间概率|time)[\s:：]*([\d\.]+)',
        '极化|polarization': r'(?:极化|polarization)[\s:：]*([\d\.]+)'
    }

    for param_name, pattern in number_patterns.items():
        # 跳过高度参数，因为已经单独处理
        if '高度' in param_name or 'height' in param_name:
            continue

        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[0])
                # 根据参数名映射到具体参数
                if '半径' in param_name or 'radius' in param_name:
                    if calculation_type == "field_strength":
                        parsed_params['rad'] = value
                elif '间距' in param_name or '采样密度' in param_name or 'gap' in param_name:
                    parsed_params['gap'] = value
                elif '功率' in param_name or 'power' in param_name or 'Pt' in param_name:
                    parsed_params['Pt'] = value
                elif '增益' in param_name or 'gain' in param_name or 'Gt' in param_name:
                    parsed_params['Gt'] = value
                elif '频率' in param_name or 'frequency' in param_name:
                    parsed_params['frequency'] = value
                elif '时间' in param_name or 'time' in param_name or '概率' in param_name:
                    parsed_params['time_percentage'] = value
                elif '极化' in param_name or 'polarization' in param_name:
                    parsed_params['signal_polarization'] = int(value)
            except:
                pass

    # 设置默认值（如果参数未解析到）
    if calculation_type == "field_strength":
        required_params = ['lat', 'lon', 'rad', 'gap', 'Pt', 'Gt', 'frequency',
                           'tx_antenna_height', 'rx_antenna_height', 'time_percentage', 'signal_polarization']
        for param in required_params:
            if param not in parsed_params or parsed_params[param] is None:
                parsed_params[param] = default_params.get(param)
    else:
        # 路径损耗计算需要移除 'lat' 和 'lon' 参数，只保留 tx_lat, tx_lon, rx_lat, rx_lon
        # 确保没有传递错误的参数
        required_params = ['tx_lat', 'tx_lon', 'rx_lat', 'rx_lon', 'gap', 'Pt', 'Gt', 'frequency',
                           'tx_antenna_height', 'rx_antenna_height', 'time_percentage', 'signal_polarization']

        # 移除可能的错误参数
        keys_to_remove = [key for key in parsed_params.keys() if key not in required_params]
        for key in keys_to_remove:
            if key not in ['lat', 'lon', 'rad']:  # 保留这些可能被误解析的字段用于调试
                del parsed_params[key]

        for param in required_params:
            if param not in parsed_params or parsed_params[param] is None:
                parsed_params[param] = default_params.get(param)

    return parsed_params, calculation_type

def vector_db_query_to_params(doc_metadata, calculation_type):
    """
    将向量数据库查询结果转换为计算参数
    类似 calculate_path_loss() 的缓存查询方式
    """
    params = {}

    if calculation_type == "path_loss":
        # 从元数据中提取路径损耗参数
        param_keys = ['tx_lat', 'tx_lon', 'rx_lat', 'rx_lon', 'gap', 'Pt', 'Gt',
                      'frequency', 'tx_antenna_height', 'rx_antenna_height',
                      'time_percentage', 'signal_polarization']

        for key in param_keys:
            metadata_key = f"param_{key}"
            if metadata_key in doc_metadata:
                try:
                    value_str = str(doc_metadata[metadata_key])
                    if '.' in value_str:
                        params[key] = float(value_str)
                    else:
                        params[key] = int(value_str)
                except:
                    params[key] = doc_metadata[metadata_key]

    elif calculation_type == "field_strength":
        # 从元数据中提取场强参数
        param_keys = ['lat', 'lon', 'rad', 'gap', 'Pt', 'Gt', 'frequency',
                      'tx_antenna_height', 'rx_antenna_height', 'time_percentage',
                      'signal_polarization']

        for key in param_keys:
            metadata_key = f"param_{key}"
            if metadata_key in doc_metadata:
                try:
                    value_str = str(doc_metadata[metadata_key])
                    if '.' in value_str:
                        params[key] = float(value_str)
                    else:
                        params[key] = int(value_str)
                except:
                    params[key] = doc_metadata[metadata_key]

    return params


def analyze_document_points(doc_metadata, calculation_type, user_question=""):
    """
    分析文档中的点数据，生成详细的分析报告
    """
    analysis_parts = []

    if calculation_type == "path_loss":
        analysis_parts.append("### 📍 路径点数据分析")

        # 检查是否有所有点数据
        if 'all_points' in doc_metadata:
            try:
                points_data = json.loads(doc_metadata['all_points'])
                total_points = len(points_data)
                analysis_parts.append(f"**总路径点数**: {total_points}个")

                # 分析点分布
                if points_data:
                    # 提取距离和损耗数据
                    distances = []
                    losses = []

                    for point in points_data[:100]:  # 只分析前100个点以避免处理过多数据
                        if 'distance' in point and point['distance'] is not None:
                            distances.append(point['distance'])
                        if 'loss' in point and point['loss'] is not None:
                            losses.append(point['loss'])

                    if distances and losses:
                        # 距离分析
                        avg_distance = sum(distances) / len(distances)
                        max_distance = max(distances)
                        min_distance = min(distances)

                        analysis_parts.append(f"**距离范围**: {min_distance:.2f} - {max_distance:.2f} km")
                        analysis_parts.append(f"**平均距离**: {avg_distance:.2f} km")

                        # 损耗分析
                        avg_loss = sum(losses) / len(losses)
                        max_loss = max(losses)
                        min_loss = min(losses)

                        analysis_parts.append(f"**损耗范围**: {min_loss:.2f} - {max_loss:.2f} dB")
                        analysis_parts.append(f"**平均损耗**: {avg_loss:.2f} dB")

                        # 分析损耗趋势
                        if len(distances) > 1 and len(losses) > 1:
                            # 简单线性趋势分析
                            try:
                                import numpy as np
                                x = np.array(distances)
                                y = np.array(losses)
                                coeffs = np.polyfit(x, y, 1)
                                slope = coeffs[0]

                                if slope > 0.5:
                                    analysis_parts.append("**损耗趋势**: 随距离增加损耗显著增大")
                                elif slope > 0.1:
                                    analysis_parts.append("**损耗趋势**: 随距离增加损耗缓慢增大")
                                else:
                                    analysis_parts.append("**损耗趋势**: 损耗变化相对平稳")
                            except:
                                analysis_parts.append("**损耗趋势**: 无法计算趋势")

                        # 关键点识别
                        high_loss_points = [p for p in points_data[:50] if
                                            'loss' in p and p['loss'] is not None and p['loss'] > avg_loss + 20]
                        if high_loss_points:
                            analysis_parts.append(f"**高损耗点**: 发现{len(high_loss_points)}个损耗显著高于平均值的点")

                            # 显示前几个高损耗点
                            analysis_parts.append("**关键高损耗点示例**:")
                            for i, point in enumerate(high_loss_points[:3]):
                                analysis_parts.append(
                                    f"  - 点{i + 1}: 距离={point.get('distance', 'N/A'):.2f}km, 损耗={point.get('loss', 'N/A'):.2f}dB")

            except Exception as e:
                analysis_parts.append(f"**点数据解析错误**: {str(e)}")

        # 检查统计信息
        if 'points_statistics' in doc_metadata:
            try:
                stats = json.loads(doc_metadata['points_statistics'])
                analysis_parts.append("**统计信息**:")
                analysis_parts.append(f"  - 总点数: {stats.get('total_points', 0)}")

                if 'coordinates_range' in stats:
                    coords = stats['coordinates_range']
                    analysis_parts.append(
                        f"  - 坐标范围: 纬度({coords.get('min_lat', 0):.4f}~{coords.get('max_lat', 0):.4f}), 经度({coords.get('min_lon', 0):.4f}~{coords.get('max_lon', 0):.4f})")

                if 'loss_range' in stats:
                    loss_range = stats['loss_range']
                    analysis_parts.append(
                        f"  - 损耗范围: {loss_range.get('min_loss', 0):.2f}~{loss_range.get('max_loss', 0):.2f} dB")

                if 'distance_range' in stats:
                    dist_range = stats['distance_range']
                    analysis_parts.append(
                        f"  - 距离范围: {dist_range.get('min_distance', 0):.2f}~{dist_range.get('max_distance', 0):.2f} km")
            except:
                analysis_parts.append("**统计信息**: 无法解析")

    elif calculation_type == "field_strength":
        analysis_parts.append("### 📍 场强点数据分析")

        # 首先打印调试信息
        analysis_parts.append(f"**文档元数据字段**: {list(doc_metadata.keys())}")

        # 检查是否有所有点数据
        if 'all_points' in doc_metadata:
            try:
                points_data = json.loads(doc_metadata['all_points'])
                total_points = len(points_data)
                analysis_parts.append(f"**总场强点数**: {total_points}个")

                if points_data and len(points_data) > 0:
                    # 打印第一个点的字段结构
                    first_point = points_data[0]
                    analysis_parts.append(f"**点数据结构**: {list(first_point.keys())}")

                # 分析点分布
                if points_data:
                    # 提取场强数据 - 根据实际字段名
                    field_strengths = []

                    for point in points_data[:100]:  # 只分析前100个点
                        if isinstance(point, dict):
                            # 尝试多种可能的字段名
                            strength = None
                            for field in ['count', 'field_strength', 'strength', 'value', 'db', 'field_strength_value']:
                                if field in point and point[field] is not None:
                                    strength = point[field]
                                    break

                            if strength is not None:
                                try:
                                    field_strengths.append(float(strength))
                                except:
                                    pass

                    if field_strengths:
                        avg_strength = sum(field_strengths) / len(field_strengths)
                        max_strength = max(field_strengths)
                        min_strength = min(field_strengths)

                        analysis_parts.append(f"**场强范围**: {min_strength:.2f} - {max_strength:.2f} dBμV/m")
                        analysis_parts.append(f"**平均场强**: {avg_strength:.2f} dBμV/m")

                        # 场强覆盖分析
                        strong_points = [s for s in field_strengths if s > 80]
                        weak_points = [s for s in field_strengths if s < 40]
                        moderate_points = [s for s in field_strengths if 40 <= s <= 80]

                        coverage_strong = len(strong_points) / len(field_strengths) * 100
                        coverage_weak = len(weak_points) / len(field_strengths) * 100
                        coverage_moderate = len(moderate_points) / len(field_strengths) * 100

                        analysis_parts.append(f"**覆盖分析**:")
                        analysis_parts.append(f"  - 强场强覆盖(>80dBμV/m): {coverage_strong:.1f}%")
                        analysis_parts.append(f"  - 中等场强覆盖(40-80dBμV/m): {coverage_moderate:.1f}%")
                        analysis_parts.append(f"  - 弱场强覆盖(<40dBμV/m): {coverage_weak:.1f}%")

                        # 覆盖评估
                        if coverage_strong > 70:
                            analysis_parts.append("**覆盖评估**: ✅ 场强覆盖优秀，通信质量良好")
                        elif coverage_moderate > 60:
                            analysis_parts.append("**覆盖评估**: ⚠️ 场强覆盖中等，部分区域可能需要优化")
                        elif coverage_weak > 40:
                            analysis_parts.append("**覆盖评估**: ❌ 场强覆盖较差，建议增强发射系统")
                        else:
                            analysis_parts.append("**覆盖评估**: 📊 场强覆盖分布不均匀")

                        # 显示点示例
                        if len(points_data) > 0:
                            analysis_parts.append("**点数据示例** (前3个点):")
                            for i, point in enumerate(points_data[:3]):
                                if isinstance(point, dict):
                                    lat = point.get('lat', 'N/A')
                                    lon = point.get('lon', 'N/A')
                                    # 查找场强值
                                    strength_value = None
                                    for field in ['count', 'field_strength', 'strength', 'value']:
                                        if field in point and point[field] is not None:
                                            strength_value = point[field]
                                            break

                                    point_info = f"  点{i + 1}: 纬度={lat}, 经度={lon}"
                                    if strength_value is not None:
                                        point_info += f", 场强={strength_value:.2f} dBμV/m"
                                    analysis_parts.append(point_info)

            except Exception as e:
                analysis_parts.append(f"**点数据解析错误**: {str(e)}")
                import traceback
                analysis_parts.append(f"**详细错误**: {traceback.format_exc()[:200]}")

        # 检查有效点数据
        if 'all_actual_points' in doc_metadata:
            try:
                actual_points = json.loads(doc_metadata['all_actual_points'])
                actual_count = len(actual_points)
                analysis_parts.append(f"**有效点数**: {actual_count}个")

                # 计算有效点比例
                total_count = len(points_data) if 'points_data' in locals() else 0
                if total_count > 0:
                    valid_ratio = actual_count / total_count * 100
                    analysis_parts.append(f"**有效点比例**: {valid_ratio:.1f}%")

            except Exception as e:
                analysis_parts.append(f"**有效点解析错误**: {str(e)}")

        # 检查统计信息
        if 'points_statistics' in doc_metadata:
            try:
                stats = json.loads(doc_metadata['points_statistics'])
                analysis_parts.append("**统计信息**:")
                analysis_parts.append(f"  - 总点数: {stats.get('total_points', 0)}")

                if 'coordinates_range' in stats:
                    coords = stats['coordinates_range']
                    analysis_parts.append(
                        f"  - 坐标范围: 纬度({coords.get('min_lat', 0):.4f}~{coords.get('max_lat', 0):.4f}), 经度({coords.get('min_lon', 0):.4f}~{coords.get('max_lon', 0):.4f})")

                if 'field_strength_range' in stats:
                    strength_range = stats['field_strength_range']
                    analysis_parts.append(
                        f"  - 场强范围: {strength_range.get('min_field_strength', 0):.2f}~{strength_range.get('max_field_strength', 0):.2f} dBμV/m")

                if 'distance_range' in stats:
                    dist_range = stats['distance_range']
                    analysis_parts.append(
                        f"  - 距离范围: {dist_range.get('min_distance', 0):.2f}~{dist_range.get('max_distance', 0):.2f} km")

            except Exception as e:
                analysis_parts.append(f"**统计信息解析错误**: {str(e)}")

        # 添加与用户问题的关联分析
    if user_question:
        analysis_parts.append("### 🔗 与您问题的关联")

        # 简单的关键词匹配分析
        question_keywords = ['场强', '覆盖', '半径', '功率', '频率', '距离', '点数据', '经纬度']
        matched_keywords = [kw for kw in question_keywords if kw in user_question]

        if matched_keywords:
            analysis_parts.append(f"**关键词匹配**: 您的问题涉及 {', '.join(matched_keywords)}")

            # 针对关键词的特定分析
            for keyword in matched_keywords:
                if keyword == '场强':
                    analysis_parts.append("  - 场强分析: 提供了详细的场强分布和覆盖评估")
                elif keyword == '覆盖':
                    analysis_parts.append("  - 覆盖分析: 基于点数据进行了覆盖区域评估")
                elif keyword == '半径':
                    analysis_parts.append("  - 半径分析: 可查看计算区域的半径范围")
                elif keyword == '功率':
                    analysis_parts.append("  - 功率分析: 发射功率影响场强分布")
                elif keyword == '点数据':
                    analysis_parts.append("  - 点数据分析: 展示了详细的点数据信息")

    return "\n".join(analysis_parts)


def show_hybrid_qa():
    """混合问答模式 - 整合文本回答和计算分析"""
    st.markdown("### 🔬 混合问答模式")
    st.info("此模式下，系统会整合文本回答和计算分析，提供全面的解决方案。")

    # 检查知识库
    if not os.path.exists(INDEX_PATH):
        st.warning("⚠️ 知识库索引文件不存在，请先构建知识库")
        st.info("请确保 faiss_index 文件夹存在且包含必要的索引文件")
        st.stop()

    # 初始化RAG组件
    rag_chain = initialize_rag_components()
    if not rag_chain:
        st.error("❌ 错误：知识库初始化失败！")
        st.stop()

    # 混合问答历史
    if 'hybrid_messages' not in st.session_state:
        st.session_state.hybrid_messages = [
            {"role": "assistant", "content": "您好！我是混合问答助手，我可以提供全面的文本分析和计算解决方案。"}]

    # 高级选项
    with st.expander("🔧 高级选项"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("检索文档数量", 1, 10, 3, key="hybrid_top_k")
        with col2:
            temperature = st.slider("回答创造性", 0.0, 1.0, 0.3, key="hybrid_temperature")

    # 显示混合问答历史
    for msg in st.session_state.hybrid_messages[1:]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "analysis" in msg:
                # 显示综合回答（包含文本回答内容）
                st.markdown(msg["analysis"])

                # 显示参考文档
                if "text_docs_info" in msg and msg["text_docs_info"]:
                    with st.expander("📄 查看知识库参考文档"):
                        for i, (doc_content, score) in enumerate(msg["text_docs_info"]):
                            st.write(f"**文档 {i + 1}** (相似度: {score:.3f}):")
                            st.text(doc_content[:300] + "..." if len(doc_content) > 300 else doc_content)
                            st.markdown("---")

                # 显示参数解析
                if "parsed_params" in msg and msg["parsed_params"]:
                    with st.expander("🔧 解析的参数"):
                        st.json(msg["parsed_params"])

                # 显示历史计算参考
                if "calc_docs_info" in msg and msg["calc_docs_info"]:
                    with st.expander("📊 历史计算参考"):
                        for i, (doc_content, score, doc_metadata) in enumerate(msg["calc_docs_info"]):
                            calc_type = doc_metadata.get('calculation_type', '未知')
                            points_count = doc_metadata.get('total_points', 0)
                            st.write(f"**历史计算 {i + 1}** (类型: {calc_type}, 相似度: {score:.3f})")
                            st.caption(
                                f"数据点数: {points_count}个, 创建时间: {doc_metadata.get('created_at', '未知')}")
                            st.text(doc_content[:200] + "..." if len(doc_content) > 200 else doc_content)
                            st.markdown("---")

                # 显示可视化状态
                if "visualization_displayed" in msg and msg["visualization_displayed"]:
                    st.success("✅ 已显示可视化计算结果")

            else:
                st.markdown(msg["content"])

    # 混合问答输入框
    if hybrid_input := st.chat_input("请输入您的问题，如：坐标xx到XX的路径损耗..."):
        # 用户消息
        st.session_state.hybrid_messages.append({"role": "user", "content": hybrid_input})
        with st.chat_message("user"):
            st.markdown(hybrid_input)

        # 处理混合问答
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在处理您的请求..."):
                try:
                    # 初始化结果容器
                    text_result = ""
                    text_docs_info = []
                    calc_info = None
                    calc_docs_info = []

                    # 1. 首先进行文本问答（RAG）- 但不单独显示界面
                    try:
                        retriever = rag_chain.retriever
                        text_docs = retriever.improved_retrieve(hybrid_input, top_k=top_k)

                        # 生成文本回答
                        text_result = rag_chain.invoke({
                            "question": hybrid_input,
                            "temperature": temperature
                        })

                        # 保存文本回答信息
                        text_docs_info = [(doc.page_content, score) for doc, score in text_docs]

                    except Exception as e:
                        text_result = f"文本回答生成遇到问题: {str(e)}"

                    # 2. 参数解析和计算（如果适用）
                    # 检查问题是否包含可解析的参数
                    has_calculatable_params = any(keyword in hybrid_input.lower() for keyword in
                                                  ['计算', '分析', '损耗', '场强', '路径', '距离',
                                                   '坐标', '纬度', '经度', '频率', '功率', '增益'])

                    if has_calculatable_params:
                        try:
                            # 参数解析
                            parsed_params, calculation_type = parse_question_parameters(hybrid_input)
                            print(parsed_params)

                            if parsed_params:
                                # 向量库相似度查询
                                similar_result = None
                                retriever = rag_chain.retriever

                                # 构建查询文本
                                query_text = f"{calculation_type}计算 "
                                param_texts = []

                                if calculation_type == "path_loss":
                                    key_params = ['tx_lat', 'tx_lon', 'rx_lat', 'rx_lon']
                                else:
                                    key_params = ['lat', 'lon', 'rad']

                                for key in key_params:
                                    if key in parsed_params and parsed_params[key] is not None:
                                        param_texts.append(f"{key}:{parsed_params[key]}")

                                if param_texts:
                                    query_text += " ".join(param_texts)

                                # 查询相似文档
                                docs = retriever.improved_retrieve(query_text, top_k=2)
                                relevant_docs = []
                                for doc, score in docs:
                                    doc_calc_type = doc.metadata.get('calculation_type', '')
                                    if doc_calc_type == calculation_type:
                                        relevant_docs.append((doc, score))
                                        calc_docs_info.append((doc.page_content, score, doc.metadata))

                                if relevant_docs:
                                    best_doc, best_score = relevant_docs[0]
                                    similar_result = {
                                        'doc': best_doc,
                                        'score': best_score,
                                        'metadata': best_doc.metadata
                                    }

                                # 执行计算和可视化
                                calculation_result = None
                                visualization_container = st.container()

                                with visualization_container:
                                    if calculation_type == "path_loss":
                                        # 准备参数
                                        path_loss_params = {}
                                        valid_params = ['tx_lat', 'tx_lon', 'rx_lat', 'rx_lon', 'gap', 'Pt', 'Gt',
                                                        'frequency', 'tx_antenna_height', 'rx_antenna_height',
                                                        'time_percentage', 'signal_polarization']

                                        for key in valid_params:
                                            if key in parsed_params and parsed_params[key] is not None:
                                                path_loss_params[key] = parsed_params[key]

                                        # 如果有相似文档，补充参数
                                        if similar_result:
                                            doc_params = vector_db_query_to_params(similar_result['metadata'],
                                                                                   calculation_type)
                                            for key, value in doc_params.items():
                                                if key in valid_params and (
                                                        key not in path_loss_params or path_loss_params[key] is None):
                                                    path_loss_params[key] = value


                                        # 执行计算
                                        if all(k in path_loss_params for k in ['tx_lat', 'tx_lon', 'rx_lat', 'rx_lon']):
                                            calculation_result = calculate_path_loss(**path_loss_params)

                                            # 显示结果
                                            show_path_loss_results(calculation_result)
                                            show_path_visualization(calculation_result)
                                            show_loss_analysis(calculation_result)

                                    elif calculation_type == "field_strength":
                                        # 准备参数
                                        print("1. 代码开始执行")
                                        field_strength_params = {}
                                        valid_params = ['lat', 'lon', 'rad', 'gap', 'Pt', 'Gt', 'frequency',
                                                        'tx_antenna_height', 'rx_antenna_height', 'time_percentage',
                                                        'signal_polarization']

                                        for key in valid_params:
                                            if key in parsed_params and parsed_params[key] is not None:
                                                field_strength_params[key] = parsed_params[key]

                                        # 如果有相似文档，补充参数
                                        if similar_result:
                                            doc_params = vector_db_query_to_params(similar_result['metadata'],
                                                                                   calculation_type)
                                            for key, value in doc_params.items():
                                                if key in valid_params and (
                                                        key not in field_strength_params or field_strength_params[
                                                    key] is None):
                                                    field_strength_params[key] = value
                                        print("2. 到达场强计算部分")
                                        if 'signal_polarization' in field_strength_params:
                                            field_strength_params['signal_pol'] = field_strength_params.pop(
                                                'signal_polarization')

                                        # 执行计算
                                        if all(k in field_strength_params for k in ['lat', 'lon', 'rad']):
                                            calculation_result = calculate_field_strength(**field_strength_params)
                                            print(calculation_result)

                                            # 显示结果
                                            show_field_strength_results(calculation_result)
                                            show_enhanced_baidu_map11(calculation_result)

                                # 分析文档点数据

                                doc_points_analysis = ""
                                if similar_result:
                                    doc_points_analysis = analyze_document_points(
                                        similar_result['metadata'],
                                        calculation_type,
                                        hybrid_input
                                    )


                                # 保存计算相关信息
                                calc_info = {
                                    'calculation_type': calculation_type,
                                    'parsed_params': parsed_params,
                                    'calculation_result': calculation_result,
                                    'similar_result': similar_result,
                                    'doc_points_analysis': doc_points_analysis,
                                    'visualization_displayed': calculation_result is not None
                                }

                        except Exception as e:
                            # 计算分析错误不影响整体流程
                            calc_info = None

                    # 3. 生成综合回答并显示
                    comprehensive_analysis = generate_comprehensive_analysis(
                        text_response=text_result,
                        text_docs_info=text_docs_info,
                        calculation_info=calc_info,
                        calc_docs_info=calc_docs_info,
                        user_question=hybrid_input
                    )

                    # 直接显示综合分析
                    st.markdown(comprehensive_analysis)

                    # 保存到历史记录
                    st.session_state.hybrid_messages.append({
                        "role": "assistant",
                        "content": f"已为您提供综合分析",
                        "analysis": comprehensive_analysis,
                        "text_docs_info": text_docs_info,
                        "calc_docs_info": calc_docs_info,
                        "parsed_params": calc_info['parsed_params'] if calc_info else None,
                        "calculation_type": calc_info['calculation_type'] if calc_info else None,
                        "visualization_displayed": calc_info and calc_info.get('visualization_displayed', False),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                except Exception as e:
                    error_msg = f"❌ 混合问答处理失败: {str(e)}"
                    st.error(error_msg)
                    import traceback
                    st.error(f"详细错误: {traceback.format_exc()}")
                    st.session_state.hybrid_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def generate_comprehensive_analysis(text_response, text_docs_info, calculation_info, calc_docs_info, user_question="",
                                    vectorstore=None, embedder=None):
    """
    生成综合分析报告 - 修复参数显示问题，结合文档知识分析
    """
    analysis_parts = []

    # 1. 开头说明
    analysis_parts.append("### 📋 综合分析报告")

    if user_question:
        analysis_parts.append(f"**用户查询**: {user_question}")

    # 2. 计算结果详细分析
    if calculation_info:
        calc_type = calculation_info.get('calculation_type', 'unknown')
        parsed_params = calculation_info.get('parsed_params', {})
        calc_result = calculation_info.get('calculation_result', {})
        similar_result = calculation_info.get('similar_result', {})

        analysis_parts.append("\n### 🔬 计算结果详细分析")

        # 显示实际使用的参数
        if parsed_params:
            analysis_parts.append("**实际使用参数**:")

            # 根据计算类型显示不同的参数
            if calc_type == "field_strength":
                field_params = [
                    ('lat', '发射点纬度'),
                    ('lon', '发射点经度'),
                    ('rad', '分析半径(km)'),
                    ('frequency', '频率(MHz)'),
                    ('Pt', '发射功率(kW)'),
                    ('Gt', '发射天线增益(dBi)'),
                    ('tx_antenna_height', '发射天线高度(m)'),
                    ('rx_antenna_height', '接收天线高度(m)'),
                    ('time_percentage', '时间概率(%)'),
                    ('signal_polarization', '信号极化')
                ]

                for param_key, param_desc in field_params:
                    if param_key in parsed_params and parsed_params[param_key] is not None:
                        analysis_parts.append(f"- {param_desc}: {parsed_params[param_key]}")

            elif calc_type == "path_loss":
                path_params = [
                    ('tx_lat', '发射点纬度'),
                    ('tx_lon', '发射点经度'),
                    ('rx_lat', '接收点纬度'),
                    ('rx_lon', '接收点经度'),
                    ('frequency', '频率(MHz)'),
                    ('Pt', '发射功率(kW)'),
                    ('Gt', '发射天线增益(dBi)'),
                    ('tx_antenna_height', '发射天线高度(m)'),
                    ('rx_antenna_height', '接收天线高度(m)')
                ]

                for param_key, param_desc in path_params:
                    if param_key in parsed_params and parsed_params[param_key] is not None:
                        analysis_parts.append(f"- {param_desc}: {parsed_params[param_key]}")

        # 显示计算结果
        if calc_result:
            analysis_parts.append("\n**计算结果统计**:")

            if calc_type == "field_strength":
                avg_strength = calc_result.get('avg_field_strength')
                max_strength = calc_result.get('max_field_strength')
                min_strength = calc_result.get('min_field_strength')

                if avg_strength is not None:
                    analysis_parts.append(f"- 平均场强: {avg_strength:.2f} dBμV/m")
                if max_strength is not None:
                    analysis_parts.append(f"- 最大场强: {max_strength:.2f} dBμV/m")
                if min_strength is not None:
                    analysis_parts.append(f"- 最小场强: {min_strength:.2f} dBμV/m")

                points_data = calc_result.get('points', [])
                if points_data:
                    analysis_parts.append(f"- 分析点数: {len(points_data)} 个")

            elif calc_type == "path_loss":
                avg_loss = calc_result.get('avg_loss')
                max_loss = calc_result.get('max_loss')
                min_loss = calc_result.get('min_loss')
                distance = calc_result.get('distance')

                if avg_loss is not None:
                    analysis_parts.append(f"- 平均损耗: {avg_loss:.2f} dB")
                if max_loss is not None:
                    analysis_parts.append(f"- 最大损耗: {max_loss:.2f} dB")
                if min_loss is not None:
                    analysis_parts.append(f"- 最小损耗: {min_loss:.2f} dB")
                if distance is not None:
                    analysis_parts.append(f"- 传播距离: {distance:.2f} km")

                points_data = calc_result.get('points', [])
                if points_data:
                    analysis_parts.append(f"- 路径点数: {len(points_data)} 个")

                    # 计算单位距离损耗
                    if distance and distance > 0 and avg_loss:
                        loss_per_km = avg_loss / distance
                        analysis_parts.append(f"- 单位距离损耗: {loss_per_km:.2f} dB/km")

    # 3. 结合技术文档的专业分析
    if text_docs_info:
        analysis_parts.append("\n### 📚 技术文档知识分析")

        # 提取与当前计算相关的文档信息
        relevant_info = []

        for doc_content, score in text_docs_info:
            if doc_content and score > 0.5:  # 只考虑相似度较高的文档
                content_str = str(doc_content)

                # 提取关键知识
                if calculation_info:
                    calc_type = calculation_info.get('calculation_type', '')

                    # 根据计算类型提取相关信息
                    if calc_type == "field_strength":
                        # 场强计算相关关键词
                        field_keywords = ['场强', '电场', '覆盖', '传播', '衰减', 'itu', 'itu-r']
                        if any(keyword in content_str.lower() for keyword in field_keywords):
                            # 提取前3行有意义的文本
                            lines = content_str.split('\n')
                            for line in lines[:10]:  # 只检查前10行
                                line_clean = line.strip()
                                if len(line_clean) > 20 and not line_clean.startswith('==='):
                                    relevant_info.append(f"{line_clean[:60]}... (可信度: {score:.3f})")
                                    break

                    elif calc_type == "path_loss":
                        # 路径损耗相关关键词
                        path_keywords = ['路径损耗', '传播损耗', '传播模型', 'itu-r p.1812', 'itu-r p.530']
                        if any(keyword in content_str.lower() for keyword in path_keywords):
                            # 提取技术描述
                            lines = content_str.split('\n')
                            for line in lines[:10]:
                                line_clean = line.strip()
                                if len(line_clean) > 20 and not line_clean.startswith('==='):
                                    relevant_info.append(f"{line_clean[:60]}... (可信度: {score:.3f})")
                                    break

        if relevant_info:
            analysis_parts.append("**文档知识参考**:")
            for i, info in enumerate(relevant_info[:3]):  # 只显示前3条
                analysis_parts.append(f"{i + 1}. {info}")
        else:
            analysis_parts.append("**技术文档**: 未找到与当前计算直接相关的技术知识")

    # 4. 对比分析：实际结果与文档知识的对比
    analysis_parts.append("\n### 🔍 知识库深度分析")

    if calculation_info and calc_result:
        calc_type = calculation_info.get('calculation_type', '')

        if calc_type == "path_loss":
            avg_loss = calc_result.get('avg_loss')
            distance = calc_result.get('distance')

            if avg_loss and distance:
                loss_per_km = avg_loss / distance if distance > 0 else 0

                analysis_parts.append("**路径损耗评估**:")

                # 基于ITU-R P.1812建议书的应用场景修正典型值
                if 'frequency' in parsed_params:
                    freq = parsed_params.get('frequency', 0)  # MHz
                    freq_ghz = freq / 1000  # 转换为GHz

                    # 基于ITU-R P.1812的适用范围进行典型值调整
                    # 该方法适用于30 MHz至6 GHz，路径长度0.25-3000 km

                    analysis_parts.append(f"- 频率: {freq} MHz ({freq_ghz:.3f} GHz)")
                    analysis_parts.append(f"- 传播距离: {distance:.2f} km")
                    analysis_parts.append(f"- 总路径损耗: {avg_loss:.2f} dB")
                    analysis_parts.append(f"- 计算单位损耗: {loss_per_km:.2f} dB/km")

                    # 根据ITU-R P.1812模型特性提供参考
                    analysis_parts.append("\n**基于ITU-R P.1812模型的参考**:")

                    # 根据频率范围提供指导性参考
                    if freq_ghz < 0.1:  # VHF低频段 (30-100 MHz)
                        if distance < 10:
                            typical_range = (4.0, 12.0)  # dB/km
                            env_desc = "VHF短距离，受地形影响较大"
                        elif distance < 100:
                            typical_range = (1.5, 6.0)  # dB/km
                            env_desc = "VHF中距离，衍射传播为主"
                        else:
                            typical_range = (0.8, 3.0)  # dB/km
                            env_desc = "VHF远距离，对流层散射可能"

                    elif freq_ghz < 1.0:  # UHF频段 (0.1-1 GHz)
                        if distance < 10:
                            typical_range = (8.0, 25.0)  # dB/km
                            env_desc = "UHF短距离，衰减较大"
                        elif distance < 100:
                            typical_range = (3.0, 15.0)  # dB/km
                            env_desc = "UHF中距离，多径效应明显"
                        else:
                            typical_range = (1.5, 8.0)  # dB/km
                            env_desc = "UHF远距离，受大气影响"

                    else:  # 微波频段 (1-6 GHz)
                        if distance < 10:
                            typical_range = (12.0, 40.0)  # dB/km
                            env_desc = "微波短距离，高衰减"
                        elif distance < 100:
                            typical_range = (5.0, 25.0)  # dB/km
                            env_desc = "微波中距离，受降雨影响"
                        else:
                            typical_range = (3.0, 15.0)  # dB/km
                            env_desc = "微波远距离，需考虑大气效应"

                    analysis_parts.append(f"- 频率分类: {env_desc}")
                    analysis_parts.append(f"- 典型损耗范围: {typical_range[0]:.1f} ~ {typical_range[1]:.1f} dB/km")

                    # 合理性评估
                    if typical_range[0] <= loss_per_km <= typical_range[1]:
                        analysis_parts.append("✅ **评估结果**: 路径损耗在典型范围内，结果合理")
                        analysis_parts.append(f"  - 符合{env_desc}的传播特性")
                    elif loss_per_km < typical_range[0]:
                        analysis_parts.append("ℹ️ **评估结果**: 路径损耗偏小，可能原因：")
                        analysis_parts.append(
                            f"  - 当前值: {loss_per_km:.2f} dB/km，典型下限: {typical_range[0]:.1f} dB/km")
                        analysis_parts.append("  - 视距传播条件良好")
                        analysis_parts.append("  - 天线高度差较小")
                        analysis_parts.append("  - 地形平坦或有利")
                    else:
                        analysis_parts.append("ℹ️ **评估结果**: 路径损耗偏大，可能原因：")
                        analysis_parts.append(
                            f"  - 当前值: {loss_per_km:.2f} dB/km，典型上限: {typical_range[1]:.1f} dB/km")
                        analysis_parts.append("  - 非视距传播（NLoS）")
                        analysis_parts.append("  - 存在显著衍射损耗")
                        analysis_parts.append("  - 地形复杂（山地、城市峡谷）")

                # 自由空间损耗参考（理论最小值）
                if 'frequency' in parsed_params:
                    freq = parsed_params.get('frequency', 0)
                    if freq > 0 and distance > 0:
                        # 自由空间路径损耗公式：Lfs = 32.44 + 20log10(d) + 20log10(f)
                        # d:距离(km), f:频率(MHz)
                        free_space_loss = 32.44 + 20 * np.log10(distance) + 20 * np.log10(freq)
                        actual_loss = avg_loss

                        analysis_parts.append(f"\n**自由空间损耗对比（理论最小值）**:")
                        analysis_parts.append(f"- 自由空间理论损耗: {free_space_loss:.2f} dB")
                        analysis_parts.append(f"- 实际计算损耗: {actual_loss:.2f} dB")
                        analysis_parts.append(f"- 额外损耗（余量）: {actual_loss - free_space_loss:.2f} dB")

                        # 根据ITU-R P.1812模型特性评估额外损耗
                        excess_loss = actual_loss - free_space_loss

                        if excess_loss < 5:
                            analysis_parts.append("✅ 接近自由空间传播（理想视距条件）")
                        elif 5 <= excess_loss < 15:
                            analysis_parts.append("✅ 轻度额外损耗（轻微衍射或反射）")
                        elif 15 <= excess_loss < 30:
                            analysis_parts.append("⚠️ 中度额外损耗（存在显著传播障碍）")
                        elif 30 <= excess_loss < 50:
                            analysis_parts.append("⚠️ 较大额外损耗（强衍射或NLoS条件）")
                        else:
                            analysis_parts.append("⚠️ 极大额外损耗（极端传播条件或参数异常）")

        elif calc_type == "field_strength":
            avg_strength = calc_result.get('avg_field_strength')

            if avg_strength is not None:
                analysis_parts.append("**场强覆盖评估**:")

                # 根据ITU-R P.1812建议书的场强计算参考
                if 'distance' in calc_result or 'rad' in parsed_params:
                    distance = calc_result.get('distance', parsed_params.get('rad', 0))
                    freq = parsed_params.get('frequency', 100)  # MHz
                    freq_ghz = freq / 1000
                    power = parsed_params.get('Pt', 1)  # kW，默认1kW

                    # ITU-R P.1812公式(70): E_p = 199.36 + 20log(f) - L_b
                    # 其中f单位为GHz，L_b为基本传输损耗
                    # 这里简化为理论估算

                    # 对于1kW ERP，自由空间下的理论场强
                    # E = 106.9 + 20log10(f_MHz) - 20log10(d_km) dBμV/m (自由空间)
                    estimated_strength_fs = 106.9 + 20 * np.log10(freq) - 20 * np.log10(distance)

                    # 根据距离和频率调整估算（考虑典型传播条件）
                    adjustment = 0
                    if distance < 10:
                        adjustment = -10  # 短距离衰减较小
                    elif distance < 50:
                        adjustment = -15
                    elif distance < 100:
                        adjustment = -20
                    else:
                        adjustment = -25 - 5 * np.log10(distance / 100)  # 长距离额外衰减

                    estimated_strength = estimated_strength_fs + adjustment

                    analysis_parts.append(f"- 距离: {distance:.1f} km")
                    analysis_parts.append(f"- 频率: {freq} MHz")
                    analysis_parts.append(f"- 发射功率: {power} kW ERP")
                    analysis_parts.append(f"- 自由空间理论场强: {estimated_strength_fs:.1f} dBμV/m")
                    analysis_parts.append(f"- 典型传播估算场强: {estimated_strength:.1f} dBμV/m")
                    analysis_parts.append(f"- 实际计算场强: {avg_strength:.1f} dBμV/m")
                    analysis_parts.append(f"- 差值: {avg_strength - estimated_strength:.1f} dB")

                    # 场强合理性评估
                    diff = avg_strength - estimated_strength

                    if abs(diff) < 10:
                        analysis_parts.append("✅ 场强值与典型估算基本一致")
                        analysis_parts.append("  符合一般传播模型预测")
                    elif diff > 10:
                        analysis_parts.append("ℹ️ 场强值偏高，可能原因：")
                        analysis_parts.append("  - 天线增益较高（方向性好）")
                        analysis_parts.append("  - 传播路径有利（高地、视距）")
                        analysis_parts.append("  - 可能存在波导或异常传播")
                    else:
                        analysis_parts.append("ℹ️ 场强值偏低，可能原因：")
                        analysis_parts.append("  - 传播路径有遮挡（NLoS）")
                        analysis_parts.append("  - 地形复杂（山谷、城市）")
                        analysis_parts.append("  - 存在显著衍射损耗")

    # 5. 专业建议
    analysis_parts.append("\n###  🎯 综合建议")

    if calculation_info:
        calc_type = calculation_info.get('calculation_type', '')

        if calc_type == "path_loss":
            analysis_parts.append("**针对当前路径损耗计算的建议**:")
            analysis_parts.append("1. **参数验证**: 确认频率、天线高度等参数符合实际设备规格")
            analysis_parts.append("2. **模型选择**: 根据地形选择合适的传播模型（如ITU-R P.1812）")
            analysis_parts.append("3. **环境因素**: 考虑地形起伏、建筑物遮挡等实际环境影响")
            analysis_parts.append("4. **实测对比**: 建议进行现场测量验证计算结果准确性")

            # 如果找到了相似历史计算
            if 'similar_result' in calculation_info and calculation_info['similar_result']:
                analysis_parts.append("5. **历史参考**: 系统已找到相似的历史计算，可参考其参数设置")

        elif calc_type == "field_strength":
            analysis_parts.append("**针对当前场强计算的建议**:")
            analysis_parts.append("1. **覆盖优化**: 根据场强分布调整发射点位置或功率")
            analysis_parts.append("2. **参数调整**: 优化天线高度、增益等参数改善覆盖效果")
            analysis_parts.append("3. **时间分析**: 考虑不同时间概率下的场强变化")
            analysis_parts.append("4. **极化匹配**: 确保发射与接收天线极化方式匹配")

            # 检查是否有必要参数
            required_fields = ['frequency', 'Pt', 'tx_antenna_height']
            missing_in_docs = [f for f in required_fields if f not in parsed_params]

            if missing_in_docs:
                analysis_parts.append(f"5. **参数补充**: 建议补充 {', '.join(missing_in_docs)} 等关键参数")

    # 6. 结论总结
    analysis_parts.append("\n### ✅ 分析总结")

    has_calc = calculation_info and calculation_info.get('calculation_result')
    has_docs = text_docs_info and len(text_docs_info) > 0

    if has_calc and has_docs:
        analysis_parts.append("✅ **综合分析完成**: 结合计算结果和技术文档进行了全面评估")
        analysis_parts.append("📊 **数据完整性**: 参数完整，结果可信")
        analysis_parts.append("🔍 **知识结合**: 已参考相关技术文档知识")

    elif has_calc:
        analysis_parts.append("✅ **计算分析完成**: 已完成参数计算和结果评估")
        analysis_parts.append("📝 **建议**: 可补充更多技术文档进行对比分析")

    elif has_docs:
        analysis_parts.append("✅ **文档分析完成**: 已检索相关技术知识")
        analysis_parts.append("🧮 **建议**: 提供具体参数可进行实际计算分析")

    # 返回结果
    return "\n".join([str(p) for p in analysis_parts if p is not None])



# 同时，在主应用中使用时，需要这样调用：
def enhanced_main_application():
    """增强的主应用示例，展示如何正确使用分析函数"""

    # 1. 首先初始化向量库（如果可用）
    vectorstore = None
    embedder = None

    try:
        from document_vector.faiss_vectorstore import BGEEmbedder, FaissVectorStore
        embedder = BGEEmbedder()
        vectorstore = FaissVectorStore(embedder, "document_vector/faiss_index")

        # 检查索引是否存在
        import os
        index_file = os.path.join("document_vector/faiss_index", "index.faiss")
        if os.path.exists(index_file):
            vectorstore.load()
            print("✅ 成功加载向量库")
        else:
            print("⚠️ 向量库索引不存在，将运行 build_vector_first.py...")
            # 可以在这里自动构建索引
            import build_vector_first
            if build_vector_first.build_knowledge_base():
                vectorstore.load()
    except Exception as e:
        print(f"⚠️ 向量库初始化失败: {str(e)}")
        print("将使用无向量库模式进行分析")

    # 2. 在需要分析时，传入向量库参数
    def process_user_query(user_question, text_response, text_docs_info,
                           calculation_info=None, calc_docs_info=None):
        """处理用户查询的增强版本"""

        # 生成综合分析报告
        analysis_report = generate_comprehensive_analysis(
            text_response=text_response,
            text_docs_info=text_docs_info,
            calculation_info=calculation_info,
            calc_docs_info=calc_docs_info,
            user_question=user_question,
            vectorstore=vectorstore,
            embedder=embedder
        )

        return analysis_report
















def show_chat_page():
    """简化的聊天页面"""

    # 只显示聊天历史，不显示其他复杂内容
    for msg in st.session_state.chat_messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 🔄 改为支持Enter键的输入框
    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:
        # 🆕 使用 text_input 并监听键盘事件
        user_input = st.text_input(
            "主聊天输入框",
            placeholder="Please enter your question...",
            key="main_chat_input",
            label_visibility="collapsed",
            on_change=handle_input_change,  # 🔧 添加回调函数
            args=()
        )


def handle_input_change():
    """处理输入框变化 - 检测Enter键"""
    if st.session_state.main_chat_input.strip():
        # 获取输入内容
        user_input = st.session_state.main_chat_input.strip()

        # 处理消息
        process_chat_message(user_input)

        # 清空输入框
        st.session_state.main_chat_input = ""


def process_chat_message(chat_input):
    """处理聊天消息的函数"""
    # 用户消息
    st.session_state.chat_messages.append({"role": "user", "content": chat_input})

    # 获取AI回复
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=st.session_state.chat_messages,
            stream=True
        )

        response_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

        # 重新运行以显示新消息
        st.rerun()

    except Exception as e:
        error_msg = f"❌ AI回复失败: {str(e)}"
        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        st.rerun()


def main():
    """主函数"""
    # ===== 初始化缓存 =====
    if 'cache_system' not in st.session_state:
        st.session_state.cache_system = init_cache_system()
    cache = st.session_state.cache_system

    # ===== 初始化页面状态 =====
    if "page_mode" not in st.session_state:
        st.session_state.page_mode = "chat"

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    if "path_loss_history" not in st.session_state:
        st.session_state.path_loss_history = []

    if "field_strength_history" not in st.session_state:
        st.session_state.field_strength_history = []

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": "你是无线电监测专家，回答相关技术问题，请用中文回答。"}
        ]

    if "kb_messages" not in st.session_state:
        st.session_state.kb_messages = [
            {"role": "system", "content": "你是基于无线电监测知识库的专业助手，请根据专业知识详细回答问题。"}
        ]

    # 🔧 在侧边栏显示缓存状态
    with st.sidebar:
        # 默认隐藏
        with st.expander("🗄️ 缓存状态", expanded=False):

            if cache.is_available():
                st.success("🟢 缓存系统已连接")

                # 获取统计信息
                stats = cache.get_cache_statistics()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("路径损耗", stats.get("path_loss_cached", 0))
                    st.metric("访问次数", stats.get("path_loss_access_total", 0))

                with col2:
                    st.metric("区域场强", stats.get("field_strength_cached", 0))
                    st.metric("访问次数", stats.get("field_strength_access_total", 0))

                # 缓存管理按钮
                if st.button("🗑️ 清理旧缓存", help="清理30天前访问次数少的缓存"):
                    result = cache.clear_old_cache(30)
                    st.success(
                        f"已清理 {result['path_loss_deleted']} 个路径损耗缓存和 {result['field_strength_deleted']} 个区域场强缓存")
                    st.rerun()

            else:
                st.error("🔴 缓存系统未连接")
                st.info("💡 系统将直接执行计算")

    # ===== 页面路由 =====
    if st.session_state.page_mode == "chat":
        show_main_page()
        #show_chat_page()

    elif st.session_state.page_mode == "path_loss":
        show_path_loss_page()

    elif st.session_state.page_mode == "field_strength":
        show_field_strength_page()

    elif st.session_state.page_mode == "knowledge":
        show_knowledge_page()



if __name__ == '__main__':
    main()