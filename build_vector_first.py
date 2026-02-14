import os
import sys
import json
from pathlib import Path
from document_vector.faiss_vectorstore import FaissVectorStore, BGEEmbedder, MongoDBDataManager
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import re
import hashlib


def smart_pdf_chunking(pdf_path, text_splitter):
    """智能PDF文档分块，确保内容的连贯性和避免过度分割"""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # 1. 识别文档结构
    structured_docs = []
    current_section = []
    section_title = ""

    for i, doc in enumerate(docs):
        page_content = doc.page_content.strip()

        # 识别章节标题（数字开头的标题，如 "1. 范围"、"2. 规范性引用文件"）
        lines = page_content.split('\n')
        for line in lines[:5]:  # 只检查前5行，通常是标题位置
            if re.match(r'^\d+\.\s+[\u4e00-\u9fa5a-zA-Z]', line.strip()):
                # 找到新章节标题
                if current_section:
                    # 保存当前章节
                    structured_docs.append({
                        'title': section_title,
                        'content': '\n'.join(current_section),
                        'source': str(pdf_path),
                        'page_range': f"{len(structured_docs) * 2 - 1}-{len(structured_docs) * 2}"
                    })

                # 开始新章节
                section_title = line.strip()
                current_section = [f"【{section_title}】"]
                break

        # 添加页面内容
        current_section.append(f"第{i + 1}页：{page_content}")

    # 添加最后一个章节
    if current_section:
        structured_docs.append({
            'title': section_title,
            'content': '\n'.join(current_section),
            'source': str(pdf_path),
            'page_range': f"{len(structured_docs) * 2 - 1}-{len(structured_docs) * 2}"
        })

    # 2. 如果没有识别到章节结构，使用原始分块
    if not structured_docs:
        # 合并相邻页面
        merged_content = ""
        for i, doc in enumerate(docs):
            merged_content += f"【第{i + 1}页】\n{doc.page_content}\n\n"

        # 使用较大的分块尺寸
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=300,
            separators=["\n\n\n", "\n\n", "\n", "。", "；", "！", "？", " ", ""]
        )

        base_doc = Document(page_content=merged_content)
        chunks = large_splitter.split_documents([base_doc])

        # 为每个块添加元数据
        for j, chunk in enumerate(chunks):
            # 生成内容哈希作为唯一标识
            content_hash = hashlib.md5(chunk.page_content[:500].encode()).hexdigest()[:8]
            chunk.metadata = {
                "source": str(pdf_path),
                "type": "pdf_document",
                "filename": os.path.basename(pdf_path),
                "chunk_id": f"{os.path.basename(pdf_path)}_{j}_{content_hash}",
                "doc_type": "技术文档"
            }

        return chunks

    # 3. 基于章节结构分块
    final_chunks = []
    for section in structured_docs:
        # 如果章节内容过长，进一步分块
        if len(section['content']) > 2000:
            section_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=200,
                separators=["\n", "。", "；", "！", "？", "，", " ", ""]
            )

            section_doc = Document(page_content=section['content'])
            sub_chunks = section_splitter.split_documents([section_doc])

            for j, chunk in enumerate(sub_chunks):
                # 生成内容哈希
                content_hash = hashlib.md5(chunk.page_content[:500].encode()).hexdigest()[:8]
                chunk.metadata = {
                    "source": str(pdf_path),
                    "type": "pdf_document",
                    "filename": os.path.basename(pdf_path),
                    "section_title": section['title'],
                    "chunk_id": f"{os.path.basename(pdf_path)}_{section['title']}_{j}_{content_hash}",
                    "page_range": section['page_range'],
                    "doc_type": "技术文档"
                }
                final_chunks.append(chunk)
        else:
            # 章节内容适中，直接作为一个块
            section_doc = Document(page_content=section['content'])
            content_hash = hashlib.md5(section['content'][:500].encode()).hexdigest()[:8]
            section_doc.metadata = {
                "source": str(pdf_path),
                "type": "pdf_document",
                "filename": os.path.basename(pdf_path),
                "section_title": section['title'],
                "chunk_id": f"{os.path.basename(pdf_path)}_{section['title']}_{content_hash}",
                "page_range": section['page_range'],
                "doc_type": "技术文档"
            }
            final_chunks.append(section_doc)

    return final_chunks


def process_text_document_with_id(file_path, text_splitter):
    """处理文本文档，添加唯一标识符"""
    loader = TextLoader(str(file_path), encoding='utf-8')
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        # 生成内容哈希
        content_hash = hashlib.md5(chunk.page_content[:500].encode()).hexdigest()[:8]
        chunk.metadata = {
            "source": str(file_path),
            "type": "text_document",
            "filename": os.path.basename(file_path),
            "chunk_id": f"{os.path.basename(file_path)}_{i}_{content_hash}",
            "doc_type": "技术文档"
        }

    return chunks
def process_mongodb_document_as_chunk(item, collection_type):
    """将单个MongoDB文档处理为一个块，包含完整的参数和结果数据（所有点数据）"""
    try:
        # 构建详细的技术描述内容
        content_parts = []

        # 根据集合类型设置元数据
        if collection_type == "field_strength":
            metadata = {
                "source": "geospatial_cache",
                "collection": "field_strength_cache",
                "calculation_type": "field_strength",
                "mongodb_id": str(item.get('_id', '')),
                "data_size": item.get('data_size', 0),
                "access_count": item.get('access_count', 0),
                "created_at": str(item.get('created_at', '')),
                "last_accessed": str(item.get('last_accessed', '')),
                "params_hash": item.get('params_hash', '')
            }

            # 添加核心计算参数描述
            params = item.get('params', {})
            if params:
                content_parts.append("=== 场强计算参数 ===")
                content_parts.append(f"发射台坐标: ({params.get('tx_lat', 'N/A')}, {params.get('tx_lon', 'N/A')})")
                content_parts.append(f"分析区域: 半径{params.get('radius', 'N/A')}公里圆形区域")
                content_parts.append(f"采样密度: {params.get('gap', 'N/A')}米间距")

                # 设备参数
                content_parts.append(f"工作频率: {params.get('frequency', 'N/A')}MHz")
                content_parts.append(f"发射功率: {params.get('Pt', 'N/A')}千瓦")
                content_parts.append(f"发射天线增益: {params.get('Gt', 'N/A')}dBi")
                content_parts.append(f"发射天线高度: {params.get('tx_antenna_height', 'N/A')}米")
                content_parts.append(f"接收天线高度: {params.get('rx_antenna_height', 'N/A')}米")

                # 传播模型参数
                content_parts.append(f"时间概率: {params.get('time_percentage', 'N/A')}%")
                content_parts.append(f"信号极化: {params.get('signal_polarization', 'N/A')}")

                # 存储参数到元数据
                for key, value in params.items():
                    metadata[f"param_{key}"] = str(value)

            # 添加详细的计算结果，包括具体点数据
            result = item.get('result', {})
            if result:
                content_parts.append("\n=== 计算结果摘要 ===")

                # 获取点数据 - 修正：根据实际情况获取
                # 从结果中获取预测点和实际点
                points = result.get('points', [])  # 预测点
                actual_points = result.get('actual_points', [])  # 实际计算点

                # 如果没有actual_points，可能所有点都是有效的
                if not actual_points and points:
                    # 检查points中是否有有效标记
                    # 或者假设points就是所有有效点
                    actual_points = points

                content_parts.append(f"预测点数: {len(points)}个")
                content_parts.append(f"有效点数: {len(actual_points)}个")

                # 摘要信息 - 修正：使用实际数据
                # 提取场强值进行统计
                all_field_strengths = []
                actual_field_strengths = []

                # 修正：根据你的图片数据，点数据可能有不同的字段名
                # 先调试查看数据格式
                debug_points_info = False
                if debug_points_info and points:
                    print(f"调试点数据结构: {type(points)}")
                    if isinstance(points, list) and len(points) > 0:
                        print(f"第一个点数据结构: {type(points[0])}")
                        print(f"第一个点字段: {points[0].keys() if isinstance(points[0], dict) else 'Not dict'}")

                # 处理预测点
                for point in points:
                    # 尝试各种可能的字段名获取场强值
                    if isinstance(point, dict):
                        # 修正：根据你的图片，场强值可能在 'count' 字段
                        field_strength = None
                        if 'count' in point:
                            field_strength = point.get('count')
                        elif 'field_strength' in point:
                            field_strength = point.get('field_strength')
                        elif 'value' in point:
                            field_strength = point.get('value')
                        elif 'strength' in point:
                            field_strength = point.get('strength')

                        if field_strength is not None:
                            all_field_strengths.append(field_strength)

                # 处理实际点
                for point in actual_points:
                    if isinstance(point, dict):
                        # 修正：同样的方式获取场强值
                        field_strength = None
                        if 'count' in point:
                            field_strength = point.get('count')
                        elif 'field_strength' in point:
                            field_strength = point.get('field_strength')
                        elif 'value' in point:
                            field_strength = point.get('value')
                        elif 'strength' in point:
                            field_strength = point.get('strength')

                        if field_strength is not None:
                            actual_field_strengths.append(field_strength)

                # 显示统计信息
                if all_field_strengths:
                    content_parts.append(
                        f"预测场强范围: {min(all_field_strengths):.2f} ~ {max(all_field_strengths):.2f} dBμV/m")
                    if actual_field_strengths:
                        content_parts.append(
                            f"实际场强范围: {min(actual_field_strengths):.2f} ~ {max(actual_field_strengths):.2f} dBμV/m")
                        content_parts.append(
                            f"平均场强: {sum(actual_field_strengths) / len(actual_field_strengths):.2f} dBμV/m")

                    # 存储结果摘要到元数据
                    metadata["max_field_strength"] = max(all_field_strengths)
                    metadata["min_field_strength"] = min(all_field_strengths)
                    if actual_field_strengths:
                        metadata["avg_field_strength"] = sum(actual_field_strengths) / len(actual_field_strengths)

                metadata["total_points"] = len(points)
                metadata["actual_points_count"] = len(actual_points)

                # 在文本内容中显示前5个点作为示例 - 修正：根据实际字段名显示
                if points:
                    content_parts.append(f"\n=== 所有预测点数据（共{len(points)}个）===")
                    content_parts.append("前5个点作为示例:")
                    for i, point in enumerate(points[:5]):
                        if isinstance(point, dict):
                            point_info = f"  点{i + 1}: 纬度={point.get('lat', 'N/A')}, 经度={point.get('lon', 'N/A')}"

                            # 修正：根据你的图片，可能没有 'rad' 字段
                            if point.get('rad') is not None:
                                point_info += f", 距离={point.get('rad')}km"
                            elif point.get('distance') is not None:
                                point_info += f", 距离={point.get('distance')}km"

                            # 修正：根据实际字段名显示场强值
                            field_strength = point.get('count') or point.get('field_strength') or point.get(
                                'value') or point.get('strength')
                            if field_strength is not None:
                                point_info += f", 场强={field_strength:.2f} dBμV/m"

                            # 修正：你的数据可能没有这些范围字段
                            if point.get('min_field_strength') is not None and point.get(
                                    'max_field_strength') is not None:
                                point_info += f", 场强范围=[{point.get('min_field_strength')}~{point.get('max_field_strength')}] dBμV/m"

                            if point.get('avg_field_strength') is not None:
                                point_info += f", 平均场强={point.get('avg_field_strength')} dBμV/m"

                            content_parts.append(point_info)
                        else:
                            content_parts.append(f"  点{i + 1}: {str(point)[:100]}...")

                if actual_points:
                    content_parts.append(f"\n=== 所有有效点数据（共{len(actual_points)}个）===")
                    content_parts.append("前5个点作为示例:")
                    for i, point in enumerate(actual_points[:5]):
                        if isinstance(point, dict):
                            point_info = f"  点{i + 1}: 纬度={point.get('lat', 'N/A')}, 经度={point.get('lon', 'N/A')}"

                            # 修正：根据你的图片，可能没有 'rad' 字段
                            if point.get('rad') is not None:
                                point_info += f", 距离={point.get('rad')}km"
                            elif point.get('distance') is not None:
                                point_info += f", 距离={point.get('distance')}km"

                            # 修正：根据实际字段名显示场强值
                            field_strength = point.get('count') or point.get('field_strength') or point.get(
                                'value') or point.get('strength')
                            if field_strength is not None:
                                point_info += f", 场强={field_strength:.2f} dBμV/m"

                            # 修正：你的数据可能没有这些范围字段
                            if point.get('min_field_strength') is not None and point.get(
                                    'max_field_strength') is not None:
                                point_info += f", 场强范围=[{point.get('min_field_strength')}~{point.get('max_field_strength')}] dBμV/m"

                            if point.get('avg_field_strength') is not None:
                                point_info += f", 平均场强={point.get('avg_field_strength')} dBμV/m"

                            content_parts.append(point_info)
                        else:
                            content_parts.append(f"  点{i + 1}: {str(point)[:100]}...")

                # 将完整点数据存储到元数据中（JSON格式）
                if points:
                    # 存储所有点数据
                    metadata["all_points"] = json.dumps(points, ensure_ascii=False)

                    # 存储点的统计信息（安全处理空值）
                    stats = {"total_points": len(points)}

                    # 提取有效数据
                    all_lats = []
                    all_lons = []
                    all_field_strengths = []

                    for point in points:
                        if isinstance(point, dict):
                            lat = point.get('lat')
                            lon = point.get('lon')
                            # 修正：根据实际字段名获取场强值
                            field_strength = point.get('count') or point.get('field_strength') or point.get(
                                'value') or point.get('strength')

                            if lat is not None:
                                all_lats.append(lat)
                            if lon is not None:
                                all_lons.append(lon)
                            if field_strength is not None:
                                all_field_strengths.append(field_strength)

                    if all_lats and all_lons:
                        stats["coordinates_range"] = {
                            "min_lat": min(all_lats),
                            "max_lat": max(all_lats),
                            "min_lon": min(all_lons),
                            "max_lon": max(all_lons)
                        }

                    if all_field_strengths:
                        stats["field_strength_range"] = {
                            "min_field_strength": min(all_field_strengths),
                            "max_field_strength": max(all_field_strengths)
                        }

                    metadata["points_statistics"] = json.dumps(stats, ensure_ascii=False)

                # 存储所有有效点数据
                if actual_points:
                    metadata["all_actual_points"] = json.dumps(actual_points, ensure_ascii=False)

        elif collection_type == "path_loss":
            metadata = {
                "source": "geospatial_cache",
                "collection": "path_loss_cache",
                "calculation_type": "path_loss",
                "mongodb_id": str(item.get('_id', '')),
                "data_size": item.get('data_size', 0),
                "access_count": item.get('access_count', 0),
                "created_at": str(item.get('created_at', '')),
                "last_accessed": str(item.get('last_accessed', '')),
                "params_hash": item.get('params_hash', '')
            }

            # 添加核心计算参数描述
            params = item.get('params', {})
            if params:
                content_parts.append("=== 路径损耗计算参数 ===")
                content_parts.append(f"发射台坐标: ({params.get('tx_lat', 'N/A')}, {params.get('tx_lon', 'N/A')})")
                content_parts.append(f"分析区域: 半径{params.get('radius', 'N/A')}公里圆形区域")
                content_parts.append(f"采样密度: {params.get('gap', 'N/A')}米间距")

                # 设备参数
                content_parts.append(f"工作频率: {params.get('frequency', 'N/A')}MHz")
                content_parts.append(f"发射功率: {params.get('Pt', 'N/A')}千瓦")
                content_parts.append(f"发射天线高度: {params.get('tx_antenna_height', 'N/A')}米")

                # 存储参数到元数据
                for key, value in params.items():
                    metadata[f"param_{key}"] = str(value)

            # 添加详细的计算结果，包括具体点数据
            result = item.get('result', {})
            if result:
                content_parts.append("\n=== 路径损耗计算结果 ===")

                # 获取点数据
                points = result.get('points', [])

                content_parts.append(f"预测点数: {len(points)}个")

                # 摘要信息
                if result.get('max_loss') is not None:
                    content_parts.append(f"最大路径损耗: {result.get('max_loss')} dB")
                if result.get('min_loss') is not None:
                    content_parts.append(f"最小路径损耗: {result.get('min_loss')} dB")
                if result.get('avg_loss') is not None:
                    content_parts.append(f"平均路径损耗: {result.get('avg_loss')} dB")
                if result.get('distance') is not None:
                    content_parts.append(f"总距离: {result.get('distance')} km")

                # 存储结果摘要到元数据
                if result.get('max_loss') is not None:
                    metadata["max_loss"] = result.get('max_loss')
                if result.get('min_loss') is not None:
                    metadata["min_loss"] = result.get('min_loss')
                if result.get('avg_loss') is not None:
                    metadata["avg_loss"] = result.get('avg_loss')

                metadata["total_points"] = len(points)
                if result.get('distance') is not None:
                    metadata["total_distance"] = result.get('distance')

                # 添加发射和接收点信息
                tx = result.get('tx', {})
                rx = result.get('rx', {})
                if tx:
                    tx_lat = tx.get('lat', 'N/A')
                    tx_lon = tx.get('lon', 'N/A')
                    content_parts.append(f"发射点: 纬度={tx_lat}, 经度={tx_lon}")
                    if tx_lat != 'N/A' and tx_lon != 'N/A':
                        metadata["tx_lat"] = tx_lat
                        metadata["tx_lon"] = tx_lon
                if rx:
                    rx_lat = rx.get('lat', 'N/A')
                    rx_lon = rx.get('lon', 'N/A')
                    content_parts.append(f"接收点: 纬度={rx_lat}, 经度={rx_lon}")
                    if rx_lat != 'N/A' and rx_lon != 'N/A':
                        metadata["rx_lat"] = rx_lat
                        metadata["rx_lon"] = rx_lon

                # 在文本内容中显示前5个点作为示例
                if points:
                    content_parts.append(f"\n=== 所有路径点数据（共{len(points)}个）===")
                    content_parts.append("前5个点作为示例:")
                    for i, point in enumerate(points[:5]):
                        point_info = f"  点{i + 1}: 纬度={point.get('lat', 'N/A')}, 经度={point.get('lon', 'N/A')}"
                        if point.get('loss') is not None:
                            point_info += f", 路径损耗={point.get('loss')} dB"
                        if point.get('distance') is not None:
                            point_info += f", 距离发射点={point.get('distance')} km"
                        content_parts.append(point_info)

                # 将完整点数据存储到元数据中（JSON格式）
                if points:
                    # 存储所有点数据
                    metadata["all_points"] = json.dumps(points, ensure_ascii=False)

                    # 存储点的统计信息（安全处理空值）
                    stats = {"total_points": len(points)}

                    # 提取有效数据
                    all_lats = [p.get('lat') for p in points if p.get('lat') is not None]
                    all_lons = [p.get('lon') for p in points if p.get('lon') is not None]
                    all_losses = [p.get('loss') for p in points if p.get('loss') is not None]
                    all_distances = [p.get('distance') for p in points if p.get('distance') is not None]

                    if all_lats:
                        stats["coordinates_range"] = {
                            "min_lat": min(all_lats),
                            "max_lat": max(all_lats),
                            "min_lon": min(all_lons),
                            "max_lon": max(all_lons)
                        }
                    if all_losses:
                        stats["loss_range"] = {
                            "min_loss": min(all_losses),
                            "max_loss": max(all_losses)
                        }
                    if all_distances:
                        stats["distance_range"] = {
                            "min_distance": min(all_distances),
                            "max_distance": max(all_distances)
                        }

                    metadata["points_statistics"] = json.dumps(stats, ensure_ascii=False)

        # 添加性能和使用统计
        content_parts.append("\n=== 缓存性能信息 ===")
        content_parts.append(f"缓存命中次数: {item.get('access_count', 0)}次")
        content_parts.append(f"数据大小: {item.get('data_size', 0)}字节")
        content_parts.append(f"创建时间: {item.get('created_at', 'N/A')}")
        content_parts.append(f"最后访问: {item.get('last_accessed', 'N/A')}")

        content = "\n".join(content_parts)

        # 创建文档对象 - 每个MongoDB文档作为一个独立的块
        doc = Document(
            page_content=content,
            metadata=metadata
        )

        return doc

    except Exception as e:
        print(f"❌ 处理MongoDB文档失败 ID={item.get('_id', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def build_knowledge_base():
    """构建高性能地理计算知识库索引（包含所有点数据）"""

    # 配置
    INDEX_PATH = "document_vector/faiss_index"
    DOCUMENTS_PATH = "document_vector/data"

    print("🚀 开始构建高性能地理计算知识库（包含所有点数据）...")

    # 1. 检查文档文件夹
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"❌ 文档文件夹 {DOCUMENTS_PATH} 不存在")
        print("正在创建文档文件夹...")
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        print(f"✅ 已创建文件夹 {DOCUMENTS_PATH}")
        print("请将您的技术文档放入此文件夹后重新运行")
        return False

    # 2. 初始化组件
    print("📦 初始化嵌入模型和数据管理器...")
    embedder = BGEEmbedder()
    vectorstore = FaissVectorStore(embedder, INDEX_PATH)
    data_manager = MongoDBDataManager()

    # 3. 加载技术文档（原有逻辑保持不变）
    # 3. 加载技术文档（使用智能分块）
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # 默认分块大小
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    print(f"📄 扫描技术文档文件夹: {DOCUMENTS_PATH}")

    # 遍历文档文件夹
    file_count = 0
    for file_path in Path(DOCUMENTS_PATH).rglob("*"):
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == '.txt':
                    # 使用新的处理函数
                    chunks = process_text_document_with_id(file_path, text_splitter)
                    documents.extend(chunks)
                    file_count += 1
                    print(f"✅ 已加载技术文档: {file_path.name} ({len(chunks)} 个片段)")

                elif file_path.suffix.lower() == '.pdf':
                    # 使用智能PDF分块
                    chunks = smart_pdf_chunking(file_path, text_splitter)
                    documents.extend(chunks)
                    file_count += 1
                    print(f"✅ 已加载技术文档: {file_path.name} ({len(chunks)} 个片段)")

                    # 显示分块统计
                    if chunks:
                        avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
                        print(f"  平均分块大小: {avg_size:.0f} 字符")
                        # 显示章节信息
                        for i, chunk in enumerate(chunks[:3]):  # 显示前3个块的标题
                            section = chunk.metadata.get('section_title', '无标题')
                            print(f"  块{i + 1}: {section[:30]}...")

            except Exception as e:
                print(f"❌ 加载失败 {file_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()

    # 4. 添加文档去重逻辑（在构建索引前）
    print("🔄 对技术文档进行去重处理...")

    # 基于chunk_id去重
    unique_documents = []
    seen_chunk_ids = set()
    duplicate_count = 0

    for doc in documents:
        chunk_id = doc.metadata.get('chunk_id', '')
        if chunk_id and chunk_id in seen_chunk_ids:
            duplicate_count += 1
            continue

        seen_chunk_ids.add(chunk_id)
        unique_documents.append(doc)

    documents = unique_documents
    print(f"✅ 去重完成: 移除了 {duplicate_count} 个重复片段")
    print(f"✅ 剩余文档数量: {len(documents)} 个")

    # 按文档来源统计
    doc_stats = {}
    for doc in documents:
        filename = doc.metadata.get('filename', 'unknown')
        doc_stats[filename] = doc_stats.get(filename, 0) + 1

    print("📊 文档分布统计:")
    for filename, count in doc_stats.items():
        print(f"  - {filename}: {count} 个片段")

    # 5. 处理高性能地理计算缓存数据 - 按文档分块
    print("🔧 处理高性能地理计算缓存数据（每个文档包含所有点数据）...")
    field_strength_count = 0
    path_loss_count = 0
    total_points_count = 0

    try:
        # 获取集合统计信息
        stats = data_manager.get_collection_stats()
        print(f"📊 缓存数据统计: 场强计算={stats.get('collection_a_count', 0)}条, "
              f"路径损耗={stats.get('collection_b_count', 0)}条")

        # 获取场强计算缓存数据
        field_strength_data = data_manager.get_collection_a_data()

        # 处理场强计算缓存数据 - 每个文档作为一个块
        for item in field_strength_data:
            doc = process_mongodb_document_as_chunk(item, "field_strength")
            if doc:
                documents.append(doc)
                field_strength_count += 1

                # 统计点数量
                total_points = doc.metadata.get('total_points', 0)
                total_points_count += total_points

                # 每处理5个文档打印一次进度
                if field_strength_count % 5 == 0:
                    print(f"✅ 已处理 {field_strength_count} 个场强计算文档（当前文档{total_points}个点）")

        print(f"🎯 场强计算文档处理完成: {field_strength_count} 个文档")

        # 获取路径损耗缓存数据
        path_loss_data = data_manager.get_collection_b_data()

        # 处理路径损耗缓存数据 - 每个文档作为一个块
        for item in path_loss_data:
            doc = process_mongodb_document_as_chunk(item, "path_loss")
            if doc:
                documents.append(doc)
                path_loss_count += 1

                # 统计点数量
                total_points = doc.metadata.get('total_points', 0)
                total_points_count += total_points

                # 每处理5个文档打印一次进度
                if path_loss_count % 5 == 0:
                    print(f"✅ 已处理 {path_loss_count} 个路径损耗文档（当前文档{total_points}个点）")

        print(f"🎯 路径损耗文档处理完成: {path_loss_count} 个文档")

    except Exception as e:
        print(f"❌ 加载地理计算缓存数据失败: {str(e)}")
        import traceback
        traceback.print_exc()

    if not documents:
        print("❌ 没有找到可用的技术文档和地理计算数据")
        return False
    print("🔄 最终检查文档多样性...")

    # 6.检查是否有大量相似片段
    similar_groups = {}
    for doc in documents:
        content = doc.page_content
        # 取前100字符作为分组依据
        key = content[:100]
        if key not in similar_groups:
            similar_groups[key] = []
        similar_groups[key].append(doc)

    # 报告相似组
    large_groups = {k: v for k, v in similar_groups.items() if len(v) > 2}
    if large_groups:
        print(f"⚠️ 发现 {len(large_groups)} 个相似内容组（每组>2个片段）:")
        for key, docs in list(large_groups.items())[:3]:  # 显示前3组
            print(f"  相似组: {len(docs)} 个片段，示例: {docs[0].page_content[:50]}...")

    # 7. 构建高性能向量索引
    try:
        print("🔄 正在构建高性能地理计算向量索引（包含所有点数据）...")
        print(f"📊 总文档数量: {len(documents)} 个")
        print(f"  - 技术文档片段: {len(documents) - field_strength_count - path_loss_count} 个")
        print(f"  - 场强计算文档: {field_strength_count} 个")
        print(f"  - 路径损耗文档: {path_loss_count} 个")
        print(f"  - 总数据点数: {total_points_count} 个点")

        vectorstore.build_index(documents)
        vectorstore.save()
        print(f"🎉 高性能地理计算知识库构建完成！")
        print(f"📁 索引已保存到: {INDEX_PATH}")
        if hasattr(vectorstore, 'get_index_info'):
            print(f"📈 索引信息: {vectorstore.get_index_info()}")

        # 显示详细统计
        if hasattr(vectorstore, 'get_cache_statistics'):
            cache_stats = vectorstore.get_cache_statistics()
            print(f"📊 缓存数据统计:")
            print(f"  - 场强计算缓存: {cache_stats.get('field_strength_cache_count', 0)} 条")
            print(f"  - 路径损耗缓存: {cache_stats.get('path_loss_cache_count', 0)} 条")
            print(f"  - 总访问次数: {cache_stats.get('total_access_count', 0)} 次")
            print(f"  - 总数据量: {cache_stats.get('total_data_size_mb', 0)} MB")

        print(f"  - 总数据点数: {total_points_count} 个点")

        # 检查点数据是否完整存储
        print("\n📋 点数据完整性检查:")
        field_strength_docs = [d for d in documents if d.metadata.get('calculation_type') == 'field_strength']
        path_loss_docs = [d for d in documents if d.metadata.get('calculation_type') == 'path_loss']

        for i, doc in enumerate(field_strength_docs[:3]):  # 检查前3个场强文档
            has_all_points = 'all_points' in doc.metadata
            points_count = doc.metadata.get('total_points', 0)
            print(f"  场强文档{i + 1}: {points_count}个点，完整存储: {has_all_points}")

        for i, doc in enumerate(path_loss_docs[:3]):  # 检查前3个路径损耗文档
            has_all_points = 'all_points' in doc.metadata
            points_count = doc.metadata.get('total_points', 0)
            print(f"  路径损耗文档{i + 1}: {points_count}个点，完整存储: {has_all_points}")

    except Exception as e:
        print(f"❌ 构建索引失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n📊 总体统计信息:")
    print(f"  - 处理技术文档: {file_count} 个")
    print(f"  - 总文档数量: {len(documents)} 个")
    print(f"  - 场强计算文档: {field_strength_count} 个")
    print(f"  - 路径损耗文档: {path_loss_count} 个")
    print(f"  - 总数据点数: {total_points_count} 个点")

    return True


def test_geospatial_knowledge_base():
    """测试地理计算知识库（检查点数据完整性）"""
    try:
        print("🧪 测试高性能地理计算知识库（检查点数据完整性）...")

        # 检查索引文件
        index_file = os.path.join("document_vector/faiss_index", "index.faiss")
        pkl_file = os.path.join("document_vector/faiss_index", "index.pkl")

        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            print("❌ 索引文件不存在")
            return False

        embedder = BGEEmbedder()
        vectorstore = FaissVectorStore(embedder, "document_vector/faiss_index")
        vectorstore.load()

        # 测试查询 - 地理计算相关
        test_queries = [
            "场强计算",
            "路径损耗",
            "点数据"
        ]

        for test_query in test_queries:
            print(f"\n🔍 测试查询: '{test_query}'")
            results = vectorstore.improved_retrieve(test_query, top_k=2)

            for i, (doc, score) in enumerate(results):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                source = doc.metadata.get('source', 'unknown')
                calc_type = doc.metadata.get('calculation_type', 'unknown')
                total_points = doc.metadata.get('total_points', 0)

                # 检查点数据完整性
                has_all_points = 'all_points' in doc.metadata

                print(f"  {i + 1}. 相似度: {score:.3f}, 来源: {source}")
                print(f"     类型: {calc_type}, 总点数: {total_points}个，完整存储: {has_all_points}")
                print(f"     内容预览: {content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ 知识库测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = build_knowledge_base()
    if success:
        test_geospatial_knowledge_base()

    print("=" * 50)
    print("🏁 高性能地理计算知识库构建完成！")
    print("所有点数据已完整存储在向量数据库中。")
    print("现在可以运行主应用程序进行智能地理计算查询。")