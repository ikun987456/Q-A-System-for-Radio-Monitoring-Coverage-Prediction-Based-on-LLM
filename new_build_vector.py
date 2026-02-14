import hashlib
import pickle
from datetime import datetime
import os
import sys
import json
from pathlib import Path
from document_vector.faiss_vectorstore import FaissVectorStore, BGEEmbedder, MongoDBDataManager
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from build_vector_first import process_mongodb_document_as_chunk
from langchain_core.documents import Document
import streamlit as st
def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


def get_document_hash(document):
    """计算文档的哈希值"""
    content = document.page_content if hasattr(document, 'page_content') else str(document)
    metadata_str = json.dumps(document.metadata, sort_keys=True) if hasattr(document, 'metadata') else ""
    combined = content + metadata_str
    return hashlib.md5(combined.encode()).hexdigest()


def load_existing_hashes(index_path="document_vector/faiss_index"):
    """加载现有的文档哈希值"""
    hashes_file = os.path.join(index_path, "document_hashes.pkl")
    if os.path.exists(hashes_file):
        try:
            with open(hashes_file, "rb") as f:
                return pickle.load(f)
        except:
            return set()
    return set()


def save_hashes(hashes, index_path="document_vector/faiss_index"):
    """保存文档哈希值"""
    hashes_file = os.path.join(index_path, "document_hashes.pkl")
    with open(hashes_file, "wb") as f:
        pickle.dump(hashes, f)


def get_mongodb_document_hash(item):
    """计算MongoDB文档的哈希值"""
    try:
        # 使用关键字段计算哈希
        key_data = {
            'mongodb_id': str(item.get('_id', '')),
            'params_hash': item.get('params_hash', ''),
            'created_at': str(item.get('created_at', '')),
            'data_size': item.get('data_size', 0)
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    except:
        return None


def incremental_update_knowledge_base():
    """
    增量更新知识库：只添加新的文档
    返回: (是否更新成功, 新增文档数量)
    """
    INDEX_PATH = "document_vector/faiss_index"
    DOCUMENTS_PATH = "document_vector/data"

    print("🔄 检查知识库增量更新...")

    # 1. 检查索引是否存在
    index_file = os.path.join(INDEX_PATH, "index.faiss")
    pkl_file = os.path.join(INDEX_PATH, "index.pkl")

    # 如果索引不存在，需要完整构建
    if not os.path.exists(index_file) or not os.path.exists(pkl_file):
        print("❌ 索引文件不存在，需要进行完整构建")
        return False, 0

    # 2. 初始化组件
    embedder = BGEEmbedder()
    vectorstore = FaissVectorStore(embedder, INDEX_PATH)

    try:
        vectorstore.load()
    except:
        print("❌ 无法加载现有索引，需要重新构建")
        return False, 0

    # 3. 加载现有的文档哈希
    existing_hashes = load_existing_hashes(INDEX_PATH)
    print(f"📊 现有文档数量: {len(existing_hashes)}")

    # 4. 收集新的文档
    new_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    # 5. 检查新的文本文件
    print(f"📄 检查新的文本文件: {DOCUMENTS_PATH}")
    file_updates = 0

    if os.path.exists(DOCUMENTS_PATH):
        for file_path in Path(DOCUMENTS_PATH).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf']:
                file_hash = get_file_hash(str(file_path))
                if file_hash and file_hash not in existing_hashes:
                    try:
                        if file_path.suffix.lower() == '.txt':
                            loader = TextLoader(str(file_path), encoding='utf-8')
                            docs = loader.load()
                            chunks = text_splitter.split_documents(docs)
                            new_documents.extend(chunks)
                            file_updates += 1
                            print(f"✅ 发现新文本文件: {file_path.name} ({len(chunks)} 个片段)")

                        elif file_path.suffix.lower() == '.pdf':
                            loader = PyPDFLoader(str(file_path))
                            docs = loader.load()
                            chunks = text_splitter.split_documents(docs)
                            new_documents.extend(chunks)
                            file_updates += 1
                            print(f"✅ 发现新PDF文件: {file_path.name} ({len(chunks)} 个片段)")

                        # 添加文件哈希到现有集合
                        existing_hashes.add(file_hash)

                    except Exception as e:
                        print(f"❌ 处理文件失败 {file_path.name}: {str(e)}")

    # 6. 检查新的MongoDB数据
    print("🔍 检查新的MongoDB数据...")
    mongo_updates = 0

    try:
        data_manager = MongoDBDataManager()

        # 检查场强计算数据
        field_strength_data = data_manager.get_collection_a_data()
        for item in field_strength_data:
            item_hash = get_mongodb_document_hash(item)
            if item_hash and item_hash not in existing_hashes:
                doc = process_mongodb_document_as_chunk(item, "field_strength")
                if doc:
                    new_documents.append(doc)
                    existing_hashes.add(item_hash)
                    mongo_updates += 1
                    if mongo_updates % 5 == 0:
                        print(f"✅ 发现新的场强计算数据: {mongo_updates} 条")

        # 检查路径损耗数据
        path_loss_data = data_manager.get_collection_b_data()
        for item in path_loss_data:
            item_hash = get_mongodb_document_hash(item)
            if item_hash and item_hash not in existing_hashes:
                doc = process_mongodb_document_as_chunk(item, "path_loss")
                if doc:
                    new_documents.append(doc)
                    existing_hashes.add(item_hash)
                    mongo_updates += 1
                    if mongo_updates % 5 == 0:
                        print(f"✅ 发现新的路径损耗数据: {mongo_updates} 条")

    except Exception as e:
        print(f"❌ 检查MongoDB数据失败: {str(e)}")

    # 7. 如果有新文档，更新索引
    if new_documents:
        print(f"🔄 发现 {len(new_documents)} 个新文档，正在更新索引...")

        try:
            # 添加新文档到现有索引
            if hasattr(vectorstore, 'add_documents'):
                vectorstore.add_documents(new_documents)
            else:
                # 如果add_documents不存在，重新构建整个索引
                print("⚠️ 当前向量库不支持增量添加，重新构建索引...")
                all_documents = vectorstore.get_all_documents() if hasattr(vectorstore, 'get_all_documents') else []
                all_documents.extend(new_documents)
                vectorstore.build_index(all_documents)

            vectorstore.save()

            # 保存更新后的哈希值
            save_hashes(existing_hashes, INDEX_PATH)

            print(f"🎉 知识库增量更新完成！新增 {len(new_documents)} 个文档")
            print(f"   - 新文本文件: {file_updates} 个")
            print(f"   - 新MongoDB数据: {mongo_updates} 条")
            print(f"   - 总文档数量: {len(existing_hashes)} 个")

            return True, len(new_documents)

        except Exception as e:
            print(f"❌ 索引更新失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, 0
    else:
        print("✅ 没有发现新的文档，知识库已是最新")
        return True, 0


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