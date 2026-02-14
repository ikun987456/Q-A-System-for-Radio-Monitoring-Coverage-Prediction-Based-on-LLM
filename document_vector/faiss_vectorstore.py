import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import faiss
import numpy as np
import pickle
from FlagEmbedding import BGEM3FlagModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
import requests
from pymongo import MongoClient
from bson.binary import Binary
import uuid
from datetime import datetime

# 知识库配置
INDEX_PATH = "faiss_index"


class MongoDBDataManager:
    def __init__(self, connection_string=None, db_name="radio_monitoring"):
        self.connection_string = connection_string or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(self.connection_string)
        self.db = self.client[db_name]

        # 直接使用集合A和B，不预设结构
        self.collection_a = self.db["field_strength_cache"]  # 集合A
        self.collection_b = self.db["path_loss_cache"]  # 集合B

    def get_collection_a_data(self, query_filter=None, limit=0):
        """获取集合A的所有数据"""
        try:
            if query_filter:
                data = list(self.collection_a.find(query_filter).limit(limit))
            else:
                data = list(self.collection_a.find())
            print(f"✅ 从集合field_strength_cache加载了 {len(data)} 条数据")
            return data
        except Exception as e:
            print(f"❌ 从集合field_strength_cache加载数据失败: {str(e)}")
            return []

    def get_collection_b_data(self, query_filter=None, limit=0):
        """获取集合B的所有数据"""
        try:
            if query_filter:
                data = list(self.collection_b.find(query_filter).limit(limit))
            else:
                data = list(self.collection_b.find())
            print(f"✅ 从集合path_loss_cache 加载了 {len(data)} 条数据")
            return data
        except Exception as e:
            print(f"❌ 从集合path_loss_cache 加载数据失败: {str(e)}")
            return []

    def search_collection_a(self, field, value, exact_match=False):
        """搜索集合A数据"""
        try:
            if exact_match:
                search_filter = {field: value}
            else:
                search_filter = {field: {"$regex": value, "$options": "i"}}

            results = list(self.collection_a.find(search_filter))
            return results
        except Exception as e:
            print(f"❌ 集合field_strength_cache搜索失败: {str(e)}")
            return []

    def search_collection_b(self, field, value, exact_match=False):
        """搜索集合B数据"""
        try:
            if exact_match:
                search_filter = {field: value}
            else:
                search_filter = {field: {"$regex": value, "$options": "i"}}

            results = list(self.collection_b.find(search_filter))
            return results
        except Exception as e:
            print(f"❌ 集合path_loss_cache搜索失败: {str(e)}")
            return []

    def get_collection_stats(self):
        """获取集合统计信息"""
        try:
            count_a = self.collection_a.count_documents({})
            count_b = self.collection_b.count_documents({})

            # 获取集合A的字段示例
            sample_a = self.collection_a.find_one()
            fields_a = list(sample_a.keys()) if sample_a else []

            # 获取集合B的字段示例
            sample_b = self.collection_b.find_one()
            fields_b = list(sample_b.keys()) if sample_b else []

            return {
                "collection_a_count": count_a,
                "collection_b_count": count_b,
                "collection_a_fields": fields_a,
                "collection_b_fields": fields_b
            }
        except Exception as e:
            print(f"❌ 获取集合统计失败: {str(e)}")
            return {}

    def insert_to_collection_a(self, data):
        """向集合A插入数据"""
        try:
            result = self.collection_a.insert_one(data)
            return result.inserted_id
        except Exception as e:
            print(f"❌ 向集合field_strength_cache插入数据失败: {str(e)}")
            return None

    def insert_to_collection_b(self, data):
        """向集合B插入数据"""
        try:
            result = self.collection_b.insert_one(data)
            return result.inserted_id
        except Exception as e:
            print(f"❌ 向集合path_loss_cache插入数据失败: {str(e)}")
            return None


class FaissVectorStore:
    def __init__(self, embedder, index_path="faiss_index"):
        self.embedder = embedder
        self.index_path = index_path
        self.index = None
        self.documents = []
        # 初始化数据管理器
        self.data_manager = MongoDBDataManager()

    def build_index(self, documents):
        """构建新的索引（覆盖现有）- 使用余弦相似度"""
        if not documents:
            raise ValueError("没有文档可用于构建索引")

        texts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        ]
        embeddings = self.embedder.embed_documents(texts)
        embeddings = np.array(embeddings).astype("float32")

        # 归一化以使用余弦相似度
        faiss.normalize_L2(embeddings)

        # 使用内积索引（归一化后内积=余弦相似度）
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.documents = documents
        self.index.add(embeddings)

    def add_documents(self, documents):
        """添加文档到现有索引（支持增量添加）- 使用余弦相似度"""
        texts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        ]
        embeddings = self.embedder.embed_documents(texts)
        embeddings = np.array(embeddings).astype("float32")

        # 归一化
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.documents = []

        self.index.add(embeddings)
        self.documents.extend(documents)

    def improved_retrieve(self, query: str, top_k: int = 5):
        """
        改进的检索方法，使用余弦相似度
        """
        query_embedding = np.array(self.embedder.embed_query(query)).astype("float32")
        # 归一化查询向量
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # 使用内积搜索（余弦相似度）
        similarities, indices = self.index.search(np.array([query_embedding]), k=top_k)

        results = []
        for idx, score in zip(indices[0], similarities[0]):
            if idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append((doc, float(score)))  # score是余弦相似度，范围[-1,1]

        return results

    def save(self):
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        with open(os.path.join(self.index_path, "index.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self):
        """加载索引和文档"""
        index_file = os.path.join(self.index_path, "index.faiss")
        pkl_file = os.path.join(self.index_path, "index.pkl")

        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")

        self.index = faiss.read_index(index_file)
        with open(pkl_file, "rb") as f:
            self.documents = pickle.load(f)

    def get_index_info(self):
        """获取索引信息"""
        if self.index is None:
            return "索引未加载"
        return f"文档数量: {len(self.documents)}, 向量维度: {self.index.d}, 索引大小: {self.index.ntotal}"

    def search(self, query, k=5):
        query_embedding = np.array(self.embedder.embed_documents([query])).astype("float32")
        D, I = self.index.search(query_embedding, k)
        return [self.documents[i] for i in I[0]]



    def geospatial_retrieve(self, query: str, top_k: int = 5):
        """
        专门检索地理计算数据
        """
        vector_results = self.improved_retrieve(query, top_k=top_k )  # 多检索一些再过滤

        geo_results = []
        for doc, score in vector_results:
            if hasattr(doc, 'metadata') and doc.metadata.get('source') == 'geospatial_cache':
                geo_results.append((doc, score))

        return geo_results[:top_k]

    def field_strength_retrieve(self, query: str, top_k: int = 5):
        """
        专门检索场强计算数据
        """
        vector_results = self.improved_retrieve(query, top_k=top_k )

        field_results = []
        for doc, score in vector_results:
            if (hasattr(doc, 'metadata') and
                    doc.metadata.get('calculation_type') == 'field_strength'):
                field_results.append((doc, score))

        return field_results[:top_k]

    def path_loss_retrieve(self, query: str, top_k: int = 5):
        """
        专门检索路径损耗计算数据
        """
        vector_results = self.improved_retrieve(query, top_k=top_k )

        path_results = []
        for doc, score in vector_results:
            if (hasattr(doc, 'metadata') and
                    doc.metadata.get('calculation_type') == 'path_loss'):
                path_results.append((doc, score))

        return path_results[:top_k]

    def mongodb_retrieve(self, query: str, top_k: int = 5):
        """
        专门检索MongoDB数据
        """
        # 使用向量检索
        vector_results = self.improved_retrieve(query, top_k=top_k)

        # 过滤出MongoDB数据
        mongo_results = []
        for doc, score in vector_results:
            if hasattr(doc, 'metadata') and doc.metadata.get('source', '').startswith('mongodb_'):
                mongo_results.append((doc, score))

        return mongo_results

    def collection_a_retrieve(self, query: str, top_k: int = 3):
        """
        专门检索集合A数据
        """
        vector_results = self.improved_retrieve(query, top_k=top_k)

        collection_a_results = []
        for doc, score in vector_results:
            if hasattr(doc, 'metadata') and doc.metadata.get('collection') == 'field_strength_cache':
                collection_a_results.append((doc, score))

        return collection_a_results

    def collection_b_retrieve(self, query: str, top_k: int = 3):
        """
        专门检索集合B数据
        """
        vector_results = self.improved_retrieve(query, top_k=top_k)

        collection_b_results = []
        for doc, score in vector_results:
            if hasattr(doc, 'metadata') and doc.metadata.get('collection') == 'path_loss_cache':
                collection_b_results.append((doc, score))

        return collection_b_results

    def data_retrieve(self, query: str, top_k: int = 3, mongo_top_k: int = 2):
        """
        数据检索：同时检索文档和MongoDB数据
        """
        # 检索文本文档
        text_results = self.improved_retrieve(query, top_k=top_k)

        # 检索相关MongoDB数据
        mongo_results = self.mongodb_retrieve(query, top_k=mongo_top_k)

        return {
            "text_docs": text_results,
            "mongo_docs": mongo_results
        }

    def find_similar_calculations(self, params: dict, top_k: int = 3):
        """
        查找相似的计算参数配置
        """
        # 构建参数描述查询
        query_parts = []
        if 'frequency' in params:
            query_parts.append(f"{params['frequency']}MHz")
        if 'tx_lat' in params and 'tx_lon' in params:
            query_parts.append(f"坐标 {params['tx_lat']} {params['tx_lon']}")
        if 'radius' in params:
            query_parts.append(f"半径 {params['radius']}公里")

        query = " ".join(query_parts)
        return self.geospatial_retrieve(query, top_k=top_k)

    def get_cache_statistics(self):
        """获取地理计算缓存统计信息"""
        if not self.documents:
            return {"total_documents": 0}

        field_strength_count = 0
        path_loss_count = 0
        total_access_count = 0
        total_data_size = 0

        for doc in self.documents:
            if hasattr(doc, 'metadata'):
                if doc.metadata.get('calculation_type') == 'field_strength':
                    field_strength_count += 1
                elif doc.metadata.get('calculation_type') == 'path_loss':
                    path_loss_count += 1

                total_access_count += doc.metadata.get('access_count', 0)
                total_data_size += doc.metadata.get('data_size', 0)

        return {
            "total_documents": len(self.documents),
            "field_strength_cache_count": field_strength_count,
            "path_loss_cache_count": path_loss_count,
            "total_access_count": total_access_count,
            "total_data_size_bytes": total_data_size,
            "total_data_size_mb": round(total_data_size / (1024 * 1024), 2)
        }

    def add_mongodb_data_to_knowledge(self, collection_name, data):
        """添加MongoDB数据到知识库"""
        try:
            # 存储到MongoDB
            if collection_name == "field_strength_cache":
                inserted_id = self.data_manager.insert_to_collection_a(data)
            elif collection_name == "path_loss_cache":
                inserted_id = self.data_manager.insert_to_collection_b(data)
            else:
                print(f"❌ 不支持的集合名称: {collection_name}")
                return False

            if inserted_id:
                # 创建文档内容
                content_parts = []
                metadata = {
                    "source": f"mongodb_collection_{collection_name.lower()}",
                    "collection": collection_name,
                    "mongodb_id": str(inserted_id)
                }

                # 动态添加所有字段
                for key, value in data.items():
                    content_parts.append(f"{key}: {value}")
                    metadata[key] = value

                content = "\n".join(content_parts)

                # 创建文档对象
                from langchain_core.documents import Document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )

                # 添加到向量索引
                self.add_documents([doc])
                return True

            return False

        except Exception as e:
            print(f"❌ 添加MongoDB数据到知识库失败: {str(e)}")
            return False

class BGEEmbedder:
    def __init__(self):
        self.model = BGEM3FlagModel('document_vector/bge-m3', use_fp16=True)

    def embed_documents(self, texts: list):
        results = self.model.encode(texts, batch_size=32, max_length=512, return_dense=True)
        if isinstance(results, dict) and "dense_vecs" in results:
            return results["dense_vecs"].tolist()
        else:
            raise ValueError(f"Unexpected embedding result format: {type(results)}")

    def embed_query(self, text: str):
        """嵌入单个查询（优化处理）"""
        if not text:
            return [0.0] * 1024

        results = self.model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        if isinstance(results, dict) and "dense_vecs" in results:
            return results["dense_vecs"][0].tolist()
        raise ValueError(f"Unexpected embedding result format: {type(results)}")


class DeepSeekLLM:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]


class RAGChain(Runnable):
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.data_prompt_template = PromptTemplate.from_template(
            """你是一个专业的无线电监测技术专家。基于以下参考文档和数据库数据，请详细回答用户的问题。

参考文档：
{text_context}

数据库数据：
{mongo_context}

用户问题：{question}

请根据参考文档和数据库数据提供专业、准确的回答。重点关注数值数据、设备参数、监测结果等信息。

如果文档和数据库数据中没有相关信息，请说明并提供你的专业建议。

请按照以下结构组织回答：
**要求**：
        1. **简洁明了**：直接回答问题，不超过150字
        2. **重点突出**：只说核心要点，不要详细展开
        3. **结构简单**：用段落或简单列表，不要复杂格式
        4. **专业准确**：确保信息准确，术语正确
"""
        )

    def invoke(self, input):
        query = input["question"]
        temperature = input.get("temperature", 0.7)

        # 使用数据检索
        data_results = self.retriever.data_retrieve(
            query,
            top_k=input.get("top_k", 3),
            mongo_top_k=input.get("mongo_top_k", 2)
        )

        # 构建文本上下文
        text_context_parts = []
        for doc, score in data_results["text_docs"]:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            text_context_parts.append(f"[来源: {source}, 相似度: {score:.3f}] {content}")

        text_context = "\n".join(text_context_parts) if text_context_parts else "暂无相关文档"

        # 构建MongoDB数据上下文
        mongo_context_parts = []
        for doc, score in data_results["mongo_docs"]:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            collection = doc.metadata.get('collection', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            mongo_context_parts.append(f"[集合: {collection}, 相似度: {score:.3f}] {content}")

        mongo_context = "\n".join(mongo_context_parts) if mongo_context_parts else "暂无相关数据库数据"

        # 生成回答
        prompt_value = self.data_prompt_template.invoke({
            "text_context": text_context,
            "mongo_context": mongo_context,
            "question": query
        })

        return self.llm.generate(prompt_value.to_string())

    def data_invoke(self, input):
        """数据调用，返回详细结果"""
        query = input["question"]

        # 数据检索
        data_results = self.retriever.data_retrieve(
            query,
            top_k=input.get("top_k", 3),
            mongo_top_k=input.get("mongo_top_k", 2)
        )

        # 生成回答
        response = self.invoke(input)

        return {
            "answer": response,
            "text_docs": data_results["text_docs"],
            "mongo_docs": data_results["mongo_docs"]
        }