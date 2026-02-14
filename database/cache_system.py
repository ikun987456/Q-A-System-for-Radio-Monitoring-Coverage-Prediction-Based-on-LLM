"""
计算结果缓存系统
"""
import pymongo
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class CalculationCache:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """初始化MongoDB缓存系统"""
        try:
            self.client = pymongo.MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000
            )
            # 测试连接
            self.client.admin.command('ping')

            self.db = self.client['radio_monitoring']
            self.path_loss_cache = self.db['path_loss_cache']
            self.field_strength_cache = self.db['field_strength_cache']

            self._create_indexes()
            self.connected = True

        except Exception as e:
            print(f"❌ MongoDB连接失败: {e}")
            self.connected = False
            self.client = None

    def _create_indexes(self):
        """创建数据库索引"""
        try:
            # 参数哈希唯一索引
            self.path_loss_cache.create_index("params_hash", unique=True, background=True)
            self.field_strength_cache.create_index("params_hash", unique=True, background=True)

            # 时间索引
            self.path_loss_cache.create_index("created_at", background=True)
            self.field_strength_cache.create_index("created_at", background=True)

        except Exception as e:
            print(f"⚠️ 创建索引失败: {e}")

    def is_available(self) -> bool:
        """检查缓存系统是否可用"""
        return self.connected and self.client is not None

    def get_params_hash(self, params: Dict[str, Any]) -> str:
        """生成参数哈希值"""

        def normalize_params(obj):
            if isinstance(obj, dict):
                return {k: normalize_params(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, float):
                return round(obj, 6)  # 保留6位小数避免精度问题
            elif isinstance(obj, list):
                return [normalize_params(item) for item in obj]
            else:
                return obj

        normalized = normalize_params(params)
        params_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(params_str.encode('utf-8')).hexdigest()

    # 路径损耗缓存
    def cache_path_loss_result(self, params: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """缓存路径损耗计算结果"""
        if not self.is_available():
            return False

        try:
            params_hash = self.get_params_hash(params)

            cache_doc = {
                "params_hash": params_hash,
                "params": params,
                "result": result,
                "calculation_type": "path_loss",
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 1
            }

            self.path_loss_cache.replace_one(
                {"params_hash": params_hash},
                cache_doc,
                upsert=True
            )

            print(f"✅ 路径损耗结果已缓存: {params_hash[:8]}...")
            return True

        except Exception as e:
            print(f"❌ 缓存路径损耗结果失败: {e}")
            return False

    def get_cached_path_loss(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存的路径损耗结果"""
        if not self.is_available():
            return None

        try:
            params_hash = self.get_params_hash(params)
            cached = self.path_loss_cache.find_one({"params_hash": params_hash})

            if cached:
                # 更新访问统计
                self.path_loss_cache.update_one(
                    {"params_hash": params_hash},
                    {
                        "$inc": {"access_count": 1},
                        "$set": {"last_accessed": datetime.now()}
                    }
                )
                return cached["result"]

            return None

        except Exception as e:
            print(f"❌ 获取路径损耗缓存失败: {e}")
            return None

    # 区域场强缓存
    def cache_field_strength_result(self, params: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """缓存区域场强计算结果"""
        if not self.is_available():
            return False

        try:
            params_hash = self.get_params_hash(params)

            cache_doc = {
                "params_hash": params_hash,
                "params": params,
                "result": result,
                "calculation_type": "field_strength",
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 1,
                "data_size": len(str(result))
            }

            self.field_strength_cache.replace_one(
                {"params_hash": params_hash},
                cache_doc,
                upsert=True
            )

            return True

        except Exception as e:
            print(f"❌ 缓存区域场强结果失败: {e}")
            return False

    def get_cached_field_strength(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存的区域场强结果"""
        if not self.is_available():
            return None

        try:
            params_hash = self.get_params_hash(params)
            cached = self.field_strength_cache.find_one({"params_hash": params_hash})

            if cached:
                self.field_strength_cache.update_one(
                    {"params_hash": params_hash},
                    {
                        "$inc": {"access_count": 1},
                        "$set": {"last_accessed": datetime.now()}
                    }
                )
                return cached["result"]

            return None

        except Exception as e:
            print(f"❌ 获取区域场强缓存失败: {e}")
            return None

    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.is_available():
            return {
                "path_loss_cached": 0,
                "field_strength_cached": 0,
                "path_loss_access_total": 0,
                "field_strength_access_total": 0,
                "status": "disconnected"
            }

        try:
            stats = {
                "path_loss_cached": self.path_loss_cache.count_documents({}),
                "field_strength_cached": self.field_strength_cache.count_documents({}),
                "status": "connected"
            }

            # 总访问次数
            path_access = list(self.path_loss_cache.aggregate([
                {"$group": {"_id": None, "total": {"$sum": "$access_count"}}}
            ]))

            field_access = list(self.field_strength_cache.aggregate([
                {"$group": {"_id": None, "total": {"$sum": "$access_count"}}}
            ]))

            stats["path_loss_access_total"] = path_access[0]["total"] if path_access else 0
            stats["field_strength_access_total"] = field_access[0]["total"] if field_access else 0

            return stats

        except Exception as e:
            return {"error": str(e)}

    def clear_old_cache(self, days: int = 30) -> Dict[str, int]:
        """清理旧缓存"""
        if not self.is_available():
            return {"path_loss_deleted": 0, "field_strength_deleted": 0}

        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            path_deleted = self.path_loss_cache.delete_many({
                "created_at": {"$lt": cutoff_date},
                "access_count": {"$lt": 5}
            })

            field_deleted = self.field_strength_cache.delete_many({
                "created_at": {"$lt": cutoff_date},
                "access_count": {"$lt": 3}
            })

            return {
                "path_loss_deleted": path_deleted.deleted_count,
                "field_strength_deleted": field_deleted.deleted_count
            }

        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")
            return {"path_loss_deleted": 0, "field_strength_deleted": 0}
