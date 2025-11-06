"""
缓存管理器 - Phase 3.3
支持两级缓存：内存（LRU）+ 磁盘（SQLite）
"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from cachetools import LRUCache
import threading


class CacheManager:
    """智能缓存管理器"""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        memory_size: int = 100,      # 内存缓存最多100个
        db_size_mb: int = 1000,      # 磁盘缓存最大1GB
        ttl_days: int = 7            # 缓存有效期7天
    ):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            memory_size: 内存缓存容量
            db_size_mb: 磁盘缓存大小限制（MB）
            ttl_days: 缓存有效期（天）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "cache.db"
        self.ttl_seconds = ttl_days * 86400
        self.max_db_size = db_size_mb * 1024 * 1024
        
        # 内存缓存（LRU）
        self.memory_cache = LRUCache(maxsize=memory_size)
        self.cache_lock = threading.Lock()
        
        # 初始化数据库
        self._init_database()
        
        # 统计信息
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    def _init_database(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                file_name TEXT,
                model_key TEXT,
                temperature REAL,
                top_p REAL,
                max_tokens INTEGER,
                result_json TEXT,
                created_at INTEGER,
                accessed_at INTEGER,
                access_count INTEGER DEFAULT 0,
                size_bytes INTEGER
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at 
            ON cache(accessed_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON cache(created_at)
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Cache database initialized: {self.db_path}")
    
    def generate_cache_key(
        self,
        file_path: str,
        model_key: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """
        生成缓存键
        
        基于文件内容哈希 + 模型参数
        """
        # 读取文件内容并计算哈希
        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # 参数哈希
        param_str = f"{model_key}_{prompt}_{temperature}_{top_p}_{max_tokens}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        
        cache_key = f"{file_hash}_{param_hash}"
        return cache_key
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存
        
        先查内存，再查磁盘
        
        Returns:
            {
                "images": [...],
                "markdown": "...",
                "metadata": {...}
            }
            或 None
        """
        self.stats["total_requests"] += 1
        
        # 1. 查询内存缓存
        with self.cache_lock:
            if cache_key in self.memory_cache:
                self.stats["memory_hits"] += 1
                print(f"⚡ Memory cache hit: {cache_key}")
                return self.memory_cache[cache_key]
        
        # 2. 查询磁盘缓存
        result = self._get_from_disk(cache_key)
        if result is not None:
            self.stats["disk_hits"] += 1
            print(f"💾 Disk cache hit: {cache_key}")
            
            # 放入内存缓存
            with self.cache_lock:
                self.memory_cache[cache_key] = result
            
            return result
        
        # 3. 未命中
        self.stats["misses"] += 1
        print(f"❌ Cache miss: {cache_key}")
        return None
    
    def _get_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从磁盘缓存获取"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询缓存
        cursor.execute("""
            SELECT result_json, created_at, access_count
            FROM cache
            WHERE cache_key = ?
        """, (cache_key,))
        
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            return None
        
        result_json, created_at, access_count = row
        
        # 检查是否过期
        if time.time() - created_at > self.ttl_seconds:
            print(f"⏰ Cache expired: {cache_key}")
            cursor.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            conn.close()
            return None
        
        # 更新访问记录
        cursor.execute("""
            UPDATE cache
            SET accessed_at = ?, access_count = ?
            WHERE cache_key = ?
        """, (int(time.time()), access_count + 1, cache_key))
        
        conn.commit()
        conn.close()
        
        # 反序列化
        return json.loads(result_json)
    
    def set(
        self,
        cache_key: str,
        result: Dict[str, Any],
        file_name: str,
        model_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ):
        """
        保存缓存
        
        同时保存到内存和磁盘
        """
        # 1. 保存到内存
        with self.cache_lock:
            self.memory_cache[cache_key] = result
        
        # 2. 保存到磁盘
        self._set_to_disk(
            cache_key, result, file_name, 
            model_key, temperature, top_p, max_tokens
        )
        
        # 3. 清理过期缓存
        self._cleanup_if_needed()
    
    def _set_to_disk(
        self,
        cache_key: str,
        result: Dict[str, Any],
        file_name: str,
        model_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ):
        """保存到磁盘缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result_json = json.dumps(result, ensure_ascii=False)
        size_bytes = len(result_json.encode('utf-8'))
        now = int(time.time())
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache
            (cache_key, file_name, model_key, temperature, top_p, max_tokens,
             result_json, created_at, accessed_at, access_count, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """, (
            cache_key, file_name, model_key, temperature, top_p, max_tokens,
            result_json, now, now, size_bytes
        ))
        
        conn.commit()
        conn.close()
        
        print(f"💾 Saved to cache: {cache_key} ({size_bytes / 1024:.1f}KB)")
    
    def _cleanup_if_needed(self):
        """清理缓存（如果超过大小限制）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查数据库大小
        cursor.execute("SELECT SUM(size_bytes) FROM cache")
        total_size = cursor.fetchone()[0] or 0
        
        if total_size > self.max_db_size:
            print(f"🧹 Cleaning up cache (current: {total_size / 1024 / 1024:.1f}MB)")
            
            # 删除最少访问的 20%
            cursor.execute("""
                DELETE FROM cache
                WHERE cache_key IN (
                    SELECT cache_key FROM cache
                    ORDER BY access_count ASC, accessed_at ASC
                    LIMIT (SELECT COUNT(*) * 0.2 FROM cache)
                )
            """)
            
            conn.commit()
            deleted = cursor.rowcount
            print(f"🧹 Cleaned up {deleted} cache entries")
        
        conn.close()
    
    def clear_all(self):
        """清空所有缓存"""
        with self.cache_lock:
            self.memory_cache.clear()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        
        print("🧹 All cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache")
        disk_count, disk_size = cursor.fetchone()
        disk_size = disk_size or 0
        
        conn.close()
        
        total = self.stats["total_requests"]
        if total > 0:
            hit_rate = (self.stats["memory_hits"] + self.stats["disk_hits"]) / total
        else:
            hit_rate = 0.0
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_count": disk_count,
            "disk_cache_size_mb": disk_size / 1024 / 1024,
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "misses": self.stats["misses"],
            "total_requests": total,
            "hit_rate": f"{hit_rate * 100:.1f}%"
        }
    
    def print_stats(self):
        """打印缓存统计"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 Cache Statistics")
        print("="*60)
        print(f"Memory cache: {stats['memory_cache_size']} entries")
        print(f"Disk cache: {stats['disk_cache_count']} entries ({stats['disk_cache_size_mb']:.1f}MB)")
        print(f"Total requests: {stats['total_requests']}")
        print(f"  - Memory hits: {stats['memory_hits']} ⚡")
        print(f"  - Disk hits: {stats['disk_hits']} 💾")
        print(f"  - Misses: {stats['misses']} ❌")
        print(f"Hit rate: {stats['hit_rate']}")
        print("="*60 + "\n")


# 全局缓存实例
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器单例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager