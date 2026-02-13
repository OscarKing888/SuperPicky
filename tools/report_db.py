#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky ReportDB - SQLite 报告数据库封装
替代原有的 CSV 报告存储，提供更高效的查询和更新操作。

Usage:
    db = ReportDB("/path/to/photos")
    db.insert_photo({"filename": "IMG_1234", "has_bird": 1, ...})
    photo = db.get_photo("IMG_1234")
    db.close()
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


# Schema 版本，用于未来升级
SCHEMA_VERSION = "1"

# 所有列定义（有序），用于 CREATE TABLE 和数据验证
PHOTO_COLUMNS = [
    # (列名, SQLite 类型, 默认值)
    ("filename",      "TEXT NOT NULL UNIQUE", None),
    ("has_bird",      "INTEGER", 0),          # 0=no, 1=yes
    ("confidence",    "REAL", 0.0),
    ("head_sharp",    "REAL", None),
    ("left_eye",      "REAL", None),
    ("right_eye",     "REAL", None),
    ("beak",          "REAL", None),
    ("nima_score",    "REAL", None),
    ("is_flying",     "INTEGER", 0),          # 0=no, 1=yes
    ("flight_conf",   "REAL", None),
    ("rating",        "INTEGER", 0),          # -1/0/1/2/3
    ("focus_status",  "TEXT", None),           # BEST/GOOD/BAD/WORST
    ("focus_x",       "REAL", None),
    ("focus_y",       "REAL", None),
    ("adj_sharpness", "REAL", None),
    ("adj_topiq",     "REAL", None),
    ("created_at",    "TEXT", None),
    ("updated_at",    "TEXT", None),
]

# 列名集合，用于快速查找
COLUMN_NAMES = {col[0] for col in PHOTO_COLUMNS}


class ReportDB:
    """SQLite 报告数据库封装。

    每个照片处理目录拥有一个独立的数据库文件：
        <directory>/.superpicky/report.db

    线程安全：设置 check_same_thread=False，支持工作线程写入。
    WAL 模式：支持读写并发。
    """

    DB_FILENAME = "report.db"

    def __init__(self, directory: str):
        """
        初始化数据库连接。

        Args:
            directory: 照片目录路径（数据库存储在 .superpicky/ 子目录下）
        """
        self.directory = directory
        self._superpicky_dir = os.path.join(directory, ".superpicky")
        self.db_path = os.path.join(self._superpicky_dir, self.DB_FILENAME)

        # 确保 .superpicky 目录存在
        os.makedirs(self._superpicky_dir, exist_ok=True)

        # 连接数据库
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row  # 支持按列名访问

        # 启用 WAL 模式和外键
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # 初始化 Schema
        self._init_schema()

    def _init_schema(self):
        """创建表和索引（如果不存在）。"""
        # 构建 CREATE TABLE 语句
        col_defs = []
        for name, type_def, _ in PHOTO_COLUMNS:
            col_defs.append(f"    {name} {type_def}")

        create_sql = (
            "CREATE TABLE IF NOT EXISTS photos (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            + ",\n".join(col_defs)
            + "\n)"
        )

        with self._conn:
            self._conn.execute(create_sql)

            # 索引
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_photos_filename "
                "ON photos(filename)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_photos_rating "
                "ON photos(rating)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_photos_has_bird "
                "ON photos(has_bird)"
            )

            # 元数据表
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # 初始化元数据
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION)
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                ("directory_path", self.directory)
            )

    # ==========================================================================
    #  写入操作
    # ==========================================================================

    def insert_photo(self, data: dict) -> None:
        """
        插入或更新一条照片记录。

        如果 filename 已存在则更新，否则插入新记录。
        自动处理 CSV 兼容的数据格式转换（如 "yes"/"no" → 1/0）。

        Args:
            data: 照片数据字典，键为列名
        """
        cleaned = self._clean_data(data)
        now = _now_iso()
        cleaned.setdefault("created_at", now)
        cleaned["updated_at"] = now

        # 仅保留合法列
        columns = [k for k in cleaned if k in COLUMN_NAMES]
        values = [cleaned[k] for k in columns]

        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)

        # INSERT OR REPLACE
        update_clause = ", ".join(
            f"{c} = excluded.{c}" for c in columns if c != "filename"
        )

        sql = (
            f"INSERT INTO photos ({col_str}) VALUES ({placeholders}) "
            f"ON CONFLICT(filename) DO UPDATE SET {update_clause}"
        )

        self._conn.execute(sql, values)
        self._conn.commit()

    def insert_photos_batch(self, photos: List[dict]) -> int:
        """
        批量插入或更新照片记录。

        使用事务包裹，性能优于逐条插入。

        Args:
            photos: 照片数据字典列表

        Returns:
            成功插入/更新的记录数
        """
        if not photos:
            return 0

        now = _now_iso()
        count = 0

        with self._conn:
            for data in photos:
                cleaned = self._clean_data(data)
                cleaned.setdefault("created_at", now)
                cleaned["updated_at"] = now

                columns = [k for k in cleaned if k in COLUMN_NAMES]
                values = [cleaned[k] for k in columns]

                placeholders = ", ".join(["?"] * len(columns))
                col_str = ", ".join(columns)

                update_clause = ", ".join(
                    f"{c} = excluded.{c}" for c in columns if c != "filename"
                )

                sql = (
                    f"INSERT INTO photos ({col_str}) VALUES ({placeholders}) "
                    f"ON CONFLICT(filename) DO UPDATE SET {update_clause}"
                )

                self._conn.execute(sql, values)
                count += 1

        return count

    # ==========================================================================
    #  查询操作
    # ==========================================================================

    def get_photo(self, filename: str) -> Optional[dict]:
        """
        按 filename 查询单条记录。

        Args:
            filename: 照片文件名（不含扩展名）

        Returns:
            照片数据字典，未找到返回 None
        """
        cursor = self._conn.execute(
            "SELECT * FROM photos WHERE filename = ?", (filename,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_photos(self) -> List[dict]:
        """
        获取所有照片记录。

        Returns:
            照片数据字典列表
        """
        cursor = self._conn.execute("SELECT * FROM photos ORDER BY filename")
        return [dict(row) for row in cursor.fetchall()]

    def get_bird_photos(self) -> List[dict]:
        """
        获取所有有鸟的照片记录（has_bird=1）。

        Returns:
            有鸟照片数据字典列表
        """
        cursor = self._conn.execute(
            "SELECT * FROM photos WHERE has_bird = 1 ORDER BY filename"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_photos_by_rating(self, rating: int) -> List[dict]:
        """
        按评分查询照片。

        Args:
            rating: 评分 (-1/0/1/2/3)

        Returns:
            照片数据字典列表
        """
        cursor = self._conn.execute(
            "SELECT * FROM photos WHERE rating = ? ORDER BY filename",
            (rating,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> dict:
        """
        获取评分统计信息。

        Returns:
            包含统计数据的字典，如:
            {
                "total": 217,
                "has_bird": 180,
                "flying": 15,
                "by_rating": {0: 50, 1: 60, 2: 45, 3: 25}
            }
        """
        stats = {}

        # 总数
        row = self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()
        stats["total"] = row[0]

        # 有鸟数
        row = self._conn.execute(
            "SELECT COUNT(*) FROM photos WHERE has_bird = 1"
        ).fetchone()
        stats["has_bird"] = row[0]

        # 飞行数
        row = self._conn.execute(
            "SELECT COUNT(*) FROM photos WHERE is_flying = 1"
        ).fetchone()
        stats["flying"] = row[0]

        # 按评分统计
        cursor = self._conn.execute(
            "SELECT rating, COUNT(*) as cnt FROM photos GROUP BY rating ORDER BY rating"
        )
        stats["by_rating"] = {row[0]: row[1] for row in cursor.fetchall()}

        return stats

    def count(self) -> int:
        """返回总记录数。"""
        row = self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()
        return row[0]

    def exists(self) -> bool:
        """数据库文件是否存在。"""
        return os.path.exists(self.db_path)

    # ==========================================================================
    #  更新操作
    # ==========================================================================

    def update_photo(self, filename: str, data: dict) -> bool:
        """
        按 filename 更新指定字段。

        Args:
            filename: 照片文件名
            data: 要更新的字段字典（仅包含需要更新的字段）

        Returns:
            是否成功更新
        """
        cleaned = self._clean_data(data)
        cleaned["updated_at"] = _now_iso()

        # 仅保留合法列，排除 filename 和 id
        columns = [k for k in cleaned if k in COLUMN_NAMES and k not in ("filename", "id")]
        if not columns:
            return False

        values = [cleaned[k] for k in columns]
        set_clause = ", ".join(f"{c} = ?" for c in columns)

        sql = f"UPDATE photos SET {set_clause} WHERE filename = ?"
        values.append(filename)

        cursor = self._conn.execute(sql, values)
        self._conn.commit()
        return cursor.rowcount > 0

    def update_ratings_batch(self, updates: List[dict]) -> int:
        """
        批量更新评分及相关数据。

        用于重新评星场景（PostAdjustmentEngine）。

        Args:
            updates: 更新数据列表，每个字典必须包含 "filename" 键，
                     以及要更新的字段（如 rating, adj_sharpness, adj_topiq）

        Returns:
            成功更新的记录数
        """
        if not updates:
            return 0

        now = _now_iso()
        count = 0

        with self._conn:
            for upd in updates:
                filename = upd.get("filename")
                if not filename:
                    continue

                cleaned = self._clean_data(upd)
                cleaned["updated_at"] = now

                columns = [k for k in cleaned if k in COLUMN_NAMES and k not in ("filename", "id")]
                if not columns:
                    continue

                values = [cleaned[k] for k in columns]
                set_clause = ", ".join(f"{c} = ?" for c in columns)

                sql = f"UPDATE photos SET {set_clause} WHERE filename = ?"
                values.append(filename)

                cursor = self._conn.execute(sql, values)
                if cursor.rowcount > 0:
                    count += 1

        return count

    # ==========================================================================
    #  元数据操作
    # ==========================================================================

    def get_meta(self, key: str) -> Optional[str]:
        """获取元数据值。"""
        cursor = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """设置元数据值。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value)
        )
        self._conn.commit()

    # ==========================================================================
    #  同步预留
    # ==========================================================================

    def get_updated_since(self, since: str) -> List[dict]:
        """
        获取指定时间之后更新的记录（增量同步用）。

        Args:
            since: ISO 8601 时间字符串

        Returns:
            更新记录列表
        """
        cursor = self._conn.execute(
            "SELECT * FROM photos WHERE updated_at > ? ORDER BY updated_at",
            (since,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # ==========================================================================
    #  连接管理
    # ==========================================================================

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ==========================================================================
    #  内部方法
    # ==========================================================================

    @staticmethod
    def _clean_data(data: dict) -> dict:
        """
        清洗输入数据，处理 CSV 兼容格式转换。

        转换规则：
        - "yes"/"no" → 1/0（仅对 has_bird, is_flying 字段）
        - "-" 或空字符串 → None
        - 数值字符串 → 对应的 float/int
        """
        cleaned = {}
        for key, value in data.items():
            # 跳过非法列名
            if key not in COLUMN_NAMES:
                continue

            # 布尔/yes-no 字段（优先处理，"-"/None/空 → 0）
            if key in ("has_bird", "is_flying"):
                if value is None or value == "-" or value == "":
                    cleaned[key] = 0
                elif isinstance(value, str):
                    cleaned[key] = 1 if value.lower() in ("yes", "1", "true") else 0
                else:
                    cleaned[key] = 1 if value else 0
                continue

            # 处理 None 和占位符
            if value is None or value == "-" or value == "":
                cleaned[key] = None
                continue

            # 数值字段
            if key in ("confidence", "head_sharp", "left_eye", "right_eye",
                        "beak", "nima_score", "flight_conf", "focus_x",
                        "focus_y", "adj_sharpness", "adj_topiq"):
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    cleaned[key] = None
                continue

            if key == "rating":
                try:
                    cleaned[key] = int(float(value))
                except (ValueError, TypeError):
                    cleaned[key] = 0
                continue

            # 文本字段直接使用
            cleaned[key] = value

        return cleaned


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
