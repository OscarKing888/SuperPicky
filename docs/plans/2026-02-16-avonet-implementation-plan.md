# Avonet.db 替代 eBird 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用 avonet.db 完全替代 eBird API，实现 100% 离线物种地理过滤。

**Architecture:** 新建 `AvonetFilter` 类，通过 GPS 坐标查询 avonet.db 获取 class_id 集合，直接用于 `predict_bird()` 过滤。移除所有 eBird API 调用和 ebird_code 转换逻辑。

**Tech Stack:** Python 3.12, SQLite3, PyTorch

---

## Task 1: 创建 AvonetFilter 核心类

**Files:**
- Create: `birdid/avonet_filter.py`

**Step 1: 创建 avonet_filter.py 基础结构**

```python
#!/usr/bin/env python3
"""
基于 avonet.db 的离线物种过滤器
替代 eBird API，实现 100% 离线的物种地理过滤
"""

import os
import sys
import sqlite3
from typing import Optional, Set

# 路径配置
BIRDID_DIR = os.path.dirname(os.path.abspath(__file__))

def get_birdid_path(relative_path: str) -> str:
    """获取 birdid 模块内的资源路径"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'birdid', relative_path)
    return os.path.join(BIRDID_DIR, relative_path)


class AvonetFilter:
    """基于 avonet.db 的离线物种过滤器"""

    # 区域代码 → 边界框映射 (south, north, west, east)
    REGION_BOUNDS = {
        # 大洲/大区
        'GLOBAL': (-90, 90, -180, 180),

        # 亚太地区
        'AU': (-44, -10, 112, 155),      # 澳大利亚
        'NZ': (-47.5, -34, 166, 179),    # 新西兰
        'CN': (18, 54, 73, 135),         # 中国
        'JP': (24, 46, 122, 154),        # 日本
        'KR': (33, 39, 124, 132),        # 韩国
        'TW': (21.5, 25.5, 119, 122.5),  # 台湾
        'TH': (5.5, 20.5, 97.5, 105.5),  # 泰国
        'MY': (0.5, 7.5, 99.5, 119.5),   # 马来西亚
        'SG': (1.1, 1.5, 103.6, 104.1),  # 新加坡
        'ID': (-11, 6, 95, 141),         # 印度尼西亚
        'PH': (4.5, 21, 116, 127),       # 菲律宾
        'VN': (8, 23.5, 102, 110),       # 越南
        'IN': (6, 36, 68, 98),           # 印度

        # 澳大利亚各州
        'AU-QLD': (-29, -10, 138, 154),
        'AU-NSW': (-37.5, -28, 141, 154),
        'AU-VIC': (-39.2, -34, 141, 150),
        'AU-TAS': (-43.7, -39.5, 143.5, 148.5),
        'AU-SA': (-38, -26, 129, 141),
        'AU-WA': (-35, -13.5, 112.5, 129),
        'AU-NT': (-26, -10.5, 129, 138),
        'AU-ACT': (-35.95, -35.1, 148.75, 149.4),

        # 美洲
        'US': (24, 49, -125, -66),       # 美国本土
        'CA': (41, 84, -141, -52),       # 加拿大
        'MX': (14, 33, -118, -86),       # 墨西哥
        'BR': (-34, 6, -74, -34),        # 巴西
        'AR': (-55, -21, -74, -53),      # 阿根廷
        'CL': (-56, -17, -76, -66),      # 智利
        'CO': (-4.5, 13, -79, -66.5),    # 哥伦比亚
        'PE': (-18.5, -0.5, -81.5, -68), # 秘鲁
        'EC': (-5, 1.5, -81, -75),       # 厄瓜多尔
        'CR': (8, 11.5, -86, -82.5),     # 哥斯达黎加

        # 欧洲
        'GB': (49, 61, -8, 2),           # 英国
        'FR': (41, 51.5, -5.5, 10),      # 法国
        'DE': (47, 55.5, 5.5, 15.5),     # 德国
        'ES': (36, 44, -9.5, 4.5),       # 西班牙
        'IT': (35.5, 47.5, 6.5, 19),     # 意大利
        'NO': (57.5, 71.5, 4, 31.5),     # 挪威
        'SE': (55, 69.5, 10.5, 24.5),    # 瑞典
        'FI': (59.5, 70.5, 19.5, 32),    # 芬兰
        'PL': (49, 55, 14, 24.5),        # 波兰
        'TR': (36, 42.5, 26, 45),        # 土耳其

        # 非洲
        'ZA': (-35, -22, 16, 33),        # 南非
        'KE': (-5, 5, 33.5, 42),         # 肯尼亚
        'TZ': (-12, -1, 29, 41),         # 坦桑尼亚
        'EG': (22, 32, 24.5, 37),        # 埃及
        'MA': (27.5, 36, -13, -1),       # 摩洛哥
    }

    def __init__(self, db_path: str = None):
        """
        初始化 Avonet 过滤器

        Args:
            db_path: 数据库路径，默认自动定位 birdid/data/avonet.db
        """
        if db_path is None:
            db_path = get_birdid_path('data/avonet.db')
        self.db_path = db_path
        self._conn = None

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """获取数据库连接（懒加载）"""
        if self._conn is None:
            if not os.path.exists(self.db_path):
                print(f"[Avonet] 数据库不存在: {self.db_path}")
                return None
            try:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            except Exception as e:
                print(f"[Avonet] 数据库连接失败: {e}")
                return None
        return self._conn

    def is_available(self) -> bool:
        """检查 avonet.db 是否可用"""
        return self._get_connection() is not None

    def get_species_by_gps(self, lat: float, lon: float) -> Optional[Set[int]]:
        """
        根据 GPS 坐标查询该位置可能出现的物种 class_ids

        Args:
            lat: 纬度 (-90 ~ 90)
            lon: 经度 (-180 ~ 180)

        Returns:
            物种 class_id 集合，如果数据库不可用返回 None
        """
        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT sm.cls
                FROM distributions d
                JOIN places p ON d.worldid = p.worldid
                JOIN sp_cls_map sm ON d.species = sm.species
                WHERE ? BETWEEN p.south AND p.north
                  AND ? BETWEEN p.west AND p.east
            """, (lat, lon))
            class_ids = {row[0] for row in cursor.fetchall()}
            return class_ids
        except Exception as e:
            print(f"[Avonet] GPS 查询失败: {e}")
            return None

    def get_species_by_region(self, region_code: str) -> Optional[Set[int]]:
        """
        根据区域代码查询物种（兼容现有 UI）

        Args:
            region_code: 区域代码，如 "AU", "AU-SA", "CN"

        Returns:
            物种 class_id 集合，如果区域未知或数据库不可用返回 None
        """
        region_code = region_code.upper()
        bounds = self.REGION_BOUNDS.get(region_code)

        if bounds is None:
            print(f"[Avonet] 未知区域代码: {region_code}")
            return None

        south, north, west, east = bounds
        # 使用边界框中心点查询
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2

        # 对于大区域，合并多个采样点的结果
        if (north - south) > 20 or (east - west) > 20:
            return self._get_species_by_bounds(south, north, west, east)

        return self.get_species_by_gps(center_lat, center_lon)

    def _get_species_by_bounds(self, south: float, north: float,
                                west: float, east: float) -> Optional[Set[int]]:
        """
        根据边界框查询物种（用于大区域）

        查询边界框内所有网格包含的物种
        """
        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT sm.cls
                FROM distributions d
                JOIN places p ON d.worldid = p.worldid
                JOIN sp_cls_map sm ON d.species = sm.species
                WHERE p.north >= ? AND p.south <= ?
                  AND p.east >= ? AND p.west <= ?
            """, (south, north, west, east))
            class_ids = {row[0] for row in cursor.fetchall()}
            return class_ids
        except Exception as e:
            print(f"[Avonet] 边界框查询失败: {e}")
            return None

    def get_supported_regions(self) -> list:
        """获取支持的区域代码列表"""
        return list(self.REGION_BOUNDS.keys())

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None


# 测试代码
if __name__ == "__main__":
    af = AvonetFilter()

    if af.is_available():
        # 测试 GPS 查询
        print("=== GPS 查询测试 ===")
        test_locations = [
            (3.0, 101.7, "吉隆坡"),
            (-33.9, 151.2, "悉尼"),
            (35.7, 139.7, "东京"),
            (51.5, -0.1, "伦敦"),
        ]
        for lat, lon, name in test_locations:
            ids = af.get_species_by_gps(lat, lon)
            if ids:
                print(f"{name} ({lat}, {lon}): {len(ids)} 个物种")

        # 测试区域查询
        print("\n=== 区域查询测试 ===")
        for region in ['AU', 'AU-SA', 'CN', 'JP']:
            ids = af.get_species_by_region(region)
            if ids:
                print(f"{region}: {len(ids)} 个物种")

        af.close()
    else:
        print("avonet.db 不可用")
```

**Step 2: 运行测试验证模块正常工作**

Run: `python birdid/avonet_filter.py`

Expected output:
```
=== GPS 查询测试 ===
吉隆坡 (3.0, 101.7): 388 个物种
悉尼 (-33.9, 151.2): XXX 个物种
...
=== 区域查询测试 ===
AU: XXX 个物种
...
```

**Step 3: Commit**

```bash
git add birdid/avonet_filter.py
git commit -m "feat: add AvonetFilter for offline species filtering"
```

---

## Task 2: 修改 bird_identifier.py 使用 AvonetFilter

**Files:**
- Modify: `birdid/bird_identifier.py`

**Step 1: 替换全局变量和懒加载函数**

在 `bird_identifier.py` 中：

1. 找到并修改全局变量（约 116-118 行）:
```python
# Before
_ebird_filter = None  # eBirdCountryFilter 单例
_species_cache = {}  # {region_code: species_set} 物种列表缓存
_gps_detected_region_cache = None  # GPS 检测的区域缓存

# After
_avonet_filter = None  # AvonetFilter 单例
```

2. 替换 `get_ebird_filter()` 为 `get_species_filter()`（约 243-256 行）:
```python
# Before
def get_ebird_filter():
    """V4.0.5: 懒加载 eBirdCountryFilter（单例模式）"""
    global _ebird_filter
    if _ebird_filter is None:
        try:
            from birdid.ebird_country_filter import eBirdCountryFilter
            ...
        except Exception as e:
            ...
    return _ebird_filter

# After
def get_species_filter():
    """懒加载 AvonetFilter（单例模式）"""
    global _avonet_filter
    if _avonet_filter is None:
        try:
            from birdid.avonet_filter import AvonetFilter
            _avonet_filter = AvonetFilter()
            if _avonet_filter.is_available():
                print("[Avonet] 离线物种过滤器已加载")
            else:
                _avonet_filter = None
        except Exception as e:
            print(f"[Avonet] 初始化失败: {e}")
            return None
    return _avonet_filter
```

3. 删除 `get_species_list_cached()` 函数（约 259-273 行）

**Step 2: 修改 predict_bird() 函数签名和过滤逻辑**

找到 `predict_bird()` 函数（约 621 行），修改：

```python
# Before
def predict_bird(
    image: Image.Image,
    top_k: int = 5,
    ebird_species_set: Optional[Set[str]] = None
) -> List[Dict]:

# After
def predict_bird(
    image: Image.Image,
    top_k: int = 5,
    species_class_ids: Optional[Set[int]] = None
) -> List[Dict]:
```

修改内部过滤逻辑（约 696-721 行）:

```python
# Before
        # eBird 过滤
        ebird_match = False
        if ebird_species_set:
            if not ebird_code and db_manager and en_name:
                ebird_code = db_manager.get_ebird_code_by_english_name(en_name)

            if ebird_code and ebird_code in ebird_species_set:
                ebird_match = True
            elif ebird_species_set:
                # 调试：显示被过滤掉的候选
                if i < 5:  # 只显示前5个被过滤的
                    print(f"[eBird过滤] 跳过: {cn_name} ({en_name}), ebird_code={ebird_code}, 置信度={confidence:.1f}%")
                continue  # 不在列表中，跳过

# After
        # Avonet 地理过滤
        region_match = False
        if species_class_ids:
            if class_id in species_class_ids:
                region_match = True
            else:
                continue  # 不在区域物种列表中，跳过
```

同时更新结果字典（约 710-720 行）:
```python
# Before
        results.append({
            ...
            'ebird_match': ebird_match,
            ...
        })

# After
        results.append({
            ...
            'region_match': region_match,
            ...
        })
```

**Step 3: 修改 identify_bird() 函数中的物种过滤逻辑**

找到 `identify_bird()` 函数中的 eBird 过滤部分（约 787-851 行），替换为：

```python
        # Avonet 地理过滤
        species_class_ids = None

        if use_ebird:  # 参数名保持兼容，实际使用 Avonet
            try:
                species_filter = get_species_filter()
                if not species_filter:
                    print("[Avonet] 离线过滤器不可用")
                else:
                    # 优先使用 GPS 坐标
                    if use_gps:
                        lat, lon, gps_msg = extract_gps_from_exif(image_path)
                        if lat and lon:
                            result['gps_info'] = {
                                'latitude': lat,
                                'longitude': lon,
                                'info': gps_msg
                            }
                            species_class_ids = species_filter.get_species_by_gps(lat, lon)
                            if species_class_ids:
                                print(f"[Avonet] GPS ({lat:.2f}, {lon:.2f}): {len(species_class_ids)} 个物种")

                    # 回退到区域代码
                    if species_class_ids is None and (region_code or country_code):
                        effective_region = region_code or country_code
                        species_class_ids = species_filter.get_species_by_region(effective_region)
                        if species_class_ids:
                            print(f"[Avonet] 区域 {effective_region}: {len(species_class_ids)} 个物种")

                    # 记录过滤信息
                    if species_class_ids:
                        result['ebird_info'] = {  # 保持键名兼容
                            'enabled': True,
                            'species_count': len(species_class_ids),
                            'data_source': 'avonet.db (offline)'
                        }

            except Exception as e:
                print(f"[Avonet] 过滤初始化失败: {e}")
```

**Step 4: 更新 predict_bird() 调用**

找到 `identify_bird()` 中调用 `predict_bird()` 的地方（约 854 行）:

```python
# Before
        results = predict_bird(image, top_k=top_k, ebird_species_set=ebird_species_set)

# After
        results = predict_bird(image, top_k=top_k, species_class_ids=species_class_ids)
```

**Step 5: 运行测试验证**

Run: `python birdid_cli.py identify test_photos/bird_with_gps.jpg`

Expected: 正常识别并显示 `[Avonet]` 相关日志

**Step 6: Commit**

```bash
git add birdid/bird_identifier.py
git commit -m "refactor: replace eBird API with AvonetFilter in bird_identifier"
```

---

## Task 3: 清理旧代码

**Files:**
- Modify: `birdid/bird_identifier.py` (移除未使用的导入和变量)
- Keep: `birdid/ebird_country_filter.py` (标记 deprecated)

**Step 1: 清理 bird_identifier.py 中的残留代码**

移除不再使用的代码：
- 删除 `OFFLINE_EBIRD_DIR` 常量（约 104 行）
- 删除 `_species_cache` 和 `_gps_detected_region_cache` 变量声明

**Step 2: 在 ebird_country_filter.py 开头添加 deprecation 警告**

```python
#!/usr/bin/env python3
"""
eBird国家鸟类过滤器

DEPRECATED: 此模块已被 avonet_filter.py 替代
保留代码仅供参考，不再被调用
"""
import warnings
warnings.warn(
    "ebird_country_filter 已弃用，请使用 avonet_filter",
    DeprecationWarning,
    stacklevel=2
)
```

**Step 3: Commit**

```bash
git add birdid/bird_identifier.py birdid/ebird_country_filter.py
git commit -m "chore: cleanup eBird references, mark ebird_country_filter deprecated"
```

---

## Task 4: 端到端测试

**Step 1: 测试 GPS 自动过滤**

```bash
# 使用带 GPS 的照片测试
python birdid_cli.py identify /path/to/photo_with_gps.jpg
```

Expected:
- 显示 `[GPS] 从 exiftool 提取: XX.XXXXXX, XX.XXXXXX`
- 显示 `[Avonet] GPS (XX.XX, XX.XX): XXX 个物种`
- 识别结果中显示 `region_match: True`

**Step 2: 测试区域代码过滤**

```bash
# 手动指定区域
python -c "
from birdid.bird_identifier import identify_bird
result = identify_bird('/path/to/photo.jpg', use_gps=False, region_code='AU-SA')
print(result)
"
```

**Step 3: 测试无过滤模式**

```bash
python -c "
from birdid.bird_identifier import identify_bird
result = identify_bird('/path/to/photo.jpg', use_ebird=False)
print(result)
"
```

**Step 4: Commit final changes if any**

```bash
git add -A
git commit -m "test: verify avonet integration works end-to-end"
```

---

## 总结

| Task | 描述 | 预计改动量 |
|------|------|-----------|
| 1 | 创建 AvonetFilter | 新建 ~150 行 |
| 2 | 修改 bird_identifier.py | 修改 ~50 行 |
| 3 | 清理旧代码 | 删除 ~10 行 |
| 4 | 端到端测试 | 测试验证 |

完成后：
- eBird API 完全移除
- 100% 离线运行
- 代码更简洁（直接 class_id 匹配）
