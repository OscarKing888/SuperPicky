# Avonet.db 替代 eBird 过滤器设计

## 概述

用 `avonet.db` 完全替代 eBird API，实现 100% 离线的物种地理过滤。

## 目标

- 完全移除 eBird API 依赖（无需网络、无需 API Key）
- 保留现有 UI 交互方式（区域选择器）
- 简化代码（直接用 class_id 匹配，无需 ebird_code 转换）

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                   bird_identifier.py                     │
│                                                          │
│   get_species_filter() → AvonetFilter                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    AvonetFilter                          │
│                  (avonet_filter.py)                      │
│                                                          │
│   • get_species_by_gps(lat, lon) → Set[int]             │
│   • get_species_by_region(region_code) → Set[int]       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      avonet.db                           │
│                    (102MB, Git LFS)                      │
│                                                          │
│   distributions: 3.3M 物种-网格映射                      │
│   places: 19,561 个 1°×1° 网格                          │
│   sp_cls_map: 11,136 物种 → OSEA class_id               │
└─────────────────────────────────────────────────────────┘
```

## 核心模块：avonet_filter.py

```python
class AvonetFilter:
    """基于 avonet.db 的离线物种过滤器"""

    def __init__(self, db_path: str = None):
        # 自动定位 birdid/data/avonet.db

    def get_species_by_gps(self, lat: float, lon: float) -> Optional[Set[int]]:
        """
        根据 GPS 坐标查询该位置可能出现的物种 class_ids

        SQL: SELECT DISTINCT cls FROM distributions d
             JOIN places p ON d.worldid = p.worldid
             JOIN sp_cls_map sm ON d.species = sm.species
             WHERE lat BETWEEN p.south AND p.north
               AND lon BETWEEN p.west AND p.east
        """

    def get_species_by_region(self, region_code: str) -> Optional[Set[int]]:
        """
        根据区域代码查询物种（兼容现有 UI）

        region_code 映射到边界框，内部调用 get_species_by_gps()
        """

    def is_available(self) -> bool:
        """检查 avonet.db 是否存在且可用"""
```

## bird_identifier.py 修改点

### 1. 替换全局变量

```python
# Before
_ebird_filter = None

# After
_avonet_filter = None
```

### 2. 替换懒加载函数

```python
# Before
def get_ebird_filter():
    from birdid.ebird_country_filter import eBirdCountryFilter
    ...

# After
def get_species_filter():
    from birdid.avonet_filter import AvonetFilter
    global _avonet_filter
    if _avonet_filter is None:
        _avonet_filter = AvonetFilter()
    return _avonet_filter
```

### 3. predict_bird() 参数类型变化

```python
# Before
ebird_species_set: Optional[Set[str]] = None  # eBird species codes

# After
species_class_ids: Optional[Set[int]] = None   # OSEA class IDs
```

### 4. predict_bird() 过滤逻辑简化

```python
# Before: 需要查 ebird_code 再匹配
if ebird_code and ebird_code in ebird_species_set:
    ...

# After: 直接用 class_id 匹配
if class_id in species_class_ids:
    ...
```

### 5. identify_bird() 获取物种列表

```python
# Before
ebird_species_set = ebird_filter.get_country_species_list(region)

# After
species_filter = get_species_filter()
if lat and lon:
    species_class_ids = species_filter.get_species_by_gps(lat, lon)
elif region_code:
    species_class_ids = species_filter.get_species_by_region(region_code)
```

## 清理项

| 文件 | 操作 |
|------|------|
| `birdid/ebird_country_filter.py` | 保留但标记 deprecated |
| `birdid/data/offline_ebird_data/` | 可删除 |
| `_gps_detected_region_cache` | 移除 |
| `_species_cache` | 可简化或移除 |

## 测试验证

```bash
# 单元测试
python -c "
from birdid.avonet_filter import AvonetFilter
af = AvonetFilter()
ids = af.get_species_by_gps(3.0, 101.7)  # 吉隆坡
print(f'吉隆坡: {len(ids)} 个物种')
ids = af.get_species_by_gps(-33.9, 151.2)  # 悉尼
print(f'悉尼: {len(ids)} 个物种')
"

# 集成测试
python birdid_cli.py identify /path/to/bird_with_gps.jpg
```

## 优势总结

| 维度 | eBird (旧) | Avonet (新) |
|------|-----------|-------------|
| 网络依赖 | 需要 | 无 |
| API Key | 需要 | 无 |
| 查询速度 | 秒级 | 毫秒级 |
| 数据覆盖 | 30天观察记录 | 物种分布范围 |
| 代码复杂度 | 高（需转换） | 低（直接匹配） |
