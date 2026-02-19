# 打包文件修改计划报告

**日期**: 2026-02-17
**状态**: ✅ 分析完成

## 1. 现状分析

通过检查 `SuperPicky.spec` 和相关目录，当前的打包配置如下：

### ✅ 已包含的新文件
目前的配置通过包含整个目录的方式，已经自动涵盖了新文件：
-   **OSEA 模型**: `models/model20240824.pth` (包含在 `models` 目录中)
-   **Avonet 数据库**: `birdid/data/avonet.db` (包含在 `birdid/data` 目录中)

### ⚠️ 需要移除/排除的文件
目前的配置会由包含整个目录而引入不需要的文件：
1.  **旧模型文件**:
    -   `birdid/models/birdid2024.pt.enc` (存在于 `birdid/models` 目录)
    -   目前 `SuperPicky.spec` 第 49 行明确包含了该目录：`(..., 'birdid/models')`
2.  **eBird 缓存/离线数据**:
    -   `birdid/data/offline_ebird_data/` (目录)
    -   目前 `SuperPicky.spec` 第 48 行包含了整个 `birdid/data` 目录，因此会包含此文件夹。
    -   注：根目录下的 `ebird_cache/` 为空，不会影响，但建议确认排除。

## 2. 修改建议 (SuperPicky.spec)

建议对 `SuperPicky.spec` 进行以下修改，以减小包体积并清理冗余文件：

### A. 移除旧模型 `birdid/models`
**操作**: 删除或注释掉第 49 行。
```python
# [删除] (os.path.join(base_path, 'birdid/models'), 'birdid/models'), 
```
**原因**: `birdid/models` 目录仅包含旧的加密模型 `birdid2024.pt.enc`，删除该行即可彻底排除旧模型。

### B. 优化 `birdid/data` 包含规则
**操作**: 将第 48 行的整体目录包含，改为指定文件包含，或添加排除过滤。
**建议方案**:
不直接包含整个 `birdid/data`，而是显式列出需要的文件：
```python
# [修改] 替代原有的 'birdid/data' 全量包含
(os.path.join(base_path, 'birdid/data/avonet.db'), 'birdid/data'),
(os.path.join(base_path, 'birdid/data/bird_reference.sqlite'), 'birdid/data'),
(os.path.join(base_path, 'birdid/data/birdinfo.json'), 'birdid/data'),
(os.path.join(base_path, 'birdid/data/osea_bird_info.json'), 'birdid/data'),
(os.path.join(base_path, 'birdid/data/osea_labels.txt'), 'birdid/data'),
# 注意：不包含 'offline_ebird_data' 目录
```
**原因**: 这样可以精确控制打包内容，确保 `avonet.db` 被包含，同时排除 `offline_ebird_data` 和其他可能的 eBird 遗留文件。

## 3. 结论
新的打包计划将：
1.  **保留** OSEA 模型 (`model20240824.pth`)。
2.  **保留** Avonet 数据库 (`avonet.db`)。
3.  **剔除** 旧模型 (`birdid2024.pt.enc`)。
4.  **剔除** eBird 离线数据缓存。

请确认是否按此计划修改 `SuperPicky.spec` 文件。
