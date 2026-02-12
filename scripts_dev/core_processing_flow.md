# core 目录处理流程说明（代码评审版）

## 1. 范围与入口
- 核心执行入口不是 `core` 本身，而是上层调用：
  - CLI: `tools/cli_processor.py` -> `PhotoProcessor.process()`
  - GUI: `ui/main_window.py` -> `PhotoProcessor.process()`
- `core` 中真正的主编排器是 `core/photo_processor.py`。

## 2. core 模块职责总览
- `photo_processor.py`
  - 主流程编排：扫描、RAW 转换、检测评分、EXIF 写入、文件整理、连拍合并、临时文件清理。
- `rating_engine.py`
  - 统一评分规则（-1/0/1/2/3 星）与原因字符串生成。
- `keypoint_detector.py`
  - 关键点检测（双眼+喙）与头部锐度计算。
- `flight_detector.py`
  - 飞版二分类（is_flying/confidence）。
- `exposure_detector.py`
  - 过曝/欠曝检测。
- `focus_point_detector.py`
  - 读取 RAW 对焦点，判断对焦点相对头部/鸟身/BBox 位置，返回评分权重。
- `burst_detector.py`
  - 连拍组检测（时间戳 + 可选 pHash）、组内最佳选择。
- `stats_formatter.py`
  - CLI/GUI 公共摘要格式化。
- `build_info.py`
  - 构建信息（commit hash）。
- `file_manager.py`、`config_manager.py`
  - 旧的文件管理与配置包装层，当前主流程（`PhotoProcessor`）未直接使用。

## 3. 批次级主流程（`PhotoProcessor.process`）
1. 扫描文件（`_scan_files`）
   - 建立 `raw_dict`、`jpg_dict`、`files_tbr`（待处理 JPEG 列表）。
2. 早期连拍检测（`_detect_bursts_early`，可选）
   - 对 RAW 读取时间戳，按时间差分组，先得到 `burst_map`。
3. RAW 转换（`_identify_raws_to_convert` + `_convert_raws`）
   - 仅对无同名 JPEG 的 RAW 进行并行转换，产出 `tmp_*.jpg`。
4. 图像处理与评分（`_process_images`）
   - 对每张待处理图执行 AI 检测、关键点、TOPIQ、飞版、曝光、对焦验证、最终评分、EXIF/CSV 更新。
5. 精选旗标计算（`_calculate_picked_flags`）
   - 只在 3 星集合上做：TOPIQ 前 x% 与锐度前 x% 交集 => `pick=1`。
6. 按评分归档（`_move_files_to_rating_folders`，可选）
   - 移动 RAW/JPEG/XMP 到评分目录；2/3 星可按鸟种分子目录；写 manifest。
7. 跨目录连拍合并（`_consolidate_burst_groups`，可选）
   - 按早期 `burst_map` 在已归档目录中二次合并到 `burst_xxx/`。
8. 清理临时 JPEG（`_cleanup_temp_files`，可选）
   - 删除前面生成的 `tmp_*.jpg`。

## 4. 单张图片处理流程（`_process_images`）
1. YOLO 首轮检测（`detect_and_draw_birds(..., skip_nima=True)`）。
2. 多鸟场景下，用 RAW 对焦点辅助重选目标鸟（可选）。
3. 早退逻辑：
   - 无鸟或低于 AI 置信度阈值，直接记 `-1/0`，写简化 EXIF，跳过后续重计算。
4. 关键点检测（眼/喙可见性 + 头部锐度）。
5. 条件 TOPIQ：
   - 仅在“有鸟 + 关键点可见性满足”时计算，减少开销。
6. 飞版检测、曝光检测（均在鸟 crop 上执行）。
7. ISO 锐度归一化：
   - 从 RAW/JPEG EXIF 读 ISO，对锐度做衰减补偿（高 ISO 防虚高）。
8. 初步评分（不带对焦权重，仅用于判定是否值得做对焦验证）。
9. 对焦点验证（`verify_focus_in_bbox`）：
   - 头部/SEG/BBox/框外映射成不同锐度与美学权重。
10. 最终评分（`rating_engine.calculate`）。
11. 写结果：
   - EXIF（rating/pick/caption/label/focus/title）
   - CSV 行更新（关键点、飞版、对焦、调整后分数）
   - 统计计数与 `file_ratings`。

## 5. 评分引擎规则（`rating_engine.calculate`）
- 基础门槛顺序：
  1. 无鸟 -> -1
  2. 置信度不足 -> 0
  3. 关键点全部不可见 -> 1
  4. 锐度低于最小阈值 -> 0
  5. 美学低于最小阈值 -> 0
- 达标判定：
  - 锐度和美学都达标 -> 基础 3 星
  - 二者其一达标 -> 基础 2 星
  - 都不达标但过最低门槛 -> 1 星
- 修正因子：
  - 眼睛可见度降权（按可见度映射 0.5~1.0）
  - 曝光问题降级
  - 对焦权重（锐度/美学双权重）
  - 飞版加成（引擎内乘法）

## 6. 输出物与目录变化
- EXIF 元数据回写（RAW 优先，必要时同步 JPEG）。
- `.superpicky/report.csv`（评分与分析数据）。
- `.superpicky/debug_crops/`（调试裁剪图，按配置/流程生成）。
- 评分目录与鸟种子目录（2/3 星时可能带 species 子目录）。
- `.superpicky_manifest.json`（用于回滚/重置场景）。

## 7. 评审观察（当前代码状态）
- 当前“有效主链”集中在 `photo_processor.py`，其余模块主要作为能力组件被调用。
- `file_manager.py`/`config_manager.py` 仍在 `core` 中，但从全局引用看未接入当前主流程，属于并存实现。
- `photo_processor._consolidate_burst_groups()` 中实例化了 `BurstDetector(use_phash=True)` 与 `exiftool_mgr`，但函数主体未实际调用这两个实例的方法，存在可清理空间。

## 8. 推荐阅读顺序（便于继续维护）
1. `core/photo_processor.py`
2. `core/rating_engine.py`
3. `core/keypoint_detector.py`
4. `core/focus_point_detector.py`
5. `core/burst_detector.py`
6. `core/flight_detector.py` + `core/exposure_detector.py`

