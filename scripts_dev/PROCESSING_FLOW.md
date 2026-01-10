# SuperPicky 处理流程关键节点

## 📋 总体流程概览

整个处理流程分为 **6个主要阶段**，每个阶段包含多个关键节点。

---

## 🔄 阶段1: 初始化与设备选择

### 1.1 应用启动
- **入口点**:
  - GUI: `main.py` → `SuperPickyMainWindow`
  - CLI: `superpicky_cli.py` → `cmd_process()`
- **初始化**:
  - 创建 `PhotoProcessor` 实例
  - 加载 `ProcessingSettings` 配置
  - 设置回调函数（日志、进度）

### 1.2 设备选择 ⚙️
- **位置**: `utils.py::get_best_device()`
- **优先级**: MPS (Apple GPU) > CUDA (NVIDIA GPU) > CPU
- **决策逻辑**:
  ```python
  if preferred_device == 'auto':
      # 自动选择最佳设备
      if mps_available: return 'mps'
      elif cuda_available: return 'cuda'
      else: return 'cpu'
  ```
- **并行策略**:
  - CPU: 使用线程池（`cpu_threads` 参数）
  - GPU: 使用队列并发（`gpu_concurrent` 参数）

### 1.3 评分引擎初始化
- **位置**: `core/photo_engine.py::create_rating_engine_from_config()`
- **功能**: 加载高级配置，初始化评分规则

---

## 📁 阶段2: 文件扫描与转换

### 2.1 文件扫描
- **位置**: `photo_processor.py::_scan_files()`
- **功能**:
  - 扫描目录中的 RAW 和 JPG 文件
  - 构建 `raw_dict` 和 `jpg_dict` 映射
  - 生成待处理文件列表 `files_tbr`
- **输出**: `(raw_dict, jpg_dict, files_tbr)`

### 2.2 RAW 文件转换
- **位置**: `photo_processor.py::_convert_raws()`
- **流程**:
  1. 识别需要转换的 RAW 文件（无对应 JPG）
  2. 并行转换（最多4线程）
  3. 使用 `rawpy` 库转换为 JPG
- **函数**: `find_bird_util.py::raw_to_jpeg()`

### 2.3 HEIF/HIF 文件转换
- **位置**: `photo_processor.py::_convert_heif_files()`
- **流程**:
  1. 识别 HEIF/HEIC/HIF 文件
  2. 并行转换（最多8线程）
  3. 使用 `pillow-heif` 转换为临时 JPG
  4. 存储映射: `heif_temp_map[原始文件名] = 临时JPG路径`
- **临时目录**: `.superpicky/temp_jpg/`

---

## 🤖 阶段3: AI检测与评分（核心阶段）

### 3.1 模型加载
- **YOLO 模型**:
  - 位置: `ai_model.py::load_yolo_model()`
  - 功能: 鸟类检测 + 分割掩码
  - 设备: 使用选定的设备（MPS/CUDA/CPU）
  
- **关键点检测模型**:
  - 位置: `core/keypoint_detector.py::get_keypoint_detector()`
  - 功能: 检测鸟类关键点（眼睛、喙等）
  - 模型: `cub200_keypoint_resnet50.pth`
  
- **飞版检测模型**:
  - 位置: `core/flight_detector.py::get_flight_detector()`
  - 功能: 判断是否为飞鸟
  - 模型: `superFlier_efficientnet.pth`
  
- **TOPIQ 美学评分模型**:
  - 位置: `iqa_scorer.py::IQAScorer._load_topiq()`
  - 功能: 计算美学评分（1-10分）
  - 模型: `cfanet_iaa_ava_res50-3cd62bb3.pth`
  - 延迟加载: 首次使用时才加载

### 3.2 单张图片处理流程

#### Phase 1: YOLO 检测
- **位置**: `photo_processor.py::_process_images_sequential()` (行530-540)
- **函数**: `ai_model.py::detect_and_draw_birds()`
- **输出**:
  - `detected`: 是否检测到鸟
  - `confidence`: AI置信度
  - `bird_bbox`: 边界框坐标
  - `bird_mask`: 分割掩码
  - `img_dims`: 图像尺寸

#### Phase 2: 关键点检测
- **位置**: `photo_processor.py::_process_images_sequential()` (行545-653)
- **流程**:
  1. 读取原图（支持 HEIF，使用 `utils.py::read_image()`）
  2. 将 bbox 从缩放尺寸转换到原图尺寸
  3. 裁剪鸟的区域
  4. 在裁剪区域上执行关键点检测
  5. 计算头部锐度、眼睛可见度等
- **输出**:
  - `head_sharpness`: 头部锐度
  - `best_eye_visibility`: 眼睛最高可见度
  - `all_keypoints_hidden`: 是否所有关键点隐藏
  - `head_center_orig`: 头部中心坐标（原图尺寸）

#### Phase 3: TOPIQ 美学评分（条件执行）
- **位置**: `photo_processor.py::_process_images_sequential()` (行655-679)
- **触发条件**:
  - `detected == True`
  - `not all_keypoints_hidden`
  - `best_eye_visibility >= 0.3`
- **函数**: `iqa_scorer.py::IQAScorer.calculate_aesthetic()`
- **流程**:
  1. 加载图片（支持 HEIF）
  2. 调整尺寸到 384x384
  3. 转换为张量
  4. 使用 TOPIQ 模型推理
  5. 返回 MOS 分数（1-10）

#### Phase 4: 飞版检测
- **位置**: `photo_processor.py::_process_images_sequential()` (行681-693)
- **触发条件**: `settings.detect_flight == True`
- **函数**: `core/flight_detector.py::FlightDetector.detect()`
- **输入**: 鸟的裁剪区域（BGR格式）
- **输出**:
  - `is_flying`: 是否飞鸟
  - `flight_confidence`: 飞行置信度
- **加成规则**: 飞鸟锐度+100，美学+0.5

#### Phase 5: 曝光检测
- **位置**: `photo_processor.py::_process_images_sequential()` (行694-708)
- **触发条件**: `settings.detect_exposure == True`
- **函数**: `core/exposure_detector.py::ExposureDetector.detect()`
- **输入**: 鸟的裁剪区域 + 曝光阈值
- **输出**:
  - `is_overexposed`: 是否过曝
  - `is_underexposed`: 是否欠曝

#### Phase 6: 对焦点验证（仅1星以上）
- **位置**: `photo_processor.py::_process_images_sequential()` (行734-767)
- **触发条件**: `preliminary_result.rating >= 1`
- **流程**:
  1. 读取 RAW 文件的 EXIF 对焦点数据
  2. 使用 `core/focus_point_detector.py::verify_focus_in_bbox()`
  3. 4层检测逻辑:
     - 头部对焦（1.1倍权重）
     - 鸟身对焦（1.0倍权重）
     - 偏移对焦（0.7倍权重）
     - 脱焦（0.5倍权重）
- **输出**:
  - `focus_sharpness_weight`: 锐度权重
  - `focus_topiq_weight`: 美学权重
  - `focus_x, focus_y`: 对焦点坐标

#### Phase 7: 最终评分计算
- **位置**: `photo_processor.py::_process_images_sequential()` (行768-782)
- **函数**: `core/rating_engine.py::RatingEngine.calculate()`
- **输入参数**:
  - `detected`, `confidence`
  - `sharpness` (原始锐度)
  - `topiq` (原始美学)
  - `focus_sharpness_weight`, `focus_topiq_weight`
  - `is_flying` (飞鸟乘法加成: 锐度×1.2, 美学×1.1)
- **评分规则**:
  - **3星**: 锐度≥阈值 AND 美学≥阈值
  - **2星**: 锐度≥阈值 OR 美学≥阈值
  - **1星**: 有鸟但未达标
  - **0星**: 技术质量差
  - **-1星**: 无鸟

#### Phase 8: EXIF 元数据写入
- **位置**: `photo_processor.py::_process_images_sequential()` (行851-925)
- **函数**: `exiftool_manager.py::batch_set_metadata()`
- **写入字段**:
  - `Rating`: 星级评分
  - `Pick`: 精选旗标
  - `Label`: 标签（Green=飞鸟, Red=头部对焦）
  - `Country`: 对焦状态（精准/鸟身/偏移/脱焦）
  - `Caption`: 详细评分报告

#### Phase 9: CSV 数据更新
- **位置**: `photo_processor.py::_update_csv_keypoint_data()`
- **文件**: `.superpicky/report.csv`
- **更新字段**:
  - `head_sharp`, `left_eye`, `right_eye`, `beak`
  - `nima_score`, `is_flying`, `flight_conf`
  - `rating`, `focus_status`, `focus_x`, `focus_y`

#### Phase 10: 调试可视化（可选）
- **位置**: `photo_processor.py::_save_debug_crop()`
- **输出目录**: `.superpicky/debug_crops/`
- **标注内容**:
  - 🟢 绿色半透明: SEG mask 鸟身区域
  - 🔵 蓝色圆圈: 头部检测区域
  - 🔴 红色十字: 对焦点位置

### 3.3 并行处理模式
- **CPU 并行**: `_process_images_parallel()` (行974-1007)
  - 使用 `ThreadPoolExecutor`
  - 线程数: `cpu_threads` (0=自动，使用CPU逻辑核心数)
  
- **GPU 并行**: `_process_images_parallel()` (行1017-1067)
  - 使用 `threading.Semaphore` 控制并发数
  - 并发数: `gpu_concurrent` (默认1=串行)

---

## ⭐ 阶段4: 精选旗标计算

### 4.1 双排名交集算法
- **位置**: `photo_processor.py::_calculate_picked_flags()` (行1620-1672)
- **流程**:
  1. 筛选所有3星照片
  2. 按美学分数排序，取 Top N%
  3. 按锐度分数排序，取 Top N%
  4. 计算交集（同时在两个Top N%中的照片）
  5. 批量写入 EXIF `Pick=1`
- **配置**: `advanced_config.picked_top_percentage` (默认10%)

---

## 📂 阶段5: 文件组织

### 5.1 文件移动
- **位置**: `photo_processor.py::_move_files_to_rating_folders()` (行1674-1752)
- **分类规则**:
  - `3星_优选/`: rating == 3
  - `2星_良好/`: rating == 2
  - `1星_普通/`: rating == 1
  - `0星_放弃/`: rating == 0 或 -1
- **优先级**: RAW > JPG/HEIF
- **Manifest**: 生成 `.superpicky_manifest.json` 记录移动历史

---

## 🧹 阶段6: 清理与收尾

### 6.1 临时文件清理
- **位置**: `photo_processor.py::_cleanup_temp_files()` (行1754-1789)
- **清理内容**:
  - RAW 转换的临时 JPG
  - HEIF 转换的临时 JPG (`.superpicky/temp_jpg/`)
- **保留选项**: `settings.keep_temp_jpg == True` 时保留并写入EXIF

### 6.2 统计信息汇总
- **位置**: `photo_processor.py::process()` (行224-236)
- **统计项**:
  - 总处理时间、平均时间
  - 最长/最短处理时间
  - 带鸟/不带鸟平均时间
  - 各星级数量、飞鸟数量、曝光问题数量

---

## 🔄 后期处理流程（重新评星）

### 流程入口
- **位置**: `post_adjustment_engine.py::PostAdjustmentEngine`
- **触发**: CLI `restar` 命令或 GUI "重新评星"功能

### 关键步骤
1. **加载 CSV**: 读取 `.superpicky/report.csv`
2. **重新计算**: 使用新阈值重新评分
3. **精选重算**: 重新计算精选旗标
4. **批量更新**: 批量写入 EXIF 和 CSV
5. **文件重分配**: 根据新评分移动文件

---

## 📊 数据流图

```
用户输入 (目录路径)
    ↓
[阶段1] 初始化
    ├─ 设备选择 (MPS/CUDA/CPU)
    ├─ 评分引擎初始化
    └─ 模型加载准备
    ↓
[阶段2] 文件准备
    ├─ 文件扫描
    ├─ RAW → JPG 转换
    └─ HEIF → JPG 转换
    ↓
[阶段3] AI处理 (每张图片)
    ├─ YOLO 检测 → bbox, mask
    ├─ 关键点检测 → 锐度, 可见度
    ├─ TOPIQ 评分 → 美学分数
    ├─ 飞版检测 → 飞行状态
    ├─ 曝光检测 → 曝光问题
    ├─ 对焦验证 → 对焦权重
    ├─ 评分计算 → 星级 (-1~3)
    ├─ EXIF 写入 → 元数据
    └─ CSV 更新 → 报告数据
    ↓
[阶段4] 精选计算
    └─ 双排名交集 → Pick=1
    ↓
[阶段5] 文件组织
    └─ 移动到分类文件夹
    ↓
[阶段6] 清理收尾
    ├─ 删除临时文件
    └─ 统计汇总
    ↓
处理完成
```

---

## 🎯 关键性能优化点

1. **并行转换**: RAW/HEIF 文件并行转换（多线程）
2. **条件执行**: TOPIQ 只在眼睛可见度≥30%时计算
3. **延迟加载**: TOPIQ 模型首次使用时才加载
4. **对焦优化**: 仅对1星以上照片做对焦检测
5. **并行推理**: CPU线程池 / GPU队列并发
6. **临时文件复用**: HEIF 预转换，避免重复转换

---

## 📝 关键配置文件

- `advanced_config.json`: 高级配置（阈值、精选百分比等）
- `.superpicky/report.csv`: 处理报告数据
- `.superpicky_manifest.json`: 文件移动历史
- `.superpicky/debug_crops/`: 调试可视化图片

---

## 🔧 关键依赖模块

- **AI模型**: `ai_model.py`, `topiq_model.py`, `nima_model.py`
- **检测器**: `core/keypoint_detector.py`, `core/flight_detector.py`, `core/exposure_detector.py`, `core/focus_point_detector.py`
- **评分引擎**: `core/rating_engine.py`
- **工具函数**: `utils.py`, `exiftool_manager.py`
- **文件管理**: `core/file_manager.py`, `temp_file_manager.py`

