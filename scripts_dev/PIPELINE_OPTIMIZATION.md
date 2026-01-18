# 流水线优化框架说明

## 📋 概述

新的流水线框架通过**面向对象的Job队列系统**实现了高效的并发处理，充分利用CPU和GPU资源，实现了流水线式的处理流程。

## 🎯 核心优化点

### 1. 流水线式HEIF转换
- **问题**: 原有实现需要等待所有HEIF文件转换完成后才开始AI推理
- **优化**: 转换一张立即传递给AI推理队列，实现流水线处理
- **效果**: 减少等待时间，提高整体吞吐量

### 2. 动态并发控制
- **问题**: 转换和推理的并发数固定，无法根据实际速度调整
- **优化**: 根据推理速度自动计算转换线程数（推理慢10倍，转换线程=推理线程/10）
- **效果**: 合理分配CPU资源，避免转换占用过多资源

### 3. 多设备并行推理
- **问题**: 只能使用单一设备（CPU或GPU）
- **优化**: 支持"全部"模式，同时使用CPU和GPU进行推理
- **效果**: 充分利用所有可用计算资源，达到最高性能

### 4. 阶段化处理
- **问题**: 所有处理步骤耦合在一起，难以优化
- **优化**: 将处理流程分解为独立阶段（转换→AI推理→EXIF写入）
- **效果**: 每个阶段可以独立优化和并发执行

## 🏗️ 架构设计

### 核心类

#### 1. `Job` - 任务单元
```python
@dataclass
class Job:
    job_id: str
    data: Any
    status: JobStatus
    result: Any
    # ...
```

#### 2. `JobQueue` - 线程安全队列
- 支持生产者-消费者模式
- 线程安全的put/get操作
- 统计信息跟踪

#### 3. `PipelineStage` - 流水线阶段抽象
- 抽象基类，定义处理接口
- 支持多工作线程并发
- 自动统计处理时间和成功率

#### 4. `Pipeline` - 流水线管理器
- 管理多个阶段的启动和停止
- 协调阶段间的数据流转
- 提供统一的统计接口

#### 5. `DeviceManager` - 设备管理器
- 检测可用设备（CPU/CUDA/MPS）
- 支持"全部"模式
- 为每个设备生成配置

### 具体实现阶段

#### 1. `HEIFConversionStage` - HEIF转换阶段
- 将HEIF/HEIC/HIF文件转换为临时JPG
- 转换完成后立即传递给下一阶段
- 动态调整并发数

#### 2. `RAWConversionStage` - RAW转换阶段
- 将RAW文件转换为JPG
- 独立运行，不阻塞主流程

#### 3. `ImageProcessingStage` - AI处理阶段
- 整合所有AI检测和评分步骤：
  - YOLO检测
  - 关键点检测
  - TOPIQ评分
  - 飞版检测
  - 曝光检测
  - 对焦点验证
  - 评分计算
- 支持多设备并行（CPU+GPU同时工作）

#### 4. `EXIFWriteStage` - EXIF写入阶段
- 批量写入元数据
- 更新CSV报告
- 线程安全的文件操作

## 📊 性能优化策略

### 1. 并发数计算

```python
def _calculate_conversion_workers(inference_workers, speed_ratio=10.0):
    # 推理慢10倍，转换线程应该是推理线程的1/10
    conversion_workers = max(1, int(inference_workers / speed_ratio))
    # 但也不要太多，避免占用过多CPU
    max_conversion = min(4, multiprocessing.cpu_count() // 2)
    return min(conversion_workers, max_conversion)
```

**示例**:
- 推理线程: 8 (CPU) + 2 (GPU) = 10
- 转换线程: max(1, 10/10) = 1，但上限为4，所以最终为1
- 如果推理线程只有2个，转换线程也是1个

### 2. 设备分配策略

**"全部"模式**:
- CPU: 使用所有逻辑核心
- CUDA: 使用配置的并发数（默认1，避免显存溢出）
- MPS: 使用配置的并发数（默认1）

**文件分配**:
- 轮询分配到不同设备
- 确保负载均衡

### 3. 流水线设计

```
HEIF文件 → [转换阶段] → [队列合并器] → [AI处理-CPU] → [EXIF写入]
                                              ↓
常规文件 → [AI处理-GPU] ────────────────────→ [EXIF写入]
```

## 🔧 使用方法

### 1. 启用流水线框架

在 `ProcessingSettings` 中设置：
```python
settings = ProcessingSettings(
    device='all',  # 使用所有设备
    use_pipeline=True,  # 启用流水线框架
    cpu_threads=0,  # 0=自动使用所有核心
    gpu_concurrent=1  # GPU并发数
)
```

### 2. 设备选择选项

- `'auto'`: 自动选择最佳设备（MPS > CUDA > CPU）
- `'cpu'`: 仅使用CPU
- `'cuda'`: 仅使用NVIDIA GPU
- `'mps'`: 仅使用Apple GPU
- `'all'`: **同时使用所有可用设备**（新功能）

### 3. 性能调优参数

```python
settings = ProcessingSettings(
    cpu_threads=8,  # CPU线程数（0=自动）
    gpu_concurrent=2,  # GPU并发数（需考虑显存）
    use_pipeline=True  # 启用流水线
)
```

## 📈 性能提升预期

### 场景1: 大量HEIF文件
- **原有方式**: 转换100张HEIF（8线程，10秒）→ AI推理（10秒）= 20秒
- **流水线方式**: 转换和推理并行，总时间 ≈ 15秒
- **提升**: ~25%

### 场景2: CPU+GPU混合
- **原有方式**: 只能使用CPU或GPU之一
- **流水线方式**: CPU和GPU同时工作
- **提升**: 接近2倍（取决于CPU和GPU性能比）

### 场景3: 转换速度优化
- **原有方式**: 转换线程固定为8，可能占用过多CPU
- **流水线方式**: 根据推理速度动态调整，转换线程=1-2
- **提升**: CPU资源更合理分配，推理速度提升

## 🐛 故障降级

如果流水线框架出现错误，会自动降级到原有处理方法：
```python
try:
    self._process_images_with_pipeline(files_tbr, raw_dict)
except Exception as e:
    self._log(f"❌ 流水线处理失败: {e}", "error")
    self._log("⚠️  降级到原有处理方法", "warning")
    self._process_images(files_tbr, raw_dict)
```

## 📝 代码结构

```
core/
├── job_queue.py          # 核心框架（Job, JobQueue, PipelineStage, Pipeline, DeviceManager）
├── pipeline_stages.py    # 具体阶段实现（HEIF转换、RAW转换、AI处理、EXIF写入）
├── pipeline_builder.py   # 流水线构建器（自动构建最优流水线）
└── photo_processor.py    # 主处理器（集成流水线框架）
```

## 🔮 未来优化方向

1. **自适应并发调整**: 根据实际处理速度动态调整各阶段并发数
2. **优先级队列**: 支持任务优先级，重要文件优先处理
3. **断点续传**: 支持中断后从断点继续处理
4. **资源监控**: 实时监控CPU/GPU/内存使用情况，自动调整并发数
5. **批处理优化**: EXIF写入支持更大的批量，减少IO开销

## ⚠️ 注意事项

1. **显存管理**: GPU并发数需要根据显存大小调整，避免OOM
2. **线程安全**: 所有共享数据结构都需要线程安全保护
3. **错误处理**: 单个任务失败不应影响整个流水线
4. **资源竞争**: CPU和GPU同时工作时，注意CPU资源分配

## 📚 相关文档

- `PROCESSING_FLOW.md`: 完整处理流程说明
- `core/job_queue.py`: 框架API文档
- `core/pipeline_builder.py`: 构建器使用说明

