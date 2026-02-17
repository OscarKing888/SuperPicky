# OSEA 模型替换 — 开发交接文档

> **分支**: 已合并到 `master` | **日期**: 2026-02-17 | **状态**: **核心替换已完成**

---

## 一、项目背景

SuperPicky 是一个鸟类摄影选片工具，内置 AI 鸟种识别功能。原模型 `birdid2024` 基于 TorchScript 格式加密部署，覆盖 10,964 种鸟类。本次工作将其替换为 OSEA（Open Species Encyclopedia of Aves）ResNet34 模型，以获得更好的识别准确率和更简洁的推理流程。

### 替换原因

| 对比项 | birdid2024 | OSEA |
|--------|-----------|------|
| 模型结构 | 未知（加密 TorchScript） | ResNet34（开放权重） |
| 预处理 | 5x Enhancement Fusion + BGR | Letterboxing 15% + RGB |
| 随机测试正确率 | 5/8 (62.5%) | 6/8 (75%) |
| 置信度异常 | 有（99.8% 误判为近似种） | 无 |
| 部署方式 | 加密 `.pt.enc` + 密钥解密 | 明文 `.pth` state_dict |

---

## 二、已完成的工作

### 2.1 Commit 历史

| Commit | 说明 |
|--------|------|
| `f71fb942` | **Merge to master**: OSEA ResNet34 model replacement |
| `457d5221` | Temperature 调整为 0.9（降低过高置信度） |
| `da00ee0a` | 预处理优化：Letterboxing 15% + T=0.7 |
| `8767bf20` | 清理 eBird 引用，标记 ebird_country_filter deprecated |
| `117868b9` | 核心替换：用 AvonetFilter 替代 eBird API |
| `9d6fd9fe` | 新增 AvonetFilter 离线物种过滤 |
| `31108ea9` | 核心替换：`predict_bird()` 改用 OSEA |
| `af7b903a` | 新增 OSEA 基础代码 |

### 2.2 关键变更

| 模块 | 变更内容 | 状态 |
|------|----------|------|
| **模型** | birdid2024 → OSEA ResNet34 | ✅ 完成 |
| **物种过滤** | eBird API → AvonetFilter (avonet.db) | ✅ 完成 |
| **预处理** | 5x Enhancement → Letterboxing 15% | ✅ 完成 |
| **Temperature** | 0.5 → 0.9（更平滑的置信度） | ✅ 完成 |
| **TTA 增强** | 测试后放弃（产生错误高置信度） | ❌ 放弃 |

### 2.3 修改文件清单

```
birdid/bird_identifier.py    # 核心改动：模型加载 + 推理逻辑 + Temperature=0.9
birdid/osea_classifier.py    # 新增：独立 OSEA 分类器
birdid/avonet_filter.py      # 新增：离线物种过滤（替代 eBird API）
birdid/ebird_country_filter.py  # 标记为 deprecated
birdid/data/osea_bird_info.json  # 新增：10,964 物种信息
birdid/data/avonet.db        # 新增：地理分布数据库 (Git LFS)
birdid_cli.py                # 新增 --model osea 参数
```

### 2.4 大文件（Git LFS 管理）

| 文件 | 大小 | 路径 | 说明 |
|------|------|------|------|
| OSEA 模型权重 | 103MB | `models/model20240824.pth` | **必需** |
| Avonet 数据库 | 102MB | `birdid/data/avonet.db` | **必需**（离线过滤） |

---

## 三、核心架构

### 3.1 模型加载 (`get_classifier()`)

```python
def get_classifier():
    if os.path.exists(MODEL_PATH):  # models/model20240824.pth
        model = models.resnet34(num_classes=11000)
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        # → OSEA ResNet34
    else:
        # 回退到旧的 birdid2024 加密模型（保留兼容）
```

### 3.2 推理流程 (`predict_bird()`)

```
输入图像
    ↓
┌─────────────────────────────────────────┐
│ YOLO 裁剪后：Letterboxing 15% 填充      │
│ 原始大图：Resize(256) → CenterCrop(224) │
└─────────────────────────────────────────┘
    ↓
RGB → torchvision.ToTensor → ImageNet Normalize
    ↓
模型推理 → Temperature=0.9 Softmax
    ↓
top-k 结果 → AvonetFilter GPS 过滤
    ↓
返回 [{"name_cn", "name_en", "confidence", ...}, ...]
```

### 3.3 AvonetFilter 离线过滤

```python
from birdid.avonet_filter import AvonetFilter

filter = AvonetFilter()
# 给定 GPS 坐标，获取该区域可能出现的物种 class_ids
species_ids = filter.get_species_by_location(lat=3.1, lon=101.7)
# 返回约 400+ class_ids（吉隆坡区域）
```

数据库结构：
- `distributions`: 3.3M 地理分布记录
- `places`: ~16,000 个 1°×1° 经纬度格子
- `sp_cls_map`: 物种 → class_id 映射

---

## 四、验证结果

| 测试项 | 命令 | 结果 |
|--------|------|------|
| 单图识别 | `birdid_cli.py identify photo.NEF` | ✅ 正确 |
| 批量识别 | `birdid_cli.py identify *.NEF` | ✅ 正确 |
| GPS 自动过滤 | 读取 EXIF GPS → AvonetFilter | ✅ 正常 |
| CLI 选片 | `superpicky_cli.py process dir --auto-identify` | ✅ 正常 |

### TTA 测试结论（已放弃）

| 测试文件 | 实际物种 | 普通预测 | TTA 预测 | 结论 |
|----------|----------|----------|----------|------|
| _Z9W0659.NEF | 蛇雕 | 蛇雕 26.6% | **船嘴鹭 87.2%** ❌ | TTA 产生错误高置信度 |

TTA 虽然提升了置信度数值，但方向可能是错误的，已回滚。

---

## 五、待完成工作

### P0 - 建议尽快完成

- [ ] **更多物种测试**: 目前主要在马来西亚鸟类上测试。建议在澳洲、中国、北美照片上各跑 20+ 张验证。
- [ ] **GUI 端到端测试**: CLI 全部通过，GUI 选片需人工验证。

### P1 - 可后续处理

- [ ] **清理旧模型代码**: `bird_identifier.py` 中仍保留 `decrypt_model()` 等旧代码作为回退。确认 OSEA 稳定后可移除。
- [ ] **置信度阈值微调**: 当前 Temperature=0.9，如用户反馈置信度仍过高/过低可继续调整。
- [ ] **代码整合**: `osea_classifier.py` 与 `bird_identifier.py` 有部分重复逻辑，可考虑整合。

### P2 - 可选优化

- [ ] **BirdID RESULTS 面板**: 选片完成后显示统计摘要（识别物种数/星级分布）。
- [ ] **PyInstaller 打包更新**: 确认 `model20240824.pth` 和 `avonet.db` 正确打包。

---

## 六、关键文件导航

```
SuperPicky2026/
├── birdid/
│   ├── bird_identifier.py       # 核心识别入口 (identify_bird → predict_bird)
│   ├── osea_classifier.py       # OSEA 独立分类器
│   ├── avonet_filter.py         # 离线 GPS 物种过滤
│   ├── ebird_country_filter.py  # [deprecated] 旧 eBird API 过滤
│   └── data/
│       ├── birdinfo.json        # 原始物种信息
│       ├── osea_bird_info.json  # OSEA 物种信息
│       ├── bird_reference.sqlite # 物种数据库
│       └── avonet.db            # 地理分布数据库 (Git LFS)
├── models/
│   └── model20240824.pth        # OSEA ResNet34 权重 (Git LFS)
├── birdid_cli.py                # CLI 识别工具
├── superpicky_cli.py            # CLI 选片工具
├── ui/
│   └── birdid_dock.py           # GUI 鸟种识别面板
├── core/
│   └── photo_processor.py       # 批量选片流程
└── birdid_server.py             # API Server (Lightroom 插件)
```

---

## 七、如何验证

```bash
# 1. 确保大文件存在
ls -la models/model20240824.pth   # 应为 103MB
ls -la birdid/data/avonet.db      # 应为 102MB

# 2. CLI 测试
python birdid_cli.py identify test_photo.jpg --top 3
# 应看到 "[BirdID] OSEA ResNet34 模型已加载"

# 3. GUI 测试
python main.py
# 打开照片 → BirdID 面板 → 点击识别
```

---

## 八、相关 PR

- **PR #17** (已合并): 修正 Windows EXIF 标题写入乱码 + CUDA 打包问题

---

## 九、联系信息

如有问题，请参考：
- `docs/plans/2026-02-16-avonet-implementation-plan.md` - AvonetFilter 实现计划
- `docs/plans/2026-02-16-avonet-replace-ebird-design.md` - eBird 替换设计文档
