# OSEA 模型对比测试设计

## 概述

集成 sun-jiao/osea 项目的 ResNet34 鸟类分类模型，与现有 birdid2024 模型进行并行对比测试，评估识别准确性差异。

## 背景

| 项目 | 模型 | 物种数 | 训练数据 |
|------|------|--------|----------|
| SuperPicky | TorchScript (加密) | 10964 | DIB-10K |
| OSEA | ResNet34 | 10964 | DIB-10K + MetaFGNet pretrain |

两个模型标签顺序完全一致，无需额外映射。

## 方案

采用独立对比脚本 `compare_models.py`，不修改现有代码。

### 文件结构

```
SuperPicky2026/
├── compare_models.py              # 对比测试脚本
├── models/
│   └── model20240824.pth          # OSEA ResNet34 权重 (103MB)
└── birdid/data/
    └── osea_labels.txt            # OSEA 标签文件
```

### 技术规格

| 参数 | 原模型 | OSEA |
|------|--------|------|
| 架构 | TorchScript | ResNet34 |
| 输入尺寸 | 224×224 | 224×224 |
| 颜色空间 | BGR | RGB |
| 正规化 mean | [0.406, 0.456, 0.485] | [0.485, 0.456, 0.406] |
| 正规化 std | [0.225, 0.224, 0.229] | [0.229, 0.224, 0.225] |

### 输出格式

终端表格显示两个模型的 Top-5 预测结果对比。

### 使用方式

```bash
python compare_models.py /path/to/image.jpg
python compare_models.py /path/to/images/ --top-k 3
```

## 参考代码

- OSEA CLI: https://github.com/sun-jiao/osea/blob/main/osea_cli/osea.py
- OSEA Mobile AI: https://github.com/sun-jiao/osea_mobile/blob/main/lib/tools/ai_tools.dart

## 后续计划

如测试效果良好，将正式集成 OSEA 模型替换现有分类模型。
