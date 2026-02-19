# OSEA 模型对比测试实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建独立脚本 `compare_models.py`，并行运行两个鸟类分类模型并输出对比结果。

**Architecture:** 加载两个模型（birdid2024 TorchScript + OSEA ResNet34），对同一图片分别推理，以表格形式展示 Top-K 预测对比。

**Tech Stack:** Python, PyTorch, torchvision, PIL, rich (终端表格)

---

### Task 1: 创建基础脚本框架

**Files:**
- Create: `compare_models.py`

**Step 1: 创建脚本文件**

创建包含 argparse 参数解析的基础框架。

**Step 2: 验证脚本可运行**

Run: `python compare_models.py --help`

Expected: 显示帮助信息

**Step 3: Commit**

```bash
git add compare_models.py
git commit -m "feat: add compare_models.py skeleton"
```

---

### Task 2: 加载鸟类信息和标签

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加标签加载函数**

- `load_bird_info()`: 加载 birdinfo.json
- `load_osea_labels()`: 加载 osea_labels.txt

**Step 2: 在 main() 中测试加载**

Run: `python compare_models.py test.jpg`

Expected: 显示物种数量 10964

**Step 3: Commit**

```bash
git add compare_models.py
git commit -m "feat: add bird info and label loading"
```

---

### Task 3: 加载 birdid2024 模型

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加模型解密和加载函数**

- `decrypt_model()`: AES 解密
- `load_birdid_model()`: 加载 TorchScript 模型

**Step 2: 验证**

Run: `python compare_models.py test.jpg`

Expected: 显示 "birdid2024 加载完成"

**Step 3: Commit**

```bash
git add compare_models.py
git commit -m "feat: add birdid2024 model loading with decryption"
```

---

### Task 4: 加载 OSEA ResNet34 模型

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加 OSEA 模型加载函数**

使用 torchvision.models.resnet34(num_classes=11000) 加载权重。

**Step 2: 验证**

Run: `python compare_models.py test.jpg`

Expected: 两个模型都加载成功

**Step 3: Commit**

```bash
git add compare_models.py
git commit -m "feat: add OSEA ResNet34 model loading"
```

---

### Task 5: 实现图像预处理函数

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加预处理函数**

- `preprocess_for_birdid()`: BGR 颜色空间，mean=[0.406, 0.456, 0.485]
- `preprocess_for_osea()`: RGB 颜色空间，mean=[0.485, 0.456, 0.406]

**Step 2: Commit**

```bash
git add compare_models.py
git commit -m "feat: add image preprocessing for both models"
```

---

### Task 6: 实现推理函数

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加推理函数**

- `predict_with_birdid()`: 使用 birdid2024 模型预测
- `predict_with_osea()`: 使用 OSEA 模型预测

返回格式: `[{rank, cn_name, en_name, confidence}, ...]`

**Step 2: Commit**

```bash
git add compare_models.py
git commit -m "feat: add prediction functions for both models"
```

---

### Task 7: 实现终端表格输出

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加表格显示函数**

使用 rich 库：
- `display_comparison()`: 显示对比表格

表格列：#, birdid2024, 置信度, OSEA, 置信度, 匹配(勾/叉)

**Step 2: Commit**

```bash
git add compare_models.py
git commit -m "feat: add rich table display for comparison results"
```

---

### Task 8: 实现主流程

**Files:**
- Modify: `compare_models.py`

**Step 1: 完善 main() 函数**

- `get_image_files()`: 获取图片列表（支持文件或目录）
- 主循环：逐张处理，显示对比，统计 Top-1 一致率

**Step 2: 验证完整流程**

Run: `python compare_models.py /path/to/test_image.jpg`

Expected: 显示对比表格

**Step 3: Commit**

```bash
git add compare_models.py
git commit -m "feat: complete compare_models.py main workflow"
```

---

### Task 9: 添加 RAW 格式支持 (可选)

**Files:**
- Modify: `compare_models.py`

**Step 1: 添加 RAW 加载支持**

- `load_image()`: 检测 RAW 扩展名，使用 rawpy 处理

**Step 2: 更新扩展名列表**

添加 .arw, .cr2, .cr3, .nef, .raf, .rw2, .dng

**Step 3: Commit**

```bash
git add compare_models.py
git commit -m "feat: add RAW format support to compare_models"
```

---

## 关键代码参考

### OSEA 预处理 (来自 osea.py)

```python
classify_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### birdid2024 预处理 (来自 bird_identifier.py)

```python
# BGR 顺序
mean = np.array([0.406, 0.456, 0.485])
std = np.array([0.225, 0.224, 0.229])
normalized = (bgr_array / 255.0 - mean) / std
```

### OSEA 模型加载 (来自 osea.py)

```python
model = models.resnet34(num_classes=11000)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
```
