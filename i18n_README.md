# SuperPicky 国际化(i18n)实现说明

## ✅ 已完成功能

### 1. 核心基础设施
- ✅ 创建了 `i18n.py` 国际化加载器
- ✅ 支持自动检测系统语言（macOS, Linux, Windows）
- ✅ 单例模式管理i18n实例
- ✅ 支持参数化翻译（如 `{count}`, `{filename}` 等）

### 2. 语言包
已创建两个完整语言包：
- ✅ `locales/zh_CN.json` - 简体中文（145+ 翻译条目）
- ✅ `locales/en_US.json` - 英文（完整翻译）

语言包结构：
```json
{
  "_meta": {
    "language_name": "简体中文",
    "language_code": "zh_CN",
    "version": "3.2.0"
  },
  "app": { ... },
  "menu": { ... },
  "buttons": { ... },
  "labels": { ... },
  "messages": { ... },
  "logs": { ... },
  "stats": { ... },
  "post_adjustment": { ... },
  "advanced_settings": { ... },
  "normalization_modes": { ... },
  "tooltips": { ... }
}
```

### 3. 已国际化的文件

#### `main.py` (主窗口)
- ✅ 窗口标题
- ✅ 菜单项（设置、高级设置、帮助、关于）
- ✅ 按钮文本（开始处理、重置目录、二次选鸟）
- ✅ 对话框消息（警告、确认、错误提示）

#### `post_adjustment_dialog.py` (二次选鸟对话框)
- ✅ 对话框标题
- ✅ 描述文本
- ✅ 统计标签
- ✅ 按钮文本（取消、应用）

#### `advanced_settings_dialog.py` (高级设置对话框)
- ✅ 语言选择器UI
- ✅ 显示语言名称而不是代码
- ✅ 正确保存语言代码到配置文件

#### `advanced_config.py` (配置管理)
- ✅ 支持语言设置的保存和加载
- ✅ 默认语言为 `zh_CN`

## 🔧 如何使用

### 用户切换语言的步骤

1. **打开高级设置**
   - 菜单栏 → 设置 → 高级设置

2. **选择语言**
   - 在"语言设置 Language Settings"部分
   - 下拉菜单显示：`简体中文` 或 `English`
   - 选择你想要的语言

3. **保存并重启**
   - 点击"保存"按钮
   - 关闭应用
   - 重新启动应用
   - ✅ 界面将显示为你选择的语言

### 开发者添加新翻译

1. **在语言包中添加翻译**

编辑 `locales/zh_CN.json`:
```json
{
  "my_section": {
    "my_key": "我的文本"
  }
}
```

编辑 `locales/en_US.json`:
```json
{
  "my_section": {
    "my_key": "My Text"
  }
}
```

2. **在代码中使用翻译**

```python
from i18n import get_i18n
from advanced_config import get_advanced_config

# 初始化（通常在 __init__ 中）
config = get_advanced_config()
i18n = get_i18n(config.language)

# 使用翻译
text = i18n.t("my_section.my_key")
```

3. **参数化翻译**

语言包：
```json
{
  "messages": {
    "processing": "正在处理第 {current}/{total} 张照片"
  }
}
```

代码：
```python
message = i18n.t("messages.processing", current=5, total=100)
# 输出: "正在处理第 5/100 张照片"
```

## 🏗️ 架构设计

### 单例模式
`get_i18n()` 使用单例模式，确保整个应用使用同一个i18n实例。这意味着：
- ✅ 性能高效（只加载一次语言包）
- ⚠️ 语言切换需要重启应用才能生效
- ✅ 这是有意的设计选择，用户已确认可接受

### 配置持久化
语言设置保存在 `advanced_config.json`:
```json
{
  ...
  "language": "zh_CN"
}
```

应用启动时：
1. 加载 `advanced_config.json`
2. 读取 `language` 字段
3. 使用该语言初始化 `i18n`
4. 所有UI组件使用 `i18n.t()` 获取翻译

### 自动语言检测
如果配置文件不存在或没有语言设置，系统会自动检测：
1. 环境变量 `LANG`
2. Python `locale` 模块
3. macOS `defaults` 命令
4. 默认使用 `zh_CN`

## 📝 待完成（可选）

以下文件包含大量日志消息，暂未国际化（优先级较低）：
- `find_bird_util.py` - 重置目录的日志消息
- `exiftool_manager.py` - EXIF操作的日志消息
- `ai_model.py` - AI模型处理的日志消息

所有日志的翻译key已经定义在语言包的 `logs` 部分，只需要修改代码调用 `i18n.t()` 即可。

## 🧪 测试

已创建测试脚本：
- `test_language_config.py` - 测试配置的保存和加载
- `test_fresh_load.py` - 模拟全新启动加载语言

运行测试：
```bash
python3 test_fresh_load.py
```

## ✨ 效果演示

### 中文界面
- 窗口标题: "慧眼选鸟 - AI智能照片筛选"
- 按钮: "开始处理", "重置目录", "二次选鸟"
- 菜单: "设置", "帮助"

### 英文界面
- 窗口标题: "SuperPicky - AI Photo Selector"
- 按钮: "Start Processing", "Reset Directory", "Post Adjustment"
- 菜单: "Settings", "Help"

## 🎯 总结

国际化实现已经完成并测试通过！用户可以：
1. 在高级设置中选择语言
2. 保存后重启应用
3. 享受完整的中英文界面

系统架构清晰，易于维护和扩展新语言。
