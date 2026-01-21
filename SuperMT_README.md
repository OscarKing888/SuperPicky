# SuperMT - 鸟类百科翻译项目

## 项目概述
使用腾讯 **Hunyuan-MT-7B** 翻译模型，将英文维基百科鸟类摘要翻译成中文。

---

## 环境配置

### 1. 创建虚拟环境
```bash
cd /Users/jameszhenyu/Documents/JamesAPPS/SuperMT
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install transformers==4.56.0 torch accelerate
```

### 3. 模型下载 (已完成)
```bash
huggingface-cli download tencent/Hunyuan-MT-7B
# 模型缓存在: ~/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B
```

---

## 数据库

**源数据库路径**: `../SuperPicky2026/birdid/data/bird_reference.sqlite`

**相关字段**:
| 字段名 | 说明 |
|---|---|
| `wikipedia_intro_en` | 英文维基百科摘要 (数据源) |
| `wikipedia_intro_zh_translated` | 翻译后的中文 (目标) |

---

## 翻译脚本

### translate_wiki.py

```python
#!/usr/bin/env python3
"""Wikipedia Translation - Hunyuan-MT-7B for Apple Silicon M3 Max"""

import sqlite3
import time
import logging
from datetime import datetime

DB_PATH = '../SuperPicky2026/birdid/data/bird_reference.sqlite'
LOG_FILE = 'translation.log'
MODEL_PATH = "tencent/Hunyuan-MT-7B"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

_model = _tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        logging.info("Loading Hunyuan-MT-7B model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16, 
            device_map="mps"  # Apple Silicon
        )
        logging.info("Model loaded successfully!")
    return _model, _tokenizer

def translate(text):
    """Translate English to Chinese using official prompt template."""
    model, tokenizer = load_model()
    messages = [
        {"role": "user", "content": f"Translate the following segment into Chinese, without additional explanation.\n\n{text}"}
    ]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        tokenized, 
        max_new_tokens=1024, 
        top_k=20, 
        top_p=0.6, 
        temperature=0.7,
        repetition_penalty=1.05,
        do_sample=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract translation (after the prompt)
    parts = result.split("\n\n")
    return parts[-1].strip() if len(parts) > 1 else result

def main():
    start = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Ensure column exists
    c.execute("PRAGMA table_info(BirdCountInfo)")
    columns = [col[1] for col in c.fetchall()]
    if 'wikipedia_intro_zh_translated' not in columns:
        c.execute("ALTER TABLE BirdCountInfo ADD COLUMN wikipedia_intro_zh_translated TEXT")
        conn.commit()
        logging.info("Added column: wikipedia_intro_zh_translated")
    
    # Get pending records
    c.execute("""
        SELECT id, english_name, wikipedia_intro_en 
        FROM BirdCountInfo 
        WHERE wikipedia_intro_en IS NOT NULL 
          AND wikipedia_intro_en != 'NOT_FOUND' 
          AND (wikipedia_intro_zh_translated IS NULL OR wikipedia_intro_zh_translated = '')
        ORDER BY id ASC
    """)
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        logging.info("All translations complete!")
        return
    
    logging.info(f"Pending translations: {len(rows)}")
    
    for i, (bid, name, en_text) in enumerate(rows, 1):
        try:
            zh = translate(en_text)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE BirdCountInfo SET wikipedia_intro_zh_translated = ? WHERE id = ?", (zh, bid))
            conn.commit()
            conn.close()
            logging.info(f"[{i}/{len(rows)}] ✅ ID {bid}: {name[:30]}...")
        except Exception as e:
            logging.error(f"[{i}/{len(rows)}] ❌ ID {bid}: {e}")
    
    elapsed = datetime.now() - start
    logging.info(f"Done! Duration: {elapsed}")

if __name__ == "__main__":
    main()
```

---

## 运行命令

### 前台运行 (可实时观察)
```bash
source .venv/bin/activate
python3 translate_wiki.py
```

### 后台运行 (长时间执行)
```bash
source .venv/bin/activate
nohup python3 translate_wiki.py > translation.log 2>&1 &
```

### 查看进度
```bash
tail -f translation.log
```

### 查看统计
```bash
sqlite3 ../SuperPicky2026/birdid/data/bird_reference.sqlite "
SELECT '待翻译' as status, COUNT(*) FROM BirdCountInfo 
  WHERE wikipedia_intro_en IS NOT NULL AND wikipedia_intro_en != 'NOT_FOUND' 
    AND (wikipedia_intro_zh_translated IS NULL OR wikipedia_intro_zh_translated = '')
UNION
SELECT '已翻译', COUNT(*) FROM BirdCountInfo 
  WHERE wikipedia_intro_zh_translated IS NOT NULL AND wikipedia_intro_zh_translated != '';
"
```

---

## 性能预估

- **硬件**: Apple M3 Max + 48GB 统一内存
- **模型**: Hunyuan-MT-7B (float16)
- **速度**: 约 5-15 tokens/秒
- **单条翻译**: 约 3-10 秒
- **全量翻译 (~10,000条)**: 约 8-24 小时

---

## 注意事项

1. **断点续传**: 脚本自动跳过已翻译的记录
2. **防休眠**: 长时间运行请使用 `caffeinate -i python3 translate_wiki.py`
3. **显存**: 约占用 14-16GB 统一内存
4. **数据库锁**: 翻译和Wiki抓取可同时运行（SQLite支持并发读）
