# SuperPicky - AI Bird Photo Culling Tool 🦅

[![Version](https://img.shields.io/badge/version-4.0.6-blue.svg)](https://github.com/jamesphotography/SuperPicky)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Windows-lightgrey.svg)](https://github.com/jamesphotography/SuperPicky/releases)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

[**中文文档 (Chinese)**](README_zh.md) | [**Release Notes**](RELEASE_NOTES.md)

**Smart AI Culling Tool for Bird Photographers**

Shoot freely, cull easily! A smart photo culling software designed specifically for bird photographers. It uses multi-model AI technology to automatically identify, rate, and filter bird photos, significantly improving post-processing efficiency.

---

## 🌟 Core Features

### 🤖 Multi-Model Synergy
- **YOLO11 Detection**: Precise bird detection and segmentation masks.
- **SuperEyes**: Detects eye visibility and calculates head sharpness.
- **SuperFlier**: Identifies flight poses for bonus points.
- **TOPIQ Aesthetics**: Assesses overall image aesthetics, composition, and lighting.

### ⭐ Smart Rating System (0-3 Stars)
| Stars | Condition | Meaning |
|-------|-----------|---------|
| ⭐⭐⭐ | Sharpness OK + Aesthetics OK | Excellent, worth editing |
| ⭐⭐ | Sharpness OK OR Aesthetics OK | Good, consider keeping |
| ⭐ | Bird found but below threshold | Average, usually delete |
| 0 | No bird / Poor quality | Delete |

### ⚙️ Skill Level Presets (New)
Automatically set thresholds based on your experience:
- **🐣 Beginner**: Sharpness>300, Aesthetics>4.5 (Keep more)
- **📷 Intermediate**: Sharpness>380, Aesthetics>4.8 (Balanced)
- **👑 Master**: Sharpness>520, Aesthetics>5.5 (Strict)

### 🏷️ Special Tags
- **Pick (Flag)**: Top 25% intersection of sharpness & aesthetics among 3-star photos.
- **Flying**: Green label for bird-in-flight photos.
- **Exposure**: Filters over/under-exposed shots (Optional).

### 📂 Auto-Organization
- **Sort by Stars**: Auto-move to 0star/1star/2star/3star folders.
- **EXIF Write**: Writes ratings, flags, and scores to RAW metadata.
- **Lightroom Compatible**: Sort and filter immediately after import.
- **Undo**: One-click reset to restore original state.

---

## 📥 Downloads

### macOS
**Apple Silicon (M1/M2/M3/M4) (v4.0.6 Beta)**
- [GitHub Download](https://github.com/jamesphotography/SuperPicky/releases/download/v4.0.6/SuperPicky_v4.0.6.dmg)
- [Google Drive (Mirror)](https://drive.google.com/file/d/1vwKMcXcZQHYSalOyXg3grOV2wYFu2W8_/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1CR1OsRRorAwC0vI5xqw7Rw?pwd=mix5) Code: mix5

**Intel (Pre-2020 Mac) (v4.0.6 Beta)**
- [GitHub Download](https://github.com/jamesphotography/SuperPicky/releases/download/v4.0.6/SuperPicky_v4.0.6_Intel.dmg)
- [Google Drive (Mirror)](https://drive.google.com/file/d/1eKw_02YlsC9Yrfi1VxOxAX6xSMzIDdQa/view?usp=drive_link)
- [Baidu Netdisk](https://pan.baidu.com/s/1hMW47CCJKaKtjtqgTiep8g?pwd=6cpu) Code: 6cpu

### Windows (v4.0.5)
**CUDA-GPU Version**
- [Google Drive (Mirror)](https://drive.google.com/file/d/17-dFw2pZKXn53zmYAZ7HQNHTyndCT76E/view?usp=drive_link)
- [Baidu Netdisk](https://pan.baidu.com/s/14tnSXnI2LIeZf4egu4xxNg?pwd=jfuz) Code: jfuz
- [Google Drive (Mirror)](https://drive.google.com/file/d/17-dFw2pZKXn53zmYAZ7HQNHTyndCT76E/view?usp=drive_link)


---

## 🚀 Quick Start

1. **Select Folder**: Drag & drop or browse for a folder with bird photos.
2. **Adjust Thresholds** (Optional): Sharpness (200-600), Aesthetics (4.0-7.0).
3. **Toggle Features**: Flight detection, Exposure check.
4. **Start**: Click to begin AI processing.
5. **Review**: Photos are organized; import to Lightroom to see ratings.

---

## 📝 Update Log

### v4.0.5 (2026-02-15)
- 🚀 **Architecture**: SQLite migration, ~1.9x speedup.
- 🌟 **Community**: Thanks @OscarKing888 for Sony ARW & UTF-8 fixes.
- 🧹 **Clean**: Unified temp files to hidden cache dir.
- 🔧 **Fixes**: Chinese path support, ExifTool deadlock, Plugin metadata.

---

## 📄 License

Open sourced under **GPL-3.0 License**.

This project uses:
- **YOLO11** by Ultralytics
- **OSEA** by Sun Jiao (github.com/sun-jiao/osea)
- **TOPIQ** by Chaofeng Chen et al.
