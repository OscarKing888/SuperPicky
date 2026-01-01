#!/bin/bash
# SuperPicky V3.7.0 - 简单打包脚本（不签名，仅用于测试）
# 作者: James Zhen Yu
# 日期: 2025-12-31

set -e  # 遇到错误立即退出

# ============================================
# 配置参数
# ============================================
VERSION="3.7.0"
APP_NAME="SuperPicky"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 辅助函数
# ============================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# 步骤1：清理旧文件
# ============================================
log_info "步骤1：清理旧的build和dist目录..."
rm -rf build dist
mkdir -p dist
log_success "清理完成"

# ============================================
# 步骤2：激活虚拟环境
# ============================================
log_info "步骤2：激活虚拟环境..."
source .venv/bin/activate
log_success "虚拟环境已激活"

# ============================================
# 步骤3：使用PyInstaller打包
# ============================================
log_info "步骤3：使用PyInstaller打包应用..."
pyinstaller SuperPicky.spec --clean --noconfirm

if [ ! -d "dist/${APP_NAME}.app" ]; then
    log_error "打包失败！未找到 dist/${APP_NAME}.app"
    exit 1
fi
log_success "PyInstaller打包完成"

# ============================================
# 步骤4：创建DMG安装包（不签名）
# ============================================
log_info "步骤4：创建DMG安装包..."
DMG_NAME="${APP_NAME}_v${VERSION}_unsigned.dmg"
DMG_PATH="dist/${DMG_NAME}"

# 删除旧的DMG
rm -f "${DMG_PATH}"

# 创建临时DMG文件夹
TEMP_DMG_DIR="dist/dmg_temp"
rm -rf "${TEMP_DMG_DIR}"
mkdir -p "${TEMP_DMG_DIR}"

# 复制应用到临时文件夹
cp -R "dist/${APP_NAME}.app" "${TEMP_DMG_DIR}/"

# 创建Applications快捷方式
ln -s /Applications "${TEMP_DMG_DIR}/Applications"

# 创建DMG（使用hdiutil）
log_info "  使用hdiutil创建DMG..."
hdiutil create -volname "${APP_NAME}" -srcfolder "${TEMP_DMG_DIR}" -ov -format UDZO "${DMG_PATH}"

# 清理临时文件夹
rm -rf "${TEMP_DMG_DIR}"
log_success "DMG创建完成: ${DMG_PATH}"

# ============================================
# 完成
# ============================================
log_success "================================================"
log_success "🎉 打包完成！（未签名版本，仅供测试）"
log_success "================================================"
log_info "应用路径: dist/${APP_NAME}.app"
log_info "DMG路径: ${DMG_PATH}"
log_info ""
log_warning "⚠️  注意: 这是未签名版本"
log_warning "  - 在另一台Mac上首次运行时需要右键->打开"
log_warning "  - 或在系统偏好设置中允许运行"
log_success "================================================"

# 显示文件大小
log_info "文件大小:"
ls -lh "${DMG_PATH}"
