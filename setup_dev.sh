#!/bin/bash
# SuperPicky 环境设置脚本 (macOS)
# 用于创建 Python 虚拟环境并安装依赖

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "SuperPicky 环境设置脚本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查 Python 版本
print_info "检查 Python 版本..."
PYTHON_CMD=""
PYTHON_VERSION=""

# 尝试不同的 Python 命令
for cmd in python3.12 python3.13 python3 python; do
    if command -v "$cmd" &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 12 ]; then
            PYTHON_CMD="$cmd"
            PYTHON_VERSION="$VERSION"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "未找到 Python 3.12 或更高版本"
    print_info "请安装 Python 3.12+："
    echo "  - 使用 Homebrew: brew install python@3.12"
    echo "  - 或从官网下载: https://www.python.org/downloads/"
    exit 1
fi

print_success "找到 Python: $PYTHON_VERSION ($PYTHON_CMD)"

# 检查虚拟环境目录
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    print_warning "虚拟环境目录已存在: $VENV_DIR"
    read -p "是否删除并重新创建? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除现有虚拟环境..."
        rm -rf "$VENV_DIR"
        print_success "已删除旧虚拟环境"
    else
        print_info "使用现有虚拟环境"
    fi
fi

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    print_info "创建 Python 虚拟环境..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    print_success "虚拟环境创建成功: $VENV_DIR"
fi

# 激活虚拟环境
print_info "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级 pip
print_info "升级 pip..."
pip install --upgrade pip --quiet
print_success "pip 已升级"

# 检查 requirements.txt
if [ ! -f "requirements.txt" ]; then
    print_error "未找到 requirements.txt 文件"
    exit 1
fi

# 安装依赖
print_info "安装依赖包（这可能需要几分钟）..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if pip install -r requirements.txt; then
    print_success "所有依赖安装完成！"
else
    print_error "依赖安装失败"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_success "环境设置完成！"
echo ""
print_info "使用方法："
echo "  1. 激活虚拟环境:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. 运行 CLI 工具:"
echo "     python superpicky_cli.py process /path/to/photos"
echo ""
echo "  3. 退出虚拟环境:"
echo "     deactivate"
echo ""