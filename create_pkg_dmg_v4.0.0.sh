#!/bin/bash
# SuperPicky V4.0.0 - PKG + DMG 完整打包脚本
# 包含: PyInstaller打包 → PKG组件 → Distribution PKG → DMG → 签名公证
# 特色: 自动安装 Lightroom 插件
# 作者: James Zhen Yu
# 日期: 2026-01-18

set -e  # 遇到错误立即退出

# ============================================
# 配置参数
# ============================================
VERSION="4.0.2"
APP_NAME="SuperPicky"
APP_NAME_CN="慧眼选鸟"
BUNDLE_ID="com.jamesphotography.superpicky"
DEVELOPER_ID="Developer ID Application: James Zhen Yu (JWR6FDB52H)"
INSTALLER_ID="Developer ID Installer: James Zhen Yu (JWR6FDB52H)"
APPLE_ID="james@jamesphotography.com.au"
TEAM_ID="JWR6FDB52H"
APP_PASSWORD=$(security find-generic-password -a "${APPLE_ID}" -s "SuperPicky-Notarize" -w)

PKG_NAME="${APP_NAME}_v${VERSION}_Installer.pkg"
DMG_NAME="${APP_NAME}_v${VERSION}.dmg"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# ============================================
# 辅助函数
# ============================================
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${PURPLE}$1${NC}"; echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ============================================
# 步骤1: 清理和准备
# ============================================
log_step "步骤 1/8: 清理旧构建"

rm -rf build dist pkg_root pkg_scripts
rm -f *.pkg *.dmg distribution.xml welcome.html conclusion.html
mkdir -p dist

log_success "清理完成"

# ============================================
# 步骤2: PyInstaller 打包
# ============================================
log_step "步骤 2/8: PyInstaller 打包应用"

log_info "激活虚拟环境..."
source .venv/bin/activate

log_info "开始 PyInstaller 打包..."
pyinstaller SuperPicky.spec --clean --noconfirm

if [ ! -d "dist/${APP_NAME}.app" ]; then
    log_error "打包失败！未找到 dist/${APP_NAME}.app"
    exit 1
fi

# 创建 .app bundle
log_info "创建 macOS 应用包..."
APP_PATH="dist/${APP_NAME}.app"

# 创建 BUNDLE 结构（如果需要）
mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

# 移动可执行文件和资源
if [ -d "dist/${APP_NAME}" ] && [ ! -f "${APP_PATH}/Contents/MacOS/${APP_NAME}" ]; then
    mv dist/${APP_NAME}/* "${APP_PATH}/Contents/MacOS/"
fi

log_success "PyInstaller 打包完成"

# 移动资源到 Contents/Resources
log_info "整理资源文件..."
if [ -d "${APP_PATH}/Contents/MacOS/SuperBirdIDPlugin.lrplugin" ]; then
    log_info "  移动 Lightroom 插件到 Resources..."
    mv "${APP_PATH}/Contents/MacOS/SuperBirdIDPlugin.lrplugin" "${APP_PATH}/Contents/Resources/"
fi

if [ -d "${APP_PATH}/Contents/MacOS/en.lproj" ]; then
    log_info "  移动 en.lproj 到 Resources..."
    mv "${APP_PATH}/Contents/MacOS/en.lproj" "${APP_PATH}/Contents/Resources/"
fi

if [ -d "${APP_PATH}/Contents/MacOS/zh-Hans.lproj" ]; then
    log_info "  移动 zh-Hans.lproj 到 Resources..."
    mv "${APP_PATH}/Contents/MacOS/zh-Hans.lproj" "${APP_PATH}/Contents/Resources/"
fi

# ============================================
# 步骤3: 代码签名
# ============================================
log_step "步骤 3/8: 代码签名"

log_info "签名嵌入的库和框架..."
find "${APP_PATH}/Contents" -type f \( -name "*.dylib" -o -name "*.so" -o -perm +111 \) \
    -exec codesign --force --sign "${DEVELOPER_ID}" --timestamp --options runtime {} \; 2>/dev/null || true

log_info "签名主应用..."
codesign --force --deep --sign "${DEVELOPER_ID}" \
    --timestamp \
    --options runtime \
    --entitlements entitlements.plist \
    "${APP_PATH}"

log_info "验证签名..."
codesign --verify --deep --strict --verbose=2 "${APP_PATH}"

log_success "代码签名完成"

# ============================================
# 步骤4: 创建 PKG 组件包
# ============================================
log_step "步骤 4/8: 创建 PKG 组件包"

mkdir -p pkg_root/Applications
mkdir -p pkg_scripts

# 复制应用（重命名为中文名）
log_info "复制应用到安装目录..."
ditto "${APP_PATH}" "pkg_root/Applications/${APP_NAME_CN}.app"

# 创建 postinstall 脚本
log_info "创建 postinstall 脚本..."
cat > pkg_scripts/postinstall << 'POSTINSTALL_EOF'
#!/bin/bash
# SuperPicky V4.0.2 - 安装后配置脚本

echo "正在配置 慧眼选鸟 SuperPicky V4.0.2..."

APP_PATH="/Applications/慧眼选鸟.app"

# 获取真实用户（而非 root）
REAL_USER=$(stat -f '%Su' /dev/console)
REAL_HOME=$(eval echo ~$REAL_USER)

echo "安装用户: $REAL_USER"
echo "用户主目录: $REAL_HOME"

# 1. 设置应用权限
chmod -R 755 "$APP_PATH"
echo "✓ 应用权限已设置"

# 2. 设置 ExifTool 可执行权限
EXIFTOOL_PATH="$APP_PATH/Contents/Frameworks/exiftools_mac/exiftool"
if [ -f "$EXIFTOOL_PATH" ]; then
    chmod +x "$EXIFTOOL_PATH"
    echo "✓ ExifTool 权限已设置"
fi

# 3. 设置 ExifTool lib 目录权限
LIB_DIR="$APP_PATH/Contents/Frameworks/exiftools_mac/lib"
if [ -d "$LIB_DIR" ]; then
    chmod -R 755 "$LIB_DIR"
fi

# 4. 安装 Lightroom 插件
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "正在检测 Lightroom 版本..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PLUGIN_SOURCE="$APP_PATH/Contents/Resources/SuperBirdIDPlugin.lrplugin"

# 检测可用的 Lightroom 版本
declare -a LR_OPTIONS
declare -a LR_PATHS
declare -a LR_NAMES

# 用户 Modules 目录（推荐）
USER_MODULES="$REAL_HOME/Library/Application Support/Adobe/Lightroom/Modules"
if [ -d "$REAL_HOME/Library/Application Support/Adobe/Lightroom" ]; then
    LR_OPTIONS+=("Lightroom 用户模块 (推荐)")
    LR_PATHS+=("$USER_MODULES")
    LR_NAMES+=("Lightroom 用户模块")
    echo "  ✓ 发现: Lightroom 用户模块目录"
fi

# Lightroom Classic 应用内 PlugIns（需要 admin）
LR_CLASSIC_PLUGINS="/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/PlugIns"
if [ -d "$LR_CLASSIC_PLUGINS" ]; then
    LR_OPTIONS+=("Lightroom Classic 应用内 (需重启LR)")
    LR_PATHS+=("$LR_CLASSIC_PLUGINS")
    LR_NAMES+=("Lightroom Classic 应用内")
    echo "  ✓ 发现: Lightroom Classic 应用"
fi

# 检测其他可能的 Lightroom 安装
for lr_app in /Applications/Adobe\ Lightroom*/Adobe\ Lightroom*.app/Contents/PlugIns; do
    if [ -d "$lr_app" ] && [[ "$lr_app" != "$LR_CLASSIC_PLUGINS" ]]; then
        app_name=$(basename "$(dirname "$(dirname "$lr_app")")" | sed 's/Adobe //')
        LR_OPTIONS+=("$app_name 应用内")
        LR_PATHS+=("$lr_app")
        LR_NAMES+=("$app_name")
        echo "  ✓ 发现: $app_name"
    fi
done

# 如果没有检测到任何 Lightroom
if [ ${#LR_OPTIONS[@]} -eq 0 ]; then
    echo "⚠ 未检测到 Lightroom 安装"
    echo "插件已保存在应用包内，您可以稍后手动安装"
    echo "插件位置: $PLUGIN_SOURCE"
else
    echo ""
    echo "检测到 ${#LR_OPTIONS[@]} 个可用安装位置"
    
    # 构建 osascript 选项列表
    OPTIONS_STR=""
    for opt in "${LR_OPTIONS[@]}"; do
        if [ -z "$OPTIONS_STR" ]; then
            OPTIONS_STR="\"$opt\""
        else
            OPTIONS_STR="$OPTIONS_STR, \"$opt\""
        fi
    done
    
    # 使用 osascript 弹出多选对话框
    echo "正在显示安装选择对话框..."
    
    SELECTED=$(osascript -e "
        set theChoices to {$OPTIONS_STR}
        set selectedItems to choose from list theChoices with title \"慧眼选鸟 - Lightroom 插件安装\" with prompt \"请选择要安装插件的 Lightroom 版本：
        
(可按住 Command 键多选)\" default items {item 1 of theChoices} with multiple selections allowed
        if selectedItems is false then
            return \"CANCELLED\"
        else
            set AppleScript's text item delimiters to \"|||\"
            return selectedItems as text
        end if
    " 2>/dev/null)
    
    if [ "$SELECTED" = "CANCELLED" ] || [ -z "$SELECTED" ]; then
        echo "用户取消了插件安装"
        echo "您可以稍后从应用包内手动复制插件"
    else
        echo "用户选择: $SELECTED"
        echo ""
        
        INSTALLED_COUNT=0
        
        # 解析用户选择并安装
        IFS='|||' read -ra SELECTED_ITEMS <<< "$SELECTED"
        for selection in "${SELECTED_ITEMS[@]}"; do
            # 查找对应的路径
            for i in "${!LR_OPTIONS[@]}"; do
                if [ "${LR_OPTIONS[$i]}" = "$selection" ]; then
                    TARGET_PATH="${LR_PATHS[$i]}"
                    TARGET_NAME="${LR_NAMES[$i]}"
                    
                    echo "正在安装到: $TARGET_NAME..."
                    
                    # 创建目录（如果不存在）
                    mkdir -p "$TARGET_PATH"
                    
                    # 删除旧版本
                    if [ -d "$TARGET_PATH/SuperBirdIDPlugin.lrplugin" ]; then
                        rm -rf "$TARGET_PATH/SuperBirdIDPlugin.lrplugin"
                    fi
                    
                    # 复制插件
                    if cp -R "$PLUGIN_SOURCE" "$TARGET_PATH/"; then
                        # 设置正确的所有者（用户目录需要）
                        if [[ "$TARGET_PATH" == "$REAL_HOME"* ]]; then
                            chown -R "$REAL_USER" "$TARGET_PATH/SuperBirdIDPlugin.lrplugin"
                        fi
                        echo "  ✓ 已安装到: $TARGET_NAME"
                        INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
                    else
                        echo "  ✗ 安装失败: $TARGET_NAME"
                    fi
                    break
                fi
            done
        done
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✓ Lightroom 插件安装完成 (共 $INSTALLED_COUNT 个位置)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
fi

# 6. 安装 Lightroom 导出预设
echo ""
echo "正在安装 Lightroom 导出预设..."
PRESET_SOURCE="$APP_PATH/Contents/Resources/SuperBirdIDPlugin.lrplugin/SuperPicky.lrtemplate"
PRESET_DIR="$REAL_HOME/Library/Application Support/Adobe/Lightroom/Export Presets/User Presets"

if [ -f "$PRESET_SOURCE" ]; then
    mkdir -p "$PRESET_DIR"
    cp "$PRESET_SOURCE" "$PRESET_DIR/"
    chown "$REAL_USER" "$PRESET_DIR/SuperPicky.lrtemplate"
    echo "✓ 导出预设已安装到: $PRESET_DIR"
else
    echo "⚠ 未找到导出预设文件，跳过"
fi

# 7. 清除隔离标记
xattr -cr "$APP_PATH" 2>/dev/null || true
echo "✓ 隔离标记已清除"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 慧眼选鸟 SuperPicky V4.0.2 安装完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 应用位置: /Applications/慧眼选鸟.app"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  Lightroom 插件首次使用说明："
echo ""
echo "   1. 打开 Lightroom → 文件 → 增效工具管理器"
echo "   2. 在左侧列表找到 SuperPicky BirdID Plugin"
echo "   3. 点击右侧「启用」按钮"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  Lightroom Plugin First-time Setup:"
echo ""
echo "   1. Open Lightroom → File → Plug-in Manager"
echo "   2. Find 'SuperPicky BirdID Plugin' in the list"
echo "   3. Click the 'Enable' button on the right"
echo ""

exit 0
POSTINSTALL_EOF

chmod +x pkg_scripts/postinstall

# 创建组件 plist 禁用 relocation（防止应用被安装到错误位置）
log_info "创建组件 plist (禁用 relocation)..."
cat > pkg_components.plist << 'COMPONENT_PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
    <dict>
        <key>BundleHasStrictIdentifier</key>
        <true/>
        <key>BundleIsRelocatable</key>
        <false/>
        <key>BundleIsVersionChecked</key>
        <false/>
        <key>BundleOverwriteAction</key>
        <string>upgrade</string>
        <key>RootRelativeBundlePath</key>
        <string>Applications/慧眼选鸟.app</string>
    </dict>
</array>
</plist>
COMPONENT_PLIST_EOF

# 构建组件包
log_info "构建 PKG 组件包..."
pkgbuild --root pkg_root \
    --scripts pkg_scripts \
    --component-plist pkg_components.plist \
    --identifier "${BUNDLE_ID}" \
    --version "${VERSION}" \
    --install-location "/" \
    "${APP_NAME}-component.pkg"

log_success "PKG 组件包创建完成"

# ============================================
# 步骤5: 创建 Distribution PKG
# ============================================
log_step "步骤 5/8: 创建 Distribution PKG"

# 创建欢迎页面
cat > welcome.html << 'WELCOME_EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        /* 支持深色和浅色模式 - 使用透明背景避免白底问题 */
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; 
            padding: 20px; 
            line-height: 1.6; 
            background: transparent; 
            color: #1a1a1a; 
        }
        @media (prefers-color-scheme: dark) {
            body { background: transparent; color: #e0e0e0; }
            h1 { color: #f1f5f9; }
            .version { color: #94a3b8; }
            h2, h3 { color: #94a3b8; }
            .highlight { color: #60a5fa; }
            li { color: #d1d5db; }
            p { color: #d1d5db; }
            strong { color: #f1f5f9; }
        }
        h1 { color: #2c3e50; margin-bottom: 5px; }
        .version { color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }
        h2, h3 { color: #34495e; }
        .highlight { color: #3498db; font-weight: bold; }
        ul { padding-left: 20px; }
        li { margin: 8px 0; color: #374151; }
        .new-badge { background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; }
    </style>
</head>
<body>
    <h1>欢迎安装 慧眼选鸟 SuperPicky</h1>
    <p class="version">版本 4.0.2</p>

    <p>本安装程序将在您的计算机上安装 <strong>慧眼选鸟 SuperPicky</strong> 和 <strong>Lightroom 插件</strong>。</p>

    <h2>V4.0.2 新功能 <span class="new-badge">NEW</span></h2>
    <ul>
        <li><span class="highlight">🦜 鸟类识别</span> - AI 自动识别鸟类物种，写入照片元数据</li>
        <li><span class="highlight">📷 Lightroom 插件</span> - 在 Lightroom 中直接识别鸟类</li>
        <li><span class="highlight">🌏 eBird 集成</span> - 基于 GPS 位置的本地鸟类过滤</li>
    </ul>

    <h3>系统要求</h3>
    <ul>
        <li>macOS 11.0 或更高版本</li>
        <li>Apple Silicon (M1/M2/M3/M4) 或 Intel 处理器</li>
        <li>约 2GB 可用磁盘空间</li>
    </ul>

    <p>点击「继续」开始安装。</p>
</body>
</html>
WELCOME_EOF

# 创建完成页面
cat > conclusion.html << 'CONCLUSION_EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        /* 支持深色和浅色模式 - 使用透明背景避免白底问题 */
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; 
            padding: 20px; 
            line-height: 1.6; 
            background: transparent; 
            color: #1a1a1a; 
        }
        @media (prefers-color-scheme: dark) {
            body { background: transparent; color: #e0e0e0; }
            h1 { color: #4ade80; }
            h2 { color: #94a3b8; }
            .success { background: transparent; border-color: #22c55e; color: #4ade80; }
            .success strong { color: #4ade80; }
            .info-box { background: transparent; border-color: #3b82f6; color: #bfdbfe; }
            .info-box strong { color: #60a5fa; }
            .info-box p { color: #d1d5db; }
            .warning { background: transparent; border-color: #f59e0b; color: #fbbf24; }
            .warning strong { color: #fbbf24; }
            .warning p { color: #d1d5db; }
            a { color: #60a5fa; }
            li { color: #d1d5db; }
            p { color: #d1d5db; }
        }
        h1 { color: #27ae60; }
        h2 { color: #34495e; }
        .success { background: transparent; border: 2px solid #27ae60; padding: 15px; border-radius: 5px; margin: 20px 0; color: #27ae60; }
        .info-box { background: transparent; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; color: #1a1a1a; }
        .warning { background: transparent; border-left: 4px solid #f59e0b; padding: 15px; margin: 15px 0; color: #856404; }
        a { color: #3498db; text-decoration: none; }
        li { color: #374151; }
        @media (prefers-color-scheme: dark) {
            li { color: #d1d5db; }
        }
    </style>
</head>
<body>
    <h1>✓ 安装成功</h1>

    <div class="success">
        <strong>慧眼选鸟 SuperPicky V4.0.2</strong> 已成功安装！
    </div>

    <h2>已安装内容</h2>
    <div class="info-box">
        <p><strong>📍 主应用:</strong> /Applications/慧眼选鸟.app</p>
        <p><strong>📍 Lightroom 插件:</strong> ~/Library/Application Support/Adobe/Lightroom/Modules/</p>
    </div>

    <h2>开始使用</h2>
    <div class="info-box">
        <p><strong>主应用:</strong></p>
        <ul>
            <li>从「启动台」找到并点击「慧眼选鸟」</li>
            <li>或前往「应用程序」文件夹</li>
        </ul>
        <p><strong>Lightroom 插件:</strong></p>
        <ul>
            <li>打开 Lightroom，选中一张照片</li>
            <li>菜单: 图库 → 增效工具 → 慧眼选鸟 - 识别当前照片</li>
        </ul>
    </div>

    <div class="warning">
        <p><strong>⚠️ 首次使用注意:</strong></p>
        <ul>
            <li>首次运行可能需要 10-30 秒加载 AI 模型</li>
            <li>使用 Lightroom 插件前需先启动主应用</li>
            <li><strong>Lightroom 插件需手动启用:</strong> 文件 → 增效工具管理器 → 找到「慧眼选鸟」→ 点击「启用」</li>
        </ul>
    </div>

    <p style="margin-top: 30px; font-size: 0.9em;">
        感谢使用慧眼选鸟！如有问题请访问 <a href="https://github.com/jamesphotography/SuperPicky">GitHub</a>
    </p>
</body>
</html>
CONCLUSION_EOF

# 创建 Distribution XML
cat > distribution.xml << DISTRIBUTION_EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>慧眼选鸟 SuperPicky</title>
    <organization>com.jamesphotography</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="false" hostArchitectures="arm64,x86_64"/>

    <welcome file="welcome.html" mime-type="text/html"/>
    <license file="LICENSE.txt" mime-type="text/plain"/>
    <conclusion file="conclusion.html" mime-type="text/html"/>

    <choices-outline>
        <line choice="default">
            <line choice="${BUNDLE_ID}"/>
        </line>
    </choices-outline>

    <choice id="default"/>
    <choice id="${BUNDLE_ID}" visible="false">
        <pkg-ref id="${BUNDLE_ID}"/>
    </choice>

    <pkg-ref id="${BUNDLE_ID}" version="${VERSION}" onConclusion="none">
        ${APP_NAME}-component.pkg
    </pkg-ref>
</installer-gui-script>
DISTRIBUTION_EOF

# 构建最终 PKG
log_info "构建 Distribution PKG..."
productbuild --distribution distribution.xml \
    --resources . \
    --package-path . \
    "${PKG_NAME}"

log_success "Distribution PKG 创建完成"

# ============================================
# 步骤6: 签名 PKG
# ============================================
log_step "步骤 6/8: 签名 PKG"

log_info "签名 PKG 安装包..."
productsign --sign "${INSTALLER_ID}" "${PKG_NAME}" "${PKG_NAME/.pkg/-signed.pkg}"
mv "${PKG_NAME/.pkg/-signed.pkg}" "${PKG_NAME}"

log_info "验证 PKG 签名..."
pkgutil --check-signature "${PKG_NAME}"

log_success "PKG 签名完成"

# ============================================
# 步骤7: 创建 DMG
# ============================================
log_step "步骤 7/8: 创建 DMG"

TEMP_DMG_DIR="dist/dmg_temp"
rm -rf "${TEMP_DMG_DIR}"
mkdir -p "${TEMP_DMG_DIR}"

# 复制 PKG 到 DMG
cp "${PKG_NAME}" "${TEMP_DMG_DIR}/"

# 复制 Lightroom 插件副本（供手动安装）
log_info "复制 Lightroom 插件副本..."
cp -R "SuperBirdIDPlugin.lrplugin" "${TEMP_DMG_DIR}/"

# 生成 PDF 安装指南
log_info "生成 PDF 安装指南..."
if [ -f "docs/安装指南_v4.0.0.html" ]; then
    # 使用 cupsfilter 或 wkhtmltopdf 生成 PDF（如果可用）
    # 备选：直接复制 HTML，用户可用浏览器打印为 PDF
    cp "docs/安装指南_v4.0.0.html" "${TEMP_DMG_DIR}/安装指南.html"
    log_info "  已复制 HTML 安装指南（可在浏览器中打印为 PDF）"
fi

# 创建网站使用教程快捷方式
log_info "创建网站快捷方式..."
cat > "${TEMP_DMG_DIR}/在线使用教程.webloc" << 'WEBLOC_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>URL</key>
    <string>https://github.com/jamesphotography/SuperPicky</string>
</dict>
</plist>
WEBLOC_EOF

# 创建插件安装说明
cat > "${TEMP_DMG_DIR}/Lightroom插件手动安装说明.txt" << 'PLUGIN_README_EOF'
慧眼选鸟 Lightroom 插件 - 手动安装说明
==========================================

如果自动安装没有成功，或者您使用的是 Lightroom Classic 以外的版本，
请按照以下步骤手动安装插件：

【Lightroom Classic】
  1. 复制 SuperBirdIDPlugin.lrplugin 文件夹到:
     ~/Library/Application Support/Adobe/Lightroom/Modules/

【Lightroom Classic (旧版)】
  1. 打开 Lightroom → 文件 → 增效工具管理器
  2. 点击「添加」
  3. 选择 SuperBirdIDPlugin.lrplugin 文件夹
  4. 重启 Lightroom

【使用方法】
  1. 先启动「慧眼选鸟」主程序，开启识鸟 API
  2. 在 Lightroom 中选中一张照片
  3. 菜单: 图库 → 增效工具 → 慧眼选鸟 - 识别当前照片

【注意事项】
  - 使用插件前需要先启动主程序
  - 确保主程序的「识鸟 API」开关已开启

==========================================
版本: 4.0.2
© 2026 James Zhen Yu
PLUGIN_README_EOF

# 创建总说明文件
cat > "${TEMP_DMG_DIR}/安装说明.txt" << README_EOF
慧眼选鸟 SuperPicky V4.0.2 安装说明
=====================================

【推荐安装方式】
双击「${PKG_NAME}」按向导安装
  - 会自动安装主应用到 /Applications
  - 会自动安装 Lightroom 插件

【手动安装 Lightroom 插件】
如果自动安装失败，请参考「Lightroom插件手动安装说明.txt」
或直接将 SuperBirdIDPlugin.lrplugin 文件夹复制到:
  ~/Library/Application Support/Adobe/Lightroom/Modules/

【首次使用】
  - 从启动台打开「慧眼选鸟」
  - Lightroom 插件: 图库 → 增效工具 → 慧眼选鸟

【问题反馈】
https://github.com/jamesphotography/SuperPicky

=====================================
© 2026 James Zhen Yu
README_EOF

# 创建 DMG
log_info "创建 DMG 镜像..."
hdiutil create -volname "${APP_NAME_CN}" \
    -srcfolder "${TEMP_DMG_DIR}" \
    -ov -format UDZO \
    "dist/${DMG_NAME}"

rm -rf "${TEMP_DMG_DIR}"

log_success "DMG 创建完成"

# ============================================
# 步骤8: 公证
# ============================================
log_step "步骤 8/8: 提交公证"

DMG_PATH="dist/${DMG_NAME}"

log_info "签名 DMG..."
codesign --force --sign "${DEVELOPER_ID}" --timestamp "${DMG_PATH}"

log_info "提交到 Apple 公证服务..."
NOTARIZE_OUTPUT=$(xcrun notarytool submit "${DMG_PATH}" \
    --apple-id "${APPLE_ID}" \
    --password "${APP_PASSWORD}" \
    --team-id "${TEAM_ID}" \
    --wait 2>&1)

echo "${NOTARIZE_OUTPUT}"

if echo "${NOTARIZE_OUTPUT}" | grep -q "status: Accepted"; then
    log_success "公证成功！"
    
    log_info "装订公证票据..."
    xcrun stapler staple "${DMG_PATH}"
    xcrun stapler validate "${DMG_PATH}"
    log_success "公证票据装订完成"
else
    log_warning "公证未完成，请检查输出"
fi

# ============================================
# 清理和总结
# ============================================
log_step "清理临时文件"

rm -rf pkg_root pkg_scripts
rm -f "${APP_NAME}-component.pkg" distribution.xml welcome.html conclusion.html pkg_components.plist

log_success "清理完成"

# ============================================
# 完成
# ============================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 SuperPicky V${VERSION} 打包完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "📦 DMG 安装包: ${BLUE}dist/${DMG_NAME}${NC}"
echo -e "📦 PKG 安装包: ${BLUE}${PKG_NAME}${NC}"
echo ""
echo -e "文件大小:"
ls -lh "dist/${DMG_NAME}" "${PKG_NAME}" 2>/dev/null || true
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
