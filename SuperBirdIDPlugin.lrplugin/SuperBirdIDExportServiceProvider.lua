local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrLogger = import 'LrLogger'
local LrHttp = import 'LrHttp'
local LrPathUtils = import 'LrPathUtils'
local LrFileUtils = import 'LrFileUtils'
local LrView = import 'LrView'
local LrBinding = import 'LrBinding'
local LrFunctionContext = import 'LrFunctionContext'

-- 版本信息
local VERSION = "v4.0.0"
local PLUGIN_NAME = "慧眼选鸟"

local myLogger = LrLogger( 'SuperBirdIDExportServiceProvider' )
myLogger:enable( "logfile" )

-- Binding helper
local bind = LrView.bind

-- Export service provider definition
local exportServiceProvider = {}

-- Required functions for Lightroom SDK
exportServiceProvider.supportsIncrementalPublish = false
exportServiceProvider.canExportVideo = false
exportServiceProvider.exportPresetDestination = "temp"

-- 不需要导出图片，只需获取原图路径
exportServiceProvider.allowFileFormats = nil
exportServiceProvider.allowColorSpaces = nil
exportServiceProvider.hideSections = { 'exportLocation', 'fileNaming', 'fileSettings', 'imageSettings', 'outputSharpening', 'metadata', 'watermarking' }

exportServiceProvider.exportPresetFields = {
    { key = 'apiUrl', default = "http://127.0.0.1:5156" },
    { key = 'topK', default = 3 },
    { key = 'useYolo', default = true },
    { key = 'useGps', default = true },
    { key = 'writeExif', default = true },
}

-- Unicode转义解码辅助函数
local function decodeUnicodeEscape(str)
    if not str then return str end

    -- 将 \uXXXX 转换为 UTF-8
    local function unicodeToUtf8(code)
        code = tonumber(code, 16)
        if code < 0x80 then
            return string.char(code)
        elseif code < 0x800 then
            return string.char(
                0xC0 + math.floor(code / 0x40),
                0x80 + (code % 0x40)
            )
        elseif code < 0x10000 then
            return string.char(
                0xE0 + math.floor(code / 0x1000),
                0x80 + (math.floor(code / 0x40) % 0x40),
                0x80 + (code % 0x40)
            )
        end
        return "?"
    end

    -- 替换所有 \uXXXX 序列
    return str:gsub("\\u(%x%x%x%x)", unicodeToUtf8)
end

-- 简单的JSON解析函数 (支持多个结果)
local function parseJSON(jsonString)
    local result = {}

    -- 提取 success 字段
    local success = string.match(jsonString, '"success"%s*:%s*([^,}]+)')
    if success then
        result.success = (success == "true")
    end

    -- 提取 results 数组中的所有结果
    local resultsBlock = string.match(jsonString, '"results"%s*:%s*%[(.-)%]')
    if resultsBlock then
        result.results = {}

        -- 使用更灵活的模式匹配每个结果对象
        for itemBlock in string.gmatch(resultsBlock, '{([^{}]*)}') do
            local item = {}

            -- 提取字段并解码Unicode
            local cn_name_raw = string.match(itemBlock, '"cn_name"%s*:%s*"([^"]*)"')
            local en_name_raw = string.match(itemBlock, '"en_name"%s*:%s*"([^"]*)"')
            local sci_name_raw = string.match(itemBlock, '"scientific_name"%s*:%s*"([^"]*)"')
            local desc_raw = string.match(itemBlock, '"description"%s*:%s*"([^"]*)"')

            item.cn_name = decodeUnicodeEscape(cn_name_raw)
            item.en_name = decodeUnicodeEscape(en_name_raw)
            item.scientific_name = decodeUnicodeEscape(sci_name_raw)
            item.description = decodeUnicodeEscape(desc_raw)

            local confStr = string.match(itemBlock, '"confidence"%s*:%s*([%d%.]+)')
            item.confidence = confStr and tonumber(confStr) or 0

            local rankStr = string.match(itemBlock, '"rank"%s*:%s*(%d+)')
            item.rank = rankStr and tonumber(rankStr) or (#result.results + 1)

            if item.cn_name then
                table.insert(result.results, item)
            end
        end
    end

    -- 提取 yolo_info (可能包含中文)
    local yolo_raw = string.match(jsonString, '"yolo_info"%s*:%s*"([^"]*)"')
    result.yolo_info = decodeUnicodeEscape(yolo_raw)

    -- 提取 gps_info
    local gpsBlock = string.match(jsonString, '"gps_info"%s*:%s*{(.-)}')
    if gpsBlock then
        result.gps_info = {}

        local lat = string.match(gpsBlock, '"latitude"%s*:%s*([%d%.%-]+)')
        local lon = string.match(gpsBlock, '"longitude"%s*:%s*([%d%.%-]+)')

        result.gps_info.latitude = lat and tonumber(lat) or nil
        result.gps_info.longitude = lon and tonumber(lon) or nil

        local region_raw = string.match(gpsBlock, '"region"%s*:%s*"([^"]*)"')
        local info_raw = string.match(gpsBlock, '"info"%s*:%s*"([^"]*)"')

        result.gps_info.region = decodeUnicodeEscape(region_raw)
        result.gps_info.info = decodeUnicodeEscape(info_raw)
    end

    -- 提取错误信息
    local error_raw = string.match(jsonString, '"error"%s*:%s*"([^"]*)"')
    result.error = decodeUnicodeEscape(error_raw)

    return result
end

-- 简单的JSON编码函数
local function encodeJSON(tbl)
    local parts = {}
    for k, v in pairs(tbl) do
        local key = '"' .. tostring(k) .. '"'
        local value
        if type(v) == "string" then
            value = '"' .. v:gsub('"', '\\"'):gsub('\\', '\\\\') .. '"'
        elseif type(v) == "boolean" then
            value = tostring(v)
        elseif type(v) == "number" then
            value = tostring(v)
        else
            value = '"' .. tostring(v) .. '"'
        end
        table.insert(parts, key .. ":" .. value)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

-- 识别单张照片并返回结果
local function recognizeSinglePhoto(photo, apiUrl, topK, useYolo, useGps)
    local LrHttp = import 'LrHttp'
    local LrFileUtils = import 'LrFileUtils'

    local photoPath = photo:getRawMetadata("path")
    local photoName = photo:getFormattedMetadata("fileName") or "Unknown"

    -- 检查文件是否存在
    if not LrFileUtils.exists(photoPath) then
        return {
            success = false,
            error = "文件不存在: " .. photoName,
            photoName = photoName
        }
    end

    -- 构建API请求
    local requestBody = encodeJSON({
        image_path = photoPath,
        use_yolo = useYolo,
        use_gps = useGps,
        top_k = topK
    })

    -- 调用API
    local response, headers = LrHttp.post(
        apiUrl .. "/recognize",
        requestBody,
        {
            { field = "Content-Type", value = "application/json" }
        }
    )

    if not response then
        return {
            success = false,
            error = "API调用失败",
            photoName = photoName
        }
    end

    -- 解析响应
    local result = parseJSON(response)
    result.photoName = photoName
    result.photo = photo

    return result
end

-- 保存识别结果到照片元数据
-- 写入 Title (鸟名) 和 Caption (学名 + 描述)
local function saveRecognitionResult(photo, species, enName, scientificName, description)
    local catalog = import('LrApplication').activeCatalog()

    -- 构建 Title 内容：中文名 (英文名)
    local title = species .. " (" .. enName .. ")"

    -- 构建 Caption 内容：只写入简介
    local caption = description or ""

    catalog:withWriteAccessDo("保存鸟类识别结果", function()
        photo:setRawMetadata("title", title)
        -- 写入 Caption (学名 + 描述)
        if caption ~= "" then
            photo:setRawMetadata("caption", caption)
        end
    end)
end


-- 显示结果选择对话框（美化版）
local function showResultSelectionDialog(results, photoName)
    local LrView = import 'LrView'
    local LrDialogs = import 'LrDialogs'
    local LrFunctionContext = import 'LrFunctionContext'
    local LrBinding = import 'LrBinding'
    local LrColor = import 'LrColor'

    local selectedIndex = nil

    LrFunctionContext.callWithContext("resultSelectionDialog", function(context)
        local f = LrView.osFactory()
        local props = LrBinding.makePropertyTable(context)

        -- 默认选中第一个
        props.selectedBird = 1

        -- 创建候选鸟种的 radio button 列表
        local candidateViews = {}

        for i, bird in ipairs(results) do
            local confidence = bird.confidence or 0
            local cnName = bird.cn_name or "未知"
            local enName = bird.en_name or ""
            
            -- 置信度颜色提示
            local confColor
            if confidence >= 50 then
                confColor = LrColor(0.2, 0.7, 0.3)  -- 绿色 - 高置信度
            elseif confidence >= 20 then
                confColor = LrColor(0.8, 0.6, 0.1)  -- 橙色 - 中置信度
            else
                confColor = LrColor(0.6, 0.6, 0.6)  -- 灰色 - 低置信度
            end

            -- 每个候选项包含：radio button + 详细信息
            candidateViews[#candidateViews + 1] = f:row {
                spacing = f:control_spacing(),
                
                f:radio_button {
                    title = "",
                    value = bind { key = 'selectedBird', object = props },
                    checked_value = i,
                    width = 20,
                },
                
                f:column {
                    spacing = 2,
                    
                    f:row {
                        f:static_text {
                            title = string.format("%d.", i),
                            font = "<system/bold>",
                            width = 20,
                        },
                        f:static_text {
                            title = cnName,
                            font = "<system/bold>",
                        },
                        f:static_text {
                            title = string.format("  %.1f%%", confidence),
                            text_color = confColor,
                            font = "<system/bold>",
                        },
                    },
                    
                    f:static_text {
                        title = "    " .. enName,
                        text_color = LrColor(0.5, 0.5, 0.5),
                        font = "<system/small>",
                    },
                },
            }
            
            -- 添加分隔线（最后一个候选后不加）
            if i < #results then
                candidateViews[#candidateViews + 1] = f:spacer { height = 6 }
                candidateViews[#candidateViews + 1] = f:separator { fill_horizontal = 1 }
                candidateViews[#candidateViews + 1] = f:spacer { height = 6 }
            end
        end

        -- 添加"跳过"选项（与候选分开）
        candidateViews[#candidateViews + 1] = f:spacer { height = 12 }
        candidateViews[#candidateViews + 1] = f:separator { fill_horizontal = 1 }
        candidateViews[#candidateViews + 1] = f:spacer { height = 8 }
        
        candidateViews[#candidateViews + 1] = f:row {
            f:radio_button {
                title = "",
                value = bind { key = 'selectedBird', object = props },
                checked_value = 0,
                width = 20,
            },
            f:static_text {
                title = "跳过此照片，不写入",
                text_color = LrColor(0.5, 0.5, 0.5),
            },
        }

        -- 构建候选列表容器
        local candidatesGroup = f:column(candidateViews)

        -- 构建完整对话框内容
        local dialogContent = f:column {
            spacing = f:control_spacing(),
            fill_horizontal = 1,

            -- 宽度占位符（确保对话框足够宽）
            f:spacer { width = 350 },

            -- 文件名标题
            f:row {
                f:static_text {
                    title = photoName .. " 的识别结果",
                    font = "<system/bold>",
                },
            },
            
            f:spacer { height = 8 },
            f:separator { fill_horizontal = 1 },
            f:spacer { height = 12 },

            -- 候选列表（不需要提示文字，用户自然理解）
            candidatesGroup,
            
            f:spacer { height = 8 },
        }

        -- 显示对话框（设置更大的宽度）
        local dialogResult = LrDialogs.presentModalDialog({
            title = PLUGIN_NAME,
            contents = dialogContent,
            actionVerb = "写入 EXIF",
            cancelVerb = "取消",
            resizable = true,
        })

        if dialogResult == "ok" then
            selectedIndex = props.selectedBird
        else
            selectedIndex = nil
        end
    end)

    return selectedIndex
end



-- UI配置
function exportServiceProvider.sectionsForTopOfDialog( f, propertyTable )
    local LrView = import 'LrView'
    local bind = LrView.bind

    return {
        {
            title = PLUGIN_NAME .. " API 配置",

            synopsis = bind { key = 'apiUrl', object = propertyTable },

            f:row {
                spacing = f:control_spacing(),

                f:static_text {
                    title = "API 地址:",
                    width = LrView.share "label_width",
                },

                f:edit_field {
                    value = bind 'apiUrl',
                    width_in_chars = 30,
                    tooltip = "慧眼选鸟 API 服务器地址，默认: http://127.0.0.1:5156",
                },
            },

            f:row {
                spacing = f:control_spacing(),

                f:static_text {
                    title = "返回结果数:",
                    width = LrView.share "label_width",
                },

                f:slider {
                    value = bind 'topK',
                    min = 1,
                    max = 10,
                    integral = true,
                    width = 200,
                },

                f:static_text {
                    title = bind 'topK',
                },
            },

            f:row {
                spacing = f:control_spacing(),

                f:checkbox {
                    title = "启用YOLO检测",
                    value = bind 'useYolo',
                    tooltip = "使用YOLO模型预检测鸟类位置",
                },
            },

            f:row {
                spacing = f:control_spacing(),

                f:checkbox {
                    title = "启用GPS定位",
                    value = bind 'useGps',
                    tooltip = "从EXIF读取GPS信息辅助识别",
                },
            },

            f:row {
                spacing = f:control_spacing(),

                f:checkbox {
                    title = "自动写入EXIF",
                    value = bind 'writeExif',
                    checked_value = true,
                    unchecked_value = false,
                    tooltip = "识别成功后自动写入鸟种名称到照片标题",
                },
            },

            f:row {
                spacing = f:control_spacing(),

                f:static_text {
                    title = "提示: 请先启动慧眼选鸟主程序",
                    text_color = import 'LrColor'( 0.5, 0.5, 0.5 ),
                },
            },
        },
    }
end

-- 主要处理函数
function exportServiceProvider.processRenderedPhotos( functionContext, exportContext )
    myLogger:info( PLUGIN_NAME .. " 识别启动 - " .. VERSION )

    local exportSettings = exportContext.propertyTable
    local apiUrl = exportSettings.apiUrl or "http://127.0.0.1:5156"
    local topK = exportSettings.topK or 3
    local useYolo = exportSettings.useYolo
    if useYolo == nil then useYolo = true end
    local useGps = exportSettings.useGps
    if useGps == nil then useGps = true end
    local writeExif = exportSettings.writeExif
    if writeExif == nil then writeExif = true end

    -- 计算照片数量
    local nPhotos = exportContext.nPhotos or 1
    myLogger:info( "待处理照片数: " .. nPhotos )

    -- 限制只处理一张照片
    if nPhotos == 0 then
        LrDialogs.message(PLUGIN_NAME,
            "没有选中要处理的照片\n\n请先选择一张照片再进行识别",
            "error")
        return
    elseif nPhotos > 1 then
        LrDialogs.message(PLUGIN_NAME,
            "一次只能识别一张照片\n\n" ..
            "当前选中: " .. nPhotos .. " 张照片\n\n" ..
            "请重新选择，只选中一张照片后再次导出",
            "warning")
        return
    end

    -- 检查API服务是否可用
    myLogger:info( "检查API服务: " .. apiUrl .. "/health" )
    local healthCheck, headers = LrHttp.get(apiUrl .. "/health")

    if not healthCheck or string.find(healthCheck, '"status"%s*:%s*"ok"') == nil then
        LrDialogs.message(PLUGIN_NAME,
            "无法连接到慧眼选鸟 API 服务\n\n" ..
            "请确保:\n" ..
            "1. 慧眼选鸟主程序已启动\n" ..
            "2. 识鸟 API 服务已开启\n" ..
            "3. 服务地址正确: " .. apiUrl,
            "error")
        return
    end

    myLogger:info( "API服务正常，开始识别..." )

    -- 处理单张照片
    for i, rendition in exportContext:renditions() do
        local photo = rendition.photo
        local result = recognizeSinglePhoto(photo, apiUrl, topK, useYolo, useGps)

        if result.success and result.results and #result.results > 0 then
            myLogger:info( "识别成功，候选数: " .. #result.results )

            -- 显示结果选择对话框
            local selectedIndex = showResultSelectionDialog(result.results, result.photoName)

            if selectedIndex and selectedIndex > 0 then
                -- 用户选择了一个结果
                local selectedBird = result.results[selectedIndex]
                local species = selectedBird.cn_name or "未知"
                local enName = selectedBird.en_name or ""
                local scientificName = selectedBird.scientific_name or ""

                if writeExif then
                    saveRecognitionResult(photo, species, enName, scientificName, selectedBird.description)
                    myLogger:info( "用户选择第" .. selectedIndex .. "名，已写入: " .. species .. " (" .. enName .. ")" )
                    -- 不再弹窗提示，直接完成
                end
            elseif selectedIndex == 0 then
                -- 用户选择跳过
                myLogger:info( "用户选择跳过此照片" )
            else
                -- 用户点击取消
                myLogger:info( "用户取消操作" )
            end

        else
            local errorMsg = result.error or "未知错误"
            myLogger:info( "识别失败: " .. errorMsg )

            -- 美化的错误消息
            local failMsg = "无法识别此照片中的鸟类\n\n" ..
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" ..
                "错误信息:\n" .. errorMsg .. "\n\n" ..
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" ..
                "可能的原因:\n" ..
                "• 照片中没有鸟类或鸟类不清晰\n" ..
                "• 图片文件损坏或格式不支持\n" ..
                "• 识别模型未正确加载"

            LrDialogs.message(PLUGIN_NAME .. " - 识别失败", failMsg, "error")
        end
        break
    end

    myLogger:info( PLUGIN_NAME .. " 识别处理完成" )
end

return exportServiceProvider
