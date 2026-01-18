#!/usr/bin/env python3
"""
SuperPicky BirdID API 服务器
提供 HTTP REST API 供外部程序调用鸟类识别功能
完全兼容 Lightroom 插件 (端口 5156)
"""

__version__ = "1.0.0"

import os
import sys
import base64
import tempfile
from io import BytesIO

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from birdid.bird_identifier import (
    identify_bird,
    predict_bird,
    load_image,
    extract_gps_from_exif,
    get_classifier,
    get_bird_info,
    get_database_manager,
    get_yolo_detector,
    YOLO_AVAILABLE
)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局配置
DEFAULT_PORT = 5156
DEFAULT_HOST = '127.0.0.1'


def get_gui_settings():
    """读取 GUI 界面设置的国家/地区过滤"""
    import re
    settings_path = os.path.expanduser('~/Documents/SuperPicky_Data/birdid_dock_settings.json')
    
    settings = {
        'use_ebird': True,
        'country_code': None,
        'region_code': None
    }
    
    if os.path.exists(settings_path):
        try:
            import json
            with open(settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            settings['use_ebird'] = data.get('use_ebird', True)
            
            # 解析国家代码（从 country_list 映射）
            # 设置文件存储的是显示名称，需要转换为代码
            country_display = data.get('selected_country', '')
            if country_display and country_display not in ('自动检测 (GPS)', '全球模式'):
                # 常见国家映射
                country_map = {
                    '澳大利亚': 'AU', '美国': 'US', '英国': 'GB', '中国': 'CN',
                    '香港': 'HK', '台湾': 'TW', '日本': 'JP'
                }
                settings['country_code'] = country_map.get(country_display)
                
                # 如果是其他国家（从"更多国家"选的），格式可能是 "国家名 (Country Name)"
                if not settings['country_code'] and '(' in country_display:
                    # 尝试加载 regions 数据匹配
                    pass
            
            # 解析区域代码
            region_display = data.get('selected_region', '')
            if region_display and region_display != '整个国家':
                # 格式: "South Australia (AU-SA)"
                match = re.search(r'\(([A-Z]{2}-[A-Z0-9]+)\)', region_display)
                if match:
                    settings['region_code'] = match.group(1)
        except Exception as e:
            print(f"[API] 读取 GUI 设置失败: {e}")
    
    return settings


def ensure_models_loaded():
    """确保模型已加载"""
    print("正在加载模型...")
    get_classifier()
    print("  分类器模型加载完成")

    get_bird_info()
    print("  鸟类信息加载完成")

    db = get_database_manager()
    if db:
        print("  数据库加载完成")

    if YOLO_AVAILABLE:
        detector = get_yolo_detector()
        if detector:
            print("  YOLO 检测器加载完成")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'service': 'SuperPicky BirdID API',
        'version': __version__,
        'yolo_available': YOLO_AVAILABLE
    })


@app.route('/recognize', methods=['POST'])
def recognize_bird():
    """
    识别鸟类

    请求体 (JSON):
    {
        "image_path": "/path/to/image.jpg",  // 图片路径（二选一）
        "image_base64": "base64_encoded_image",  // Base64编码的图片（二选一）
        "use_yolo": true,  // 是否使用YOLO裁剪（可选，默认true）
        "use_gps": true,  // 是否使用GPS过滤（可选，默认true）
        "top_k": 3  // 返回前K个结果（可选，默认3）
    }

    返回 (JSON):
    {
        "success": true,
        "results": [
            {
                "rank": 1,
                "cn_name": "白头鹎",
                "en_name": "Light-vented Bulbul",
                "scientific_name": "Pycnonotus sinensis",
                "confidence": 95.5,
                "ebird_match": true
            },
            ...
        ],
        "gps_info": {
            "latitude": 39.123,
            "longitude": 116.456,
            "info": "GPS: 39.123, 116.456"
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            print("[API] ❌ 无效的请求体")
            return jsonify({'success': False, 'error': '无效的请求体'}), 400

        # 获取图片
        image = None
        image_path = data.get('image_path')
        image_base64 = data.get('image_base64')
        temp_file = None
        
        # 日志：显示请求信息
        if image_path:
            print(f"[API] 📷 收到识别请求: {os.path.basename(image_path)}")
        elif image_base64:
            print(f"[API] 📷 收到 Base64 图片识别请求")

        if image_path:
            # 从文件路径加载
            if not os.path.exists(image_path):
                print(f"[API] ❌ 文件不存在: {image_path}")
                return jsonify({'success': False, 'error': f'文件不存在: {image_path}'}), 404
        elif image_base64:
            # 从 Base64 解码
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))

                # 创建临时文件用于 EXIF 读取
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                image.save(temp_file.name, 'JPEG')
                image_path = temp_file.name
            except Exception as e:
                return jsonify({'success': False, 'error': f'Base64解码失败: {e}'}), 400
        else:
            return jsonify({'success': False, 'error': '必须提供 image_path 或 image_base64'}), 400

        # 获取参数
        use_yolo = data.get('use_yolo', True)
        use_gps = data.get('use_gps', True)
        top_k = data.get('top_k', 3)
        
        # 读取 GUI 设置的国家/地区过滤
        gui_settings = get_gui_settings()
        country_code = data.get('country_code', gui_settings['country_code'])
        region_code = data.get('region_code', gui_settings['region_code'])
        use_ebird = data.get('use_ebird', gui_settings['use_ebird'])
        
        # 日志：显示识别参数
        print(f"[API] ⚙️  识别参数:")
        print(f"[API]     YOLO裁剪: {'✅ 是' if use_yolo else '❌ 否'}")
        print(f"[API]     GPS过滤: {'✅ 是' if use_gps else '❌ 否'}")
        print(f"[API]     eBird过滤: {'✅ 是' if use_ebird else '❌ 否'}")
        print(f"[API]     国家: {country_code or '无'}, 地区: {region_code or '无'}")

        # 执行识别
        result = identify_bird(
            image_path,
            use_yolo=use_yolo,
            use_gps=use_gps,
            top_k=top_k,
            country_code=country_code,
            region_code=region_code,
            use_ebird=use_ebird
        )
        
        # 日志：显示识别结果
        if result.get('success'):
            results = result.get('results', [])
            if results:
                top_result = results[0]
                print(f"[API] ✅ 识别成功! 第1候选: {top_result.get('cn_name', '?')} ({top_result.get('confidence', 0):.1f}%)")
            else:
                print(f"[API] ⚠️  识别完成但没有结果")
        else:
            print(f"[API] ❌ 识别失败: {result.get('error', '未知错误')}")

        # 清理临时文件
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass

        if not result['success']:
            return jsonify({
                'success': False,
                'error': result.get('error', '识别失败')
            }), 500

        # 格式化结果（兼容 Lightroom 插件格式）
        formatted_results = []
        for i, r in enumerate(result.get('results', []), 1):
            formatted_results.append({
                'rank': i,
                'cn_name': r.get('cn_name', ''),
                'en_name': r.get('en_name', ''),
                'scientific_name': r.get('scientific_name', ''),
                'confidence': float(r.get('confidence', 0)),
                'ebird_match': r.get('ebird_match', False)
            })
        
        # 智能候选筛选：根据置信度差距决定返回多少个候选
        if len(formatted_results) >= 2:
            top_confidence = formatted_results[0]['confidence']
            
            # 计算与第一名的相对差距（百分比）
            # 如果第1名 = 50%, 第2名 = 40%, 相对差距 = (50-40)/50 = 20%
            smart_results = [formatted_results[0]]  # 总是包含第1名
            
            for r in formatted_results[1:]:
                if top_confidence > 0:
                    relative_gap = (top_confidence - r['confidence']) / top_confidence * 100
                    # 如果相对差距 <= 50%，认为是"接近的候选"
                    if relative_gap <= 50:
                        smart_results.append(r)
                    else:
                        break  # 后面的差距只会更大，停止添加
            
            # 日志：显示筛选结果
            if len(smart_results) == 1:
                print(f"[API] 🎯 第1名置信度({top_confidence:.1f}%)远高于其他，仅返回1个候选")
            else:
                print(f"[API] 🎯 候选置信度接近，返回 {len(smart_results)} 个候选")
            
            formatted_results = smart_results
        
        # 如果没有结果，返回详细的错误信息
        if not formatted_results:
            ebird_info = result.get('ebird_info')
            if ebird_info and ebird_info.get('enabled'):
                region = ebird_info.get('region', '未知区域')
                species_count = ebird_info.get('species_count', 0)
                error_msg = f"eBird 区域过滤：所有候选鸟种都不在 {region} 的 {species_count} 种记录中。建议：1) 确认拍摄地点正确 2) 尝试关闭 eBird 过滤"
                print(f"[API] ⚠️  {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'ebird_info': ebird_info
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '未能识别图片中的鸟类，请确保照片中有清晰的鸟类'
                })

        response = {
            'success': True,
            'results': formatted_results,
            'yolo_info': result.get('yolo_info'),
            'gps_info': result.get('gps_info'),
            'ebird_info': result.get('ebird_info')
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/exif/write-title', methods=['POST'])
def write_exif_title():
    """
    写入鸟种名称到 EXIF Title

    请求体:
    {
        "image_path": "/path/to/image.jpg",
        "bird_name": "白头鹎"
    }
    """
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        bird_name = data.get('bird_name')

        if not image_path or not bird_name:
            return jsonify({'success': False, 'error': '缺少必需参数'}), 400

        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'文件不存在: {image_path}'}), 404

        from exiftool_manager import get_exiftool_manager
        exiftool_mgr = get_exiftool_manager()
        success = exiftool_mgr.set_metadata(image_path, {'Title': bird_name})

        return jsonify({
            'success': success,
            'message': f'已写入: {bird_name}' if success else '写入失败'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/exif/write-caption', methods=['POST'])
def write_exif_caption():
    """
    写入鸟种描述到 EXIF Caption

    请求体:
    {
        "image_path": "/path/to/image.jpg",
        "caption": "鸟种描述文本"
    }
    """
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        caption = data.get('caption')

        if not image_path or not caption:
            return jsonify({'success': False, 'error': '缺少必需参数'}), 400

        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'文件不存在: {image_path}'}), 404

        from exiftool_manager import get_exiftool_manager
        exiftool_mgr = get_exiftool_manager()
        success = exiftool_mgr.set_metadata(image_path, {'Caption-Abstract': caption})

        return jsonify({
            'success': success,
            'message': f'已写入描述' if success else '写入失败'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description='SuperPicky BirdID API 服务器')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'监听地址 (默认: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'监听端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-preload', action='store_true', help='跳过模型预加载')

    args = parser.parse_args()

    print("=" * 60)
    print(f"  SuperPicky BirdID API 服务器 v{__version__}")
    print("=" * 60)
    print(f"监听地址: http://{args.host}:{args.port}")
    print(f"健康检查: http://{args.host}:{args.port}/health")
    print(f"识别接口: POST http://{args.host}:{args.port}/recognize")
    print("=" * 60)
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)

    # 预加载模型
    if not args.no_preload:
        print("\n正在预加载模型...")
        ensure_models_loaded()
        print("模型预加载完成\n")

    # 启动服务器
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
