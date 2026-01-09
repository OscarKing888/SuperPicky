#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动安装 PyTorch CUDA 版本
检测系统 CUDA 版本并安装对应的 PyTorch
"""

import os
import sys
import subprocess
import platform

# Windows 控制台编码设置
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def check_nvidia_driver():
    """检查 NVIDIA 驱动是否安装"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, result.stdout
        return False, None
    except FileNotFoundError:
        return False, None
    except Exception as e:
        return False, str(e)

def get_cuda_version_from_nvidia_smi():
    """从 nvidia-smi 获取 CUDA 版本"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return version
    except Exception:
        pass
    return None

def get_driver_version():
    """获取驱动版本"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return version
    except Exception:
        pass
    return None

def check_current_pytorch():
    """检查当前 PyTorch 版本"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        return True, version, cuda_available, cuda_version
    except ImportError:
        return False, None, False, None
    except Exception as e:
        return False, None, False, str(e)

def select_pytorch_cuda_version(cuda_version_str):
    """根据系统 CUDA 版本选择 PyTorch CUDA 版本"""
    if not cuda_version_str:
        # 默认使用 cu118（兼容性最好）
        return 'cu118', 'https://download.pytorch.org/whl/cu118'
    
    try:
        # 解析版本号（例如 "12.1" -> 12.1）
        major, minor = map(int, cuda_version_str.split('.')[:2])
        version_float = major + minor / 10.0
        
        if version_float >= 12.1:
            return 'cu121', 'https://download.pytorch.org/whl/cu121'
        elif version_float >= 11.8:
            return 'cu118', 'https://download.pytorch.org/whl/cu118'
        else:
            # 旧版本 CUDA，使用 cu118（向后兼容）
            return 'cu118', 'https://download.pytorch.org/whl/cu118'
    except Exception:
        # 解析失败，使用默认
        return 'cu118', 'https://download.pytorch.org/whl/cu118'

def install_pytorch_cuda(cuda_version='cu118', index_url=None):
    """安装 PyTorch CUDA 版本"""
    if index_url is None:
        index_url = f'https://download.pytorch.org/whl/{cuda_version}'
    
    print(f"\n📦 正在安装 PyTorch CUDA 版本 ({cuda_version})...")
    print(f"   索引 URL: {index_url}")
    print("   这可能需要几分钟，请耐心等待...\n")
    
    # 卸载旧版本
    print("   1. 卸载旧版本...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'],
                      capture_output=True, check=False)
    except Exception:
        pass
    
    # 安装新版本
    print(f"   2. 安装 PyTorch CUDA {cuda_version} 版本...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio',
             '--index-url', index_url],
            check=True
        )
        print("   ✅ 安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 安装失败: {e}")
        return False

def verify_installation():
    """验证安装"""
    print("\n" + "=" * 60)
    print("验证安装...")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA 版本: {torch.version.cuda}")
            print(f"✅ GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA 不可用")
            return False
    except ImportError:
        print("❌ PyTorch 未正确安装")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    print("=" * 60)
    print("  PyTorch CUDA 版本自动安装脚本")
    print("=" * 60)
    print()
    
    # 步骤 1: 检查 NVIDIA 驱动
    print("[1/5] 检查 NVIDIA GPU 和驱动...")
    driver_available, driver_info = check_nvidia_driver()
    
    if not driver_available:
        print("❌ 未找到 NVIDIA 驱动程序")
        print()
        print("💡 请先安装 NVIDIA 驱动程序:")
        print("   1. 访问 https://www.nvidia.com/drivers")
        print("   2. 下载并安装最新的驱动程序")
        print("   3. 安装完成后重新运行此脚本")
        print()
        input("按 Enter 键退出...")
        return False
    
    print("✅ 找到 NVIDIA 驱动程序")
    
    driver_version = get_driver_version()
    if driver_version:
        print(f"   驱动版本: {driver_version}")
    
    cuda_version = get_cuda_version_from_nvidia_smi()
    if cuda_version:
        print(f"   CUDA 版本: {cuda_version}")
    else:
        print("   ⚠️  无法检测 CUDA 版本，将使用默认版本")
    
    # 步骤 2: 检查当前 PyTorch
    print()
    print("[2/5] 检查当前 PyTorch 版本...")
    pytorch_installed, pytorch_version, cuda_available, pytorch_cuda_version = check_current_pytorch()
    
    if pytorch_installed:
        print(f"   当前版本: {pytorch_version}")
        print(f"   CUDA 可用: {cuda_available}")
        if pytorch_cuda_version:
            print(f"   PyTorch CUDA 版本: {pytorch_cuda_version}")
        
        if cuda_available:
            print()
            print("✅ PyTorch 已安装 CUDA 版本，无需重新安装")
            verify_installation()
            return True
    else:
        print("   ⚠️  PyTorch 未安装")
    
    # 步骤 3: 选择 PyTorch CUDA 版本
    print()
    print("[3/5] 选择 PyTorch CUDA 版本...")
    pytorch_cuda, index_url = select_pytorch_cuda_version(cuda_version)
    print(f"   将安装: PyTorch CUDA {pytorch_cuda} 版本")
    print(f"   索引 URL: {index_url}")
    
    # 步骤 4: 确认安装
    print()
    print("[4/5] 准备安装...")
    print("   将卸载旧版本并安装新版本")
    response = input("   是否继续? (Y/n): ").strip().lower()
    if response and response != 'y':
        print("   已取消")
        return False
    
    # 步骤 5: 安装
    print()
    print("[5/5] 安装 PyTorch CUDA 版本...")
    success = install_pytorch_cuda(pytorch_cuda, index_url)
    
    if not success:
        print()
        print("❌ 安装失败")
        print()
        print("💡 可以尝试手动安装:")
        print(f"   pip install torch torchvision torchaudio --index-url {index_url}")
        print()
        input("按 Enter 键退出...")
        return False
    
    # 验证
    print()
    success = verify_installation()
    
    if success:
        print()
        print("=" * 60)
        print("  ✅ 安装完成！")
        print("=" * 60)
        print()
        print("  现在可以重新运行程序，应该会使用 CUDA 加速了")
        print()
    else:
        print()
        print("⚠️  安装完成，但验证失败")
        print("   请手动运行: python -c \"import torch; print(torch.cuda.is_available())\"")
        print()
    
    input("按 Enter 键退出...")
    return success

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")
        sys.exit(1)
