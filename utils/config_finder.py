# utils/config_finder.py
"""
动态配置文件查找工具
支持多种路径查找策略，自动定位配置文件
"""

import os
from typing import List, Optional
from pathlib import Path

def find_config_file(config_path: str, project_root: Optional[str] = None) -> str:
    """
    动态查找配置文件，支持多种路径查找策略

    Args:
        config_path: 配置文件路径（可以是相对路径或绝对路径）
        project_root: 项目根目录（可选，自动检测）

    Returns:
        str: 找到的配置文件的绝对路径

    Raises:
        FileNotFoundError: 如果在所有候选路径中都找不到配置文件
    """

    # 如果没有提供项目根目录，自动检测
    if project_root is None:
        # 从当前文件位置向上查找，直到找到包含特定标志文件的目录
        current_dir = Path(__file__).parent.parent
        while current_dir != current_dir.parent:
            if (current_dir / "core").exists() or (current_dir / "scripts").exists():
                project_root = str(current_dir)
                break
            current_dir = current_dir.parent
        else:
            project_root = os.getcwd()

    # 构建候选路径列表（按优先级排序）
    candidate_paths = [
        config_path,  # 1. 原始路径（可能是绝对路径）
        os.path.join(project_root, config_path),  # 2. 项目根目录 + 相对路径
        os.path.join(project_root, "examples", "configs", os.path.basename(config_path)),  # 3. examples/configs目录
        os.path.join(project_root, "configs", os.path.basename(config_path)),  # 4. configs目录
        os.path.join(os.getcwd(), config_path),  # 5. 当前工作目录 + 相对路径
        os.path.join(os.getcwd(), os.path.basename(config_path)),  # 6. 当前目录下的文件名
        os.path.join(project_root, os.path.basename(config_path)),  # 7. 项目根目录下的文件名
    ]

    # 去重并保持顺序
    unique_paths = []
    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if abs_path not in unique_paths:
            unique_paths.append(abs_path)

    # 查找第一个存在的文件
    for path in unique_paths:
        if os.path.exists(path) and os.path.isfile(path):
            print(f"✓ Found configuration file: {path}")
            return path

    # 如果都找不到，生成详细的错误信息
    error_msg = f"❌ Configuration file '{config_path}' not found.\n"
    error_msg += f"📁 Project root: {project_root}\n"
    error_msg += f"💼 Current working directory: {os.getcwd()}\n"
    error_msg += f"🔍 Searched in the following locations:\n"

    for i, path in enumerate(unique_paths, 1):
        exists_status = "✓" if os.path.exists(path) else "✗"
        error_msg += f"  {i:2d}. {exists_status} {path}\n"

    error_msg += f"\n💡 Suggestions:\n"
    error_msg += f"  • Check if the file exists in one of the searched locations\n"
    error_msg += f"  • Use absolute path: python script.py --config /full/path/to/config.yaml\n"
    error_msg += f"  • Place config in: {os.path.join(project_root, 'examples', 'configs')}\n"
    error_msg += f"  • Place config in project root: {project_root}\n"

    raise FileNotFoundError(error_msg)

def list_available_configs(project_root: Optional[str] = None) -> List[str]:
    """
    列出可用的配置文件

    Args:
        project_root: 项目根目录（可选）

    Returns:
        List[str]: 找到的配置文件路径列表
    """
    if project_root is None:
        current_dir = Path(__file__).parent.parent
        while current_dir != current_dir.parent:
            if (current_dir / "core").exists() or (current_dir / "scripts").exists():
                project_root = str(current_dir)
                break
            current_dir = current_dir.parent
        else:
            project_root = os.getcwd()

    config_dirs = [
        project_root,
        os.path.join(project_root, "examples", "configs"),
        os.path.join(project_root, "configs"),
        os.getcwd()
    ]

    config_files = []
    for config_dir in config_dirs:
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith(('.yaml', '.yml')):
                    full_path = os.path.join(config_dir, file)
                    if full_path not in config_files:
                        config_files.append(full_path)

    return sorted(config_files)

if __name__ == "__main__":
    # 测试功能
    print("🔍 Available configuration files:")
    configs = list_available_configs()
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")

    # 测试查找功能
    if configs:
        test_config = os.path.basename(configs[0])
        print(f"\n🧪 Testing search for: {test_config}")
        try:
            found = find_config_file(test_config)
            print(f"✅ Success: {found}")
        except FileNotFoundError as e:
            print(f"❌ Failed: {e}")
