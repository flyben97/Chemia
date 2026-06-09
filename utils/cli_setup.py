# utils/cli_setup.py
"""
CLI 脚本公共初始化工具
统一处理 sys.path、日志、warning suppression，减少各脚本开头的样板代码。
"""

import os
import sys
import logging
import warnings


def ensure_project_root_in_path(start_path: str = None) -> str:
    """
    确保项目根目录在 sys.path 中。
    如果 start_path 不在 sys.path，会向上查找包含 core/、models/、utils/ 的目录并插入。
    """
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
        # 默认从 utils/cli_setup.py 出发，项目根目录是上一级
        start_path = os.path.dirname(start_path)

    if start_path in sys.path:
        return start_path

    # 如果 start_path 本身看起来像项目根目录，直接插入
    if (
        os.path.exists(os.path.join(start_path, 'core')) and
        os.path.exists(os.path.join(start_path, 'models')) and
        os.path.exists(os.path.join(start_path, 'utils'))
    ):
        sys.path.insert(0, start_path)
        return start_path

    # 否则向上查找
    current_path = start_path
    while current_path != os.path.dirname(current_path):
        if (
            os.path.exists(os.path.join(current_path, 'core')) and
            os.path.exists(os.path.join(current_path, 'models')) and
            os.path.exists(os.path.join(current_path, 'utils'))
        ):
            sys.path.insert(0, current_path)
            return current_path
        current_path = os.path.dirname(current_path)

    #  fallback：插入 start_path
    sys.path.insert(0, start_path)
    return start_path


def suppress_common_warnings():
    """抑制常见库的警告，获得更干净的 CLI 输出。"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*positional args.*")


def setup_basic_logging(level: int = logging.INFO):
    """设置基础日志级别，抑制常见库的 DEBUG 输出。"""
    logging.basicConfig(level=level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('graphviz').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('rdkit').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def standard_cli_setup(start_path: str = None, log_level: int = logging.INFO) -> str:
    """
    标准 CLI 初始化：依次执行 ensure_project_root_in_path、suppress_common_warnings、setup_basic_logging。

    Returns:
        项目根目录路径
    """
    root = ensure_project_root_in_path(start_path)
    suppress_common_warnings()
    setup_basic_logging(log_level)
    return root
