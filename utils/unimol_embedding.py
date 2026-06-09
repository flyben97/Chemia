# utils/unimol_embedding.py
import numpy as np
from typing import List, Optional
import os
import sys
import contextlib
from pathlib import Path

# Explicitly suppress numba debug output when using UniMol
from .suppress_logs import suppress_debug_logs
suppress_debug_logs()


def check_model_cached(model_size: str = '84m') -> bool:
    """
    检查指定大小的UniMolV2模型是否已缓存。
    
    Args:
        model_size: 模型大小 ('84m', '164m', '310m', '570m', '1.1B')
        
    Returns:
        如果模型已缓存返回True，否则返回False
    """
    try:
        import unimol_tools
        weights_dir = Path(unimol_tools.__file__).parent / "weights" / "modelzoo" / model_size.upper()
        return weights_dir.exists() and any(weights_dir.glob("*.pt"))
    except ImportError:
        return False


def get_cache_size(model_size: str = '84m') -> Optional[str]:
    """
    获取已缓存模型的大小信息。
    
    Args:
        model_size: 模型大小
        
    Returns:
        格式化的文件大小字符串，如 "321 MB"，未缓存则返回None
    """
    try:
        import unimol_tools
        weights_dir = Path(unimol_tools.__file__).parent / "weights" / "modelzoo" / model_size.upper()
        if not weights_dir.exists():
            return None
        
        total_size = 0
        for f in weights_dir.glob("*.pt"):
            total_size += f.stat().st_size
        
        if total_size == 0:
            return None
            
        # 格式化为人类可读
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024
        return f"{total_size:.1f} TB"
    except ImportError:
        return None

@contextlib.contextmanager
def redirect_stdout_to_file(filepath):
    """
    A context manager to temporarily redirect stdout and stderr to a file.
    """
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as log_file:
        # 同时重定向 stdout 和 stderr
        # Tee(log_file, original_stdout) 可以让日志既写入文件也打印到控制台
        # 如果只想写入文件，就用 log_file
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            yield
        finally:
            # 无论成功还是失败，都恢复原始的 stdout 和 stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def get_unimol_embedding(
    smiles_list: List[str],
    model_version: str = 'v2',
    model_size: str = '84m',
    remove_hs: bool = False,
    log_dir: Optional[str] = None,
    disable_logging: bool = False,
    standardize_smiles: bool = True
) -> Optional[np.ndarray]:
    """
    使用 Uni-Mol 模型为 SMILES 列表生成分子级 embedding。

    Args:
        smiles_list: SMILES字符串列表
        model_version: 模型版本 ('v1' 或 'v2')
        model_size: 模型大小 (仅v2可用: '84m', '164m', '310m', '570m', '1.1B')
        remove_hs: 是否移除氢原子
        log_dir: 日志保存目录，如果为None且disable_logging=False，则不保存日志
        disable_logging: 是否完全禁用日志记录 (优先级高于log_dir)
        standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

    Returns:
        分子embeddings数组，如果失败则返回None
    """
    # SMILES标准化（默认启用）
    if standardize_smiles:
        from .smiles_validator import standardize_smiles as _standardize_smiles
        processed_smiles_list = [_standardize_smiles(smi) for smi in smiles_list]
    else:
        processed_smiles_list = smiles_list

    if disable_logging:
        log_filepath = os.devnull
    else:
        log_filepath = os.path.join(log_dir, 'unimol_tools.log') if log_dir else os.devnull

    with redirect_stdout_to_file(log_filepath):
        try:
            # --- 延迟导入，确保在重定向环境中进行 ---
            from unimol_tools import UniMolRepr

            if model_version == 'v1':
                model_name = 'unimolv1'
                # print(f"--- 正在初始化 Uni-Mol V1 模型... ---") # print 会被重定向
            elif model_version == 'v2':
                model_name = 'unimolv2'
                valid_sizes = ['84m', '164m', '310m', '570m', '1.1B']
                if model_size not in valid_sizes:
                    raise ValueError(f"无效的 model_size '{model_size}'。Uni-Mol V2 可选值: {valid_sizes}")
                # print(f"--- 正在初始化 Uni-Mol V2 模型 (大小: {model_size}) ---")
            else:
                raise ValueError(f"无效的 model_version '{model_version}'。请选择 'v1' 或 'v2'。")

            # 根据版本设置参数
            if model_version == 'v2':
                clf = UniMolRepr(
                    data_type='molecule',
                    model_name=model_name,
                    model_size=model_size,
                    remove_hs=remove_hs
                )
            else:  # v1
                clf = UniMolRepr(
                    data_type='molecule',
                    model_name=model_name,
                    remove_hs=remove_hs
                )

            unimol_repr = clf.get_repr(processed_smiles_list, return_atomic_reprs=False)
            # UniMol Tools 返回的是 numpy array 列表，每个元素对应一个分子的表示
            if isinstance(unimol_repr, list):
                molecule_embedding = np.array(unimol_repr)
            else:
                # 旧版本可能返回字典格式
                molecule_embedding = np.array(unimol_repr.get('cls_repr', unimol_repr))

            if molecule_embedding.shape[0] != len(smiles_list):
                # 如果返回数量不匹配，可能是某些 SMILES 处理失败
                # 尝试从 UniMolRepr 对象获取成功的 smiles 列表
                success_smiles = getattr(clf, 'smiles_list', smiles_list[:molecule_embedding.shape[0]])
                embedding_dim = molecule_embedding.shape[1]
                final_embeddings = np.full((len(smiles_list), embedding_dim), np.nan)

                smiles_to_idx_map = {smi: i for i, smi in enumerate(smiles_list)}

                for i, smi in enumerate(success_smiles):
                    if i < molecule_embedding.shape[0]:
                        original_idx = smiles_to_idx_map.get(smi)
                        if original_idx is not None:
                            final_embeddings[original_idx] = molecule_embedding[i]

                molecule_embedding = final_embeddings

            # 在退出重定向前，打印成功信息到原始控制台
            sys.stdout = sys.__stdout__
            print("Successfully generated Uni-Mol Embeddings!")

            return molecule_embedding

        except Exception as e:
            # 在退出重定向前，打印错误信息到原始控制台
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"\nA critical error occurred during Uni-Mol model initialization or prediction: {e}")
            # 将错误也写入日志文件
            with open(log_filepath, 'a') as f:
                import traceback
                f.write("\n--- CRITICAL ERROR ---\n")
                f.write(str(e))
                f.write(traceback.format_exc())
            return None

def get_unimol_embedding_single(
    smiles: str,
    model_version: str = 'v2',
    model_size: str = '84m',
    remove_hs: bool = False,
    log_dir: Optional[str] = None,
    disable_logging: bool = True,
    standardize_smiles: bool = True
) -> Optional[np.ndarray]:
    """
    使用 Uni-Mol 模型为单个 SMILES 字符串生成 embedding。
    
    简化版API，适合只需要处理单个分子的场景。

    Args:
        smiles: SMILES 字符串 (例如 'CCO' 表示乙醇)
        model_version: 模型版本 ('v1' 或 'v2')
        model_size: 模型大小 (仅v2可用: '84m', '164m', '310m', '570m', '1.1B')
        remove_hs: 是否移除氢原子
        log_dir: 日志保存目录，如果为None且不禁用日志，则不保存日志
        disable_logging: 是否完全禁用日志记录 (默认True，保持输出干净)
        standardize_smiles: 是否对SMILES进行标准化（默认True，使用RDKit canonical SMILES）

    Returns:
        一维 numpy 数组 (embedding 向量)，如果失败则返回 None
        
    Examples:
        >>> embedding = get_unimol_embedding_single('CCO')
        >>> print(embedding.shape)  # (768,)
        >>> print(embedding[:5])    # 前5个数值
    """
    embeddings = get_unimol_embedding(
        smiles_list=[smiles],
        model_version=model_version,
        model_size=model_size,
        remove_hs=remove_hs,
        log_dir=log_dir,
        disable_logging=disable_logging,
        standardize_smiles=standardize_smiles
    )
    
    if embeddings is None:
        return None
    
    # 返回第一个（也是唯一一个）embedding，展平为一维数组
    return embeddings[0]


# ... (if __name__ == '__main__' 部分保持不变)
if __name__ == '__main__':
    my_molecules = ['O=C(C)Oc1ccccc1C(=O)O', 'CCN(CC)CC', 'InvalidSMILES']

    test_log_dir = "temp_unimol_logs"
    print(f"Testing Uni-Mol with logs directed to: {test_log_dir}")
    embeddings = get_unimol_embedding(my_molecules, log_dir=test_log_dir)

    if embeddings is not None:
        print(f"\nOutput Shape: {embeddings.shape}")
        print(f"Embedding for Aspirin: {embeddings[0, :5]}")
        print(f"Embedding for TEA: {embeddings[1, :5]}")
        print(f"Embedding for Invalid SMILES: {embeddings[2, :5]}")

    log_file_path = os.path.join(test_log_dir, "unimol_tools.log")
    from rich import print as rprint # use rich for colored output
    if os.path.exists(log_file_path):
        rprint(f"\n[green]✓ Log file created at: {log_file_path}[/green]")
        with open(log_file_path, 'r') as f:
            content = f.read()
            rprint("--- Log Content Preview ---")
            rprint(content[:500] + "..." if len(content) > 500 else content)
    else:
        rprint(f"\n[red]❌ Log file not found at: {log_file_path}[/red]")
