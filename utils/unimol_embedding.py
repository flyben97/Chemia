# utils/unimol_embedding.py
import numpy as np
from typing import List, Optional
import os
import sys
import contextlib

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
    log_dir: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    使用 Uni-Mol 模型为 SMILES 列表生成分子级 embedding。
    所有输出（包括日志）将被重定向到 log_dir/unimol_tools.log。
    """
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

            clf = UniMolRepr(
                data_type='molecule',
                model_name=model_name,
                model_size=model_size if model_version == 'v2' else None,
                remove_hs=remove_hs
            )
            
            unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=False)
            molecule_embedding = np.array(unimol_repr['cls_repr'])
            
            if molecule_embedding.shape[0] != len(smiles_list):
                success_smiles = unimol_repr['smiles']
                embedding_dim = molecule_embedding.shape[1]
                final_embeddings = np.full((len(smiles_list), embedding_dim), np.nan)
                
                smiles_to_idx_map = {smi: i for i, smi in enumerate(smiles_list)}
                
                for i, smi in enumerate(success_smiles):
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