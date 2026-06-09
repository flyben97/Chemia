# utils/transformer_embeddings.py
# (Formerly embedder.py)
#
# =============================================================================
# 运行前请确保已安装所有依赖:
#
# 1. 激活你的 Conda 环境 (e.g., `conda activate chemia`)
# 2. 安装核心库:
#    pip install torch transformers
#
# 3. (重要!) 安装 MolT5 需要的 sentencepiece 库:
#    pip install sentencepiece
# 4. (推荐) 安装 tqdm 以显示进度条:
#    pip install tqdm
# =============================================================================

import numpy as np
from typing import List, Optional, Type
from pathlib import Path
import os
import json
import shutil
from datetime import datetime
import time

# --- 1. 设备配置：自动选择 GPU 或 CPU ---
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 使用设备 (transformer_embeddings): {DEVICE} ---")
except ImportError:
    DEVICE = None
    print("--- 警告: PyTorch 未安装，transformer embedding 功能将不可用 ---")

# --- 1.5 项目级缓存配置 ---
PROJECT_ROOT = Path(__file__).parent.parent  # 项目根目录
PRETRAINED_MODELS_DIR = PROJECT_ROOT / "pretrained_models"
PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 镜像源列表 (优先使用国内镜像)
HF_MIRRORS = [
    'https://huggingface-mirror.com',   # 镜像源 1 (国内，推荐，对大文件支持更好)
    'https://hf-mirror.com',            # 镜像源 2 (国内)
    'https://huggingface.co',           # 官方源 (国外)
]

print(f"--- 预训练模型缓存目录: {PRETRAINED_MODELS_DIR} ---")

# --- 2. 核心辅助函数 ---

def _get_embeddings_in_batches(
    model,
    tokenizer,
    smiles_list: List[str],
    device: torch.device,
    batch_size: int = 32,
    model_type: str = "Transformer"
) -> np.ndarray:
    """
    辅助函数，通过分批处理SMILES列表来计算嵌入，以节省显存。
    """
    model.eval()
    all_embeddings_list = []

    try:
        from tqdm import tqdm
        progress_bar = tqdm(range(0, len(smiles_list), batch_size), desc=f"[{model_type}] Processing batches")
    except ImportError:
        progress_bar = range(0, len(smiles_list), batch_size)

    with torch.no_grad():
        for i in progress_bar:
            batch_smiles = smiles_list[i : i + batch_size]

            inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            batch_embedding = outputs.last_hidden_state.mean(dim=1)

            all_embeddings_list.append(batch_embedding.cpu())

    final_embeddings = torch.cat(all_embeddings_list, dim=0)
    return final_embeddings.numpy()


def _download_model_with_retry(
    model_name: str,
    model_type: str,
    tokenizer_class: Type,
    model_class: Type,
    timeout: int = 30
):
    """
    尝试从多个 HuggingFace 镜像源下载模型，自动切换镜像。

    Args:
        model_name: 模型名称
        model_type: 模型类型（用于日志）
        tokenizer_class: 分词器类
        model_class: 模型类
        timeout: 超时时间（秒）

    Returns:
        (tokenizer, model) 或 (None, None)
    """
    tokenizer_args = {'legacy': False} if tokenizer_class.__name__ == 'T5Tokenizer' else {}

    # 尝试 HuggingFace 镜像源
    for mirror_idx, mirror_url in enumerate(HF_MIRRORS, 1):
        try:
            print(f"[{model_type}] 尝试从 HuggingFace 镜像源 {mirror_idx}/{len(HF_MIRRORS)} 下载: {mirror_url}")

            # 设置环境变量
            original_hf_endpoint = os.environ.get('HF_ENDPOINT')
            if mirror_url != 'https://huggingface.co':
                os.environ['HF_ENDPOINT'] = mirror_url

            # 下载模型和分词器
            print(f"[{model_type}] 正在下载分词器...")
            tokenizer = tokenizer_class.from_pretrained(model_name, **tokenizer_args)

            print(f"[{model_type}] 正在下载模型...")
            model = model_class.from_pretrained(model_name)

            print(f"[{model_type}] ✅ 从 {mirror_url} 下载成功！")

            # 恢复环境变量
            if original_hf_endpoint is not None:
                os.environ['HF_ENDPOINT'] = original_hf_endpoint
            elif 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']

            return tokenizer, model

        except Exception as e:
            print(f"[{model_type}] ❌ 镜像源 {mirror_idx} 失败: {str(e)[:100]}")

            # 恢复环境变量
            if original_hf_endpoint is not None:
                os.environ['HF_ENDPOINT'] = original_hf_endpoint
            elif 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']

            if mirror_idx < len(HF_MIRRORS):
                print(f"[{model_type}] 尝试下一个镜像源...")
                time.sleep(1)  # 等待 1 秒后重试
            continue

    print(f"[{model_type}] ❌ 所有下载源都失败，无法下载模型")
    return None, None


def _get_model_and_tokenizer(
    model_name: str,
    model_type: str,
    cache_subdir: str,
    tokenizer_class: Type,
    model_class: Type,
    smiles_list: List[str],
    batch_size: int
):
    """
    通用函数，负责加载或下载模型和分词器，并计算嵌入。

    优先从项目的 pretrained_models 文件夹加载，如果不存在则自动下载。
    """
    if DEVICE is None:
        print(f"[{model_type}] 错误: PyTorch 未安装，无法使用该模型")
        return None

    # 使用项目级缓存目录
    local_model_path = PRETRAINED_MODELS_DIR / cache_subdir / model_name.replace("/", "_")

    model, tokenizer = None, None

    # 1. 尝试从项目缓存加载
    if local_model_path.exists():
        print(f"\n[{model_type}] 📦 从项目缓存加载模型...")
        print(f"[{model_type}] 📍 缓存路径: {local_model_path}")
        try:
            tokenizer_args = {'legacy': False} if tokenizer_class.__name__ == 'T5Tokenizer' else {}
            print(f"[{model_type}] 正在加载分词器...")
            tokenizer = tokenizer_class.from_pretrained(local_model_path, local_files_only=True, **tokenizer_args)
            print(f"[{model_type}] 正在加载模型...")
            model = model_class.from_pretrained(local_model_path, local_files_only=True).to(DEVICE)
            print(f"[{model_type}] ✅ 模型加载成功！")
        except Exception as e:
            print(f"[{model_type}] ⚠️  本地模型加载失败: {e}")
            print(f"[{model_type}] 将删除损坏的缓存并重新下载...")
            shutil.rmtree(local_model_path, ignore_errors=True)
            model, tokenizer = None, None

    # 2. 如果本地加载失败或不存在，则从 HuggingFace 下载
    if model is None or tokenizer is None:
        print(f"\n[{model_type}] 📥 首次使用，正在下载预训练模型...")
        print(f"[{model_type}] 模型: {model_name}")
        print(f"[{model_type}] 这可能需要几分钟，请耐心等待...")

        tokenizer, model = _download_model_with_retry(
            model_name, model_type, tokenizer_class, model_class
        )

        if model is None or tokenizer is None:
            print(f"[{model_type}] ❌ 模型下载失败")
            return None

        # 将模型保存到项目缓存
        print(f"\n[{model_type}] 💾 正在保存模型到项目缓存...")
        local_model_path.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[{model_type}] 保存分词器...")
            tokenizer.save_pretrained(local_model_path)
            print(f"[{model_type}] 保存模型...")
            model.save_pretrained(local_model_path)

            # 创建模型信息文件
            with open(local_model_path / "model_info.json", "w") as f:
                json.dump({
                    "model_name": model_name,
                    "download_time": datetime.now().isoformat(),
                }, f, indent=2)
            print(f"[{model_type}] ✅ 模型已成功缓存到: {local_model_path}")

        except Exception as e:
            print(f"[{model_type}] ❌ 保存模型时发生错误: {e}")
            return None

    # 3. 使用加载好的模型和分词器进行分批计算
    print(f"[{model_type}] 开始计算嵌入，批次大小: {batch_size}")
    embeddings = _get_embeddings_in_batches(model, tokenizer, smiles_list, DEVICE, batch_size, model_type)
    print(f"[{model_type}] 计算完成。")

    # 清理显存
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


# --- 3. 模型封装函数 (已全部修改) ---

def get_chemberta_embedding(
    smiles_list: List[str],
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 32
) -> Optional[np.ndarray]:
    """使用 ChemBERTa 模型为 SMILES 列表计算 embedding (分批处理)。"""
    from transformers import AutoTokenizer, AutoModel
    print(f"\n[ChemBERTa] 正在获取 '{model_name}' 的 embedding...")
    return _get_model_and_tokenizer(
        model_name=model_name,
        model_type="ChemBERTa",
        cache_subdir="chemberta_models",
        tokenizer_class=AutoTokenizer,
        model_class=AutoModel,
        smiles_list=smiles_list,
        batch_size=batch_size
    )

def get_molt5_embedding(
    smiles_list: List[str],
    model_name: str = "laituan245/molt5-large",
    batch_size: int = 32
) -> Optional[np.ndarray]:
    """使用 MolT5 模型为 SMILES 列表计算 embedding (分批处理)。"""
    from transformers import T5Tokenizer, T5EncoderModel
    print(f"\n[MolT5] 正在获取 '{model_name}' 的 embedding...")
    return _get_model_and_tokenizer(
        model_name=model_name,
        model_type="MolT5",
        cache_subdir="molt5_models",
        tokenizer_class=T5Tokenizer,
        model_class=T5EncoderModel,
        smiles_list=smiles_list,
        batch_size=batch_size
    )

def get_chemroberta_embedding(
    smiles_list: List[str],
    model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k",
    batch_size: int = 32
) -> Optional[np.ndarray]:
    """使用 ChemRoBERTa 模型为 SMILES 列表计算 embedding (分批处理)。"""
    from transformers import AutoTokenizer, AutoModel
    print(f"\n[ChemRoBERTa] 正在获取 '{model_name}' 的 embedding...")
    return _get_model_and_tokenizer(
        model_name=model_name,
        model_type="ChemRoBERTa",
        cache_subdir="chemroberta_models",
        tokenizer_class=AutoTokenizer,
        model_class=AutoModel,
        smiles_list=smiles_list,
        batch_size=batch_size
    )

def get_biogpt_embedding(
    smiles_list: List[str],
    model_name: str = "microsoft/biogpt",
    batch_size: int = 16
) -> Optional[np.ndarray]:
    """
    使用 BioGPT 模型为 SMILES 列表计算 embedding (分批处理)。

    BioGPT 在生物医学文献上预训练，适合生物医学应用。
    注意: BioGPT 模型较大，建议使用较小的 batch_size。
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\n[BioGPT] 正在获取 '{model_name}' 的 embedding...")
    return _get_model_and_tokenizer(
        model_name=model_name,
        model_type="BioGPT",
        cache_subdir="biogpt_models",
        tokenizer_class=AutoTokenizer,
        model_class=AutoModelForCausalLM,
        smiles_list=smiles_list,
        batch_size=batch_size
    )

def get_roberta_embedding(
    smiles_list: List[str],
    model_name: str = "roberta-base",
    batch_size: int = 32
) -> Optional[np.ndarray]:
    """
    使用 RoBERTa 模型为 SMILES 列表计算 embedding (分批处理)。

    RoBERTa 是改进的 BERT，性能更好，适合通用特征提取。
    """
    from transformers import AutoTokenizer, AutoModel
    print(f"\n[RoBERTa] 正在获取 '{model_name}' 的 embedding...")
    return _get_model_and_tokenizer(
        model_name=model_name,
        model_type="RoBERTa",
        cache_subdir="roberta_models",
        tokenizer_class=AutoTokenizer,
        model_class=AutoModel,
        smiles_list=smiles_list,
        batch_size=batch_size
    )

def get_matbert_embedding(
    material_formulas: List[str],
    model_name: str = "M3L/MatBERT",
    batch_size: int = 32
) -> Optional[np.ndarray]:
    """
    使用 MatBERT 模型为材料公式列表计算 embedding (分批处理)。

    MatBERT 是专为材料科学设计的预训练模型，用于处理材料化学公式。

    Args:
        material_formulas: 材料化学公式列表，例如 ['Fe(NO3)3• 9H2O', 'La0.85Ag0.15Mn1−yAlyO3']
        model_name: 模型名称
        batch_size: 批处理大小

    Returns:
        np.ndarray: Embedding 矩阵

    示例:
        >>> formulas = ['Fe(NO3)3• 9H2O', 'La0.85Ag0.15Mn1−yAlyO3']
        >>> embeddings = get_matbert_embedding(formulas)
        >>> print(embeddings.shape)  # (2, 768)
    """
    from transformers import AutoTokenizer, AutoModel
    print(f"\n[MatBERT] 正在获取 '{model_name}' 的 embedding...")
    return _get_model_and_tokenizer(
        model_name=model_name,
        model_type="MatBERT",
        cache_subdir="matbert_models",
        tokenizer_class=AutoTokenizer,
        model_class=AutoModel,
        smiles_list=material_formulas,  # 使用相同的处理逻辑
        batch_size=batch_size
    )

# --- 4. 主执行部分 ---
if __name__ == '__main__':

    # 准备一个更大的SMILES列表用于压力测试
    base_molecules = [
        'CCO', 'c1ccccc1', 'O=C(C)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    ]
    molecules_to_test = base_molecules * 250  # 模拟一个包含1000个分子的大型数据集

    print(f"\n准备为 {len(molecules_to_test)} 个分子计算 embedding...")

    all_embeddings = {}

    # --- 依次调用每个模型，并指定一个合适的batch_size ---
    # 你可以根据你的GPU显存调整这个值。
    # 对于16GB GPU，64通常是一个安全且高效的选择。如果还OOM，可以设为32, 16, 8...
    BS = 64

    # 原有模型
    print("\n" + "="*80)
    print("加载原有模型...")
    print("="*80)
    all_embeddings['chemberta'] = get_chemberta_embedding(molecules_to_test, batch_size=BS)
    all_embeddings['molt5'] = get_molt5_embedding(molecules_to_test, batch_size=BS)
    all_embeddings['chemroberta'] = get_chemroberta_embedding(molecules_to_test, batch_size=BS)

    # 新增模型
    print("\n" + "="*80)
    print("加载新增模型...")
    print("="*80)
    all_embeddings['biogpt'] = get_biogpt_embedding(molecules_to_test, batch_size=16)  # BioGPT 使用较小的 batch_size
    all_embeddings['roberta'] = get_roberta_embedding(molecules_to_test, batch_size=BS)

    # --- 打印结果摘要 ---
    print("\n\n" + "="*80)
    print("           Transformer-based Embedding 结果摘要")
    print("="*80)
    for name, embeddings in all_embeddings.items():
        if embeddings is not None:
            num_molecules, embedding_dim = embeddings.shape
            print(f"✅ 模型: {name:<15} | 形状: ({num_molecules}, {embedding_dim:<4}) | 预览: {np.round(embeddings[0, :3], 4)}")
        else:
            print(f"❌ 模型: {name:<15} | 生成失败")
    print("="*80)
