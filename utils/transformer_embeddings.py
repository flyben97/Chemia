# utils/transformer_embeddings.py
# (Formerly embedder.py)
#
# =============================================================================
# 运行前请确保已安装所有依赖:
#
# 1. 激活你的 Conda 环境 (e.g., `conda activate craft`)
# 2. 安装核心库:
#    pip install torch transformers
#
# 3. (重要!) 安装 MolT5 需要的 sentencepiece 库:
#    pip install sentencepiece
# =============================================================================

import torch
import numpy as np
from typing import List, Optional

# --- 1. 设备配置：自动选择 GPU 或 CPU ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 使用设备 (transformer_embeddings): {DEVICE} ---")


# --- 2. 模型封装函数 ---

def get_chemberta_embedding(
    smiles_list: List[str], 
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
) -> Optional[np.ndarray]:
    """
    使用 ChemBERTa 模型为 SMILES 列表计算 embedding。
    使用所有 token 的 last_hidden_state 的平均值作为分子表示。
    """
    print(f"\n[ChemBERTa] 正在计算 '{model_name}' 的 embedding...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            
            outputs = model(**inputs)
            # 平均池化策略
            embedding = outputs.last_hidden_state.mean(dim=1)
            
        print("[ChemBERTa] 计算完成。")
        return embedding.cpu().numpy()

    except Exception as e:
        print(f"[ChemBERTa] 发生错误: {e}")
        return None


def get_molt5_embedding(
    smiles_list: List[str], 
    model_name: str = "laituan245/molt5-base"
) -> Optional[np.ndarray]:
    """
    使用 MolT5 模型的 Encoder 部分为 SMILES 列表计算 embedding。
    """
    print(f"\n[MolT5] 正在计算 '{model_name}' 的 embedding...")
    try:
        from transformers import T5Tokenizer, T5EncoderModel
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        print("[MolT5] 计算完成。")
        return embedding.cpu().numpy()

    except Exception as e:
        print(f"[MolT5] 发生错误: {e}")
        return None

def get_chemroberta_embedding(
    smiles_list: List[str], 
    model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k"
) -> Optional[np.ndarray]:
    """
    使用基于 RoBERTa 的化学模型为 SMILES 列表计算 embedding。
    """
    print(f"\n[ChemRoBERTa] 正在计算 '{model_name}' 的 embedding...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        print("[ChemRoBERTa] 计算完成。")
        return embedding.cpu().numpy()

    except Exception as e:
        print(f"[ChemRoBERTa] 发生错误: {e}")
        return None


# --- 3. 主执行部分 ---
if __name__ == '__main__':
    
    # 准备一批 SMILES 数据用于测试
    molecules_to_test = [
        'CCO',                                      # 乙醇
        'c1ccccc1',                                 # 苯
        'O=C(C)Oc1ccccc1C(=O)O',                     # 阿司匹林
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'              # 咖啡因
    ]
    
    print(f"\n准备为 {len(molecules_to_test)} 个分子计算 embedding...")
    
    all_embeddings = {}

    # --- 依次调用每个模型 ---
    all_embeddings['chemberta'] = get_chemberta_embedding(molecules_to_test)
    all_embeddings['molt5'] = get_molt5_embedding(molecules_to_test)
    all_embeddings['chemroberta'] = get_chemroberta_embedding(molecules_to_test)
    
    # --- 打印结果摘要 ---
    print("\n\n" + "="*60)
    print("           Transformer-based Embedding 结果摘要")
    print("="*60)
    for name, embeddings in all_embeddings.items():
        if embeddings is not None:
            # 获取 embedding 的维度信息
            num_molecules, embedding_dim = embeddings.shape
            print(f"✅ 模型: {name:<12} | 形状: ({num_molecules}, {embedding_dim})")
            # 打印第一个分子的 embedding 的前5个维度作为预览
            print(f"   - 预览 (第一个分子): {np.round(embeddings[0, :5], 4)}")
        else:
            print(f"❌ 模型: {name:<12} | 生成失败")
    print("="*60)