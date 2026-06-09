"""
Transformer模型实现
支持分子序列和特征的Transformer架构
"""

import math
import numpy as np
from typing import Optional, List, Union, Dict, Any, Tuple
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.nn import MultiheadAttention, LayerNorm, Dropout, Linear
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. Transformer models will not be available.")

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        BertModel, BertTokenizer, BertConfig,
        RobertaModel, RobertaTokenizer, RobertaConfig,
        GPT2Model, GPT2Tokenizer, GPT2Config,
        T5Model, T5Tokenizer, T5Config
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not installed. Pre-trained models will not be available.")


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SMILESTokenizer:
    """SMILES分词器"""

    def __init__(self, vocab_size: int = 1000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # 基础SMILES字符集
        self.base_chars = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # 原子
            '(', ')', '[', ']', '=', '#', '+', '-',           # 结构符号
            '1', '2', '3', '4', '5', '6', '7', '8', '9',      # 数字
            'c', 'n', 'o', 's', 'p',                         # 芳香原子
            '@', '/', '\\', '%'                              # 立体化学
        ]

        # 特殊token
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

        # 构建词汇表
        self.char_to_idx = self.special_tokens.copy()
        for i, char in enumerate(self.base_chars):
            self.char_to_idx[char] = i + len(self.special_tokens)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # 填充到指定词汇表大小
        while len(self.char_to_idx) < vocab_size:
            dummy_token = f'<DUMMY_{len(self.char_to_idx)}>'
            self.char_to_idx[dummy_token] = len(self.char_to_idx)
            self.idx_to_char[len(self.idx_to_char)] = dummy_token

    def encode(self, smiles: str) -> List[int]:
        """编码SMILES字符串"""
        tokens = [self.special_tokens['<BOS>']]

        for char in smiles:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.special_tokens['<UNK>'])

        tokens.append(self.special_tokens['<EOS>'])

        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.special_tokens['<PAD>']] * (self.max_length - len(tokens)))

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """解码token序列"""
        chars = []
        for token in tokens:
            if token in self.idx_to_char:
                char = self.idx_to_char[token]
                if char not in ['<PAD>', '<BOS>', '<EOS>']:
                    chars.append(char)
        return ''.join(chars)

    def batch_encode(self, smiles_list: List[str]) -> torch.Tensor:
        """批量编码"""
        encoded = [self.encode(smiles) for smiles in smiles_list]
        return torch.tensor(encoded, dtype=torch.long)


class SMILESTransformer(nn.Module):
    """基于SMILES序列的Transformer模型"""

    def __init__(self,
                 vocab_size: int = 1000,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_length: int = 512,
                 output_dim: int = 1,
                 task_type: str = 'regression'):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer models")

        self.d_model = d_model
        self.task_type = task_type
        self.max_length = max_length

        # 分词器
        self.tokenizer = SMILESTokenizer(vocab_size, max_length)

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
                module.weight.data.uniform_(-initrange, initrange)

    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """
        前向传播

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            预测结果
        """
        # 编码SMILES
        src = self.tokenizer.batch_encode(smiles_list)  # [batch_size, seq_len]
        src = src.to(next(self.parameters()).device)

        # 创建padding mask
        pad_token = self.tokenizer.special_tokens['<PAD>']
        src_key_padding_mask = (src == pad_token)

        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Transformer编码
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, d_model]

        # 全局平均池化（忽略padding位置）
        mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, d_model]

        # 分类/回归
        output = self.classifier(pooled)  # [batch_size, output_dim]

        return output


class FeatureTransformer(nn.Module):
    """基于特征的Transformer模型"""

    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 output_dim: int = 1,
                 task_type: str = 'regression'):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer models")

        self.d_model = d_model
        self.task_type = task_type

        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]

        Returns:
            预测结果
        """
        if x.dim() == 2:
            # 如果是2D，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]

        # 位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Transformer编码
        output = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

        # 全局平均池化
        pooled = output.mean(dim=1)  # [batch_size, d_model]

        # 分类/回归
        output = self.classifier(pooled)  # [batch_size, output_dim]

        return output


class PretrainedTransformer(nn.Module):
    """基于预训练模型的Transformer"""

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 freeze_backbone: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for pre-trained models")

        self.model_name = model_name
        self.task_type = task_type

        # 加载预训练模型
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logging.warning(f"Failed to load {model_name}, using BERT as fallback: {e}")
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.backbone = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 获取隐藏层维度
        hidden_size = self.config.hidden_size

        # 输出层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """
        前向传播

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            预测结果
        """
        # 分词
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # 移动到正确的设备
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # 通过backbone
        outputs = self.backbone(**encoded)

        # 使用[CLS] token的表示或平均池化
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # 平均池化
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded['attention_mask']
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        # 分类/回归
        output = self.classifier(pooled_output)

        return output


class MultiModalTransformer(nn.Module):
    """多模态Transformer（SMILES + 特征）"""

    def __init__(self,
                 vocab_size: int = 1000,
                 feature_dim: int = 100,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_length: int = 512,
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 fusion_method: str = 'concat'):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer models")

        self.d_model = d_model
        self.task_type = task_type
        self.fusion_method = fusion_method

        # SMILES分支
        self.smiles_tokenizer = SMILESTokenizer(vocab_size, max_length)
        self.smiles_embedding = nn.Embedding(vocab_size, d_model)
        self.smiles_pos_encoder = PositionalEncoding(d_model, dropout, max_length)

        smiles_encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.smiles_encoder = TransformerEncoder(smiles_encoder_layers, num_layers)

        # 特征分支
        self.feature_projection = nn.Linear(feature_dim, d_model)

        feature_encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.feature_encoder = TransformerEncoder(feature_encoder_layers, num_layers // 2)

        # 融合层
        if fusion_method == 'concat':
            fusion_input_dim = d_model * 2
        elif fusion_method == 'attention':
            self.cross_attention = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            fusion_input_dim = d_model
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            fusion_input_dim = d_model
        else:
            fusion_input_dim = d_model

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, smiles_list: List[str], features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            smiles_list: SMILES字符串列表
            features: 额外特征 [batch_size, feature_dim]

        Returns:
            预测结果
        """
        device = next(self.parameters()).device

        # SMILES分支
        smiles_tokens = self.smiles_tokenizer.batch_encode(smiles_list).to(device)
        pad_token = self.smiles_tokenizer.special_tokens['<PAD>']
        smiles_mask = (smiles_tokens == pad_token)

        smiles_emb = self.smiles_embedding(smiles_tokens) * math.sqrt(self.d_model)
        smiles_emb = self.smiles_pos_encoder(smiles_emb.transpose(0, 1)).transpose(0, 1)

        smiles_output = self.smiles_encoder(smiles_emb, src_key_padding_mask=smiles_mask)

        # SMILES池化
        smiles_mask_float = (~smiles_mask).float().unsqueeze(-1)
        smiles_pooled = (smiles_output * smiles_mask_float).sum(dim=1) / smiles_mask_float.sum(dim=1)

        # 特征分支
        features = features.to(device)
        feature_emb = self.feature_projection(features).unsqueeze(1)  # [batch_size, 1, d_model]
        feature_output = self.feature_encoder(feature_emb)
        feature_pooled = feature_output.squeeze(1)  # [batch_size, d_model]

        # 融合
        if self.fusion_method == 'concat':
            fused = torch.cat([smiles_pooled, feature_pooled], dim=-1)
        elif self.fusion_method == 'attention':
            # 使用交叉注意力
            smiles_attended, _ = self.cross_attention(
                smiles_pooled.unsqueeze(1),
                feature_pooled.unsqueeze(1),
                feature_pooled.unsqueeze(1)
            )
            fused = smiles_attended.squeeze(1)
        elif self.fusion_method == 'gated':
            # 门控融合
            gate_input = torch.cat([smiles_pooled, feature_pooled], dim=-1)
            gate_weights = self.gate(gate_input)
            fused = smiles_pooled * gate_weights + feature_pooled * (1 - gate_weights)
        else:
            # 简单相加
            fused = smiles_pooled + feature_pooled

        # 输出
        output = self.classifier(fused)

        return output


def create_transformer_model(model_type: str,
                           vocab_size: int = 1000,
                           input_dim: int = 100,
                           d_model: int = 512,
                           nhead: int = 8,
                           num_layers: int = 6,
                           dim_feedforward: int = 2048,
                           dropout: float = 0.1,
                           max_length: int = 512,
                           output_dim: int = 1,
                           task_type: str = 'regression',
                           **kwargs) -> nn.Module:
    """
    Transformer模型工厂函数

    Args:
        model_type: 模型类型 ('smiles', 'feature', 'pretrained', 'multimodal')
        vocab_size: 词汇表大小（SMILES模型）
        input_dim: 输入特征维度（特征模型）
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: 层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout率
        max_length: 最大序列长度
        output_dim: 输出维度
        task_type: 任务类型
        **kwargs: 其他参数

    Returns:
        Transformer模型
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for Transformer models")

    model_type = model_type.lower()

    if model_type == 'smiles':
        return SMILESTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length,
            output_dim=output_dim,
            task_type=task_type
        )

    elif model_type == 'feature':
        return FeatureTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim,
            task_type=task_type
        )

    elif model_type == 'pretrained':
        model_name = kwargs.get('model_name', 'bert-base-uncased')
        freeze_backbone = kwargs.get('freeze_backbone', False)
        return PretrainedTransformer(
            model_name=model_name,
            output_dim=output_dim,
            task_type=task_type,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )

    elif model_type == 'multimodal':
        feature_dim = kwargs.get('feature_dim', input_dim)
        fusion_method = kwargs.get('fusion_method', 'concat')
        return MultiModalTransformer(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length,
            output_dim=output_dim,
            task_type=task_type,
            fusion_method=fusion_method
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 测试函数
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # 测试SMILES Transformer
        print("Testing SMILES Transformer...")
        smiles_model = create_transformer_model(
            model_type='smiles',
            vocab_size=100,
            d_model=128,
            nhead=4,
            num_layers=2,
            output_dim=1
        )

        test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
        output = smiles_model(test_smiles)
        print(f"SMILES Transformer output shape: {output.shape}")

        # 测试特征Transformer
        print("\nTesting Feature Transformer...")
        feature_model = create_transformer_model(
            model_type='feature',
            input_dim=50,
            d_model=128,
            nhead=4,
            num_layers=2,
            output_dim=1
        )

        test_features = torch.randn(3, 50)
        output = feature_model(test_features)
        print(f"Feature Transformer output shape: {output.shape}")

        # 测试多模态Transformer
        print("\nTesting MultiModal Transformer...")
        multimodal_model = create_transformer_model(
            model_type='multimodal',
            vocab_size=100,
            feature_dim=20,
            d_model=128,
            nhead=4,
            num_layers=2,
            output_dim=1,
            fusion_method='attention'
        )

        test_features_mm = torch.randn(3, 20)
        output = multimodal_model(test_smiles, test_features_mm)
        print(f"MultiModal Transformer output shape: {output.shape}")

        print("\n✅ All Transformer models created successfully!")

    else:
        print("PyTorch not available, skipping tests")
