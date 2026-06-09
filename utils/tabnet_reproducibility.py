#!/usr/bin/env python3
"""
TabNet模型复现性增强工具
确保TabNet模型的完整保存和精确复现
"""

import os
import json
import torch
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from rich.console import Console

console = Console()

class TabNetReproducibilityManager:
    """TabNet模型复现性管理器"""

    def __init__(self):
        self.console = console

    def set_reproducible_environment(self, seed: int = 42):
        """设置完全可复现的环境"""
        # 设置NumPy随机种子
        np.random.seed(seed)

        # 设置PyTorch随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 设置确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 设置环境变量
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.console.print(f"[green]✓ 设置可复现环境 (seed={seed})[/green]")

    def save_tabnet_model_complete(self,
                                   model: Union[TabNetRegressor, TabNetClassifier],
                                   model_path: str,
                                   training_params: Optional[Dict] = None,
                                   model_config: Optional[Dict] = None,
                                   additional_metadata: Optional[Dict] = None) -> str:
        """
        完整保存TabNet模型，包括所有必要的复现信息

        Args:
            model: 训练好的TabNet模型
            model_path: 保存路径（不含扩展名）
            training_params: 训练参数
            model_config: 模型配置参数
            additional_metadata: 额外的元数据

        Returns:
            实际保存的模型文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 1. 保存TabNet模型本身
        model.save_model(model_path)
        actual_model_path = model_path + '.zip'

        # 2. 保存完整的元数据
        metadata = {
            'model_type': model.__class__.__name__,
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__,
            'device_used': str(model.device),
            'save_timestamp': str(torch.utils.data.get_worker_info()),
            'model_config': model_config or {},
            'training_params': training_params or {},
            'additional_metadata': additional_metadata or {}
        }

        # 保存模型的网络配置
        if hasattr(model, 'network'):
            network_config = {
                'input_dim': getattr(model.network, 'input_dim', None),
                'output_dim': getattr(model.network, 'output_dim', None),
                'n_d': getattr(model.network, 'n_d', None),
                'n_a': getattr(model.network, 'n_a', None),
                'n_steps': getattr(model.network.tabnet.encoder, 'n_steps', None) if hasattr(model.network, 'tabnet') else None,
                'gamma': getattr(model.network.tabnet.encoder, 'gamma', None) if hasattr(model.network, 'tabnet') else None,
            }
            metadata['network_config'] = network_config

        # 保存元数据到JSON文件
        metadata_path = model_path + '_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, default=str)

        # 3. 保存随机状态
        random_state_path = model_path + '_random_state.pkl'
        random_states = {
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            random_states['torch_cuda_state'] = torch.cuda.get_rng_state_all()

        with open(random_state_path, 'wb') as f:
            pickle.dump(random_states, f)

        self.console.print(f"[green]✓ 完整保存TabNet模型到:[/green]")
        self.console.print(f"  - 模型文件: [dim]{actual_model_path}[/dim]")
        self.console.print(f"  - 元数据: [dim]{metadata_path}[/dim]")
        self.console.print(f"  - 随机状态: [dim]{random_state_path}[/dim]")

        return actual_model_path

    def load_tabnet_model_complete(self,
                                   model_path: str,
                                   restore_random_state: bool = False,
                                   task_type: Optional[str] = None) -> Union[TabNetRegressor, TabNetClassifier]:
        """
        完整加载TabNet模型，包括所有复现信息

        Args:
            model_path: 模型文件路径（可以是.zip文件或基础路径）
            restore_random_state: 是否恢复随机状态
            task_type: 任务类型，如果为None则自动推断

        Returns:
            加载的TabNet模型
        """
        # 处理路径
        if model_path.endswith('.zip'):
            base_path = model_path[:-4]
            actual_model_path = model_path
        else:
            base_path = model_path
            actual_model_path = model_path + '.zip'

        if not os.path.exists(actual_model_path):
            raise FileNotFoundError(f"TabNet模型文件未找到: {actual_model_path}")

        # 1. 加载元数据
        metadata_path = base_path + '_metadata.json'
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.console.print(f"[green]✓ 加载元数据:[/green] [dim]{metadata_path}[/dim]")
        else:
            self.console.print("[yellow]⚠️  未找到元数据文件，使用默认推断[/yellow]")

        # 2. 恢复随机状态（如果需要）
        if restore_random_state:
            random_state_path = base_path + '_random_state.pkl'
            if os.path.exists(random_state_path):
                with open(random_state_path, 'rb') as f:
                    random_states = pickle.load(f)

                np.random.set_state(random_states['numpy_state'])
                torch.set_rng_state(random_states['torch_state'])
                if torch.cuda.is_available() and 'torch_cuda_state' in random_states:
                    torch.cuda.set_rng_state_all(random_states['torch_cuda_state'])

                self.console.print("[green]✓ 恢复随机状态[/green]")
            else:
                self.console.print("[yellow]⚠️  未找到随机状态文件[/yellow]")

        # 3. 推断任务类型
        if task_type is None:
            model_type = metadata.get('model_type', '')
            if 'Regressor' in model_type:
                task_type = 'regression'
            elif 'Classifier' in model_type:
                task_type = 'classification'
            else:
                # 尝试从网络配置推断
                network_config = metadata.get('network_config', {})
                output_dim = network_config.get('output_dim', 1)
                task_type = 'regression' if output_dim == 1 else 'classification'

        # 4. 创建并加载模型
        try:
            if task_type == 'regression':
                model = TabNetRegressor()
            else:
                model = TabNetClassifier()

            model.load_model(actual_model_path)

            self.console.print(f"[green]✓ 成功加载TabNet{task_type}模型:[/green] [dim]{actual_model_path}[/dim]")

            # 验证加载的模型配置
            if metadata.get('network_config'):
                expected_config = metadata['network_config']
                if hasattr(model, 'network'):
                    actual_input_dim = getattr(model.network, 'input_dim', None)
                    expected_input_dim = expected_config.get('input_dim')
                    if expected_input_dim and actual_input_dim != expected_input_dim:
                        self.console.print(f"[yellow]⚠️  输入维度不匹配: 期望{expected_input_dim}, 实际{actual_input_dim}[/yellow]")

            return model

        except Exception as e:
            self.console.print(f"[red]❌ 加载TabNet模型失败: {e}[/red]")
            raise

def enhance_tabnet_save_function():
    """增强现有的TabNet保存函数"""

    enhanced_code = '''
def save_tabnet_model_enhanced(model, model_path_base, model_name, training_params=None, console=None):
    """
    增强的TabNet模型保存函数，确保完整的复现性

    Args:
        model: TabNet模型实例
        model_path_base: 基础保存路径（不含扩展名）
        model_name: 模型名称
        training_params: 训练参数字典
        console: Rich控制台对象

    Returns:
        实际保存的模型文件路径
    """
    from utils.tabnet_reproducibility import TabNetReproducibilityManager

    if console is None:
        from rich.console import Console
        console = Console()

    # 使用增强的保存方法
    manager = TabNetReproducibilityManager()

    # 提取模型配置
    model_config = {
        'n_d': getattr(model, 'n_d', None),
        'n_a': getattr(model, 'n_a', None),
        'n_steps': getattr(model, 'n_steps', None),
        'gamma': getattr(model, 'gamma', None),
        'lambda_sparse': getattr(model, 'lambda_sparse', None),
        'seed': getattr(model, 'seed', None),
        'device_name': str(getattr(model, 'device', 'cpu'))
    }

    actual_path = manager.save_tabnet_model_complete(
        model=model,
        model_path=model_path_base,
        training_params=training_params,
        model_config=model_config,
        additional_metadata={'model_name': model_name}
    )

    console.print(f"[green]✓ 增强保存TabNet模型:[/green] [dim]{actual_path}[/dim]")
    return actual_path
'''

    return enhanced_code

if __name__ == "__main__":
    print("TabNet复现性增强工具")
    print("="*50)

    # 演示增强的保存和加载
    manager = TabNetReproducibilityManager()

    # 设置可复现环境
    manager.set_reproducible_environment(42)

    # 创建测试数据
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype(np.float32)
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    X_test = X_test.astype(np.float32)

    # 训练模型
    model = TabNetRegressor(n_d=8, n_a=8, n_steps=3, seed=42, verbose=0, device_name='cpu')
    model.fit(X_train, y_train, max_epochs=20, patience=5)

    # 使用增强的保存方法
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'enhanced_tabnet_model')

        training_params = {'max_epochs': 20, 'patience': 5}
        saved_path = manager.save_tabnet_model_complete(
            model=model,
            model_path=model_path,
            training_params=training_params
        )

        # 测试加载
        loaded_model = manager.load_tabnet_model_complete(
            model_path=saved_path,
            restore_random_state=True
        )

        # 验证复现性
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)

        max_diff = np.max(np.abs(pred_original - pred_loaded))
        console.print(f"[cyan]预测差异: {max_diff:.10f}[/cyan]")

        if max_diff < 1e-8:
            console.print("[green]✅ 完美复现![/green]")
        else:
            console.print("[yellow]⚠️  存在微小差异[/yellow]")
