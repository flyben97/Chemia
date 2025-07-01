# examples/predict_from_files_example.py

import subprocess
import os
import sys
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel

# 初始化 rich 控制台，用于美化输出
console = Console()

def run_file_mode_prediction(
    model_path: str,
    config_path: str,
    input_file: str,
    output_file: str,
    scaler_path: str = None,
    encoder_path: str = None,
    verbose: bool = False
):
    """
    一个Python封装函数，用于以“文件模式”调用命令行预测脚本，并提供简洁的用户界面。
    
    Args:
        model_path (str): 模型文件的直接路径。
        config_path (str): run_config.json 文件的直接路径。
        input_file (str): 用于预测的输入CSV文件路径。
        output_file (str): 保存预测结果的输出CSV文件路径。
        scaler_path (str, optional): 标准化器（scaler）对象的直接路径。
        encoder_path (str, optional): 标签编码器（encoder）对象的直接路径。
        verbose (bool): 如果为 True, 则显示详细的特征生成日志。
    """
    # 构建将要执行的命令行命令
    command = [
        "python",
        "utils/predictor.py",
        "--model_path", model_path,
        "--config_path", config_path,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    
    # 只有在提供了有效路径时，才添加可选参数
    if scaler_path:
        command.extend(["--scaler_path", scaler_path])
    if encoder_path:
        command.extend(["--encoder_path", encoder_path])
    if verbose:
        command.append("--verbose")

    # 美化UI：打印一个简洁的标题和格式化后的多行命令
    console.print("\n[bold blue]---> Executing Prediction Command[/bold blue]")
    formatted_command = f"$ {command[0]} {command[1]} \\"
    i = 2
    while i < len(command):
        arg_name = command[i]
        if (i + 1 < len(command)) and not command[i+1].startswith('--'):
            arg_value = command[i+1]
            formatted_command += f"\n    {arg_name} {arg_value} \\"
            i += 2
        else:
            formatted_command += f"\n    {arg_name} \\"
            i += 1
    console.print(f"[dim]{formatted_command.rstrip(' \\')}[/dim]\n")
    
    try:
        # 确保子进程在项目的根目录运行，以避免模块导入问题
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 创建一个包含 FORCE_COLOR 的环境，以强制子进程输出颜色
        process_env = os.environ.copy()
        process_env["FORCE_COLOR"] = "1"
        
        # 执行子进程并捕获其输出
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            cwd=project_root,
            env=process_env
        )
        
        # 使用低级的 sys.stdout.write 来打印已包含颜色代码的输出，避免二次渲染
        if result.stdout:
            sys.stdout.write(result.stdout)
        
        if result.stderr:
            console.print(f"[bold red]Errors:[/bold red]\n{result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        # 如果子进程执行失败，打印错误信息
        console.print(f"\n[bold red]FATAL ERROR: Prediction script failed with exit code {e.returncode}[/bold red]")
        if e.stdout:
            console.print("[bold yellow]--- stdout from failed process ---[/bold yellow]")
            sys.stdout.write(e.stdout)
        if e.stderr:
            console.print("[bold red]--- stderr from failed process ---[/bold red]")
            sys.stderr.write(e.stderr)

def create_prediction_input_for_test_folder() -> str:
    """
    创建一个虚拟的CSV输入文件，其结构严格匹配 'test' 文件夹中配置文件所定义的需求。
    这包括SMILES列和正确数量的预计算特征列。
    """
    console.print("[dim]Creating a dummy input file for 'test' folder scenario...[/dim]")
    
    num_samples = 2
    num_precomputed_features = 35  # 来自于配置文件中的 "feature_columns": "4:39"

    # SMILES 数据
    smiles_data = {
        'Reactant1': ['CCO', 'c1ccccc1'],
        'Reactant2': ['CC(=O)O', 'CNC(=O)N'],
        'Ligand': ['O=C(C)Oc1ccccc1C(=O)O', 'CCN(CC)CC']
    }
    df_smiles = pd.DataFrame(smiles_data)

    # 虚拟的前置列，以确保预计算特征的列索引正确
    dummy_cols_before_features = {
        'id': [f'pred_{i+1}' for i in range(num_samples)],
        'rxn_id': [f"rxn_pred_{i}" for i in range(num_samples)],
        'catalyst': [f"cat_pred_{i}" for i in range(num_samples)],
        'solvent': [f"sol_pred_{i}" for i in range(num_samples)]
    }
    df_before = pd.DataFrame(dummy_cols_before_features)

    # 虚拟的预计算特征矩阵
    feature_matrix = np.random.rand(num_samples, num_precomputed_features)
    feature_names = [f'precomp_feat_{i+1}' for i in range(num_precomputed_features)]
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)

    # 按正确的顺序拼接所有部分
    df_final = pd.concat([df_before, df_features, df_smiles], axis=1)

    os.makedirs("data", exist_ok=True)
    dummy_path = "data/prediction_input_for_test.csv"
    df_final.to_csv(dummy_path, index=False)
    console.print(f"[dim]Dummy input file created at: '{dummy_path}'[/dim]")
    return dummy_path

if __name__ == "__main__":
    
    # --- 配置 'test' 文件夹的预测任务 ---
    
    # 1. 定义 'test' 文件夹中所有相关文件的路径
    TEST_FOLDER_DIR = "test"
    MODEL_FILE = os.path.join(TEST_FOLDER_DIR, "xgboost_model.json")
    CONFIG_FILE = os.path.join(TEST_FOLDER_DIR, "run_config.json")
    SCALER_FILE = os.path.join(TEST_FOLDER_DIR, "processed_dataset_scaler.joblib")
    ENCODER_FILE = os.path.join(TEST_FOLDER_DIR, "processed_dataset_label_encoder.joblib")
    
    # 2. 检查所有必需文件是否存在
    required_files = [MODEL_FILE, CONFIG_FILE]
    if not all(os.path.exists(f) for f in required_files):
        console.print(Panel(
            "One or more required files were not found in the 'test' directory.\n"
            f"Please ensure the following files exist:\n"
            f"  - [cyan]{MODEL_FILE}[/cyan]\n"
            f"  - [cyan]{CONFIG_FILE}[/cyan]\n",
            title="[bold red]File Not Found Error[/bold red]",
            border_style="red"
        ))
    else:
        # 3. 创建一个符合配置要求的虚拟输入文件
        input_file_path = create_prediction_input_for_test_folder()
        
        # 4. 定义输出文件的完整路径 (我们希望结果保存在'test'文件夹内)
        output_file_path = os.path.join(TEST_FOLDER_DIR, "predictions_on_new_data.csv")

        # 5. 执行预测
        run_file_mode_prediction(
            model_path=MODEL_FILE,
            config_path=CONFIG_FILE,
            input_file=input_file_path,
            output_file=output_file_path,
            scaler_path=SCALER_FILE if os.path.exists(SCALER_FILE) else None,
            encoder_path=ENCODER_FILE if os.path.exists(ENCODER_FILE) else None,
            verbose=False  # 设置为 True 可以看到所有特征生成的详细日志
        )