#!/usr/bin/env python3
"""
批量工作流程运行器

允许用户将多个 YAML 配置文件放在一个文件夹下，系统依次读取并执行。
支持完整工作流程（训练+预测）或仅训练模式。

使用示例:
    # 批量执行所有配置文件
    python scripts/run_batch_workflow.py --config-dir configs/

    # 仅训练模式
    python scripts/run_batch_workflow.py --config-dir configs/ --training-only

    # 指定输出目录
    python scripts/run_batch_workflow.py --config-dir configs/ --output-dir batch_results/

    # 指定文件模式（默认 *.yaml）
    python scripts/run_batch_workflow.py --config-dir configs/ --pattern "*.yml"
"""


import sys
import yaml

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# --- 标准 CLI 初始化 ---
from utils.cli_setup import standard_cli_setup
from utils.constants import DEFAULT_CONSOLE_WIDTH, DEFAULT_BATCH_OUTPUT_PREFIX
project_root = standard_cli_setup()
# --- END ---

from core.run_manager import RunManager
from utils.io_handler import CHEMIA_BANNER

console = Console(width=DEFAULT_CONSOLE_WIDTH, highlight=False)


class BatchWorkflowRunner:
    """批量工作流程运行器"""

    def __init__(
        self,
        config_dir: str,
        output_dir: Optional[str] = None,
        training_only: bool = False,
        pattern: str = "*.yaml"
    ):
        """
        初始化批量运行器

        Args:
            config_dir: 配置文件目录
            output_dir: 输出目录（可选）
            training_only: 是否仅训练模式
            pattern: 文件匹配模式
        """
        self.config_dir = Path(config_dir)
        self.training_only = training_only
        self.pattern = pattern

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{DEFAULT_BATCH_OUTPUT_PREFIX}_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 验证配置目录
        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")

        if not self.config_dir.is_dir():
            raise NotADirectoryError(f"不是目录: {self.config_dir}")

    def find_config_files(self) -> List[Path]:
        """
        查找配置文件

        Returns:
            List[Path]: 配置文件列表
        """
        config_files = sorted(self.config_dir.glob(self.pattern))

        if not config_files:
            console.print(f"[yellow]警告: 在 {self.config_dir} 中未找到匹配 {self.pattern} 的文件[/yellow]")
            return []

        return config_files

    def validate_config(self, config_path: Path) -> bool:
        """
        验证配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            bool: 是否有效
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                console.print(f"[red]✗ {config_path.name}: 配置不是字典[/red]")
                return False

            # 检查必要的字段
            required_fields = ['data', 'task_type', 'features', 'split_config', 'training']
            missing_fields = [field for field in required_fields if field not in config]

            if missing_fields:
                console.print(f"[red]✗ {config_path.name}: 缺少必要字段 {missing_fields}[/red]")
                return False

            return True

        except Exception as e:
            console.print(f"[red]✗ {config_path.name}: 配置文件错误 - {str(e)}[/red]")
            return False

    def run_single_config(self, config_path: Path, index: int, total: int) -> Dict[str, Any]:
        """
        运行单个配置文件

        Args:
            config_path: 配置文件路径
            index: 当前索引
            total: 总数

        Returns:
            Dict[str, Any]: 运行结果
        """
        result = {
            'config_file': config_path.name,
            'status': 'pending',
            'output_dir': None,
            'error': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }

        try:
            console.print(f"\n[cyan]{'='*80}[/cyan]")
            console.print(f"[cyan]处理 [{index}/{total}]: {config_path.name}[/cyan]")
            console.print(f"[cyan]{'='*80}[/cyan]")

            # 创建实验输出目录
            config_name = config_path.stem
            exp_output_dir = self.output_dir / config_name
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            result['output_dir'] = str(exp_output_dir)

            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 运行工作流程
            console.print(f"[green]✓ 配置文件已加载[/green]")

            run_manager = RunManager(
                config=config,
                output_dir=str(exp_output_dir),
                training_only=self.training_only
            )

            console.print(f"[green]✓ 运行管理器已初始化[/green]")

            # 执行工作流程
            if self.training_only:
                console.print(f"[blue]→ 开始训练...[/blue]")
                run_manager.run_training_only()
            else:
                console.print(f"[blue]→ 开始完整工作流程...[/blue]")
                run_manager.run_full_workflow()

            result['status'] = 'success'
            console.print(f"[green]✓ {config_path.name} 执行成功[/green]")

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            console.print(f"[red]✗ {config_path.name} 执行失败: {str(e)}[/red]")

        finally:
            result['end_time'] = datetime.now().isoformat()

        return result

    def run_batch(self) -> Dict[str, Any]:
        """
        运行批量工作流程

        Returns:
            Dict[str, Any]: 批量运行结果
        """
        # 打印欢迎信息
        console.print(CHEMIA_BANNER)
        console.print(Panel(
            "[bold cyan]批量工作流程运行器[/bold cyan]",
            expand=False
        ))

        # 查找配置文件
        config_files = self.find_config_files()

        if not config_files:
            console.print("[red]错误: 未找到配置文件[/red]")
            return {
                'status': 'failed',
                'total': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }

        console.print(f"\n[cyan]找到 {len(config_files)} 个配置文件[/cyan]")
        console.print(f"[cyan]输出目录: {self.output_dir}[/cyan]")
        console.print(f"[cyan]模式: {'仅训练' if self.training_only else '完整工作流程'}[/cyan]\n")

        # 验证配置文件
        valid_configs = []
        for config_path in config_files:
            if self.validate_config(config_path):
                valid_configs.append(config_path)

        if not valid_configs:
            console.print("[red]错误: 没有有效的配置文件[/red]")
            return {
                'status': 'failed',
                'total': len(config_files),
                'successful': 0,
                'failed': len(config_files),
                'results': []
            }

        console.print(f"[green]✓ {len(valid_configs)} 个配置文件有效[/green]\n")

        # 运行配置文件
        results = []
        successful = 0
        failed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]处理配置文件...",
                total=len(valid_configs)
            )

            for index, config_path in enumerate(valid_configs, 1):
                result = self.run_single_config(config_path, index, len(valid_configs))
                results.append(result)

                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1

                progress.update(task, advance=1)

        # 生成总结
        summary = {
            'status': 'completed',
            'total': len(valid_configs),
            'successful': successful,
            'failed': failed,
            'output_dir': str(self.output_dir),
            'start_time': results[0]['start_time'] if results else None,
            'end_time': results[-1]['end_time'] if results else None,
            'results': results
        }

        # 打印总结
        self._print_summary(summary)

        # 保存结果
        self._save_results(summary)

        return summary

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """打印总结"""
        console.print(f"\n[cyan]{'='*80}[/cyan]")
        console.print("[cyan]批量处理总结[/cyan]")
        console.print(f"[cyan]{'='*80}[/cyan]\n")

        # 创建总结表格
        table = Table(title="执行结果")
        table.add_column("配置文件", style="cyan")
        table.add_column("状态", style="magenta")
        table.add_column("输出目录", style="green")

        for result in summary['results']:
            status_str = "✓ 成功" if result['status'] == 'success' else "✗ 失败"
            status_color = "green" if result['status'] == 'success' else "red"
            output_dir = result['output_dir'] or "N/A"

            table.add_row(
                result['config_file'],
                f"[{status_color}]{status_str}[/{status_color}]",
                output_dir
            )

        console.print(table)

        # 打印统计信息
        console.print(f"\n[cyan]统计信息:[/cyan]")
        console.print(f"  总数: {summary['total']}")
        console.print(f"  [green]成功: {summary['successful']}[/green]")
        console.print(f"  [red]失败: {summary['failed']}[/red]")
        console.print(f"  输出目录: {summary['output_dir']}")

    def _save_results(self, summary: Dict[str, Any]) -> None:
        """保存结果到 JSON 文件"""
        results_file = self.output_dir / "batch_results.json"

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            console.print(f"\n[green]✓ 结果已保存到: {results_file}[/green]")

        except Exception as e:
            console.print(f"\n[yellow]⚠ 保存结果失败: {str(e)}[/yellow]")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量工作流程运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量执行所有配置文件
  python scripts/run_batch_workflow.py --config-dir configs/

  # 仅训练模式
  python scripts/run_batch_workflow.py --config-dir configs/ --training-only

  # 指定输出目录
  python scripts/run_batch_workflow.py --config-dir configs/ --output-dir batch_results/

  # 指定文件模式
  python scripts/run_batch_workflow.py --config-dir configs/ --pattern "*.yml"
        """
    )

    parser.add_argument(
        '--config-dir',
        required=True,
        help='配置文件目录'
    )

    parser.add_argument(
        '--output-dir',
        default=None,
        help=f'输出目录（可选，默认为 {DEFAULT_BATCH_OUTPUT_PREFIX}_TIMESTAMP）'
    )

    parser.add_argument(
        '--training-only',
        action='store_true',
        help='仅训练模式（不进行预测）'
    )

    parser.add_argument(
        '--pattern',
        default='*.yaml',
        help='文件匹配模式（默认 *.yaml）'
    )

    args = parser.parse_args()

    try:
        runner = BatchWorkflowRunner(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            training_only=args.training_only,
            pattern=args.pattern
        )

        summary = runner.run_batch()

        # 返回适当的退出码
        sys.exit(0 if summary['failed'] == 0 else 1)

    except Exception as e:
        console.print(f"[red]错误: {str(e)}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()
