# main.py
import argparse
from rich.console import Console

console = Console()

def main():
    """
    The sole entry point of the application.
    It parses the config file path and passes it to the run manager.
    """
    from core.config_loader import load_config
    from core.run_manager import start_experiment_run

    parser = argparse.ArgumentParser(description="Run ML experiments based on a configuration file.")
    parser.add_argument(
        '--config', 
        type=str, 
        default="config.yaml", 
        help="Path to the experiment configuration YAML file."
    )
    args = parser.parse_args()

    try:
        # 1. Load configuration
        config = load_config(args.config)

        # 2. Start the experiment run with the loaded config
        start_experiment_run(config)
        
        console.print("\n==========================================================")
        console.print("==========    ML PIPELINE FINISHED SUCCESSFULLY   ==========")
        console.print("============================================================")

    except (FileNotFoundError, ValueError) as e:
        console.print(f"\n[bold red]An error occurred during setup: {e}[/bold red]")
        console.print("[bold yellow]Please check your configuration file and file paths.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False)


if __name__ == "__main__":
    main()