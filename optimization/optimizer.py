# optimization/optimizer.py
import pandas as pd
import logging
import sys
import os
import contextlib
from bayes_opt import BayesianOptimization
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED

# 获取专为优化过程设置的、只写入文件的记录器
opt_logger = logging.getLogger("optimization_logger")

@contextlib.contextmanager
def suppress_console_output():
    """
    A context manager to temporarily suppress stdout and logging to the console.
    This does NOT affect loggers specifically configured to write to files.
    """
    original_stdout = sys.stdout
    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        sys.stdout = open(os.devnull, 'w')
        root_logger.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
        root_logger.setLevel(original_level)


class BayesianReactionOptimizer:
    def __init__(self, predictor, space_loader, opt_config, fixed_components=None, feature_gen_config=None, output_dir=None):
        self.predictor = predictor
        self.space_loader = space_loader
        self.config = opt_config
        self.fixed_components = fixed_components or {}
        self.feature_gen_config = feature_gen_config
        self.output_dir = output_dir
        opt_logger.info("Optimizer initialized in robust mode.")

    def objective_function(self, **kwargs):
        dynamic_indices = {key: int(round(val)) for key, val in kwargs.items()}

        result_df = None

        with suppress_console_output():
            try:
                reaction_df = self.space_loader.build_reaction_df(
                    dynamic_indices,
                    fixed_components=self.fixed_components,
                    feature_gen_config=self.feature_gen_config,
                    output_dir=self.output_dir
                )

                result_df = self.predictor.predict_from_df(reaction_df)

            except Exception as e:
                opt_logger.error(f"Error in objective function (suppressed block): {e}", exc_info=True)
                return -999.0

        try:
            if result_df is None or result_df.empty:
                opt_logger.warning("Prediction returned None or empty DataFrame. Returning low score.")
                return -999.0

            if 'prediction' not in result_df.columns:
                 opt_logger.warning("Returned DataFrame is missing the 'prediction' column. Returning low score.")
                 return -999.0

            prediction = result_df['prediction'].iloc[0]
            final_pred = float(prediction) if pd.notna(prediction) else -999.0
            opt_logger.info(f"Objective score: {final_pred:.4f} for params: {dynamic_indices}")
            return final_pred

        except (KeyError, IndexError) as e:
            opt_logger.warning(f"Data access failed after prediction ({e}), returning low score.")
            return -999.0
        except Exception as e:
            opt_logger.error(f"Error processing result in objective function: {e}", exc_info=True)
            return -999.0

    def run(self):
        opt_logger.info("Setting up Bayesian Optimization...")

        if not self.space_loader.pbounds:
            raise ValueError("Search space is empty. Please set at least one component's mode to 'search' in the config.")

        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.space_loader.pbounds,
            random_state=self.config['random_state'],
            verbose=0
        )

        opt_logger.info(f"Running Optimization for {self.config['n_iter']} iterations "
                     f"(plus {self.config['init_points']} initial points)...")

        console = Console()
        console.print("Running Bayesian optimization... (This may take a while, detailed progress in log file)")

        optimizer.maximize(
            init_points=self.config['init_points'],
            n_iter=self.config['n_iter']
        )

        console.print("✓ Optimization completed!")

        opt_logger.info(f"Completed {len(optimizer.res)} optimization iterations")

        final_results = sorted(optimizer.res, key=lambda x: x.get('target', -999), reverse=True)
        opt_logger.info("\n" + "="*50 + "\nTop 5 Results (Full Details):\n" + "="*50)
        for i, res in enumerate(final_results[:5]):
            opt_logger.info(f"  Rank {i+1}: Score = {res.get('target', 'N/A'):.4f}, Raw Params = {res.get('params', {})}")
        opt_logger.info("="*50 + "\n")

        opt_logger.info("Optimization Finished! Generating final report.")
        result_df = self._report_and_save_results(optimizer, console)

        return result_df

    def _report_and_save_results(self, optimizer, console: Console):
        """
        Formats the optimization results, saves them, and prints a summary table to the console.
        Now includes fixed components in the final report.
        """
        results = []

        target_col_name = self.predictor.config.get('data', {})\
            .get('single_file_config', {}).get('target_col', 'prediction')
        predicted_col_name = f"predicted_{target_col_name}"

        if not optimizer.res:
            opt_logger.warning("No results found in optimizer.res list. Cannot generate a report.")
            console.print("[yellow]Warning: Bayesian Optimization produced no results. The model may be failing every prediction.[/yellow]")
            return pd.DataFrame()

        sorted_res = sorted(optimizer.res, key=lambda x: x.get('target', -999), reverse=True)

        # --- START OF FIX: Main logic change is here ---
        for res in sorted_res:
            params = res.get('params', {})
            condition = {predicted_col_name: res.get('target', -999.0)}
            dynamic_indices = {key.replace('_idx',''): int(round(val)) for key, val in params.items()}

            # Iterate through ALL components defined in the space loader to build the full condition
            for name, component in self.space_loader.components.items():
                details = component['details']
                capitalized_name = name.capitalize()

                idx = None
                # Case 1: It was a dynamic (searched) component
                if name.lower() in dynamic_indices:
                    idx = dynamic_indices[name.lower()]

                # Case 2: It was a fixed component
                elif details['mode'] == 'fixed':
                    if 'index' in details:
                        idx = details['index']
                    elif 'row_index' in details and 'data' in component and component['data'] is not None:
                         # For 'row_index', we need to find the corresponding 'Index' value from the dataframe
                         df = component['data']
                         row_idx = details['row_index']
                         if 0 <= row_idx < len(df):
                             idx = df.iloc[row_idx]['Index']
                         else:
                             opt_logger.warning(f"row_index {row_idx} is out of bounds for component '{name}'. Skipping.")
                             continue
                    elif 'value' in details:
                        # For fixed values without a file (e.g., Temperature=80), directly add them
                        condition[capitalized_name] = details['value']
                        continue # Move to next component

                # If we determined an index (either dynamic or fixed), get the readable value from its file
                if idx is not None and 'data' in component and component['data'] is not None:
                    df = component['data']
                    info_row_series_list = df[df['Index'] == idx]
                    if not info_row_series_list.empty:
                        info_row_series = info_row_series_list.iloc[0]

                        # Add row_index (position in DataFrame/File)
                        # Assuming default RangeIndex from read_csv, the index is the row position (0-based)
                        row_indices = info_row_series_list.index.tolist()
                        if row_indices:
                            condition[f"{capitalized_name}_row_index"] = row_indices[0]

                        # Also add the internal Index value for reference
                        condition[f"{capitalized_name}_Index"] = idx

                        display_col = details.get('display_col')
                        if display_col and display_col in info_row_series:
                            condition[capitalized_name] = info_row_series[display_col]
                        else: # Fallback if display_col is not specified or not found
                            condition[capitalized_name] = f"Index_{idx}"

            # Also add fixed components from the main config (e.g., fixed SMILES)
            if self.fixed_components:
                for comp_name, comp_value in self.fixed_components.items():
                     condition[comp_name.capitalize()] = comp_value

            results.append(condition)
        # --- END OF FIX ---

        if not results:
            opt_logger.warning("Result list is empty after processing. Cannot generate report.")
            console.print("[yellow]Warning: Failed to parse optimization results. Check `optimization_run.log` for details.[/yellow]")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        top_k = self.config.get('top_k_results', 10)

        # Deduplicate to show unique top conditions
        condition_cols = [col for col in results_df.columns if col != predicted_col_name]
        if condition_cols:
            # Reorder columns to have the score first, then dynamic, then fixed
            all_component_names = [name.capitalize() for name in self.space_loader.components.keys()]
            if self.fixed_components:
                 all_component_names.extend([name.capitalize() for name in self.fixed_components.keys()])

            # Prioritize columns that exist in the dataframe
            ordered_cols = [predicted_col_name] + [col for col in all_component_names if col in results_df.columns]
            # Add any other columns that might have been missed
            ordered_cols.extend([col for col in results_df.columns if col not in ordered_cols])

            final_df = results_df.drop_duplicates(subset=condition_cols, keep='first')[ordered_cols].head(top_k)
        else:
            final_df = results_df.head(top_k)

        console.print("\n[bold green]--- Top Optimized Conditions ---[/bold green]")

        if final_df.empty:
            console.print("[yellow]No valid conditions to display.[/yellow]")
            return final_df

        unique_scores = final_df[predicted_col_name].nunique()
        if unique_scores == 1:
            score_value = final_df[predicted_col_name].iloc[0]
            console.print(f"[yellow]Warning: All optimization attempts resulted in the same score ({score_value:.4f}).[/yellow]")
            console.print("[yellow]This suggests the model is consistently failing to produce meaningful predictions.[/yellow]")

        table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
        table.add_column("Rank", style="cyan", justify="right")

        for col in final_df.columns:
            if col == predicted_col_name:
                table.add_column(f"Score ({target_col_name})", style="bold yellow", justify="right")
            else:
                table.add_column(col, overflow="fold") # Use 'fold' to wrap long text like SMILES

        for i, row in enumerate(final_df.itertuples(index=False), 1):
            row_data = [str(i)]
            for j, item in enumerate(row):
                if final_df.columns[j] == predicted_col_name:
                    row_data.append(f"{item:.4f}")
                else:
                    # Shorten long SMILES for display, but full value is in CSV
                    item_str = str(item)
                    if len(item_str) > 60 and 'c' in item_str: # Heuristic for SMILES
                        item_str = item_str[:57] + "..."
                    row_data.append(item_str)
            table.add_row(*row_data)

        console.print(table)

        # Save results to CSV files
        self._save_results_to_csv(results_df, final_df, target_col_name, predicted_col_name)

        return final_df

    def _save_results_to_csv(self, all_results_df, top_results_df, target_col_name, predicted_col_name):
        """
        Save optimization results to CSV files.

        Args:
            all_results_df: DataFrame with all optimization results
            top_results_df: DataFrame with top-K results (deduplicated)
            target_col_name: Name of the target column (e.g., 'ee')
            predicted_col_name: Name of the predicted column (e.g., 'predicted_ee')
        """
        if self.output_dir is None:
            opt_logger.warning("No output directory specified, skipping CSV export.")
            return

        import os

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Prepare column order for CSV export
        # Start with Rank and Score, then reaction components
        csv_columns = []

        # Add Rank column to all results (sorted by score)
        all_results_with_rank = all_results_df.copy()
        all_results_with_rank['Rank'] = range(1, len(all_results_with_rank) + 1)

        # Rename predicted column to match display format
        score_col_name = f"Score ({target_col_name})"
        all_results_with_rank = all_results_with_rank.rename(columns={predicted_col_name: score_col_name})

        # Define preferred column order (matching the table display)
        preferred_order = ['Rank', score_col_name]

        # Add component columns in a logical order
        component_order = []
        if 'Ligand' in all_results_with_rank.columns:
            component_order.append('Ligand')
        if 'Catalyst' in all_results_with_rank.columns:
            component_order.append('Catalyst')
        if 'Base' in all_results_with_rank.columns:
            component_order.append('Base')
        if 'Solvent' in all_results_with_rank.columns:
            component_order.append('Solvent')
        if 'Temperature' in all_results_with_rank.columns:
            component_order.append('Temperature')

        # Add reactant columns
        if 'Reactant1' in all_results_with_rank.columns:
            component_order.append('Reactant1')
        if 'Reactant2' in all_results_with_rank.columns:
            component_order.append('Reactant2')

        # Add any remaining columns
        remaining_cols = [col for col in all_results_with_rank.columns
                         if col not in preferred_order + component_order]

        final_column_order = preferred_order + component_order + remaining_cols

        # Reorder columns
        available_columns = [col for col in final_column_order if col in all_results_with_rank.columns]
        all_results_ordered = all_results_with_rank[available_columns]

        # Save all results
        all_results_file = os.path.join(self.output_dir, "optimization_all_results.csv")
        all_results_ordered.to_csv(all_results_file, index=False, encoding='utf-8')
        opt_logger.info(f"Saved all {len(all_results_ordered)} optimization results to: {all_results_file}")

        # Save top-K results (deduplicated)
        if not top_results_df.empty:
            top_results_with_rank = top_results_df.copy()
            top_results_with_rank['Rank'] = range(1, len(top_results_with_rank) + 1)
            top_results_with_rank = top_results_with_rank.rename(columns={predicted_col_name: score_col_name})

            # Reorder columns for top results
            available_top_columns = [col for col in final_column_order if col in top_results_with_rank.columns]
            top_results_ordered = top_results_with_rank[available_top_columns]

            top_results_file = os.path.join(self.output_dir, f"optimization_top_{len(top_results_ordered)}_results.csv")
            top_results_ordered.to_csv(top_results_file, index=False, encoding='utf-8')
            opt_logger.info(f"Saved top {len(top_results_ordered)} optimization results to: {top_results_file}")

        # Print file locations to console
        from rich.console import Console
        console = Console()
        console.print(f"\n[bold green]Results saved to CSV files:[/bold green]")
        console.print(f"  📄 All results ({len(all_results_ordered)} entries): [cyan]{all_results_file}[/cyan]")
        if not top_results_df.empty:
            console.print(f"  🏆 Top results ({len(top_results_ordered)} entries): [cyan]{top_results_file}[/cyan]")
