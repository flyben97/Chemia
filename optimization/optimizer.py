# optimization/optimizer.py
import pandas as pd
import logging
from bayes_opt import BayesianOptimization

class BayesianReactionOptimizer:
    def __init__(self, predictor, space_loader, opt_config, fixed_components=None, feature_gen_config=None, output_dir=None):
        """
        Initializes the optimizer.

        Args:
            predictor (Predictor): An instance of the configured Predictor class.
            space_loader (SearchSpaceLoader): An instance of the configured loader.
            opt_config (dict): Configuration dictionary for Bayesian Optimization.
            fixed_components (dict): Fixed SMILES components for the reaction.
            feature_gen_config (dict): Feature generation configuration from training.
            output_dir (str): Output directory for feature generation logs.
        """
        self.predictor = predictor
        self.space_loader = space_loader
        self.config = opt_config
        self.fixed_components = fixed_components or {}
        self.feature_gen_config = feature_gen_config
        self.output_dir = output_dir
        logging.info("Optimizer initialized in robust mode. Features are recalculated each iteration for correctness.")

    def objective_function(self, **kwargs):
        """
        The black-box function that Bayesian Optimization will try to maximize.
        It builds a full reaction DataFrame for each iteration and passes it
        to the reliable high-level predictor API.
        """
        # bayes_opt passes lowercase keys with '_idx' suffix, convert them
        dynamic_indices = {key: int(round(val)) for key, val in kwargs.items()}
        
        try:
            # 1. Build a complete DataFrame that mimics the original training data structure.
            #    The space_loader is responsible for creating a "perfect fake".
            reaction_df = self.space_loader.build_reaction_df(
                dynamic_indices, 
                fixed_components=self.fixed_components,
                feature_gen_config=self.feature_gen_config,
                output_dir=self.output_dir
            )
            
            logging.info(f"Built reaction DF: shape={reaction_df.shape}, columns={len(reaction_df.columns)}")
            logging.info(f"First few columns: {list(reaction_df.columns)[:10]}")
            
            # 2. Use the high-level, robust predict_from_df method.
            #    This method handles all feature generation and processing internally.
            #    Include precomputed features from optimization space components
            result_df = self.predictor.predict_from_df(reaction_df)
            
            logging.info(f"Prediction result: {type(result_df)}, shape={result_df.shape if result_df is not None else 'None'}")
            
            if result_df is None or result_df.empty:
                logging.warning("Prediction returned None or empty DataFrame")
                return -999.0
            
            # Extract the single prediction value
            prediction = result_df['prediction'].iloc[0]
            final_pred = float(prediction) if pd.notna(prediction) else -999.0
            logging.info(f"Final prediction value: {final_pred}")
            return final_pred

        except KeyError as e:
            # This can happen if an index is out of bounds
            logging.warning(f"Index lookup failed ({e}), returning low score.")
            return -999.0
        except Exception as e:
            # Catch any other error during the process
            logging.error(f"Error in objective function: {e}", exc_info=True)
            return -999.0

    def run(self):
        """
        Executes the Bayesian Optimization process.
        """
        logging.info("Setting up Bayesian Optimization...")
        
        if not self.space_loader.pbounds:
            raise ValueError("Search space is empty. Please set at least one component's mode to 'search' in the config.")
            
        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.space_loader.pbounds,
            random_state=self.config['random_state'],
            verbose=0  # Reduce console output, details go to log
        )
        
        logging.info(f"Running Optimization for {self.config['n_iter']} iterations "
                     f"(plus {self.config['init_points']} initial points)...")
        
        print("Running Bayesian optimization... (detailed progress in log file)")
        
        # Run optimization with minimal terminal output
        total_iterations = self.config['init_points'] + self.config['n_iter']
        optimizer.maximize(
            init_points=self.config['init_points'], 
            n_iter=self.config['n_iter']
        )
        
        # Log detailed results after completion
        logging.info(f"Completed {len(optimizer.res)} optimization iterations")
        for i, res in enumerate(optimizer.res):
            logging.info(f"Iteration {i+1:>3}: Score = {res['target']:>8.4f}, Params = {res['params']}")
        
        # Log final summary
        for i, res in enumerate(optimizer.res[-min(5, len(optimizer.res)):]):
            logging.info(f"Final Top {i+1}: Score = {res['target']:.4f}, Params = {res['params']}")
        
        print("✓ Optimization completed!")
        logging.info("Optimization Finished!")
        result_df = self._report_and_save_results(optimizer)
        
        return result_df

    def _report_and_save_results(self, optimizer):
        """Formats the optimization results into a readable and generic DataFrame."""
        results = []
        
        # Get the original target column name from the predictor's config
        target_col_name = self.predictor.config.get('data', {})\
            .get('single_file_config', {}).get('target_col', 'prediction')
        predicted_col_name = f"predicted_{target_col_name}"

        for res in optimizer.res:
            params = res['params']
            condition = {predicted_col_name: res['target']}
            dynamic_indices = {key: int(round(val)) for key, val in params.items()}
            
            # Build readable condition details for ALL search components
            for name, component in self.space_loader.components.items():
                details = component['details']
                capitalized_name = name.capitalize()
                
                # Get the index for this component
                idx = None
                if name.lower() in dynamic_indices:
                    idx = dynamic_indices[name.lower()]
                elif details['mode'] == 'fixed':
                    if 'index' in details:
                        idx = details['index']
                    elif 'value' in details:
                        condition[capitalized_name] = details['value']
                        continue
                
                # If we have an index and data file, extract the display value
                if 'data' in component and component['data'] is not None and idx is not None:
                    df = component['data']
                    info_row_series = df[df['Index'] == idx].iloc[0]
                    
                    display_col = details.get('display_col')
                    if display_col and display_col in info_row_series:
                        condition[capitalized_name] = info_row_series[display_col]

            results.append(condition)

        if not results:
            logging.warning("No results to process, returning empty DataFrame")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        top_k = self.config.get('top_k_results', 10)
        
        # Sort by the dynamic column name and handle duplicates
        results_df = results_df.sort_values(by=predicted_col_name, ascending=False)
        
        condition_cols = [col for col in results_df.columns if col != predicted_col_name]
        results_df['condition_id'] = results_df[condition_cols].astype(str).apply(lambda row: hash(tuple(row)), axis=1)
        
        # Process results step by step to avoid type checking issues
        from typing import cast
        deduplicated_df = cast(pd.DataFrame, results_df.drop_duplicates(subset='condition_id', keep='first'))
        top_results_df = cast(pd.DataFrame, deduplicated_df.head(top_k))
        final_df = cast(pd.DataFrame, top_results_df.drop(columns=['condition_id']))
        
        return final_df