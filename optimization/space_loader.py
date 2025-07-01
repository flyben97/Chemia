# optimization/space_loader.py
import pandas as pd
import logging
from typing import Dict, Any, Optional

class SearchSpaceLoader:
    def __init__(self, components_config: Dict[str, Any]):
        self.config = components_config
        self.components: Dict[str, Dict[str, Any]] = {}
        self.pbounds: Dict[str, tuple] = {}
        logging.info("Loading and parsing reaction components from config...")
        self._process_components()

    def _process_components(self):
        for name, details in self.config.items():
            mode = details.get('mode')
            if not mode: raise ValueError(f"Component '{name}' is missing 'mode' field.")
            self.components[name] = {'details': details, 'data': None}
            if 'file' in details:
                self._load_data_for_component_if_needed(name)
            
            if mode == 'search':
                df = self.components[name].get('data')
                if df is not None:
                    min_idx, max_idx = df['Index'].min(), df['Index'].max()
                    self.pbounds[name.lower()] = (min_idx, max_idx)
                    logging.info(f"  - Loaded SEARCH component '{name}' with {len(df)} options.")
                else:
                    pass
            elif mode == 'fixed':
                logging.info(f"  - Registered FIXED component '{name}'.")

    def _load_data_for_component_if_needed(self, name: str):
        if self.components[name].get('data') is not None: return
        details = self.components[name]['details']
        file_path = details.get('file')
        if not file_path: return
        try:
            sep = details.get('sep', ',')
            df = pd.read_csv(file_path, sep=sep, engine='python' if sep != ',' else 'c')
            if 'Index' not in df.columns: raise KeyError(f"Required 'Index' column not found in '{file_path}'.")
            self.components[name]['data'] = df
        except Exception as e:
            logging.error(f"FATAL: Error loading file '{file_path}' for component '{name}': {e}")
            raise
    
    def build_reaction_df(self, dynamic_indices: Dict[str, int], fixed_components: Optional[Dict[str, str]] = None, 
                         feature_gen_config: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Builds a single-row DataFrame that mimics the structure of the original
        training data, using config directives to select relevant SMILES and feature columns.
        
        Args:
            dynamic_indices: Indices for search components
            fixed_components: Fixed SMILES components from optimization config
            feature_gen_config: Feature generation configuration from training
            output_dir: Output directory for feature generation logs
        """
        data_row = {}
        smiles_for_feature_gen = {}
        
        # 1. Process fixed components first (these are usually SMILES)
        if fixed_components:
            for comp_name, comp_value in fixed_components.items():
                data_row[comp_name] = comp_value
                # If this is a SMILES column that needs feature generation
                if feature_gen_config and comp_name in feature_gen_config.get('per_smiles_col_generators', {}):
                    smiles_for_feature_gen[comp_name] = comp_value
        
        # 2. Process dynamic/search components
        for name, component in self.components.items():
            details = component['details']
            capitalized_name = name.capitalize()
            idx = None

            # Determine the index or value for the component
            if name.lower() in dynamic_indices:
                idx = dynamic_indices[name.lower()]
            elif details['mode'] == 'fixed':
                if 'row_index' in details:
                    # Use row_index to select from the data file
                    df = component.get('data')
                    if df is not None:
                        row_idx = details['row_index']
                        if 0 <= row_idx < len(df):
                            # Get the Index value from the specified row
                            idx = df.iloc[row_idx]['Index']
                            logging.info(f"  - Fixed component '{name}' using row {row_idx} (Index={idx})")
                        else:
                            raise ValueError(f"row_index {row_idx} is out of range for component '{name}' (0-{len(df)-1})")
                    else:
                        raise ValueError(f"Component '{name}' has row_index but no data file loaded")
                elif 'index' in details:
                    idx = details['index']
                elif 'value' in details:
                    # For fixed components, only add to data_row if it's a SMILES column
                    if (feature_gen_config and 
                        capitalized_name in feature_gen_config.get('per_smiles_col_generators', {})):
                        data_row[capitalized_name] = details['value']
                        smiles_for_feature_gen[capitalized_name] = details['value']
                    continue
            else: # Fallback for search components
                if 'data' in component and component['data'] is not None:
                    idx = component['data']['Index'].min()

            # If we have an index and data file, extract the information
            if 'data' in component and component['data'] is not None and idx is not None:
                df = component['data']
                info_row_series = df[df['Index'] == idx].iloc[0]
                
                # Add the display column ONLY if it's a SMILES column that needs feature generation
                display_col = details.get('display_col')
                if display_col:
                    smiles_value = info_row_series[display_col]
                    
                    # Check if this is a SMILES column that needs feature generation
                    if (feature_gen_config and 
                        capitalized_name in feature_gen_config.get('per_smiles_col_generators', {})):
                        data_row[capitalized_name] = smiles_value
                        smiles_for_feature_gen[capitalized_name] = smiles_value
                        logging.info(f"  - Added SMILES column: {capitalized_name} = {smiles_value}")
                
                # Add pre-computed features IF specified
                if details.get('is_feature_source'):
                    slice_spec = details.get('feature_slice')
                    if not slice_spec:
                        raise ValueError(f"Component '{name}' is marked as 'is_feature_source' but lacks 'feature_slice'.")
                    
                    parts = slice_spec.split(':')
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if len(parts) > 1 and parts[1] else len(info_row_series)
                    
                    # Add ONLY the sliced columns as features with proper prefixes
                    feature_data = info_row_series.iloc[start:end]
                    
                    # Special handling for temperature - use 'Temp' instead of 'temperature_Temp'
                    if name.lower() == 'temperature':
                        for col, val in feature_data.to_dict().items():
                            if col == 'Temp':
                                data_row['Temp'] = val
                                logging.info(f"  - Added temperature feature: Temp = {val}")
                    else:
                        # Add component prefix to feature names to match training data format
                        prefix = name.lower() + "_"
                        prefixed_features = {prefix + col: val for col, val in feature_data.to_dict().items()}
                        data_row.update(prefixed_features)
                        logging.info(f"  - Added {len(prefixed_features)} features with prefix '{prefix}'")

        # 3. Generate features for SMILES columns - THIS IS THE KEY FIX
        if smiles_for_feature_gen and feature_gen_config:
            logging.info(f"Generating features for SMILES columns: {list(smiles_for_feature_gen.keys())}")
            
            from utils.feature_generator import generate_features
            import os
            
            # Ensure output directory exists
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            for smiles_col, smiles_value in smiles_for_feature_gen.items():
                if smiles_col in feature_gen_config.get('per_smiles_col_generators', {}):
                    feature_configs = feature_gen_config['per_smiles_col_generators'][smiles_col]
                    
                    # Generate features for this single SMILES
                    generated_df = generate_features([smiles_value], feature_configs, output_dir=output_dir)
                    
                    if not generated_df.empty:
                        # Add generated features with column prefix to avoid name conflicts
                        feature_data = generated_df.iloc[0].to_dict()
                        prefixed_feature_data = {f"{smiles_col}_{col}": val for col, val in feature_data.items()}
                        data_row.update(prefixed_feature_data)
                        logging.info(f"  - Generated {len(feature_data)} features for {smiles_col}")

        # 4. Add dummy target column if not present
        target_col = 'ee'  # This should match the training target column
        if target_col not in data_row:
            data_row[target_col] = 0.0

        result_df = pd.DataFrame([data_row])
        logging.info(f"Built reaction DataFrame with {len(data_row)} total columns")
        logging.info(f"DataFrame columns: {sorted(list(data_row.keys()))}")
        return result_df

    def get_feature_source_components(self):
        """
        Returns a list of component names that are marked as feature sources.
        """
        feature_sources = []
        for name, component in self.components.items():
            details = component['details']
            if details.get('is_feature_source', False):
                feature_sources.append(name.lower())
        return feature_sources
    
    def get_expected_feature_prefixes(self, include_smiles_prefixes=True, smiles_cols=None):
        """
        Returns expected feature prefixes based on configuration.
        
        Args:
            include_smiles_prefixes: Whether to include SMILES column prefixes
            smiles_cols: List of SMILES column names
        """
        prefixes = []
        
        # Add component feature prefixes
        for name in self.get_feature_source_components():
            prefixes.append(f"{name}_")
        
        # Add SMILES feature prefixes if requested
        if include_smiles_prefixes and smiles_cols:
            for smiles_col in smiles_cols:
                prefixes.append(f"{smiles_col}_")
        
        return prefixes