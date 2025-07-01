# utils/predictor_api.py
import pandas as pd
from typing import Dict, Any, Optional
import numpy as np
import logging

from core.run_manager import process_dataframe

class Predictor:
    def __init__(self, model, scaler, label_encoder, run_config: Dict[str, Any], output_dir: str = "."):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.config = run_config
        self.output_dir = output_dir
        
        # Setup dedicated logger for predictor (file only, no terminal output)
        self.logger = logging.getLogger("predictor_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent terminal output
        
        # Only add file handler if not already configured
        if not self.logger.handlers:
            log_file = f"{output_dir}/predictor.log"
            import os
            os.makedirs(output_dir, exist_ok=True)
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def predict_from_df(self, df: pd.DataFrame, skip_precomputed_features: bool = False) -> Optional[pd.DataFrame]:
        """
        High-level API: Takes a complete, correctly structured DataFrame,
        processes it, and returns predictions.
        
        Args:
            df: Input DataFrame with reaction data
            skip_precomputed_features: If True, skip loading precomputed features 
                                     and only generate SMILES features
        """
        try:
            # Log detailed debugging info to file, show minimal progress to user
            self.logger.info("=== PREDICTOR DEBUG ===")
            self.logger.info(f"Input DF shape: {df.shape}")
            self.logger.info(f"Input DF columns: {list(df.columns)}")
            self.logger.info(f"First row preview: {df.iloc[0].to_dict()}")
            
            # Simple terminal progress indicator
            print("Processing prediction input...")
            
            # Get config info
            common_cfg = self.config.get('data', {}).get('single_file_config', {}).copy()
            feature_gen_cfg = self.config.get('features', {})
            smiles_cols = common_cfg.get('smiles_col', [])
            if isinstance(smiles_cols, str):
                smiles_cols = [smiles_cols]
                
            self.logger.info(f"SMILES columns expected: {smiles_cols}")
            
            # Check if this DataFrame already contains ALL the features we need
            # Look for SMILES features with column prefixes
            has_all_smiles_features = True
            feature_summary = {}
            for smiles_col in smiles_cols:
                prefixed_features = [col for col in df.columns if col.startswith(f"{smiles_col}_")]
                if not prefixed_features:
                    has_all_smiles_features = False
                    self.logger.info(f"Missing features for {smiles_col}")
                    break
                else:
                    feature_summary[smiles_col] = len(prefixed_features)
                    self.logger.info(f"Found {len(prefixed_features)} features for {smiles_col}")
            
            # Count numeric columns (potential features)
            numeric_cols = [col for col in df.columns if col not in smiles_cols and col != 'ee' and 
                           hasattr(df[col], 'dtype') and pd.api.types.is_numeric_dtype(df[col])]
            self.logger.info(f"Found {len(numeric_cols)} numeric feature columns")
            
            # Show concise summary to terminal
            if feature_summary:
                total_features = sum(feature_summary.values()) + len(numeric_cols)
                print(f"  ✓ Feature check: {total_features} total features ({len(feature_summary)} SMILES components)")
            else:
                print(f"  • Dataset: {df.shape[0]} samples, {df.shape[1]} columns")
            
            # If we have all SMILES features AND numeric features, treat this as a complete feature DataFrame
            if has_all_smiles_features and len(numeric_cols) > 10:  # Reasonable threshold
                self.logger.info("✓ DataFrame appears to contain all features, skipping feature generation")
                print("  ✓ Using pre-computed features directly")
                
                # Extract only the feature columns (exclude SMILES text columns and target)
                feature_cols = []
                for col in df.columns:
                    if col in smiles_cols or col == 'ee':
                        continue  # Skip SMILES text and target columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        feature_cols.append(col)
                
                X_new = df[feature_cols].values
                self.logger.info(f"Using {len(feature_cols)} feature columns directly: {feature_cols[:10]}...")
                self.logger.info(f"Feature matrix shape: {X_new.shape}")
                
            else:
                self.logger.info("DataFrame missing features, will generate them...")
                print("  • Generating molecular features...")
                # Fall back to the original feature generation process
                if skip_precomputed_features:
                    common_cfg.pop('precomputed_features', None)
                
                # Look for pre-generated SMILES features with column prefixes
                has_prefixed_smiles_features = False
                for smiles_col in smiles_cols:
                    prefixed_features = [col for col in df.columns if col.startswith(f"{smiles_col}_")]
                    if prefixed_features:
                        has_prefixed_smiles_features = True
                        self.logger.info(f"Found {len(prefixed_features)} prefixed features for {smiles_col}")
                        break
                
                # If we already have prefixed SMILES features, skip SMILES generation
                if has_prefixed_smiles_features:
                    feature_gen_cfg = {}  # Skip SMILES feature generation
                    self.logger.info("Detected pre-generated SMILES features, skipping re-generation")
                    
                    # Modify precomputed feature configuration to include ALL feature columns
                    if common_cfg.get('precomputed_features'):
                        potential_feature_cols = []
                        smiles_feature_cols = []
                        
                        for col in df.columns:
                            is_smiles_feature = any(col.startswith(f"{smiles_col}_") for smiles_col in smiles_cols)
                            if is_smiles_feature:
                                smiles_feature_cols.append(col)
                            elif '_' in col and col not in smiles_cols:
                                if (col in df.columns and 
                                    hasattr(df[col], 'dtype') and 
                                    pd.api.types.is_numeric_dtype(df[col]) and
                                    col not in ['Index']):
                                    potential_feature_cols.append(col)
                        
                        all_feature_cols = potential_feature_cols + smiles_feature_cols
                        
                        if all_feature_cols:
                            common_cfg['precomputed_features']['feature_columns'] = all_feature_cols
                            self.logger.info(f"Auto-detected {len(potential_feature_cols)} component features + {len(smiles_feature_cols)} SMILES features")
                
                self.logger.info(f"Calling process_dataframe with common_cfg: {common_cfg}")
                
                X_new, _, _, _ = process_dataframe(
                    df=df.copy(), common_cfg=common_cfg,
                    feature_gen_cfg=feature_gen_cfg, output_dir=self.output_dir
                )
                
                self.logger.info(f"Feature processing result: shape={X_new.shape}")
                print(f"  ✓ Feature matrix generated: {X_new.shape}")
                
                if X_new.shape[0] == 0:
                    self.logger.warning("Feature processing resulted in an empty dataset.")
                    print("  ⚠ Warning: No valid features generated")
                    return pd.DataFrame()

            # Scale and predict
            X_new_scaled = self.scaler.transform(X_new) if self.scaler else X_new
            self.logger.info(f"After scaling: shape={X_new_scaled.shape}")
            
            predictions = self.model.predict(X_new_scaled)
            self.logger.info(f"Model predictions: {predictions}")
            print(f"  ✓ Predictions generated for {len(predictions)} samples")
            
            # Use the input df for the output, as it has all the original info
            results_df = df.copy()
            results_df['prediction'] = predictions
            self.logger.info(f"Returning results DF: shape={results_df.shape}")

            return results_df
        except Exception as e:
            # Provide more context in the error message
            self.logger.error(f"Error during prediction in Predictor API (predict_from_df): {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"  ✗ Prediction failed: {e}")
            return None