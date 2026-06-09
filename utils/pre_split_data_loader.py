"""
Pre-split Data Loader Module

Handles loading of pre-split datasets (train/test or train/val/test)
for CHEMIA training pipeline.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PreSplitDataLoader:
    """Loader for pre-split datasets"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pre-split data loader

        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.source_mode = self.data_config.get('source_mode', 'single_file')

        # Validate source mode
        valid_modes = ['single_file', 'pre_split_cv', 'pre_split_t_v_t', 'pre_split_train_test']
        if self.source_mode not in valid_modes:
            raise ValueError(f"Invalid source_mode: {self.source_mode}. Must be one of {valid_modes}")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data based on source mode

        Returns:
            Dictionary with keys 'train', 'val' (optional), 'test'
        """
        if self.source_mode == 'pre_split_train_test':
            return self._load_train_test_split()
        elif self.source_mode == 'pre_split_t_v_t':
            return self._load_train_val_test_split()
        elif self.source_mode == 'pre_split_cv':
            return self._load_cv_split()
        elif self.source_mode == 'single_file':
            raise ValueError("Use standard data loading for single_file mode")
        else:
            raise ValueError(f"Unsupported source_mode: {self.source_mode}")

    def _load_train_test_split(self) -> Dict[str, pd.DataFrame]:
        """
        Load train/test split

        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        config_section = self.data_config.get('pre_split_train_test_config', {})

        train_path = config_section.get('train_path')
        test_path = config_section.get('test_path')

        if not train_path or not test_path:
            raise ValueError("pre_split_train_test_config must specify train_path and test_path")

        # Load files
        train_df = self._load_csv_file(train_path, 'train')
        test_df = self._load_csv_file(test_path, 'test')

        # Validate columns
        self._validate_columns(train_df, test_df, config_section)

        logger.info(f"✓ Loaded train/test split:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")

        return {
            'train': train_df,
            'test': test_df
        }

    def _load_train_val_test_split(self) -> Dict[str, pd.DataFrame]:
        """
        Load train/val/test split

        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames
        """
        config_section = self.data_config.get('pre_split_t_v_t_config', {})

        train_path = config_section.get('train_path')
        val_path = config_section.get('valid_path')
        test_path = config_section.get('test_path')

        if not train_path or not test_path:
            raise ValueError("pre_split_t_v_t_config must specify train_path and test_path")

        # Load files
        train_df = self._load_csv_file(train_path, 'train')
        test_df = self._load_csv_file(test_path, 'test')

        val_df = None
        if val_path:
            val_df = self._load_csv_file(val_path, 'validation')

        # Validate columns
        self._validate_columns(train_df, test_df, config_section, val_df)

        logger.info(f"✓ Loaded train/val/test split:")
        logger.info(f"  Train: {len(train_df)} samples")
        if val_df is not None:
            logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")

        result = {
            'train': train_df,
            'test': test_df
        }

        if val_df is not None:
            result['val'] = val_df

        return result

    def _load_cv_split(self) -> Dict[str, pd.DataFrame]:
        """
        Load cross-validation split

        Returns:
            Dictionary with fold information
        """
        config_section = self.data_config.get('pre_split_cv_config', {})

        # This would load multiple fold files
        # Implementation depends on specific CV format
        raise NotImplementedError("CV split loading not yet implemented")

    def _load_csv_file(self, filepath: str, split_name: str) -> pd.DataFrame:
        """
        Load a CSV file with error handling

        Args:
            filepath: Path to CSV file
            split_name: Name of the split (for logging)

        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{split_name} file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"✓ Loaded {split_name} data from {filepath}")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            raise RuntimeError(f"Error loading {split_name} file {filepath}: {e}")

    def _validate_columns(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                         config_section: Dict[str, Any],
                         val_df: Optional[pd.DataFrame] = None):
        """
        Validate that required columns exist in all datasets

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            config_section: Configuration section
            val_df: Validation DataFrame (optional)
        """
        smiles_cols = config_section.get('smiles_col', [])
        target_col = config_section.get('target_col')

        # Check target column
        if target_col:
            for df, name in [(train_df, 'train'), (test_df, 'test')]:
                if target_col not in df.columns:
                    raise ValueError(f"Target column '{target_col}' not found in {name} data")

            if val_df is not None and target_col not in val_df.columns:
                raise ValueError(f"Target column '{target_col}' not found in validation data")

        # Check SMILES columns
        if smiles_cols:
            for col in smiles_cols:
                for df, name in [(train_df, 'train'), (test_df, 'test')]:
                    if col not in df.columns:
                        raise ValueError(f"SMILES column '{col}' not found in {name} data")

                if val_df is not None and col not in val_df.columns:
                    raise ValueError(f"SMILES column '{col}' not found in validation data")

        logger.info("✓ Column validation passed")

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information

        Returns:
            Dictionary with configuration details
        """
        if self.source_mode == 'pre_split_train_test':
            config_section = self.data_config.get('pre_split_train_test_config', {})
        elif self.source_mode == 'pre_split_t_v_t':
            config_section = self.data_config.get('pre_split_t_v_t_config', {})
        else:
            config_section = {}

        return {
            'source_mode': self.source_mode,
            'smiles_columns': config_section.get('smiles_col', []),
            'target_column': config_section.get('target_col'),
            'precomputed_features': config_section.get('precomputed_features', {})
        }


def load_pre_split_data(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load pre-split data

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with loaded DataFrames
    """
    loader = PreSplitDataLoader(config)
    return loader.load_data()


def get_data_splits_info(data_splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Get information about data splits

    Args:
        data_splits: Dictionary of DataFrames

    Returns:
        Dictionary with split information
    """
    info = {}

    for split_name, df in data_splits.items():
        info[split_name] = {
            'samples': len(df),
            'columns': list(df.columns),
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }

    return info


# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = {
        'data': {
            'source_mode': 'pre_split_train_test',
            'pre_split_train_test_config': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'smiles_col': ['Catalyst', 'Reactant1', 'Reactant2'],
                'target_col': 'yield',
                'precomputed_features': {
                    'feature_columns': None
                }
            }
        }
    }

    # This would load the data if files exist
    # loader = PreSplitDataLoader(example_config)
    # data_splits = loader.load_data()
    # print(get_data_splits_info(data_splits))
