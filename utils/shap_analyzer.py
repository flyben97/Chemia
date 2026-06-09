# utils/shap_analyzer.py
"""
SHAP (SHapley Additive exPlanations) Analysis Module

Provides SHAP-based model interpretability for tree-based and other supported models.
Supports XGBoost, LightGBM, CatBoost, Random Forest, and other tree-based models.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Union, Tuple, List
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not installed. Install with: pip install shap")


class SHAPAnalyzer:
    """
    SHAP-based model interpretability analyzer.

    Supports:
    - Tree-based models: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting
    - Linear models: Ridge, Lasso, ElasticNet (via permutation explainer)
    - Neural networks: TabNet, ANN (via permutation explainer)
    """

    # Models that support TreeExplainer (fastest)
    TREE_EXPLAINER_MODELS = {
        'xgboost', 'xgbregressor', 'xgbclassifier',
        'lightgbm', 'lgbmregressor', 'lgbmclassifier',
        'catboost', 'catboostregressor', 'catboostclassifier',
        'randomforest', 'randomforestregressor', 'randomforestclassifier',
        'extratrees', 'extratreesregressor', 'extratreesclassifier',
        'gradientboosting', 'gradientboostingregressor', 'gradientboostingclassifier',
        'histgradientboosting', 'histgradientboostingregressor', 'histgradientboostingclassifier',
    }

    # Models that support KernelExplainer (slower but universal)
    KERNEL_EXPLAINER_MODELS = {
        'ridge', 'lasso', 'elasticnet', 'bayesianridge',
        'svr', 'svc', 'knn', 'kneighborsregressor', 'kneighborsclassifier',
        'kernelridge', 'gaussianprocess', 'gaussianprocessregressor', 'gaussianprocessclassifier',
        'tabnet', 'tabnetregressor', 'tabnetclassifier',
        'ann', 'mlpregressor', 'mlpclassifier',
    }

    def __init__(self, model, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP analyzer.

        Args:
            model: Trained model instance
            X_train: Training data for background dataset
            feature_names: List of feature names (optional)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        self.explainer = None
        self.shap_values = None
        self.model_type = self._get_model_type()

    def _get_model_type(self) -> str:
        """Determine the model type."""
        model_class_name = self.model.__class__.__name__.lower()
        return model_class_name

    def _supports_tree_explainer(self) -> bool:
        """Check if model supports TreeExplainer."""
        return self.model_type in self.TREE_EXPLAINER_MODELS

    def _supports_kernel_explainer(self) -> bool:
        """Check if model supports KernelExplainer."""
        return self.model_type in self.KERNEL_EXPLAINER_MODELS

    def create_explainer(self, max_samples: int = 100, use_kernel: bool = False) -> bool:
        """
        Create SHAP explainer for the model.

        Args:
            max_samples: Maximum samples for background dataset
            use_kernel: Force use of KernelExplainer even if TreeExplainer is available

        Returns:
            True if explainer created successfully, False otherwise
        """
        try:
            # Use subset of training data as background
            if len(self.X_train) > max_samples:
                background_indices = np.random.choice(len(self.X_train), max_samples, replace=False)
                X_background = self.X_train[background_indices]
            else:
                X_background = self.X_train

            # Try TreeExplainer first (faster)
            if self._supports_tree_explainer() and not use_kernel:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    print(f"✓ Created TreeExplainer for {self.model_type}")
                    return True
                except Exception as e:
                    print(f"[WARNING] TreeExplainer failed: {e}. Trying KernelExplainer...")

            # Fall back to KernelExplainer
            if self._supports_kernel_explainer() or use_kernel:
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    X_background,
                    link="identity"
                )
                print(f"✓ Created KernelExplainer for {self.model_type}")
                return True

            # Try PermutationExplainer as last resort
            self.explainer = shap.PermutationExplainer(
                self.model.predict,
                X_background
            )
            print(f"✓ Created PermutationExplainer for {self.model_type}")
            return True

        except Exception as e:
            print(f"✗ Failed to create explainer: {e}")
            return False

    def explain(self, X_test: np.ndarray, max_samples: Optional[int] = None) -> bool:
        """
        Calculate SHAP values for test data.

        Args:
            X_test: Test data to explain
            max_samples: Maximum samples to explain (for performance)

        Returns:
            True if explanation successful, False otherwise
        """
        if self.explainer is None:
            print("✗ Explainer not created. Call create_explainer() first.")
            return False

        try:
            # Limit samples for performance
            if max_samples and len(X_test) > max_samples:
                X_explain = X_test[:max_samples]
            else:
                X_explain = X_test

            self.shap_values = self.explainer.shap_values(X_explain)
            print(f"✓ Calculated SHAP values for {len(X_explain)} samples")
            return True

        except Exception as e:
            print(f"✗ Failed to calculate SHAP values: {e}")
            return False

    def plot_summary(self, output_path: Optional[str] = None, plot_type: str = "bar") -> Optional[str]:
        """
        Create SHAP summary plot.

        Args:
            output_path: Path to save plot
            plot_type: "bar" or "violin"

        Returns:
            Path to saved plot or None
        """
        if self.shap_values is None:
            print("✗ SHAP values not calculated. Call explain() first.")
            return None

        try:
            plt.figure(figsize=(12, 8))

            if plot_type == "bar":
                shap.summary_plot(
                    self.shap_values,
                    self.X_train[:len(self.shap_values)],
                    feature_names=self.feature_names,
                    plot_type="bar",
                    show=False
                )
            else:  # violin
                shap.summary_plot(
                    self.shap_values,
                    self.X_train[:len(self.shap_values)],
                    feature_names=self.feature_names,
                    show=False
                )

            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Summary plot saved to {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            print(f"✗ Failed to create summary plot: {e}")
            return None

    def plot_dependence(self, feature_idx: Union[int, str], output_path: Optional[str] = None) -> Optional[str]:
        """
        Create SHAP dependence plot for a specific feature.

        Args:
            feature_idx: Feature index or name
            output_path: Path to save plot

        Returns:
            Path to saved plot or None
        """
        if self.shap_values is None:
            print("✗ SHAP values not calculated. Call explain() first.")
            return None

        try:
            # Convert feature name to index if needed
            if isinstance(feature_idx, str):
                feature_idx = self.feature_names.index(feature_idx)

            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.X_train[:len(self.shap_values)],
                feature_names=self.feature_names,
                show=False
            )

            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Dependence plot saved to {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            print(f"✗ Failed to create dependence plot: {e}")
            return None

    def plot_force(self, sample_idx: int = 0, output_path: Optional[str] = None) -> Optional[str]:
        """
        Create SHAP force plot for a specific sample.

        Args:
            sample_idx: Index of sample to explain
            output_path: Path to save plot

        Returns:
            Path to saved plot or None
        """
        if self.shap_values is None:
            print("✗ SHAP values not calculated. Call explain() first.")
            return None

        try:
            plt.figure(figsize=(14, 4))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                self.X_train[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )

            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Force plot saved to {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            print(f"✗ Failed to create force plot: {e}")
            return None

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            print("✗ SHAP values not calculated. Call explain() first.")
            return pd.DataFrame()

        try:
            # Handle multi-output case (classification)
            if isinstance(self.shap_values, list):
                shap_values = np.abs(self.shap_values[0]).mean(axis=0)
            else:
                shap_values = np.abs(self.shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': shap_values
            }).sort_values('Importance', ascending=False)

            return importance_df.head(top_n)

        except Exception as e:
            print(f"✗ Failed to get feature importance: {e}")
            return pd.DataFrame()

    def generate_report(self, X_test: np.ndarray, output_dir: str = "./shap_analysis") -> str:
        """
        Generate comprehensive SHAP analysis report.

        Args:
            X_test: Test data to analyze
            output_dir: Directory to save report files

        Returns:
            Path to output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating SHAP Analysis Report")
        print(f"{'='*60}")

        # Create explainer
        if not self.create_explainer():
            print("✗ Failed to create explainer")
            return output_dir

        # Calculate SHAP values
        if not self.explain(X_test):
            print("✗ Failed to calculate SHAP values")
            return output_dir

        # Generate plots
        print("\n[1/4] Creating summary plot (bar)...")
        self.plot_summary(
            os.path.join(output_dir, "01_summary_bar.png"),
            plot_type="bar"
        )

        print("[2/4] Creating summary plot (violin)...")
        self.plot_summary(
            os.path.join(output_dir, "02_summary_violin.png"),
            plot_type="violin"
        )

        print("[3/4] Creating dependence plots...")
        importance_df = self.get_feature_importance(top_n=5)
        for idx, row in importance_df.iterrows():
            feature_name = row['Feature']
            self.plot_dependence(
                feature_name,
                os.path.join(output_dir, f"03_dependence_{feature_name}.png")
            )

        print("[4/4] Creating force plots...")
        for i in range(min(3, len(X_test))):
            self.plot_force(
                i,
                os.path.join(output_dir, f"04_force_sample_{i}.png")
            )

        # Save feature importance
        print("\n[5/5] Saving feature importance...")
        importance_df = self.get_feature_importance(top_n=len(self.feature_names))
        importance_path = os.path.join(output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to {importance_path}")

        # Create summary report
        report_path = os.path.join(output_dir, "SHAP_ANALYSIS_REPORT.txt")
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SHAP ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Number of Features: {len(self.feature_names)}\n")
            f.write(f"Number of Samples Analyzed: {len(X_test)}\n\n")
            f.write("Top 10 Most Important Features:\n")
            f.write("-"*60 + "\n")
            for idx, row in importance_df.head(10).iterrows():
                f.write(f"{row['Feature']:30s} {row['Importance']:10.6f}\n")
            f.write("\n" + "="*60 + "\n")

        print(f"✓ Report saved to {report_path}")
        print(f"\n✓ SHAP analysis complete! Results saved to: {output_dir}")

        return output_dir


def analyze_model_with_shap(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "./shap_analysis"
) -> str:
    """
    Convenience function to perform complete SHAP analysis.

    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        feature_names: Feature names
        output_dir: Output directory for results

    Returns:
        Path to output directory
    """
    analyzer = SHAPAnalyzer(model, X_train, feature_names)
    return analyzer.generate_report(X_test, output_dir)
