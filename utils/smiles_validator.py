#!/usr/bin/env python3
"""
SMILES Validation Module

This module provides functions to validate SMILES strings in datasets,
ensuring that columns specified as SMILES actually contain valid molecular representations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Try to import RDKit for SMILES validation
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. SMILES validation will be limited.")

console = Console()

def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a single SMILES string is valid
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(smiles, str):
        return False
    
    if not smiles or smiles.strip() == "":
        return False
    
    # Basic format checks
    smiles = smiles.strip()
    
    # Check for obviously invalid characters or patterns
    if any(char in smiles for char in ['\n', '\r', '\t']):
        return False
    
    # If RDKit is available, use it for proper validation
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
    else:
        # Basic validation without RDKit
        # Check for basic SMILES characters
        valid_chars = set("CNOPSFClBrI0123456789()[]@+-=#$%./\\")
        return all(c in valid_chars for c in smiles)

def validate_smiles_column(df: pd.DataFrame, column_name: str, 
                          sample_size: int = 100, 
                          min_valid_ratio: float = 0.8) -> Dict[str, Any]:
    """
    Validate a SMILES column in a DataFrame
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to validate
        sample_size: Number of samples to check (for performance)
        min_valid_ratio: Minimum ratio of valid SMILES required
        
    Returns:
        dict: Validation results including statistics and invalid samples
    """
    results = {
        'column_name': column_name,
        'total_samples': len(df),
        'valid_count': 0,
        'invalid_count': 0,
        'empty_count': 0,
        'valid_ratio': 0.0,
        'is_valid_column': False,
        'invalid_samples': [],
        'sample_size_checked': 0,
        'error_message': None
    }
    
    # Check if column exists
    if column_name not in df.columns:
        results['error_message'] = f"Column '{column_name}' not found in DataFrame"
        return results
    
    # Get the column data
    column_data = df[column_name]
    
    # Sample data for validation (for performance on large datasets)
    if len(column_data) > sample_size:
        sample_indices = np.random.choice(len(column_data), size=sample_size, replace=False)
        sample_data = column_data.iloc[sample_indices]
        sample_df_subset = df.iloc[sample_indices]
    else:
        sample_data = column_data
        sample_df_subset = df
    
    results['sample_size_checked'] = len(sample_data)
    
    # Validate each SMILES in the sample
    for idx, smiles in enumerate(sample_data):
        original_idx = sample_df_subset.index[idx]
        
        # Check for empty/null values
        if pd.isna(smiles) or smiles == "" or smiles is None:
            results['empty_count'] += 1
            results['invalid_samples'].append({
                'index': original_idx,
                'value': str(smiles),
                'error': 'Empty or null value'
            })
            continue
        
        # Validate SMILES
        if is_valid_smiles(smiles):
            results['valid_count'] += 1
        else:
            results['invalid_count'] += 1
            results['invalid_samples'].append({
                'index': original_idx,
                'value': str(smiles)[:50] + "..." if len(str(smiles)) > 50 else str(smiles),
                'error': 'Invalid SMILES format'
            })
    
    # Calculate statistics
    total_checked = results['valid_count'] + results['invalid_count'] + results['empty_count']
    if total_checked > 0:
        results['valid_ratio'] = results['valid_count'] / total_checked
        results['is_valid_column'] = results['valid_ratio'] >= min_valid_ratio
    
    return results

def validate_smiles_columns(df: pd.DataFrame, smiles_columns: List[str],
                           sample_size: int = 100,
                           min_valid_ratio: float = 0.8,
                           show_details: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate multiple SMILES columns in a DataFrame
    
    Args:
        df: DataFrame to validate
        smiles_columns: List of column names that should contain SMILES
        sample_size: Number of samples to check per column
        min_valid_ratio: Minimum ratio of valid SMILES required
        show_details: Whether to show detailed validation results
        
    Returns:
        tuple: (all_valid, detailed_results_dict)
    """
    if not RDKIT_AVAILABLE:
        console.print("[yellow]⚠️  Warning: RDKit not available. SMILES validation will be basic.[/yellow]")
    
    all_results = {}
    all_valid = True
    
    console.print(f"\n[bold cyan]🔍 Validating SMILES columns...[/bold cyan]")
    
    for column in smiles_columns:
        console.print(f"  • Checking column: [magenta]{column}[/magenta]")
        
        results = validate_smiles_column(df, column, sample_size, min_valid_ratio)
        all_results[column] = results
        
        if results['error_message']:
            console.print(f"    [red]❌ {results['error_message']}[/red]")
            all_valid = False
            continue
        
        if results['is_valid_column']:
            console.print(f"    [green]✅ Valid ({results['valid_count']}/{results['sample_size_checked']} samples, {results['valid_ratio']:.1%})[/green]")
        else:
            console.print(f"    [red]❌ Invalid ({results['valid_count']}/{results['sample_size_checked']} samples, {results['valid_ratio']:.1%})[/red]")
            all_valid = False
    
    # Show detailed results if requested
    if show_details and not all_valid:
        console.print(f"\n[bold red]❌ SMILES Validation Failed![/bold red]")
        
        # Create detailed report
        for column, results in all_results.items():
            if not results['is_valid_column'] and not results['error_message']:
                console.print(f"\n[bold yellow]⚠️  Issues in column '{column}':[/bold yellow]")
                
                # Show invalid samples
                invalid_samples = results['invalid_samples'][:10]  # Show first 10 invalid samples
                if invalid_samples:
                    table = Table(title=f"Invalid SMILES in '{column}' (showing first 10)")
                    table.add_column("Row Index", style="cyan")
                    table.add_column("Value", style="red")
                    table.add_column("Error", style="yellow")
                    
                    for sample in invalid_samples:
                        table.add_row(
                            str(sample['index']),
                            sample['value'],
                            sample['error']
                        )
                    
                    console.print(table)
                
                # Show recommendations
                console.print(f"\n[bold blue]💡 Recommendations for '{column}':[/bold blue]")
                if results['empty_count'] > 0:
                    console.print(f"  • Remove or fill {results['empty_count']} empty/null values")
                if results['invalid_count'] > 0:
                    console.print(f"  • Fix {results['invalid_count']} invalid SMILES strings")
                console.print(f"  • Expected format: Valid SMILES strings (e.g., 'CCO', 'c1ccccc1')")
    
    return all_valid, all_results

def suggest_potential_smiles_columns(df: pd.DataFrame, 
                                   threshold: float = 0.7) -> List[str]:
    """
    Suggest which columns might contain SMILES based on content analysis
    
    Args:
        df: DataFrame to analyze
        threshold: Minimum ratio of valid SMILES to consider a column
        
    Returns:
        list: Column names that might contain SMILES
    """
    potential_columns = []
    
    console.print(f"\n[bold cyan]🔍 Analyzing columns for potential SMILES content...[/bold cyan]")
    
    for column in df.columns:
        # Skip obviously non-SMILES columns
        if df[column].dtype in ['int64', 'float64', 'bool']:
            continue
        
        # Check a sample of the column
        sample_size = min(50, len(df))
        sample_data = df[column].head(sample_size)
        
        valid_count = 0
        total_count = 0
        
        for value in sample_data:
            if pd.notna(value) and value != "":
                total_count += 1
                if is_valid_smiles(str(value)):
                    valid_count += 1
        
        if total_count > 0:
            valid_ratio = valid_count / total_count
            if valid_ratio >= threshold:
                potential_columns.append(column)
                console.print(f"  • [green]{column}[/green]: {valid_ratio:.1%} valid SMILES")
    
    return potential_columns

def create_smiles_validation_report(df: pd.DataFrame, 
                                  smiles_columns: List[str],
                                  output_path: Optional[str] = None) -> str:
    """
    Create a comprehensive SMILES validation report
    
    Args:
        df: DataFrame to validate
        smiles_columns: List of SMILES column names
        output_path: Optional path to save the report
        
    Returns:
        str: Report content
    """
    all_valid, results = validate_smiles_columns(df, smiles_columns, show_details=False)
    
    report_lines = []
    report_lines.append("SMILES Validation Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated on: {pd.Timestamp.now()}")
    report_lines.append(f"Dataset shape: {df.shape}")
    report_lines.append(f"SMILES columns checked: {len(smiles_columns)}")
    report_lines.append("")
    
    for column, result in results.items():
        report_lines.append(f"Column: {column}")
        report_lines.append("-" * 30)
        if result['error_message']:
            report_lines.append(f"Error: {result['error_message']}")
        else:
            report_lines.append(f"Total samples: {result['total_samples']}")
            report_lines.append(f"Samples checked: {result['sample_size_checked']}")
            report_lines.append(f"Valid SMILES: {result['valid_count']}")
            report_lines.append(f"Invalid SMILES: {result['invalid_count']}")
            report_lines.append(f"Empty values: {result['empty_count']}")
            report_lines.append(f"Valid ratio: {result['valid_ratio']:.1%}")
            report_lines.append(f"Column status: {'VALID' if result['is_valid_column'] else 'INVALID'}")
        report_lines.append("")
    
    report_lines.append(f"Overall validation: {'PASSED' if all_valid else 'FAILED'}")
    
    report_content = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_content)
        console.print(f"[green]✅ Report saved to: {output_path}[/green]")
    
    return report_content

# Example usage functions
def quick_smiles_check(df: pd.DataFrame, smiles_columns: List[str]) -> bool:
    """
    Quick SMILES validation check (returns True/False)
    
    Args:
        df: DataFrame to check
        smiles_columns: List of SMILES column names
        
    Returns:
        bool: True if all columns are valid, False otherwise
    """
    all_valid, _ = validate_smiles_columns(df, smiles_columns, show_details=False)
    return all_valid

if __name__ == "__main__":
    # Example usage
    print("SMILES Validator Module")
    print("Import this module to use SMILES validation functions")
    
    # Test with sample data
    sample_data = {
        'valid_smiles': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCC'],
        'invalid_smiles': ['invalid', '123', '', 'not_smiles'],
        'mixed_column': ['CCO', 'invalid', 'c1ccccc1', ''],
        'numbers': [1, 2, 3, 4]
    }
    
    test_df = pd.DataFrame(sample_data)
    
    print("\nTesting with sample data:")
    print(test_df)
    
    # Test validation
    all_valid, results = validate_smiles_columns(test_df, ['valid_smiles', 'invalid_smiles', 'mixed_column'])
    
    print(f"\nValidation result: {'PASSED' if all_valid else 'FAILED'}") 