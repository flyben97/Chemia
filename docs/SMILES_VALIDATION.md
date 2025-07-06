# SMILES Validation Feature

## Overview

The SMILES validation feature automatically checks if columns specified as SMILES in your configuration actually contain valid SMILES strings. This helps prevent errors during feature generation and ensures data quality.

## Key Features

- **Automatic Validation**: Validates SMILES columns when data is loaded
- **RDKit-based**: Uses RDKit for accurate SMILES validation
- **Configurable**: Adjust validation strictness and behavior
- **Detailed Reporting**: Shows validation statistics and invalid samples
- **Smart Suggestions**: Suggests potential SMILES columns in your data

## Configuration

Add the `smiles_validation` section to your configuration file:

```yaml
data:
  single_file_config:
    smiles_col: ["Catalyst", "Solvent", "Reagent"]
    
    # SMILES Validation Configuration
    smiles_validation:
      enabled: true                # Enable/disable validation (default: true)
      min_valid_ratio: 0.8        # Minimum ratio of valid SMILES (default: 0.8 = 80%)
      sample_size: 200             # Number of samples to check (default: 200)
      strict_mode: true            # Raise error on failure (default: true)
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable SMILES validation |
| `min_valid_ratio` | float | `0.8` | Minimum ratio of valid SMILES required (0.0-1.0) |
| `sample_size` | integer | `200` | Number of samples to check for validation |
| `strict_mode` | boolean | `true` | If true, raise error on validation failure; if false, continue with warning |

## Usage Scenarios

### 1. Strict Validation (Recommended)
```yaml
smiles_validation:
  enabled: true
  min_valid_ratio: 0.8
  strict_mode: true
```

### 2. Lenient Validation (For Experimental Data)
```yaml
smiles_validation:
  enabled: true
  min_valid_ratio: 0.5
  strict_mode: false
```

### 3. Quick Validation (For Large Datasets)
```yaml
smiles_validation:
  enabled: true
  sample_size: 50
  min_valid_ratio: 0.8
```

### 4. Disable Validation
```yaml
smiles_validation:
  enabled: false
```

## Example Output

When validation runs, you'll see output like:

```
🔍 SMILES Validation:
  - Checking 3 SMILES columns: ['Catalyst', 'Solvent', 'Reagent']
  • Checking column: Catalyst
    ✅ Valid (95/100 samples, 95.0%)
  • Checking column: Solvent
    ❌ Invalid (45/100 samples, 45.0%)
  • Checking column: Reagent
    ✅ Valid (88/100 samples, 88.0%)
```

## Error Handling

If validation fails, the system will:

1. Show detailed error information
2. Display invalid samples
3. Suggest potential solutions
4. Suggest alternative SMILES columns (if found)

## Common Error Solutions

1. **Column not found**: Check your `smiles_col` configuration
2. **Invalid SMILES**: Clean your data or use SMILES standardization
3. **Low valid ratio**: Adjust `min_valid_ratio` or fix data quality
4. **Wrong column**: Use suggested alternative columns

## Examples

See `examples/configs/smiles_validation_example.yaml` for a complete configuration example.

## Performance Notes

- Validation samples a subset of your data for performance
- Increase `sample_size` for more thorough validation
- Decrease `sample_size` for faster validation of large datasets
- Validation uses RDKit when available, falls back to basic checks otherwise 