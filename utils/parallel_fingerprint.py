"""
Parallel molecular fingerprint calculation module.
Optimized for high-performance batch processing of Morgan fingerprints and other molecular features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
from rich.console import Console
from rich.progress import Progress, TaskID

# Handle RDKit imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, Descriptors

    # Import fingerprint generators
    rdFingerprintGenerator = None
    rdMolDescriptors = None
    HAS_NEW_FP_GENERATOR = False

    try:
        import importlib
        rdFingerprintGenerator = importlib.import_module('rdkit.Chem.rdFingerprintGenerator')
        HAS_NEW_FP_GENERATOR = True
    except ImportError:
        try:
            rdMolDescriptors = importlib.import_module('rdkit.Chem.rdMolDescriptors')
            HAS_NEW_FP_GENERATOR = False
        except ImportError:
            pass

    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RDKit not available: {e}")
    RDKIT_AVAILABLE = False
    rdFingerprintGenerator = None
    rdMolDescriptors = None
    HAS_NEW_FP_GENERATOR = False

console = Console()

def _calculate_single_morgan_fingerprint(args: Tuple[str, int, int]) -> Tuple[str, Optional[np.ndarray]]:
    """
    Calculate Morgan fingerprint for a single SMILES string.

    Args:
        args: Tuple of (smiles, radius, nBits)

    Returns:
        Tuple of (smiles, fingerprint_array or None if failed)
    """
    smiles, radius, nBits = args

    if not RDKIT_AVAILABLE or rdFingerprintGenerator is None:
        return smiles, None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = morgan_gen.GetFingerprint(mol)
        fp_array = np.array(fp, dtype=np.int8)  # Use int8 to save memory

        return smiles, fp_array
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to calculate Morgan fingerprint for {smiles}: {e}[/yellow]")
        return smiles, None

def _calculate_single_fingerprint_batch(args: Tuple[List[str], str, Dict[str, Any]]) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Calculate fingerprints for a batch of SMILES strings in a single process.
    This reduces the overhead of process creation for small batches.

    Args:
        args: Tuple of (smiles_batch, fp_type, config)

    Returns:
        List of (smiles, fingerprint_array or None) tuples
    """
    smiles_batch, fp_type, config = args
    results = []

    if not RDKIT_AVAILABLE:
        return [(smiles, None) for smiles in smiles_batch]

    for smiles in smiles_batch:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append((smiles, None))
                continue

            fp_array = None
            if fp_type.lower() == "morgan" and rdFingerprintGenerator is not None:
                radius = config.get('radius', 2)
                nBits = config.get('nBits', 2048)
                morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
                fp = morgan_gen.GetFingerprint(mol)
                fp_array = np.array(fp, dtype=np.int8)
            elif fp_type.lower() == "maccs":
                fp = MACCSkeys.GenMACCSKeys(mol)
                fp_array = np.array(fp, dtype=np.int8)
            elif fp_type.lower() == "rdkit" and rdFingerprintGenerator is not None:
                nBits = config.get('nBits', 2048)
                rdk_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7, fpSize=nBits)
                fp = rdk_gen.GetFingerprint(mol)
                fp_array = np.array(fp, dtype=np.int8)
            elif fp_type.lower() == "atompair" and rdFingerprintGenerator is not None:
                nBits = config.get('nBits', 2048)
                ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)
                fp = ap_gen.GetFingerprint(mol)
                fp_array = np.array(fp, dtype=np.int8)
            elif fp_type.lower() == "torsion" and rdFingerprintGenerator is not None:
                nBits = config.get('nBits', 2048)
                torsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)
                fp = torsion_gen.GetFingerprint(mol)
                fp_array = np.array(fp, dtype=np.int8)

            results.append((smiles, fp_array))

        except Exception as e:
            results.append((smiles, None))

    return results

def calculate_morgan_fingerprints_parallel(
    smiles_list: List[str],
    radius: int = 2,
    nBits: int = 2048,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate Morgan fingerprints for a list of SMILES using parallel processing.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius (default: 2)
        nBits: Number of bits in fingerprint (default: 2048)
        n_jobs: Number of parallel processes. If None, uses CPU count - 1
        batch_size: Batch size for processing. If None, automatically determined

    Returns:
        pandas.DataFrame with SMILES as index and fingerprint bits as columns
    """
    if not RDKIT_AVAILABLE:
        console.print("[bold red]Error: RDKit not available. Cannot calculate Morgan fingerprints.[/bold red]")
        return pd.DataFrame(index=pd.Index(smiles_list))

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    if batch_size is None:
        # Automatically determine batch size based on list length and number of jobs
        batch_size = max(1, len(smiles_list) // (n_jobs * 4))

    console.print(f"[green]Calculating Morgan fingerprints in parallel...[/green]")
    console.print(f"[blue]Parameters: radius={radius}, nBits={nBits}, n_jobs={n_jobs}, batch_size={batch_size}[/blue]")

    # Create batches
    smiles_batches = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        smiles_batches.append(batch)

    # Prepare arguments for parallel processing
    config = {'radius': radius, 'nBits': nBits}
    batch_args = [(batch, "morgan", config) for batch in smiles_batches]

    # Process in parallel
    all_results = []
    with Progress() as progress:
        task = progress.add_task("Processing batches...", total=len(batch_args))

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_calculate_single_fingerprint_batch, args) for args in batch_args]

            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
                progress.advance(task)

    # Process results
    feature_matrix = []
    failed_count = 0
    zero_row = [0] * nBits

    # Create column names
    columns = [f"morgan_{i+1}" for i in range(nBits)]

    for smiles, fp_array in all_results:
        if fp_array is None:
            feature_matrix.append(zero_row)
            failed_count += 1
        else:
            feature_matrix.append(fp_array.tolist())

    if failed_count > 0:
        console.print(f"[yellow]Warning: Failed to calculate fingerprints for {failed_count} molecules. Filled with zeros.[/yellow]")

    # Create DataFrame
    df = pd.DataFrame(feature_matrix, index=pd.Index(smiles_list), columns=pd.Index(columns))

    console.print(f"[green]✓ Successfully calculated Morgan fingerprints. Shape: {df.shape}[/green]")
    return df

def calculate_fingerprints_parallel(
    smiles_list: List[str],
    fp_type: str,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate various types of molecular fingerprints using parallel processing.

    Args:
        smiles_list: List of SMILES strings
        fp_type: Type of fingerprint ("morgan", "maccs", "rdkit", "atompair", "torsion")
        n_jobs: Number of parallel processes
        batch_size: Batch size for processing
        **kwargs: Additional parameters (radius, nBits, etc.)

    Returns:
        pandas.DataFrame with fingerprints
    """
    if not RDKIT_AVAILABLE:
        console.print(f"[bold red]Error: RDKit not available. Cannot calculate {fp_type} fingerprints.[/bold red]")
        return pd.DataFrame(index=pd.Index(smiles_list))

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    if batch_size is None:
        batch_size = max(1, len(smiles_list) // (n_jobs * 4))

    # Special handling for Morgan fingerprints (most optimized)
    if fp_type.lower() == "morgan":
        return calculate_morgan_fingerprints_parallel(
            smiles_list,
            radius=kwargs.get('radius', 2),
            nBits=kwargs.get('nBits', 2048),
            n_jobs=n_jobs,
            batch_size=batch_size
        )

    console.print(f"[green]Calculating {fp_type.upper()} fingerprints in parallel...[/green]")
    console.print(f"[blue]Parameters: n_jobs={n_jobs}, batch_size={batch_size}[/blue]")

    # Create batches
    smiles_batches = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        smiles_batches.append(batch)

    # Prepare arguments
    config = kwargs.copy()
    batch_args = [(batch, fp_type, config) for batch in smiles_batches]

    # Process in parallel
    all_results = []
    with Progress() as progress:
        task = progress.add_task(f"Processing {fp_type} fingerprints...", total=len(batch_args))

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_calculate_single_fingerprint_batch, args) for args in batch_args]

            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
                progress.advance(task)

    # Determine dimensions
    nBits = kwargs.get('nBits', 2048)
    if fp_type.lower() == "maccs":
        nBits = 167  # MACCS keys are actually 167 bits in RDKit

    # Process results
    feature_matrix = []
    failed_count = 0
    zero_row = [0] * nBits
    columns = [f"{fp_type}_{i+1}" for i in range(nBits)]

    for smiles, fp_array in all_results:
        if fp_array is None:
            feature_matrix.append(zero_row)
            failed_count += 1
        else:
            feature_matrix.append(fp_array.tolist())

    if failed_count > 0:
        console.print(f"[yellow]Warning: Failed to calculate {fp_type} fingerprints for {failed_count} molecules.[/yellow]")

    # Create DataFrame
    df = pd.DataFrame(feature_matrix, index=pd.Index(smiles_list), columns=pd.Index(columns))

    console.print(f"[green]✓ Successfully calculated {fp_type.upper()} fingerprints. Shape: {df.shape}[/green]")
    return df

def benchmark_parallel_vs_sequential(
    smiles_list: List[str],
    fp_type: str = "morgan",
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark parallel vs sequential fingerprint calculation.

    Args:
        smiles_list: List of SMILES strings
        fp_type: Type of fingerprint to benchmark
        **kwargs: Additional parameters

    Returns:
        Dictionary with benchmark results
    """
    import time
    from .mol_fp_features import calculate_molecular_features

    console.print(f"[cyan]Benchmarking {fp_type} fingerprint calculation...[/cyan]")
    console.print(f"[blue]Dataset size: {len(smiles_list)} molecules[/blue]")

    # Sequential calculation
    console.print("[yellow]Running sequential calculation...[/yellow]")
    start_time = time.time()

    sequential_results = []
    for smiles in smiles_list:
        result = calculate_molecular_features(
            smiles,
            fp_type=fp_type,
            descriptors=False,
            **kwargs
        )
        sequential_results.append(result)

    sequential_time = time.time() - start_time

    # Parallel calculation
    console.print("[yellow]Running parallel calculation...[/yellow]")
    start_time = time.time()

    parallel_df = calculate_fingerprints_parallel(smiles_list, fp_type, **kwargs)

    parallel_time = time.time() - start_time

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    results = {
        'dataset_size': len(smiles_list),
        'fp_type': fp_type,
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'cpu_count': mp.cpu_count()
    }

    console.print(f"[green]Benchmark Results:[/green]")
    console.print(f"  Sequential time: {sequential_time:.2f} seconds")
    console.print(f"  Parallel time: {parallel_time:.2f} seconds")
    console.print(f"  Speedup: {speedup:.2f}x")
    console.print(f"  CPU cores: {mp.cpu_count()}")

    return results

if __name__ == "__main__":
    # Test with some example molecules
    test_smiles = [
        'CCO',  # Ethanol
        'c1ccccc1',  # Benzene
        'O=C(C)Oc1ccccc1C(=O)O',  # Aspirin
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    ] * 100  # Replicate for testing

    console.print(f"[cyan]Testing parallel fingerprint calculation with {len(test_smiles)} molecules[/cyan]")

    # Test Morgan fingerprints
    morgan_df = calculate_morgan_fingerprints_parallel(test_smiles)
    console.print(f"Morgan fingerprints shape: {morgan_df.shape}")

    # Test other fingerprint types
    for fp_type in ["maccs", "rdkit"]:
        fp_df = calculate_fingerprints_parallel(test_smiles, fp_type)
        console.print(f"{fp_type.upper()} fingerprints shape: {fp_df.shape}")

    # Run benchmark
    benchmark_results = benchmark_parallel_vs_sequential(test_smiles[:50])  # Smaller set for benchmark
