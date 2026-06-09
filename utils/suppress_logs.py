#!/usr/bin/env python3
"""
Utility module to suppress debug logs and warnings

Import this module at the beginning of any script to clean up output:
    from utils.suppress_logs import suppress_debug_logs
    suppress_debug_logs()

Or simply import it (it will automatically apply suppressions):
    import utils.suppress_logs
"""

import warnings
import logging
import os


def suppress_debug_logs():
    """
    Suppress debug logs and warnings from various libraries
    """
    # Suppress common warning categories
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    # Suppress specific library warnings
    warnings.filterwarnings("ignore", message=".*positional args.*")  # graphviz
    warnings.filterwarnings("ignore", message=".*matplotlib.*")
    warnings.filterwarnings("ignore", message=".*rdkit.*")

    # Configure logging levels for specific libraries
    library_loggers = [
        'matplotlib',
        'matplotlib.font_manager',
        'matplotlib.pyplot',
        'matplotlib.figure',
        'graphviz',
        'graphviz._tools',
        'PIL',
        'PIL.Image',
        'rdkit',
        'rdkit.Chem',
        'urllib3',
        'requests',
        # Add numba loggers to suppress SSA debug output
        'numba',
        'numba.core',
        'numba.core.ssa',
        'numba.core.types',
        'numba.core.compiler',
        'numba.core.pythonapi',
        'numba.core.lowering',
        'numba.core.analysis',
        'numba.typed',
        'numba.cuda'
    ]

    for logger_name in library_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set root logger to INFO level to avoid DEBUG spam
    # but only if it hasn't been configured yet
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)
    else:
        # If already configured, just set level
        root_logger.setLevel(logging.INFO)

    # Suppress matplotlib DEBUG specifically
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'


def suppress_numba_logs():
    """
    Additional function to specifically suppress Numba compilation logs
    This is especially useful when using UniMol which depends on Numba
    """
    try:
        import numba
        # Set numba logging to WARNING level
        numba.config.DISABLE_JIT = False  # Keep JIT enabled

        # Configure numba logging
        numba_loggers = [
            'numba',
            'numba.core',
            'numba.core.ssa',
            'numba.core.types',
            'numba.core.compiler',
            'numba.core.pythonapi',
            'numba.core.lowering',
            'numba.core.analysis',
            'numba.typed',
            'numba.cuda'
        ]

        for logger_name in numba_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
            logging.getLogger(logger_name).propagate = False

    except ImportError:
        pass


def suppress_optuna_logs():
    """
    Additional function to suppress Optuna optimization logs
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass


# Side-effects removed: callers should explicitly invoke suppress_debug_logs() when needed.
# Example:
#   from utils.suppress_logs import suppress_debug_logs
#   suppress_debug_logs()
