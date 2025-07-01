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
        'requests'
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


def suppress_optuna_logs():
    """
    Additional function to suppress Optuna optimization logs
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass


# Automatically apply suppressions when module is imported
suppress_debug_logs() 