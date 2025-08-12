#!/usr/bin/env python3
"""
Utility functions for diameter measurements
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def log_diameter_statistics(diameters: np.ndarray, prefix: str = "Diameter", method: str = None):
    """
    Log diameter measurement statistics.
    
    Args:
        diameters: Array of diameter measurements
        prefix: Prefix for log message (e.g., "Diameter", "Gradient-based")
        method: Optional method name to include in the message
    """
    non_zero = diameters[diameters > 0]
    
    # Build message prefix
    if method:
        message_prefix = f"{prefix} ({method})"
    else:
        message_prefix = prefix
    
    if len(non_zero) > 0:
        logger.info(f"{message_prefix} statistics: "
                   f"min={np.min(non_zero):.1f}, "
                   f"max={np.max(non_zero):.1f}, "
                   f"mean={np.mean(non_zero):.1f}, "
                   f"median={np.median(non_zero):.1f} pixels")
        logger.info(f"Total points: {len(diameters)}, Non-zero: {len(non_zero)}")
    else:
        logger.warning(f"All {message_prefix.lower()} measurements are zero!")