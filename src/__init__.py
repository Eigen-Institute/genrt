"""
genrt - Machine Learning Accelerator for RT-TDDFT

A universal quantum dynamics model for accelerating Real-Time Time-Dependent
Density Functional Theory simulations using E3NN equivariant networks and
Mamba state space models.
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import utils

__all__ = ["data", "models", "utils", "__version__"]
