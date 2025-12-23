"""Integration modules for Rose Glass Recovery"""

from .rose_glass_bridge import (
    RoseGlassBridge,
    GCTVariables,
    get_bridge,
    ML_AVAILABLE,
    CORE_VERSION
)

__all__ = [
    'RoseGlassBridge',
    'GCTVariables', 
    'get_bridge',
    'ML_AVAILABLE',
    'CORE_VERSION'
]
