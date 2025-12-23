"""
Rose Glass Core Integration
===========================

Integrates rose-glass-recovery with the core Rose Glass ML pipeline.

When rose-glass is available, uses full ML models.
Falls back to regex-based extraction when ML unavailable.
"""

import sys
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Track integration status
ML_AVAILABLE = False
CORE_VERSION = None


@dataclass
class GCTVariables:
    """GCT variable container - mirrors core definition"""
    psi: float  # Internal consistency
    rho: float  # Accumulated wisdom
    q: float    # Moral/emotional activation
    f: float    # Social belonging
    tau: float = 0.5  # Temporal depth
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'psi': self.psi,
            'rho': self.rho,
            'q': self.q,
            'f': self.f,
            'tau': self.tau
        }


# Try to import core ML models
try:
    # Add potential paths
    import os
    potential_paths = [
        os.path.expanduser('~/rose-glass'),
        os.path.expanduser('~/rose-glass/ml-models'),
        '/Users/chris/rose-glass',
        '/Users/chris/rose-glass/ml-models',
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    from realtime_gct_pipeline import RealtimeGCTPipeline, CoherenceSnapshot
    from gct_ml_framework import GCTVariables as CoreGCTVariables
    
    ML_AVAILABLE = True
    CORE_VERSION = "2.1"
    logger.info(f"Rose Glass ML core loaded (v{CORE_VERSION})")
    
except ImportError as e:
    logger.warning(f"Rose Glass ML core not available: {e}")
    logger.warning("Falling back to regex-based extraction")
    ML_AVAILABLE = False


class RoseGlassBridge:
    """
    Bridge between rose-glass-recovery and core Rose Glass ML.
    
    Provides consistent interface regardless of whether ML is available.
    """
    
    def __init__(self, 
                 use_ml: bool = True,
                 model_dir: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the bridge.
        
        Args:
            use_ml: Whether to use ML models (if available)
            model_dir: Directory containing trained models
            device: Device for ML inference ("cpu" or "cuda")
        """
        self.use_ml = use_ml and ML_AVAILABLE
        self.device = device
        
        if self.use_ml:
            try:
                self.pipeline = RealtimeGCTPipeline(
                    model_dir=model_dir or "/Users/chris/GCT-ML-Lab/models",
                    device=device
                )
                logger.info("ML pipeline initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ML pipeline: {e}")
                self.use_ml = False
                self.pipeline = None
        else:
            self.pipeline = None
            
        # Regex fallback patterns (from recovery_translator.py)
        self._init_regex_patterns()
    
    def _init_regex_patterns(self):
        """Initialize regex patterns for fallback mode"""
        import re
        
        self.isolation_patterns = [
            re.compile(r'\b(alone|lonely|isolated|nobody|no one|by myself)\b', re.I),
            re.compile(r'\b(don\'t understand|doesn\'t get it|can\'t relate)\b', re.I),
        ]
        
        self.connection_patterns = [
            re.compile(r'\b(we|us|our|together)\b', re.I),
            re.compile(r'\b(sponsor|group|meeting|fellowship|support)\b', re.I),
        ]
        
        self.activation_patterns = [
            re.compile(r'\b(stressed|stress|anxious|anxiety|worried)\b', re.I),
            re.compile(r'\b(angry|frustrated|upset|scared)\b', re.I),
        ]
        
        self.wisdom_patterns = [
            re.compile(r'\b(learned|realized|understood|remember when)\b', re.I),
            re.compile(r'\b(pattern|cycle|used to|in the past)\b', re.I),
        ]
    
    def extract_variables(self, text: str) -> GCTVariables:
        """
        Extract GCT variables from text.
        
        Uses ML models if available, falls back to regex.
        """
        if self.use_ml and self.pipeline:
            return self._extract_ml(text)
        else:
            return self._extract_regex(text)
    
    def _extract_ml(self, text: str) -> GCTVariables:
        """Extract using ML models"""
        import torch
        
        with torch.no_grad():
            # Get predictions from each model
            psi, _ = self.pipeline.psi_trainer.predict(text)
            rho, _ = self.pipeline.rho_trainer.predict(text)
            q, _ = self.pipeline.q_trainer.predict(text)
            f, _ = self.pipeline.f_trainer.predict(text)
        
        # Estimate tau (not in current ML pipeline, use heuristic)
        tau = self._estimate_tau(text)
        
        return GCTVariables(
            psi=float(psi),
            rho=float(rho),
            q=float(q),
            f=float(f),
            tau=tau
        )
    
    def _extract_regex(self, text: str) -> GCTVariables:
        """Extract using regex patterns (fallback)"""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = max(len(words), 1)
        
        # f-dimension: social belonging
        isolation_count = sum(
            len(p.findall(text_lower)) for p in self.isolation_patterns
        )
        connection_count = sum(
            len(p.findall(text_lower)) for p in self.connection_patterns
        )
        total = isolation_count + connection_count
        f = connection_count / total if total > 0 else 0.5
        
        # q-dimension: emotional activation
        activation_count = sum(
            len(p.findall(text_lower)) for p in self.activation_patterns
        )
        exclamations = text.count('!')
        q = min(0.3 + activation_count * 0.15 + exclamations * 0.1, 1.0)
        
        # rho-dimension: wisdom
        wisdom_count = sum(
            len(p.findall(text_lower)) for p in self.wisdom_patterns
        )
        rho = min(0.3 + wisdom_count * 0.15, 1.0)
        
        # psi-dimension: consistency (harder with regex)
        # Check for contradiction markers
        contradiction_pairs = [
            (r"i'm fine", r"but|however"),
            (r"under control", r"can't|struggling"),
        ]
        import re
        contradictions = 0
        for pos, neg in contradiction_pairs:
            if re.search(pos, text_lower) and re.search(neg, text_lower):
                contradictions += 1
        psi = max(0.8 - contradictions * 0.2, 0.3)
        
        # tau: temporal depth
        tau = self._estimate_tau(text)
        
        return GCTVariables(
            psi=psi,
            rho=rho,
            q=q,
            f=f,
            tau=tau
        )
    
    def _estimate_tau(self, text: str) -> float:
        """Estimate temporal depth from text"""
        import re
        text_lower = text.lower()
        
        past_markers = len(re.findall(
            r'\b(years ago|used to|back then|learned|history|pattern)\b',
            text_lower
        ))
        present_markers = len(re.findall(
            r'\b(right now|today|tonight|this moment|urgent)\b',
            text_lower
        ))
        
        if past_markers > present_markers:
            return min(0.5 + past_markers * 0.1, 1.0)
        elif present_markers > past_markers:
            return max(0.5 - present_markers * 0.1, 0.1)
        return 0.5
    
    def get_coherence(self, variables: GCTVariables) -> float:
        """Calculate coherence from variables"""
        # Biological optimization on q
        km, ki = 0.3, 2.0
        q_opt = variables.q / (km + variables.q + (variables.q ** 2 / ki))
        
        # Coherence formula
        coherence = (
            variables.psi + 
            variables.rho * variables.psi + 
            q_opt + 
            variables.f * variables.psi
        ) / 4
        
        return min(coherence, 1.0)
    
    @property
    def is_ml_mode(self) -> bool:
        """Check if using ML mode"""
        return self.use_ml
    
    @property
    def mode_description(self) -> str:
        """Human-readable mode description"""
        if self.use_ml:
            return f"ML Mode (Rose Glass Core v{CORE_VERSION})"
        return "Regex Mode (Fallback)"


# Convenience function
def get_bridge(use_ml: bool = True) -> RoseGlassBridge:
    """Get a configured bridge instance"""
    return RoseGlassBridge(use_ml=use_ml)


# Test
if __name__ == "__main__":
    bridge = get_bridge()
    print(f"Mode: {bridge.mode_description}")
    
    test_text = "I'm doing fine but nobody understands what I'm going through. I'm all alone in this."
    
    variables = bridge.extract_variables(test_text)
    coherence = bridge.get_coherence(variables)
    
    print(f"\nTest: {test_text[:50]}...")
    print(f"Variables: {variables.to_dict()}")
    print(f"Coherence: {coherence:.3f}")
