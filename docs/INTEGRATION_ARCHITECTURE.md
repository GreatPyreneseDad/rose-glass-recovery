# Rose Glass Recovery - Integration Architecture

## Current State

`rose-glass-recovery` is currently a **standalone** repo with regex-based pattern matching. This was intentional for the initial proof-of-concept, but production deployment should integrate with the full ML infrastructure from the core `rose-glass` repo.

## Integration Options

### Option 1: Import from Core (Recommended)

Install `rose-glass` as dependency and use its ML extractors:

```python
# requirements.txt
rose-glass @ git+https://github.com/GreatPyreneseDad/rose-glass.git

# recovery_translator.py
from rose_glass.ml_models import (
    create_psi_model,
    create_rho_model, 
    create_q_model,
    create_f_model,
    RealtimeGCTPipeline
)

class RecoveryTranslator:
    def __init__(self):
        # Use core ML pipeline
        self.pipeline = RealtimeGCTPipeline()
        
        # Add recovery-specific calibrations
        self.recovery_calibration = RecoveryCalibration()
```

### Option 2: Submodule

Add `rose-glass` as git submodule:

```bash
git submodule add https://github.com/GreatPyreneseDad/rose-glass.git lib/rose-glass
```

### Option 3: Copy ML Models

For fully standalone deployment, copy the ML models:
- `ml-models/psi_consistency_model.py`
- `ml-models/rho_wisdom_model.py`
- `ml-models/q_moral_activation_model.py`
- `ml-models/f_social_belonging_model.py`
- `ml-models/realtime_gct_pipeline.py`

## Recovery-Specific Enhancements

Regardless of integration approach, `rose-glass-recovery` adds:

### 1. Recovery Calibrations
- Trauma-informed defaults
- Neurodivergent support (autism, ADHD)
- Addiction-specific vocabulary

### 2. Recovery Feature Extractors
```python
class RecoveryFeatureExtractor:
    """Addiction-specific features that augment core ML"""
    
    RECOVERY_MARKERS = [
        'sponsor', 'meeting', 'step', 'clean', 'sober',
        'relapse', 'trigger', 'craving', 'recovery'
    ]
    
    ISOLATION_MARKERS = [
        'alone', 'nobody', 'no one', 'by myself',
        "doesn't understand", "can't relate"
    ]
```

### 3. Session Trajectory Analysis
```python
class SessionAnalyzer:
    """Track patterns across session - not in core repo"""
    
    def analyze_trajectory(self, snapshots: List[CoherenceSnapshot]):
        # f-dimension trend
        # q-gradient spikes
        # Isolation marker accumulation
        pass
```

### 4. Clinical Insight Generation
```python
class ClinicalInsightGenerator:
    """Translate ML outputs to clinical language"""
    
    def generate_insight(self, variables: GCTVariables) -> ClinicalInsight:
        # Recovery-specific interpretation
        # Counselor-facing language
        # Intervention window estimation
        pass
```

## ML Model Training for Recovery

The core ML models should be fine-tuned on addiction recovery data:

### Training Data Sources
1. **Reddit r/stopdrinking** - 279,688 posts (Kramer et al. 2024)
2. **Reddit r/OpiatesRecovery** - Recovery transition data (Lu et al. 2019)
3. **Clinical transcripts** - With consent and IRB approval

### Fine-Tuning Approach
```python
# Fine-tune q-model on recovery emotional patterns
q_model, q_trainer = create_q_model()
q_trainer.train(
    train_data=recovery_labeled_data,
    calibration="trauma_informed"
)
```

## Recommended Next Steps

1. **Short-term**: Keep regex version for demos, note ML integration path
2. **Medium-term**: Add rose-glass as dependency, use core ML pipeline
3. **Long-term**: Fine-tune models on recovery-specific data, publish research

## File Structure After Integration

```
rose-glass-recovery/
├── src/
│   ├── core/
│   │   ├── recovery_translator.py     # Uses rose-glass ML
│   │   ├── recovery_calibrations.py   # Recovery-specific params
│   │   └── clinical_insight.py        # Clinical translation
│   ├── extractors/
│   │   ├── recovery_features.py       # Addiction vocabulary
│   │   └── isolation_detector.py      # f-dimension enhancement
│   ├── session/
│   │   ├── trajectory_analyzer.py     # Multi-statement tracking
│   │   └── intervention_estimator.py  # Window prediction
│   └── integrations/
│       └── eleos_connector.py         # Eleos Health API
├── models/
│   └── recovery_finetuned/            # Fine-tuned weights
├── tests/
└── docs/
```

## Current Limitations

The regex-based version has these limitations vs ML integration:

| Feature | Regex Version | ML Integration |
|---------|---------------|----------------|
| Accuracy | ~70% | Target 85%+ |
| Nuance | Limited | Captures subtlety |
| Context | None | Sentence-level |
| Training | Fixed rules | Learnable |
| Speed | Fast | Requires GPU for best perf |

The regex version is suitable for:
- Demos and proof-of-concept
- Resource-constrained environments
- Explainability (transparent rules)

ML integration is needed for:
- Production clinical deployment
- Research validation
- Subtle pattern detection
- Fine-grained gradient tracking
