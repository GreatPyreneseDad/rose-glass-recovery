# Rose Glass Recovery

**Translation Framework for Addiction Counseling & Behavioral Health**

> *"The counselor already knows. Rose Glass makes it visible."*

## Architecture

This repo provides **recovery-specific calibrations and clinical insights** on top of the core [Rose Glass](https://github.com/GreatPyreneseDad/rose-glass) ML framework.

```
┌─────────────────────────────────────────────────────────────┐
│                   rose-glass-recovery                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Recovery        │  │ Clinical        │  │ Session     │ │
│  │ Calibrations    │  │ Insights        │  │ Trajectory  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           └────────────────────┼───────────────────┘        │
│                                │                            │
│                    ┌───────────▼───────────┐               │
│                    │   RoseGlassBridge     │               │
│                    │   (ML or Regex)       │               │
│                    └───────────┬───────────┘               │
└────────────────────────────────┼────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     rose-glass (core)   │
                    │  ┌────┐ ┌────┐ ┌────┐  │
                    │  │ Ψ  │ │ ρ  │ │ q  │  │
                    │  │ ML │ │ ML │ │ ML │  │
                    │  └────┘ └────┘ └────┘  │
                    │  ┌────┐ ┌────┐ ┌────┐  │
                    │  │ f  │ │ τ  │ │ λ  │  │
                    │  │ ML │ │ ML │ │ ML │  │
                    │  └────┘ └────┘ └────┘  │
                    └─────────────────────────┘
```

### Two Modes

**ML Mode** (when `rose-glass` is installed):
- Full neural network extractors for Ψ, ρ, q, f
- Transformer-based embeddings (RoBERTa)
- Biological optimization with learned parameters
- ~85% accuracy target

**Regex Mode** (standalone fallback):
- Pattern matching for key markers
- No external dependencies
- ~70% accuracy
- Fast, explainable

```python
from src.integrations import get_bridge

# Auto-detects available mode
bridge = get_bridge()
print(bridge.mode_description)  # "ML Mode (Rose Glass Core v2.1)" or "Regex Mode (Fallback)"
```

## The Problem

Addiction counselors develop intuition through pattern recognition across hundreds of clients. They *feel* when someone is heading toward relapse. But:

- That intuition is dismissed as "soft" or "subjective"
- Documentation systems capture *content* ("discussed triggers") not *pattern* ("isolation language increasing")  
- By the time "objective" metrics show risk (missed appointments, positive test), the window for intervention has closed

## The Solution

Rose Glass translates counselor intuition into visible dimensions that clinical teams can act on—**without judgment, without diagnosis, without pathologizing**.

```python
from src.core import RecoveryTranslator

translator = RecoveryTranslator()

# Analyze session transcript
analysis = translator.analyze_session(transcript)

# What the counselor felt but couldn't prove:
print(analysis.get_clinical_insight())
```

Output:
```
⚠️  PATTERN SHIFT DETECTED

f-dimension collapse: Client shifted from "we" and "my sponsor" to "I" and 
"nobody understands" over last 3 sessions.

q-gradient spike: Emotional activation doubles when employment discussed.

Estimated intervention window: 7-14 days before crisis point
```

## Installation

### Standalone (Regex Mode)
```bash
git clone https://github.com/GreatPyreneseDad/rose-glass-recovery.git
cd rose-glass-recovery
pip install -r requirements.txt
```

### With ML (Recommended)
```bash
# First install core
git clone https://github.com/GreatPyreneseDad/rose-glass.git ~/rose-glass
pip install -r ~/rose-glass/requirements.txt

# Then install recovery
git clone https://github.com/GreatPyreneseDad/rose-glass-recovery.git
cd rose-glass-recovery
pip install -r requirements.txt
```

## Quick Start

```python
from src.core import RecoveryTranslator

# Initialize with trauma-informed defaults
translator = RecoveryTranslator(
    calibration="trauma_informed",
    enable_gradient_tracking=True
)

# Single statement analysis
result = translator.analyze_statement(
    "I'm fine. Everything's under control. I don't need to call my sponsor."
)

print(f"Ψ (consistency): {result.psi:.2f}")  # Low - contradiction signals
print(f"f (belonging): {result.f:.2f}")      # Low - isolation from support
print(f"Authenticity: {result.authenticity_score:.2f}")  # Performance detected
```

### Session Tracking

```python
# Track patterns across session
translator.start_session()

translator.add_to_session("I've been doing great this week")
translator.add_to_session("Work has been stressful but I'm handling it")
translator.add_to_session("I don't really need meetings anymore")
translator.add_to_session("Nobody at work understands what I'm going through")

# Get trajectory analysis
trajectory = translator.get_session_trajectory()

print(f"f-dimension trend: {trajectory.f_trend}")  # DECLINING
print(f"Isolation markers: {trajectory.isolation_markers}")
print(f"Intervention window: {trajectory.estimated_window}")
```

## The Four Dimensions (Recovery Context)

| Dimension | What It Measures | Risk Signal |
|-----------|------------------|-------------|
| **Ψ (Psi)** | Narrative consistency | Contradictions, fragmented story |
| **ρ (Rho)** | Wisdom integration | No connection to past patterns |
| **q** | Emotional activation | Spike around specific topics |
| **f** | Social belonging | "We" → "I" shift, isolation language |

### Extended Dimensions

| Dimension | What It Measures | What It Reveals |
|-----------|------------------|-----------------|
| **τ (Tau)** | Temporal depth | Living in crisis present vs. integrated timeline |
| **λ (Lambda)** | Lens interference | How cultural context shapes expression |
| **∇q/∇t** | Gradient tracking | Escalation trajectory, intervention window |

## Neurodivergent Calibrations

Many addiction clients are neurodivergent. Standard "affect detection" pathologizes their communication:

```python
# Autism: Values logical consistency, direct communication
translator = RecoveryTranslator(calibration="autism_spectrum")

# ADHD: Values associative connections, high engagement variability
translator = RecoveryTranslator(calibration="adhd")

# Trauma: Tactical communication, heightened awareness
translator = RecoveryTranslator(calibration="trauma_informed")
```

## Academic Validation

Rose Glass dimensions are validated against peer-reviewed research:

| Finding | Source | Rose Glass Mapping |
|---------|--------|-------------------|
| "Stressed" has 40% similarity to "drink" | Kramer et al. 2024 | q-dimension |
| "Bored" has 45% similarity to "craving" | Kramer et al. 2024 | q-dimension |
| Social isolation predicts relapse | Lu et al. 2019 | f-dimension |
| 78.48% accuracy predicting escalation | ResearchGate 2024 | Ψ-dimension |

See [docs/ACADEMIC_VALIDATION.md](docs/ACADEMIC_VALIDATION.md) for complete mapping.

## Ethical Framework

### What We Don't Do
- **No Profiling**: Never infer identity or demographics
- **No Judgment**: Translation without quality assessment
- **No Surveillance**: Consensual therapeutic contexts only
- **No Replacement**: Augments counselor intuition, never replaces

### What We Do
- **Validate Intuition**: Give language to what counselors already sense
- **Enable Advocacy**: Documentation for clinical decision-making
- **Respect Dignity**: All communication patterns are valid
- **Cultural Humility**: Multiple lenses for multiple contexts

## Project Structure

```
rose-glass-recovery/
├── src/
│   ├── core/
│   │   └── recovery_translator.py    # Main translation engine
│   ├── calibrations/                  # Neurodivergent + cultural
│   └── integrations/
│       └── rose_glass_bridge.py      # ML/Regex bridge
├── docs/
│   ├── ACADEMIC_VALIDATION.md
│   ├── INTEGRATION_ARCHITECTURE.md
│   └── ETHICS.md
├── tests/
└── README.md
```

## Related Repositories

| Repo | Purpose |
|------|---------|
| [rose-glass](https://github.com/GreatPyreneseDad/rose-glass) | Core v2.1 ML framework |
| [RoseGlassLE](https://github.com/GreatPyreneseDad/RoseGlassLE) | Law enforcement extensions |
| [rose-looking-glass](https://github.com/GreatPyreneseDad/rose-looking-glass) | API-ready implementation |
| [emotionally-informed-rag](https://github.com/GreatPyreneseDad/emotionally-informed-rag) | Legal RAG with Rose Glass |

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{roseglassrecovery2025,
  author = {MacGregor bin Joseph, Christopher},
  title = {Rose Glass Recovery: Translation Framework for Addiction Counseling},
  year = {2025},
  version = {1.0},
  url = {https://github.com/GreatPyreneseDad/rose-glass-recovery}
}
```

---

**"The counselor already knows. Rose Glass just makes it visible—without judgment, without measurement, with full respect for the dignity of every person in recovery."**
