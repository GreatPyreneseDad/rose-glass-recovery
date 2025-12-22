"""Test the recovery translator with concerning patterns"""

from recovery_translator import RecoveryTranslator

translator = RecoveryTranslator(calibration="trauma_informed")

print("=" * 60)
print("ROSE GLASS RECOVERY - CONCERNING PATTERN TEST")
print("=" * 60)

# Statements showing escalating isolation and performance
concerning_statements = [
    "I'm fine, really. Everything is totally under control now.",
    "I don't need meetings anymore. I've got this figured out.",
    "Nobody at work understands what I'm dealing with. I'm alone in this.",
    "My sponsor doesn't get it either. I stopped calling.",
    "Tonight is going to be hard. I'm all alone and stressed about money.",
]

translator.start_session()

for i, statement in enumerate(concerning_statements, 1):
    print(f"\n--- Statement {i} ---")
    print(f'"{statement}"')
    
    visibility = translator.add_to_session(statement)
    
    print(f"\nΨ={visibility.psi:.2f}  ρ={visibility.rho:.2f}  "
          f"q={visibility.q:.2f}  f={visibility.f:.2f}  τ={visibility.tau:.2f}")
    print(f"Authenticity: {visibility.authenticity_score:.2f}")
    if visibility.isolation_markers:
        print(f"Isolation markers: {visibility.isolation_markers}")
    if visibility.activation_triggers:
        print(f"Activation topics: {visibility.activation_triggers}")

print("\n" + "=" * 60)
print("SESSION TRAJECTORY ANALYSIS")
print("=" * 60)

trajectory = translator.get_session_trajectory()
print(f"\nf-dimension trend: {trajectory.f_trend.value}")
print(f"q-dimension trend: {trajectory.q_trend.value}")
print(f"Ψ-dimension trend: {trajectory.psi_trend.value}")

print(f"\nAll isolation markers: {trajectory.isolation_markers}")
print(f"All activation topics: {trajectory.activation_topics}")

if trajectory.intervention_recommended:
    print(f"\n⚠️  INTERVENTION PATTERN DETECTED")
    print(f"Reason: {trajectory.intervention_reason}")
    print(f"Estimated window: {trajectory.estimated_window}")

# Final clinical insight
print("\n" + "=" * 60)
print("CLINICAL INSIGHT - FINAL STATEMENT")
print("=" * 60)

final_visibility = translator.current_session[-1]
insight = translator.get_clinical_insight(final_visibility)
print(insight.get_narrative())
