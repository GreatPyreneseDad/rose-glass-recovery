"""
Rose Glass Recovery: Translation Framework for Addiction Counseling
===================================================================

Translates communication patterns into visible dimensions that validate
counselor intuition—without judgment, without measurement, with full 
respect for the dignity of every person in recovery.

Author: Christopher MacGregor bin Joseph
Version: 1.0
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Pattern-based risk indication (NOT diagnosis)"""
    STABLE = "stable"
    WATCH = "watch"
    CONCERN = "concern"
    URGENT = "urgent"


class TrendDirection(Enum):
    """Dimension trajectory"""
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class PatternVisibility:
    """
    What Rose Glass sees in a statement.
    These are patterns, not measurements.
    """
    psi: float  # Internal consistency (0-1)
    rho: float  # Wisdom integration (0-1)
    q: float    # Emotional activation (0-1)
    f: float    # Social belonging (0-1)
    
    # Extended dimensions
    tau: float = 0.5  # Temporal depth
    
    # Derived insights
    authenticity_score: float = 0.5
    isolation_markers: List[str] = field(default_factory=list)
    activation_triggers: List[str] = field(default_factory=list)
    
    @property
    def coherence(self) -> float:
        """Overall pattern coherence through recovery lens"""
        return (self.psi + self.rho * self.psi + 
                self._optimize_q(self.q) + self.f * self.psi) / 4
    
    def _optimize_q(self, q_raw: float) -> float:
        """Biological optimization prevents extreme readings"""
        km, ki = 0.3, 2.0
        if q_raw <= 0:
            return 0
        return q_raw / (km + q_raw + (q_raw ** 2 / ki))


@dataclass
class SessionTrajectory:
    """Tracks patterns across a session"""
    f_trend: TrendDirection
    q_trend: TrendDirection
    psi_trend: TrendDirection
    
    isolation_markers: List[str]
    activation_topics: List[str]
    
    estimated_window: Optional[str] = None  # "7-14 days", "immediate", etc.
    intervention_recommended: bool = False
    intervention_reason: Optional[str] = None


@dataclass 
class ClinicalInsight:
    """
    Translates patterns into language counselors can use.
    NOT a diagnosis. NOT a treatment recommendation.
    """
    summary: str
    dimension_notes: Dict[str, str]
    pattern_shifts: List[str]
    considerations: List[str]
    
    risk_level: RiskLevel
    confidence: float  # In translation, not prediction
    
    def get_narrative(self) -> str:
        """Human-readable clinical insight"""
        narrative = f"PATTERN TRANSLATION (confidence: {self.confidence:.0%})\n"
        narrative += "=" * 50 + "\n\n"
        narrative += f"{self.summary}\n\n"
        
        narrative += "DIMENSION NOTES:\n"
        for dim, note in self.dimension_notes.items():
            narrative += f"  • {dim}: {note}\n"
        
        if self.pattern_shifts:
            narrative += "\nPATTERN SHIFTS:\n"
            for shift in self.pattern_shifts:
                narrative += f"  ⚠️ {shift}\n"
        
        if self.considerations:
            narrative += "\nCONSIDERATIONS (not recommendations):\n"
            for c in self.considerations:
                narrative += f"  → {c}\n"
        
        narrative += f"\nRisk Pattern: {self.risk_level.value.upper()}"
        
        return narrative


class RecoveryTranslator:
    """
    Main translation engine for addiction recovery contexts.
    
    Validates counselor intuition by making patterns visible.
    Does NOT diagnose, judge, or recommend treatment.
    """
    
    # Isolation language markers
    ISOLATION_MARKERS = [
        r'\b(alone|lonely|isolated|nobody|no one|by myself)\b',
        r'\b(don\'t understand|doesn\'t get it|can\'t relate)\b',
        r'\b(on my own|all by myself|just me)\b',
        r'\b(no friends|lost .* friends|pushed .* away)\b',
    ]
    
    # Social connection markers
    CONNECTION_MARKERS = [
        r'\b(we|us|our|together)\b',
        r'\b(sponsor|group|meeting|fellowship)\b',
        r'\b(support|helped|supporting)\b',
        r'\b(family|friend|partner|community)\b',
    ]
    
    # Emotional activation triggers
    ACTIVATION_TRIGGERS = [
        (r'\b(work|job|boss|employment|fired|unemployed)\b', 'employment'),
        (r'\b(family|parent|mother|father|sibling|child)\b', 'family'),
        (r'\b(relationship|partner|spouse|divorce|breakup)\b', 'relationship'),
        (r'\b(money|debt|bills|rent|mortgage|financial)\b', 'financial'),
        (r'\b(health|sick|pain|doctor|hospital)\b', 'health'),
        (r'\b(trauma|abuse|assault|violence)\b', 'trauma'),
    ]
    
    # Consistency contradiction patterns
    CONTRADICTION_PATTERNS = [
        (r"i'm fine", r"but|however|except|although"),
        (r"under control", r"can't|struggling|hard"),
        (r"don't need", r"maybe|sometimes|probably"),
        (r"doing great", r"but|except|although|however"),
    ]
    
    # Temporal markers
    PAST_INTEGRATION = [
        r'\b(learned|realized|understood|remember when)\b',
        r'\b(used to|back then|in the past|years ago)\b',
        r'\b(pattern|cycle|always|every time)\b',
    ]
    
    PRESENT_CRISIS = [
        r'\b(right now|today|tonight|this moment)\b',
        r'\b(need|want|have to|must)\b',
        r'\b(can\'t wait|immediately|urgent)\b',
    ]
    
    def __init__(self, 
                 calibration: str = "trauma_informed",
                 enable_gradient_tracking: bool = True):
        """
        Initialize recovery translator.
        
        Args:
            calibration: Cultural/context calibration to use
            enable_gradient_tracking: Track patterns over time
        """
        self.calibration = calibration
        self.gradient_enabled = enable_gradient_tracking
        
        # Load calibration parameters
        self.params = self._load_calibration(calibration)
        
        # Session tracking
        self.current_session: Optional[List[PatternVisibility]] = None
        self.session_start: Optional[datetime] = None
    
    def _load_calibration(self, name: str) -> Dict:
        """Load calibration parameters"""
        calibrations = {
            "trauma_informed": {
                "km": 0.25,  # Lower saturation - expect activation
                "ki": 1.5,   # More inhibition - prevent overwhelming
                "f_weight": 1.2,  # Social connection matters more
                "psi_tolerance": 0.3,  # Accept some inconsistency
            },
            "autism_spectrum": {
                "km": 0.35,
                "ki": 2.5,
                "f_weight": 0.8,  # Don't over-weight social
                "psi_tolerance": 0.1,  # Value logical consistency
            },
            "adhd": {
                "km": 0.2,
                "ki": 1.2,
                "f_weight": 1.0,
                "psi_tolerance": 0.5,  # Accept topic shifts
            },
            "veteran": {
                "km": 0.3,
                "ki": 2.0,
                "f_weight": 1.1,
                "psi_tolerance": 0.2,  # Value directness
            }
        }
        return calibrations.get(name, calibrations["trauma_informed"])
    
    def analyze_statement(self, text: str) -> PatternVisibility:
        """
        Analyze a single statement for pattern visibility.
        
        Args:
            text: The statement to analyze
            
        Returns:
            PatternVisibility with dimension readings
        """
        text_lower = text.lower()
        
        # Calculate f-dimension (social belonging)
        isolation_count = sum(
            len(re.findall(pattern, text_lower)) 
            for pattern in self.ISOLATION_MARKERS
        )
        connection_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.CONNECTION_MARKERS
        )
        
        total_social = isolation_count + connection_count
        if total_social > 0:
            f = connection_count / total_social
        else:
            f = 0.5  # Neutral if no markers
        
        # Track isolation markers found
        isolation_markers = []
        for pattern in self.ISOLATION_MARKERS:
            matches = re.findall(pattern, text_lower)
            isolation_markers.extend(matches)
        
        # Calculate q-dimension (emotional activation)
        activation_triggers = []
        activation_score = 0
        for pattern, topic in self.ACTIVATION_TRIGGERS:
            if re.search(pattern, text_lower):
                activation_triggers.append(topic)
                activation_score += 0.15
        
        # Exclamation and intensity markers
        exclamations = len(re.findall(r'[!?]{1,}', text))
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        intensity_markers = exclamations * 0.1 + caps_words * 0.05
        
        q = min(0.3 + activation_score + intensity_markers, 1.0)
        
        # Calculate psi-dimension (internal consistency)
        contradiction_score = 0
        for positive, negative in self.CONTRADICTION_PATTERNS:
            if re.search(positive, text_lower) and re.search(negative, text_lower):
                contradiction_score += 0.2
        
        psi = max(0.8 - contradiction_score, 0.2)
        
        # Calculate rho-dimension (wisdom integration)
        past_refs = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.PAST_INTEGRATION
        )
        
        rho = min(0.3 + past_refs * 0.15, 1.0)
        
        # Calculate tau (temporal depth)
        present_refs = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.PRESENT_CRISIS
        )
        
        if past_refs > present_refs:
            tau = min(0.5 + past_refs * 0.1, 1.0)
        elif present_refs > past_refs:
            tau = max(0.5 - present_refs * 0.1, 0.1)
        else:
            tau = 0.5
        
        # Calculate authenticity
        # High psi + low q + denial language = possible performance
        denial_patterns = [
            r'\b(fine|okay|good|great|perfect)\b.*\b(really|totally|completely)\b',
            r'\bdon\'t need\b',
            r'\bunder control\b',
        ]
        denial_count = sum(
            1 for p in denial_patterns if re.search(p, text_lower)
        )
        
        if psi > 0.7 and q < 0.3 and denial_count > 0:
            authenticity = 0.4  # Possible performance
        elif psi < 0.5 and q > 0.6:
            authenticity = 0.8  # Struggling but authentic
        else:
            authenticity = 0.6
        
        return PatternVisibility(
            psi=psi,
            rho=rho,
            q=q,
            f=f,
            tau=tau,
            authenticity_score=authenticity,
            isolation_markers=isolation_markers,
            activation_triggers=activation_triggers
        )
    
    def get_clinical_insight(self, 
                            visibility: PatternVisibility,
                            context: Optional[str] = None) -> ClinicalInsight:
        """
        Generate clinical insight from pattern visibility.
        
        This is TRANSLATION, not diagnosis.
        """
        dimension_notes = {}
        pattern_shifts = []
        considerations = []
        
        # Analyze f-dimension
        if visibility.f < 0.3:
            dimension_notes["f (Social Belonging)"] = (
                f"Low ({visibility.f:.2f}) - Isolation language present: "
                f"{', '.join(visibility.isolation_markers[:3]) if visibility.isolation_markers else 'subtle markers'}"
            )
            pattern_shifts.append("Social connection may be weakening")
            considerations.append("Gentle inquiry about support system")
        elif visibility.f > 0.7:
            dimension_notes["f (Social Belonging)"] = (
                f"Strong ({visibility.f:.2f}) - Connection language active"
            )
        else:
            dimension_notes["f (Social Belonging)"] = f"Moderate ({visibility.f:.2f})"
        
        # Analyze q-dimension
        if visibility.q > 0.7:
            dimension_notes["q (Emotional Activation)"] = (
                f"High ({visibility.q:.2f}) - Topics: "
                f"{', '.join(visibility.activation_triggers) if visibility.activation_triggers else 'general intensity'}"
            )
            pattern_shifts.append("Emotional activation elevated")
            considerations.append("Create space for processing without solving")
        elif visibility.q < 0.2:
            dimension_notes["q (Emotional Activation)"] = (
                f"Low ({visibility.q:.2f}) - May indicate suppression or stability"
            )
        else:
            dimension_notes["q (Emotional Activation)"] = f"Moderate ({visibility.q:.2f})"
        
        # Analyze psi-dimension
        if visibility.psi < 0.5:
            dimension_notes["Ψ (Consistency)"] = (
                f"Low ({visibility.psi:.2f}) - Possible contradictions in narrative"
            )
            pattern_shifts.append("Narrative consistency fragmented")
            considerations.append("Explore without challenging - may indicate internal conflict")
        else:
            dimension_notes["Ψ (Consistency)"] = f"Stable ({visibility.psi:.2f})"
        
        # Analyze authenticity
        if visibility.authenticity_score < 0.5:
            dimension_notes["Authenticity Pattern"] = (
                f"Performance indicators ({visibility.authenticity_score:.2f}) - "
                "Client may be presenting expected narrative"
            )
            considerations.append("Create safety for authentic expression")
        
        # Analyze tau (temporal)
        if visibility.tau < 0.3:
            dimension_notes["τ (Temporal)"] = (
                f"Present-focused ({visibility.tau:.2f}) - Living in immediate crisis"
            )
            considerations.append("Grounding before exploration")
        elif visibility.tau > 0.7:
            dimension_notes["τ (Temporal)"] = (
                f"Integrated ({visibility.tau:.2f}) - Connecting to broader timeline"
            )
        
        # Determine risk pattern
        risk_score = 0
        if visibility.f < 0.3:
            risk_score += 2
        if visibility.q > 0.7:
            risk_score += 1
        if visibility.psi < 0.5:
            risk_score += 1
        if visibility.authenticity_score < 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            risk_level = RiskLevel.URGENT
        elif risk_score >= 3:
            risk_level = RiskLevel.CONCERN
        elif risk_score >= 2:
            risk_level = RiskLevel.WATCH
        else:
            risk_level = RiskLevel.STABLE
        
        # Generate summary
        if risk_level == RiskLevel.URGENT:
            summary = (
                "Multiple pattern indicators suggest client may be approaching crisis. "
                "Isolation language combined with narrative fragmentation and possible "
                "performance of recovery warrant close attention."
            )
        elif risk_level == RiskLevel.CONCERN:
            summary = (
                "Pattern shifts detected that may indicate increasing strain. "
                "Consider exploring underlying dynamics with care."
            )
        elif risk_level == RiskLevel.WATCH:
            summary = (
                "Some pattern variations noted. Continue monitoring for trends."
            )
        else:
            summary = (
                "Patterns appear stable. Continue supportive engagement."
            )
        
        return ClinicalInsight(
            summary=summary,
            dimension_notes=dimension_notes,
            pattern_shifts=pattern_shifts,
            considerations=considerations,
            risk_level=risk_level,
            confidence=0.7  # Always acknowledge translation uncertainty
        )
    
    def start_session(self) -> None:
        """Start tracking a new session"""
        self.current_session = []
        self.session_start = datetime.now()
    
    def add_to_session(self, text: str) -> PatternVisibility:
        """Add a statement to current session and analyze"""
        if self.current_session is None:
            self.start_session()
        
        visibility = self.analyze_statement(text)
        self.current_session.append(visibility)
        return visibility
    
    def get_session_trajectory(self) -> SessionTrajectory:
        """Analyze trajectory across current session"""
        if not self.current_session or len(self.current_session) < 2:
            return SessionTrajectory(
                f_trend=TrendDirection.STABLE,
                q_trend=TrendDirection.STABLE,
                psi_trend=TrendDirection.STABLE,
                isolation_markers=[],
                activation_topics=[],
            )
        
        # Calculate trends
        first_half = self.current_session[:len(self.current_session)//2]
        second_half = self.current_session[len(self.current_session)//2:]
        
        avg_f_first = sum(v.f for v in first_half) / len(first_half)
        avg_f_second = sum(v.f for v in second_half) / len(second_half)
        
        avg_q_first = sum(v.q for v in first_half) / len(first_half)
        avg_q_second = sum(v.q for v in second_half) / len(second_half)
        
        avg_psi_first = sum(v.psi for v in first_half) / len(first_half)
        avg_psi_second = sum(v.psi for v in second_half) / len(second_half)
        
        def get_trend(first: float, second: float, threshold: float = 0.1) -> TrendDirection:
            if second - first > threshold:
                return TrendDirection.RISING
            elif first - second > threshold:
                return TrendDirection.DECLINING
            return TrendDirection.STABLE
        
        f_trend = get_trend(avg_f_first, avg_f_second)
        q_trend = get_trend(avg_q_first, avg_q_second)
        psi_trend = get_trend(avg_psi_first, avg_psi_second)
        
        # Collect all markers
        all_isolation = []
        all_triggers = []
        for v in self.current_session:
            all_isolation.extend(v.isolation_markers)
            all_triggers.extend(v.activation_triggers)
        
        # Determine if intervention recommended
        intervention_recommended = False
        intervention_reason = None
        estimated_window = None
        
        if f_trend == TrendDirection.DECLINING and avg_f_second < 0.4:
            intervention_recommended = True
            intervention_reason = "f-dimension declining with isolation markers increasing"
            estimated_window = "7-14 days"
        
        if q_trend == TrendDirection.RISING and avg_q_second > 0.7:
            intervention_recommended = True
            intervention_reason = "q-dimension escalating"
            estimated_window = "immediate attention"
        
        return SessionTrajectory(
            f_trend=f_trend,
            q_trend=q_trend,
            psi_trend=psi_trend,
            isolation_markers=list(set(all_isolation)),
            activation_topics=list(set(all_triggers)),
            estimated_window=estimated_window,
            intervention_recommended=intervention_recommended,
            intervention_reason=intervention_reason
        )
    
    def end_session(self) -> Optional[SessionTrajectory]:
        """End current session and return final trajectory"""
        if self.current_session is None:
            return None
        
        trajectory = self.get_session_trajectory()
        self.current_session = None
        self.session_start = None
        return trajectory


def demo():
    """Demonstrate recovery translator"""
    translator = RecoveryTranslator(calibration="trauma_informed")
    
    print("=" * 60)
    print("ROSE GLASS RECOVERY - PATTERN TRANSLATION DEMO")
    print("=" * 60)
    
    # Example statements showing different patterns
    statements = [
        "I'm doing great, everything is totally under control.",
        "Work has been really stressful lately, but I'm handling it on my own.",
        "I don't really need to go to meetings anymore. Nobody there understands.",
        "I talked to my sponsor yesterday about how I've been feeling anxious.",
        "I can't stop thinking about how I used to handle stress. Tonight is going to be hard.",
    ]
    
    translator.start_session()
    
    for i, statement in enumerate(statements, 1):
        print(f"\n--- Statement {i} ---")
        print(f'"{statement}"')
        
        visibility = translator.add_to_session(statement)
        insight = translator.get_clinical_insight(visibility)
        
        print(f"\nΨ={visibility.psi:.2f}  ρ={visibility.rho:.2f}  "
              f"q={visibility.q:.2f}  f={visibility.f:.2f}  τ={visibility.tau:.2f}")
        print(f"Authenticity: {visibility.authenticity_score:.2f}")
        if visibility.isolation_markers:
            print(f"Isolation markers: {visibility.isolation_markers}")
        if visibility.activation_triggers:
            print(f"Activation topics: {visibility.activation_triggers}")
    
    print("\n" + "=" * 60)
    print("SESSION TRAJECTORY")
    print("=" * 60)
    
    trajectory = translator.get_session_trajectory()
    print(f"\nf-dimension trend: {trajectory.f_trend.value}")
    print(f"q-dimension trend: {trajectory.q_trend.value}")
    print(f"Ψ-dimension trend: {trajectory.psi_trend.value}")
    
    if trajectory.intervention_recommended:
        print(f"\n⚠️  INTERVENTION PATTERN DETECTED")
        print(f"Reason: {trajectory.intervention_reason}")
        print(f"Estimated window: {trajectory.estimated_window}")
    
    # Final insight on last statement
    print("\n" + "=" * 60)
    print("FINAL STATEMENT INSIGHT")
    print("=" * 60)
    
    final_visibility = translator.current_session[-1] if translator.current_session else None
    if final_visibility:
        final_insight = translator.get_clinical_insight(final_visibility)
        print(final_insight.get_narrative())


if __name__ == "__main__":
    demo()
