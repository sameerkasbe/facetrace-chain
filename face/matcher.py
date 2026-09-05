"""Face similarity comparison, transparent multi-factor confidence scoring, and probabilistic match classification."""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import numpy as np

from utils.config import get_config


class MatchClassification(str, Enum):
    HIGH_SIMILARITY = "High-Confidence Face Match"
    POSSIBLE = "Possible Visual Match"
    NO_MATCH = "No Reliable Match"


PROBABILISTIC_DISCLAIMER = (
    "Algorithmic Verification Notice: Facial similarity scores are probabilistic metrics "
    "calculated from high-dimensional deep learning embeddings. They indicate algorithmic visual "
    "correlation rather than absolute biological certainty."
)


@dataclass
class MatchAssessment:
    """Detailed biometric assessment separating face similarity from overall verification confidence."""
    face_similarity: float           # 0.0 to 1.0 pure embedding cosine similarity
    overall_confidence: float        # 0.0 to 1.0 composite confidence (similarity + quality + detection)
    classification: MatchClassification
    badge_label: str                 # "🟢 HIGH CONFIDENCE MATCH", "🟡 POSSIBLE MATCH", etc.
    quality_label: str               # "High", "Medium", "Low"
    detection_confidence_pct: int
    similarity_pct: int
    confidence_pct: int
    explanation: str

    @property
    def is_match(self) -> bool:
        return self.classification in {MatchClassification.HIGH_SIMILARITY, MatchClassification.POSSIBLE}


class FaceMatcher:
    """Computes cosine similarity between facial embeddings and calculates calibrated confidence."""

    def __init__(
        self,
        match_threshold: Optional[float] = None,
        possible_threshold: Optional[float] = None
    ):
        config = get_config()
        self.match_threshold = (
            match_threshold
            if match_threshold is not None
            else getattr(config, "face_match_high_threshold", 0.65)
        )
        self.possible_threshold = (
            possible_threshold
            if possible_threshold is not None
            else getattr(config, "face_match_medium_threshold", 0.45)
        )
        if self.possible_threshold > self.match_threshold:
            self.possible_threshold = round(self.match_threshold * 0.75, 4)

    @staticmethod
    def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Calculates cosine similarity between two 1D normalized embeddings.
        
        Range: [0.0, 1.0].
        """
        a = embedding_a.flatten()
        b = embedding_b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cos_sim))

    def assess_match(
        self,
        similarity_score: float,
        quality_score: float = 1.0,
        detection_confidence: float = 0.95
    ) -> MatchAssessment:
        """Calculates transparent multi-factor confidence and produces an explainable assessment."""
        # Clean bounds
        sim = max(0.0, min(1.0, similarity_score))
        qual = max(0.0, min(1.0, quality_score))
        det = max(0.0, min(1.0, detection_confidence))

        # Scientific multi-factor confidence weighting:
        # 75% facial embedding similarity + 15% image quality + 10% detection confidence
        composite_conf = (0.75 * sim) + (0.15 * qual) + (0.10 * det)
        composite_conf = max(0.0, min(1.0, composite_conf))

        sim_pct = int(round(sim * 100))
        conf_pct = int(round(composite_conf * 100))
        det_pct = int(round(det * 100))

        if qual >= 0.8:
            qual_label = "High"
        elif qual >= 0.5:
            qual_label = "Medium"
        else:
            qual_label = "Low"

        # Classification based on primary embedding similarity against calibrated thresholds
        if sim >= self.match_threshold:
            classification = MatchClassification.HIGH_SIMILARITY
            badge = "🟢 HIGH CONFIDENCE MATCH"
            explanation = (
                f"High-confidence visual face match based on {sim_pct}% deep embedding similarity "
                f"after landmark alignment. Target image quality is {qual_label}."
            )
        elif sim >= self.possible_threshold:
            classification = MatchClassification.POSSIBLE
            badge = "🟡 POSSIBLE MATCH"
            explanation = (
                f"Possible visual correlation with {sim_pct}% embedding similarity. "
                f"Pose, illumination, or resolution differences may affect verification certainty."
            )
        else:
            classification = MatchClassification.NO_MATCH
            badge = "🔴 NO RELIABLE MATCH"
            explanation = (
                f"Facial similarity ({sim_pct}%) is below verification threshold ({int(round(self.possible_threshold*100))}%). "
                f"Not considered a reliable match."
            )

        return MatchAssessment(
            face_similarity=round(sim, 4),
            overall_confidence=round(composite_conf, 4),
            classification=classification,
            badge_label=badge,
            quality_label=qual_label,
            detection_confidence_pct=det_pct,
            similarity_pct=sim_pct,
            confidence_pct=conf_pct,
            explanation=explanation
        )

    def classify(self, similarity_score: float) -> Tuple[MatchClassification, str]:
        """Backward-compatible classification method."""
        assessment = self.assess_match(similarity_score)
        return assessment.classification, f"{assessment.similarity_pct}%"
