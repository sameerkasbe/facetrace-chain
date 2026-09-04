"""Face similarity comparison and probabilistic match classification."""
from enum import Enum
from typing import Tuple
import numpy as np

class MatchClassification(str, Enum):
    HIGH_SIMILARITY = "High Similarity Match"
    POSSIBLE = "Possible Match"
    NO_MATCH = "No Match"

PROBABILISTIC_DISCLAIMER = (
    "Algorithmic Verification Notice: Facial similarity scores are probabilistic metrics "
    "calculated from high-dimensional deep learning embeddings. They indicate algorithmic visual "
    "correlation rather than absolute biological certainty."
)

class FaceMatcher:
    """Computes cosine similarity between facial embeddings and classifies matches."""

    def __init__(self, match_threshold: float = 0.85, possible_threshold: float = 0.65):
        self.match_threshold = match_threshold
        self.possible_threshold = possible_threshold

    @staticmethod
    def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Calculates cosine similarity between two 1D normalized embeddings.
        
        Range: [-1.0, 1.0]. Typically for face verification: [0.0, 1.0].
        """
        a = embedding_a.flatten()
        b = embedding_b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        # Cosine similarity
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to realistic bounds
        return max(0.0, min(1.0, cos_sim))

    def classify(self, similarity_score: float) -> Tuple[MatchClassification, str]:
        """Classifies a similarity score based on configured thresholds."""
        score_pct = int(round(similarity_score * 100))
        if similarity_score >= self.match_threshold:
            return MatchClassification.HIGH_SIMILARITY, f"{score_pct}%"
        elif similarity_score >= self.possible_threshold:
            return MatchClassification.POSSIBLE, f"{score_pct}%"
        else:
            return MatchClassification.NO_MATCH, f"{score_pct}%"
