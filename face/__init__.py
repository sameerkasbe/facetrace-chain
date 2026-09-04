"""Face detection, encoding, and similarity matching module."""
from .detector import FaceDetector, FaceDetectionResult, FaceDetectionError
from .encoder import FaceEncoder
from .matcher import FaceMatcher, MatchClassification

__all__ = [
    "FaceDetector",
    "FaceDetectionResult",
    "FaceDetectionError",
    "FaceEncoder",
    "FaceMatcher",
    "MatchClassification",
]
