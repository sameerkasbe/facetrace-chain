"""Face detection, quality evaluation, encoding, and similarity matching module."""
from .detector import FaceDetector, FaceDetectionResult, FaceDetectionError
from .encoder import FaceEncoder
from .matcher import FaceMatcher, MatchClassification, MatchAssessment, PROBABILISTIC_DISCLAIMER
from .recognizer import BaseFaceRecognizer, InsightFaceRecognizer, SFaceRecognizer, get_face_recognizer
from .quality import FaceQualityAnalyzer, FaceQualityReport

__all__ = [
    "FaceDetector",
    "FaceDetectionResult",
    "FaceDetectionError",
    "FaceEncoder",
    "FaceMatcher",
    "MatchClassification",
    "MatchAssessment",
    "PROBABILISTIC_DISCLAIMER",
    "BaseFaceRecognizer",
    "InsightFaceRecognizer",
    "SFaceRecognizer",
    "get_face_recognizer",
    "FaceQualityAnalyzer",
    "FaceQualityReport",
]
