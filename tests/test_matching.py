"""Unit tests for face matching, cosine similarity, and face detection validation."""
import pytest
import numpy as np
import cv2
from face.matcher import FaceMatcher, MatchClassification
from face.detector import FaceDetector, FaceDetectionError

def test_cosine_similarity_identical_vectors():
    matcher = FaceMatcher(match_threshold=0.85, possible_threshold=0.65)
    vec = np.random.randn(128)
    vec = vec / np.linalg.norm(vec)

    sim = matcher.compute_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5
    classification, score_str = matcher.classify(sim)
    assert classification == MatchClassification.HIGH_SIMILARITY
    assert score_str == "100%"


def test_cosine_similarity_orthogonal_vectors():
    matcher = FaceMatcher()
    v1 = np.zeros(128)
    v1[0] = 1.0
    v2 = np.zeros(128)
    v2[1] = 1.0

    sim = matcher.compute_similarity(v1, v2)
    assert abs(sim - 0.0) < 1e-5
    classification, score_str = matcher.classify(sim)
    assert classification == MatchClassification.NO_MATCH
    assert score_str == "0%"


def test_classification_thresholds():
    matcher = FaceMatcher(match_threshold=0.85, possible_threshold=0.65)

    # Above match threshold
    assert matcher.classify(0.92)[0] == MatchClassification.HIGH_SIMILARITY
    assert matcher.classify(0.85)[0] == MatchClassification.HIGH_SIMILARITY

    # Possible match zone
    assert matcher.classify(0.84)[0] == MatchClassification.POSSIBLE
    assert matcher.classify(0.65)[0] == MatchClassification.POSSIBLE

    # Below possible threshold
    assert matcher.classify(0.64)[0] == MatchClassification.NO_MATCH
    assert matcher.classify(0.42)[0] == MatchClassification.NO_MATCH


def test_no_face_detected_validation(tmp_path):
    detector = FaceDetector()
    # Create blank black image without faces
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    blank_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank_path), blank)

    with pytest.raises(FaceDetectionError) as exc_info:
        detector.detect_primary_face(str(blank_path))
    assert "Unable to detect a face in this image." in str(exc_info.value)


def test_multi_face_detected_validation():
    detector = FaceDetector()
    multi_path = "sample_data/sample_multi_face.jpg"
    
    with pytest.raises(FaceDetectionError) as exc_info:
        detector.detect_primary_face(multi_path, enforce_single=True)
    assert "Multiple faces detected" in str(exc_info.value)
