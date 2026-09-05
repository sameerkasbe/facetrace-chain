"""Unit tests for face matching, ArcFace & SFace recognition abstraction, face quality checks, and candidate verification."""
import pytest
import numpy as np
import cv2
from pathlib import Path

from face.matcher import FaceMatcher, MatchClassification, MatchAssessment
from face.detector import FaceDetector, FaceDetectionError
from face.recognizer import BaseFaceRecognizer, InsightFaceRecognizer, SFaceRecognizer, get_face_recognizer
from face.encoder import FaceEncoder
from face.quality import FaceQualityAnalyzer, FaceQualityReport
from search.candidate_collector import CandidateCollector
from models.match_record import CandidateResult


def test_cosine_similarity_identical_vectors():
    matcher = FaceMatcher(match_threshold=0.65, possible_threshold=0.45)
    vec = np.random.randn(512)
    vec = vec / np.linalg.norm(vec)

    sim = matcher.compute_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5
    classification, score_str = matcher.classify(sim)
    assert classification == MatchClassification.HIGH_SIMILARITY
    assert score_str == "100%"


def test_cosine_similarity_orthogonal_vectors():
    matcher = FaceMatcher()
    v1 = np.zeros(512)
    v1[0] = 1.0
    v2 = np.zeros(512)
    v2[1] = 1.0

    sim = matcher.compute_similarity(v1, v2)
    assert abs(sim - 0.0) < 1e-5
    classification, score_str = matcher.classify(sim)
    assert classification == MatchClassification.NO_MATCH
    assert score_str == "0%"


def test_classification_thresholds_and_multi_factor_assessment():
    matcher = FaceMatcher(match_threshold=0.65, possible_threshold=0.45)

    # Above match threshold
    res_high = matcher.assess_match(similarity_score=0.85, quality_score=0.9, detection_confidence=0.98)
    assert res_high.classification == MatchClassification.HIGH_SIMILARITY
    assert "HIGH CONFIDENCE MATCH" in res_high.badge_label
    assert res_high.is_match is True

    # Possible match zone
    res_pos = matcher.assess_match(similarity_score=0.55, quality_score=0.8, detection_confidence=0.9)
    assert res_pos.classification == MatchClassification.POSSIBLE
    assert "POSSIBLE MATCH" in res_pos.badge_label
    assert res_pos.is_match is True

    # Below possible threshold
    res_no = matcher.assess_match(similarity_score=0.30, quality_score=0.8, detection_confidence=0.9)
    assert res_no.classification == MatchClassification.NO_MATCH
    assert "NO RELIABLE MATCH" in res_no.badge_label
    assert res_no.is_match is False


def test_no_face_detected_validation(tmp_path):
    detector = FaceDetector()
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


def test_face_quality_analyzer_sharp_vs_blurry():
    # 1. Clear sharp face
    sharp_img = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
    report_sharp = FaceQualityAnalyzer.analyze(sharp_img)
    assert report_sharp.resolution == (150, 150)
    assert report_sharp.resolution_status == "Good"

    # 2. Artificially blurred image
    blurry_img = cv2.GaussianBlur(sharp_img, (35, 35), 0)
    report_blurry = FaceQualityAnalyzer.analyze(blurry_img)
    assert report_blurry.blur_status in {"Blurry", "Moderate Focus"}
    assert len(report_blurry.warnings) > 0


def test_face_quality_analyzer_low_resolution():
    small_crop = np.random.randint(60, 180, (40, 40, 3), dtype=np.uint8)
    report = FaceQualityAnalyzer.analyze(small_crop)
    assert report.resolution_status == "Low Resolution"
    assert report.face_size_status == "Small Face"
    assert report.overall_quality in {"MEDIUM", "LOW"}


def test_sface_recognizer_fallback():
    sface = SFaceRecognizer()
    assert sface.embedding_dim == 128
    assert "SFace" in sface.name

    detector = FaceDetector()
    sample_path = "sample_data/sample_portrait_a.jpg"
    face_res = detector.detect_primary_face(sample_path, enforce_single=True)

    emb = sface.generate_embedding(face_res)
    assert emb.shape == (128,)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4


def test_insightface_arcface_primary():
    arcface = InsightFaceRecognizer()
    assert arcface.embedding_dim == 512
    assert "ArcFace" in arcface.name

    detector = FaceDetector()
    sample_path = "sample_data/sample_portrait_a.jpg"
    face_res = detector.detect_primary_face(sample_path, enforce_single=True)

    emb = arcface.generate_embedding(face_res)
    assert emb.shape == (512,)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4


def test_face_encoder_wrapper_delegation():
    encoder = FaceEncoder(engine_name="arcface")
    assert encoder.embedding_dim == 512

    encoder_sface = FaceEncoder(engine_name="sface")
    assert encoder_sface.embedding_dim == 128


def test_candidate_collector_category_separation():
    candidates = [
        CandidateResult(
            title="John Doe Instagram Profile",
            source_url="https://instagram.com/johndoe",
            image_url="https://example.com/p1.jpg",
            source_domain="instagram.com",
            content_type="profile",
            category="Profiles"
        ),
        CandidateResult(
            title="Tech Conference Article Post",
            source_url="https://example.com/posts/tech-conf",
            image_url="https://example.com/p2.jpg",
            source_domain="example.com",
            content_type="post",
            category="Posts & Images"
        ),
        CandidateResult(
            title="Viral TikTok Clip",
            source_url="https://tiktok.com/@user/video/123",
            image_url="https://example.com/p3.jpg",
            source_domain="tiktok.com",
            content_type="reel",
            category="Reels & Videos"
        )
    ]

    categorized = CandidateCollector.separate_by_category(candidates)
    assert len(categorized["profiles"]) == 1
    assert categorized["profiles"][0].title == "John Doe Instagram Profile"
    assert len(categorized["posts_images"]) == 1
    assert len(categorized["reels_videos"]) == 1
    assert categorized["reels_videos"][0].title == "Viral TikTok Clip"


def test_face_detector_infinite_or_nan_values_safe(monkeypatch):
    """Verifies that non-finite (inf/nan) outputs from neural detectors do not cause OverflowError."""
    detector = FaceDetector()
    test_img = np.zeros((400, 400, 3), dtype=np.uint8)

    class MockYN:
        def setInputSize(self, sz):
            pass

        def detect(self, img):
            # 15 elements: x, y, w, h, 10 landmark coordinates, score
            degenerate_faces = np.array([
                [float("inf"), 50.0, float("inf"), 100.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
                [10.0, 20.0, float("nan"), 40.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90],
                [50.0, 60.0, 80.0, 90.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.88]
            ], dtype=np.float32)
            return True, degenerate_faces

    monkeypatch.setattr(detector, "detector", MockYN())
    monkeypatch.setattr(detector, "use_yunet", True)

    results = detector.detect_all_faces(test_img)
    # The two invalid detections with inf and nan should be skipped, and only the valid one should remain
    assert len(results) == 1
    box, conf, _ = results[0]
    assert box == (50, 60, 80, 90)
    assert abs(conf - 0.88) < 1e-4

