"""Unit tests for deterministic SHA-256 fingerprinting and tamper detection."""
import copy
import pytest
from utils.hashing import (
    canonicalize_url,
    canonicalize_metadata,
    create_content_fingerprint,
    verify_content_fingerprint,
    canonicalize_evidence_package,
    hash_evidence_package,
    verify_evidence_package,
)
from models.evidence_package import EvidencePackage


def test_canonicalize_url():
    url1 = "HTTPS://Example.COM/Path/To/Post/?utm_source=twitter&b=2&a=1"
    url2 = "https://example.com/Path/To/Post?a=1&b=2"
    assert canonicalize_url(url1) == canonicalize_url(url2)


def test_fingerprint_determinism():
    url = "https://commons.wikimedia.org/wiki/File:Portrait_1.jpg"
    content_id = "wiki_101"
    img_bytes = b"sample_raw_image_data_buffer_12345"
    metadata = {"author": "John Doe", "license": "CC-BY-SA", "year": 2024}

    hash1, bytes32_1 = create_content_fingerprint(url, content_id, img_bytes, metadata)
    hash2, bytes32_2 = create_content_fingerprint(url, content_id, img_bytes, metadata)

    assert hash1 == hash2
    assert len(hash1) == 64
    assert bytes32_1 == "0x" + hash1
    assert verify_content_fingerprint(hash1, url, content_id, img_bytes, metadata) is True


def test_tamper_detection_on_image_modification():
    url = "https://commons.wikimedia.org/wiki/File:Portrait_1.jpg"
    content_id = "wiki_101"
    img_bytes_orig = b"original_image_bytes"
    img_bytes_tampered = b"tampered_image_bytes"
    metadata = {"author": "Alice"}

    orig_hash, _ = create_content_fingerprint(url, content_id, img_bytes_orig, metadata)
    tampered_hash, _ = create_content_fingerprint(url, content_id, img_bytes_tampered, metadata)

    assert orig_hash != tampered_hash
    assert verify_content_fingerprint(orig_hash, url, content_id, img_bytes_tampered, metadata) is False


def test_tamper_detection_on_metadata_modification():
    url = "https://example.com/post/1"
    content_id = "post_1"
    img_bytes = b"image_content"
    meta_a = {"author": "Alice", "status": "verified"}
    meta_b = {"author": "Eve", "status": "verified"}

    hash_a, _ = create_content_fingerprint(url, content_id, img_bytes, meta_a)
    hash_b, _ = create_content_fingerprint(url, content_id, img_bytes, meta_b)

    assert hash_a != hash_b


def test_tamper_detection_on_url_modification():
    url_a = "https://legitimate-source.org/content/123"
    url_b = "https://impostor-source.org/content/123"
    content_id = "item_123"
    img_bytes = b"image_content"

    hash_a, _ = create_content_fingerprint(url_a, content_id, img_bytes)
    hash_b, _ = create_content_fingerprint(url_b, content_id, img_bytes)

    assert hash_a != hash_b


def test_evidence_package_canonicalization_and_hashing():
    pkg1 = EvidencePackage.create(
        input_image_hash="112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
        face_detected=True,
        search_provider="Google Lens via SerpAPI",
        search_timestamp="2026-09-04T12:00:00Z",
        result_count=5,
        matched_title="Official Biography Portrait",
        matched_source_url="https://en.wikipedia.org/wiki/Portrait",
        matched_image_url="https://upload.wikimedia.org/wikipedia/commons/portrait.jpg",
        matched_source_domain="en.wikipedia.org",
        matched_platform="Wikipedia",
        similarity_score=0.9123,
        threshold=0.85,
        verification_result="MATCH",
        case_id="FACETRACE-TEST001"
    )
    # Fix verification_timestamp for determinism test
    pkg1.verification_timestamp = "2026-09-04T12:05:00Z"

    pkg2 = copy.deepcopy(pkg1)

    hash1, bytes32_1 = hash_evidence_package(pkg1)
    hash2, bytes32_2 = hash_evidence_package(pkg2)

    assert hash1 == hash2
    assert len(hash1) == 64
    assert bytes32_1 == "0x" + hash1
    assert verify_evidence_package(pkg1, hash1) is True
    assert verify_evidence_package(pkg1, bytes32_1) is True


def test_evidence_package_tampering_fields():
    base_pkg = EvidencePackage.create(
        input_image_hash="112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
        face_detected=True,
        search_provider="Google Lens via SerpAPI",
        search_timestamp="2026-09-04T12:00:00Z",
        result_count=5,
        matched_title="Official Biography Portrait",
        matched_source_url="https://en.wikipedia.org/wiki/Portrait",
        matched_image_url="https://upload.wikimedia.org/wikipedia/commons/portrait.jpg",
        matched_source_domain="en.wikipedia.org",
        matched_platform="Wikipedia",
        similarity_score=0.9123,
        threshold=0.85,
        verification_result="MATCH",
        case_id="FACETRACE-TEST001"
    )
    orig_hash, _ = hash_evidence_package(base_pkg)

    # 1. Tamper similarity score
    p_sim = copy.deepcopy(base_pkg)
    p_sim.face_verification["similarity_score"] = 0.9999
    assert hash_evidence_package(p_sim)[0] != orig_hash

    # 2. Tamper platform
    p_plat = copy.deepcopy(base_pkg)
    p_plat.matched_content["platform"] = "Instagram"
    assert hash_evidence_package(p_plat)[0] != orig_hash

    # 3. Tamper source URL
    p_url = copy.deepcopy(base_pkg)
    p_url.matched_content["source_url"] = "https://forged-evidence.org"
    assert hash_evidence_package(p_url)[0] != orig_hash

    # 4. Tamper verification result
    p_res = copy.deepcopy(base_pkg)
    p_res.face_verification["result"] = "NO_MATCH"
    assert hash_evidence_package(p_res)[0] != orig_hash


def test_canonicalize_evidence_package_with_numpy_types():
    """Ensures numpy scalars (float32, int64, bool_), arrays, and nested None serialize deterministically."""
    import numpy as np

    raw_data = {
        "case_id": "FACETRACE-NUMPY-TEST",
        "timestamp": "2026-09-04T12:00:00Z",
        "scores": [np.float32(0.8500001), np.float64(0.9999999)],
        "indices": [np.int32(1), np.int64(42)],
        "flags": {"verified": np.bool_(True), "flag_none": None},
        "vector_sample": np.array([0.1, 0.2, 0.3], dtype=np.float32)
    }

    canon_str = canonicalize_evidence_package(raw_data)
    assert isinstance(canon_str, str)
    assert "flag_none" not in canon_str  # None is cleanly filtered
    assert "FACETRACE-NUMPY-TEST" in canon_str

    # Determinism test
    hash1, bytes32_1 = hash_evidence_package(raw_data)
    hash2, bytes32_2 = hash_evidence_package(raw_data)
    assert hash1 == hash2
    assert bytes32_1 == bytes32_2
