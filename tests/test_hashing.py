"""Unit tests for deterministic SHA-256 fingerprinting and tamper detection."""
import pytest
from utils.hashing import (
    canonicalize_url,
    canonicalize_metadata,
    create_content_fingerprint,
    verify_content_fingerprint,
)

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
