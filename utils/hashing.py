"""Deterministic cryptographic hashing and tamper-evident fingerprinting."""
import hashlib
import json
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Tuple, Dict, Any, Optional

def canonicalize_url(url: str) -> str:
    """Canonicalize a URL to ensure deterministic string representation.
    
    Normalizes scheme and netloc to lowercase, strips trailing slashes,
    sorts query parameters, and removes tracking query arguments (utm_*).
    """
    if not url:
        return ""
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    scheme = parsed.scheme.lower() or "https"
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
        
    # Filter out common tracking query parameters
    filtered_queries = []
    for k, v in sorted(parse_qsl(parsed.query)):
        if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}:
            filtered_queries.append((k, v))
            
    query = urlencode(filtered_queries)
    return urlunparse((scheme, netloc, path, "", query, ""))


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _clean_for_canonical_json(obj: Any, unstable_keys: Optional[set] = None) -> Any:
    """Recursively converts objects to JSON-serializable native types with deterministic floats."""
    if unstable_keys is None:
        unstable_keys = {"download_latency", "client_ip", "local_temp_path", "runtime_id"}

    if isinstance(obj, dict):
        return {
            str(k): _clean_for_canonical_json(v, unstable_keys)
            for k, v in sorted(obj.items())
            if k not in unstable_keys and v is not None
        }
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_canonical_json(item, unstable_keys) for item in obj]
    elif hasattr(obj, "ndim") and obj.ndim > 0 and hasattr(obj, "tolist"):
        return [_clean_for_canonical_json(item, unstable_keys) for item in obj.tolist()]
    elif hasattr(obj, "item"):  # numpy scalar (np.float32, np.int64, np.bool_, etc.)
        val = obj.item()
        if isinstance(val, float):
            return round(val, 6)
        return val
    elif isinstance(obj, float):
        return round(obj, 6)
    elif isinstance(obj, (bool, int, str)):
        return obj
    elif hasattr(obj, "isoformat"):  # datetime / date objects
        return obj.isoformat()
    elif obj is None:
        return None
    return str(obj)


def canonicalize_metadata(metadata: Optional[Dict[str, Any]] = None) -> str:
    """Serialize metadata dict into a deterministic JSON string with sorted keys."""
    if not metadata:
        return "{}"
    
    cleaned = _clean_for_canonical_json(metadata)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def create_content_fingerprint(
    source_url: str,
    content_identifier: str,
    image_bytes: bytes,
    metadata: Optional[Dict[str, Any]] = None,
    title: str = ""
) -> Tuple[str, str]:
    """Generates a deterministic SHA-256 fingerprint from canonical content components.
    
    Components:
      1. Canonical Source URL
      2. Content Identifier (e.g. candidate ID, image filename, or post ID)
      3. Title or headline description
      4. SHA-256 of the raw image bytes
      5. Canonical JSON representation of stable metadata
      
    Returns:
      (hex_hash, bytes32_hex): The 64-char hex digest and '0x'-prefixed 32-byte representation.
    """
    canon_url = canonicalize_url(source_url)
    canon_ident = str(content_identifier).strip()
    canon_title = str(title).strip()
    image_hash = compute_sha256_bytes(image_bytes)
    canon_meta = canonicalize_metadata(metadata)
    
    payload = f"{canon_url}\n{canon_ident}\n{canon_title}\n{image_hash}\n{canon_meta}"
    hex_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    bytes32_hex = "0x" + hex_digest
    return hex_digest, bytes32_hex


def verify_content_fingerprint(
    expected_hex_hash: str,
    source_url: str,
    content_identifier: str,
    image_bytes: bytes,
    metadata: Optional[Dict[str, Any]] = None,
    title: str = ""
) -> bool:
    """Verifies if the recalculated fingerprint matches the expected hash."""
    recalculated_hex, _ = create_content_fingerprint(
        source_url=source_url,
        content_identifier=content_identifier,
        image_bytes=image_bytes,
        metadata=metadata,
        title=title
    )
    return recalculated_hex.lower() == expected_hex_hash.lower()


def canonicalize_evidence_package(evidence_data: Any) -> str:
    """Serializes an EvidencePackage or dict into a canonical deterministic JSON string.
    
    Ensures sorted keys, consistent key/value separation with no extra whitespace,
    and exclusion of ephemeral/unstable runtime fields.
    """
    if hasattr(evidence_data, "to_dict"):
        d = evidence_data.to_dict()
    elif isinstance(evidence_data, dict):
        d = dict(evidence_data)
    else:
        raise TypeError(f"Expected EvidencePackage or dict, got {type(evidence_data)}")

    cleaned = _clean_for_canonical_json(d)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_evidence_package(evidence_data: Any) -> Tuple[str, str]:
    """Generates deterministic SHA-256 fingerprint for a Search Evidence Package.
    
    Returns:
        (hex_hash, bytes32_hex): The 64-char hex digest and '0x'-prefixed 32-byte representation.
    """
    canon_json = canonicalize_evidence_package(evidence_data)
    hex_digest = hashlib.sha256(canon_json.encode("utf-8")).hexdigest()
    bytes32_hex = "0x" + hex_digest
    return hex_digest, bytes32_hex


def verify_evidence_package(evidence_data: Any, expected_hex_hash: str) -> bool:
    """Verifies whether recalculated SHA-256 of the evidence package matches the expected hash."""
    recalculated_hex, _ = hash_evidence_package(evidence_data)
    clean_expected = expected_hex_hash.lower().replace("0x", "").strip()
    return recalculated_hex.lower() == clean_expected

