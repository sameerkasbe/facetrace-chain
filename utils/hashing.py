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


def canonicalize_metadata(metadata: Optional[Dict[str, Any]] = None) -> str:
    """Serialize metadata dict into a deterministic JSON string with sorted keys."""
    if not metadata:
        return "{}"
    
    # Exclude unstable runtime/ephemeral keys
    unstable_keys = {"download_latency", "client_ip", "local_temp_path", "runtime_id"}
    stable_dict = {
        str(k): v for k, v in metadata.items()
        if k not in unstable_keys and v is not None
    }
    return json.dumps(stable_dict, sort_keys=True, separators=(",", ":"))


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
