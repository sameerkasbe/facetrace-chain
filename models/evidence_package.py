"""Structured Search Evidence Package data model with deterministic canonical serialization."""
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Tuple


@dataclass
class EvidencePackage:
    """Tamper-evident discovery evidence package containing verified match data.
    
    Fields follow the required specification:
      - case_id: Unique case identifier (FACETRACE-<unique-id>)
      - verification_timestamp: ISO-8601 UTC timestamp
      - input: Input face metadata and image hash
      - search: Discovery search engine metadata and result counts
      - matched_content: Discovered web/social content and source URLs
      - face_verification: Biometric similarity scoring and classification
    """
    case_id: str
    verification_timestamp: str
    input: Dict[str, Any]
    search: Dict[str, Any]
    matched_content: Dict[str, Any]
    face_verification: Dict[str, Any]

    @classmethod
    def create(
        cls,
        input_image_hash: str,
        face_detected: bool,
        search_provider: str,
        search_timestamp: str,
        result_count: int,
        matched_title: str,
        matched_source_url: str,
        matched_image_url: str,
        matched_source_domain: str,
        matched_platform: str,
        similarity_score: float,
        threshold: float,
        verification_result: str,
        case_id: str = "",
        extra_input_meta: Dict[str, Any] = None,
        extra_search_meta: Dict[str, Any] = None,
        extra_content_meta: Dict[str, Any] = None,
    ) -> "EvidencePackage":
        """Factory method to construct a canonical EvidencePackage."""
        if not case_id:
            uid = uuid.uuid4().hex[:12].upper()
            case_id = f"FACETRACE-{uid}"

        now_iso = datetime.now(timezone.utc).isoformat()

        input_data = {
            "image_hash": input_image_hash,
            "face_detected": bool(face_detected)
        }
        if extra_input_meta:
            input_data.update(extra_input_meta)

        search_data = {
            "provider": search_provider,
            "search_timestamp": search_timestamp,
            "result_count": int(result_count)
        }
        if extra_search_meta:
            search_data.update(extra_search_meta)

        matched_data = {
            "title": matched_title,
            "source_url": matched_source_url,
            "image_url": matched_image_url,
            "source_domain": matched_source_domain,
            "platform": matched_platform
        }
        if extra_content_meta:
            matched_data.update(extra_content_meta)

        face_data = {
            "similarity_score": round(float(similarity_score), 4),
            "threshold": round(float(threshold), 4),
            "result": verification_result
        }

        return cls(
            case_id=case_id,
            verification_timestamp=now_iso,
            input=input_data,
            search=search_data,
            matched_content=matched_data,
            face_verification=face_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Serializes the evidence package to a canonical, deterministic JSON string."""
        from utils.hashing import canonicalize_evidence_package
        return canonicalize_evidence_package(self)

    def compute_hash(self) -> Tuple[str, str]:
        """Computes deterministic SHA-256 hash of the canonical JSON representation.
        
        Returns:
            (hex_digest, bytes32_hex): 64-char hex string and '0x'-prefixed 32-byte representation.
        """
        from utils.hashing import hash_evidence_package
        return hash_evidence_package(self)
