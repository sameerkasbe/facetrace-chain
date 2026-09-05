"""Data models for candidate discovery, classification, and verified match records."""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from datetime import datetime, timezone


@dataclass
class CandidateResult:
    """Discovered candidate result from the live reverse search provider."""
    title: str
    source_url: str
    image_url: str
    source_domain: str
    platform: str = "Web"
    snippet: str = ""
    author: str = ""
    publication_date: str = ""
    search_provider: str = ""
    discovery_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_identifier: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Classification enrichment
    content_type: str = "webpage"      # profile, post, image, reel, video, webpage
    category: str = "Posts & Images"   # Profiles, Posts & Images, Reels & Videos

    # Biometric Matching enrichment
    similarity_score: float = 0.0
    overall_confidence: float = 0.0
    similarity_status: str = "Pending"
    confidence_badge: str = ""
    quality_label: str = "Unknown"
    match_explanation: str = ""
    face_detected: bool = False
    face_confidence: float = 0.0
    is_verified_match: bool = False
    is_live: bool = True
    downloaded_image_bytes: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("downloaded_image_bytes", None)
        return d


@dataclass
class MatchRecord:
    """Selected match record verified and ready for blockchain anchoring."""
    source_url: str
    source_title: str
    image_url: str
    similarity_score: float
    similarity_status: str
    discovery_timestamp: str
    content_hash: str                  # 64-character hex string
    content_bytes32: str               # 0x-prefixed 32-byte hex for Solidity
    content_identifier: str
    source_domain: str
    platform: str = "Web"
    content_type: str = "post"
    category: str = "Posts & Images"
    overall_confidence: float = 0.0
    confidence_badge: str = ""
    quality_label: str = ""
    match_explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_bytes: Optional[bytes] = None
    evidence_package: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("image_bytes", None)
        return d
