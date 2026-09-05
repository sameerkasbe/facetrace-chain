"""Data models for on-chain blockchain records and re-verification results."""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED ✓"
    TAMPERING_DETECTED = "TAMPERING DETECTED ✗"
    NOT_FOUND = "NOT FOUND ON-CHAIN"
    FAILED = "VERIFICATION FAILED"


@dataclass
class BlockchainStoreResult:
    """Result of anchoring a content fingerprint on the blockchain."""
    success: bool
    content_hash: str
    tx_hash: str
    block_number: int
    timestamp: int
    submitter: str
    source_reference: str
    gas_used: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationRecord:
    """Outcome of querying and re-verifying a content fingerprint on-chain."""
    status: VerificationStatus
    content_hash: str
    recalculated_hash: str
    hashes_match: bool
    on_chain_found: bool
    source_reference: Optional[str] = None
    on_chain_timestamp: Optional[int] = None
    submitter: Optional[str] = None
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    details: str = ""
    error_message: Optional[str] = None
    stored_similarity: float = 0.0

    @property
    def is_authentic(self) -> bool:
        return self.status == VerificationStatus.VERIFIED and self.hashes_match and self.on_chain_found

    @property
    def timestamp(self) -> Optional[int]:
        return self.on_chain_timestamp

    @property
    def stored_url(self) -> Optional[str]:
        return self.source_reference

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["is_authentic"] = self.is_authentic
        return d
