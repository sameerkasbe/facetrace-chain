"""Cryptographic re-verification engine and tampering demonstration."""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import copy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.match_record import MatchRecord
from models.verification_record import VerificationRecord, VerificationStatus
from models.evidence_package import EvidencePackage
from utils.hashing import (
    create_content_fingerprint,
    hash_evidence_package,
    canonicalize_evidence_package
)
from .client import BlockchainClient


class BlockchainVerifier:
    """Verifies content integrity against immutable on-chain records."""

    def __init__(self, client: Optional[BlockchainClient] = None):
        self.client = client or BlockchainClient()

    def verify_record(
        self,
        record: MatchRecord,
        expected_original_hash: Optional[str] = None
    ) -> VerificationRecord:
        """Recalculates the SHA-256 fingerprint of the record/evidence and validates it on-chain."""
        # Check if record contains structured EvidencePackage
        if getattr(record, "evidence_package", None):
            recalculated_hex, _ = hash_evidence_package(record.evidence_package)
            source_ref = record.source_url
        else:
            recalculated_hex, _ = create_content_fingerprint(
                source_url=record.source_url,
                content_identifier=record.content_identifier,
                image_bytes=record.image_bytes or b"",
                metadata=record.metadata,
                title=record.source_title
            )
            source_ref = record.source_url

        expected = (expected_original_hash or record.content_hash).lower().replace("0x", "").strip()
        recalculated_clean = recalculated_hex.lower().replace("0x", "").strip()
        hashes_match = (recalculated_clean == expected)

        # Query blockchain with original expected hash to locate on-chain record
        on_chain_data = self.client.query_verification(expected)

        if not hashes_match:
            return VerificationRecord(
                status=VerificationStatus.TAMPERING_DETECTED,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=False,
                on_chain_found=on_chain_data is not None,
                source_reference=source_ref,
                tx_hash=on_chain_data.get("tx_hash") if on_chain_data else None,
                block_number=None,
                details=(
                    "TAMPERING DETECTED: The recalculated evidence package fingerprint does not match "
                    "the anchored blockchain fingerprint. Content, metadata, or source URL was modified."
                )
            )

        if on_chain_data is not None and on_chain_data.get("exists"):
            return VerificationRecord(
                status=VerificationStatus.VERIFIED,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=True,
                on_chain_found=True,
                source_reference=on_chain_data.get("source_reference"),
                on_chain_timestamp=on_chain_data.get("timestamp"),
                submitter=on_chain_data.get("submitter"),
                details="AUTHENTIC — DATA HAS NOT BEEN MODIFIED. Content fingerprint matches immutable on-chain record."
            )
        else:
            return VerificationRecord(
                status=VerificationStatus.NOT_FOUND,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=True,
                on_chain_found=False,
                source_reference=source_ref,
                details="NOT FOUND: Local fingerprint recalculated successfully, but no matching record exists on-chain."
            )

    def verify_raw_evidence(
        self,
        evidence_data: Union[Dict[str, Any], EvidencePackage, str],
        expected_original_hash: Optional[str] = None
    ) -> VerificationRecord:
        """Verifies raw evidence JSON/dictionary directly against blockchain."""
        import json
        if isinstance(evidence_data, str):
            try:
                evidence_data = json.loads(evidence_data)
            except Exception:
                pass

        recalculated_hex, _ = hash_evidence_package(evidence_data)
        
        # Extract metadata helpers
        source_ref = None
        stored_sim = 0.0
        if isinstance(evidence_data, dict):
            source_ref = evidence_data.get("matched_content", {}).get("source_url")
            stored_sim = float(evidence_data.get("face_verification", {}).get("similarity_score", 0.0))
            if expected_original_hash is None:
                expected_original_hash = evidence_data.get("content_hash") or evidence_data.get("evidence_hash")
        elif hasattr(evidence_data, "matched_content"):
            source_ref = getattr(evidence_data, "matched_content", {}).get("source_url")
            stored_sim = float(getattr(evidence_data, "face_verification", {}).get("similarity_score", 0.0))

        expected = (expected_original_hash or recalculated_hex).lower().replace("0x", "").strip()
        recalculated_clean = recalculated_hex.lower().replace("0x", "").strip()
        hashes_match = (recalculated_clean == expected)

        on_chain_data = self.client.query_verification(expected)

        if not hashes_match:
            return VerificationRecord(
                status=VerificationStatus.TAMPERING_DETECTED,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=False,
                on_chain_found=on_chain_data is not None,
                source_reference=source_ref,
                stored_similarity=stored_sim,
                error_message="Hash mismatch: Recalculated evidence hash differs from expected original.",
                details="HASH MISMATCH — DATA HAS BEEN MODIFIED. Recalculated evidence hash differs from original."
            )

        if on_chain_data is not None and on_chain_data.get("exists"):
            return VerificationRecord(
                status=VerificationStatus.VERIFIED,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=True,
                on_chain_found=True,
                source_reference=on_chain_data.get("source_reference") or source_ref,
                on_chain_timestamp=on_chain_data.get("timestamp"),
                submitter=on_chain_data.get("submitter"),
                stored_similarity=stored_sim,
                details="AUTHENTIC — DATA HAS NOT BEEN MODIFIED."
            )
        else:
            return VerificationRecord(
                status=VerificationStatus.NOT_FOUND,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=True,
                on_chain_found=False,
                source_reference=source_ref,
                stored_similarity=stored_sim,
                error_message="Hash not found on blockchain network.",
                details="NOT FOUND: Evidence package hash is valid, but record not found on this blockchain network."
            )

    def verify_evidence_package(
        self,
        evidence_data: Union[Dict[str, Any], EvidencePackage, str],
        expected_original_hash: Optional[str] = None
    ) -> VerificationRecord:
        """Alias and first-class method for verifying evidence packages."""
        return self.verify_raw_evidence(evidence_data, expected_original_hash=expected_original_hash)

    def simulate_tampering(
        self,
        original_record: MatchRecord,
        modified_field: str = "source_title",
        tampered_value: str = "Unauthorized Alteration [TAMPERED]"
    ) -> Tuple[MatchRecord, VerificationRecord]:
        """Creates a tampered version of a MatchRecord/EvidencePackage and demonstrates detection."""
        tampered_record = copy.deepcopy(original_record)
        
        if getattr(tampered_record, "evidence_package", None):
            pkg = copy.deepcopy(tampered_record.evidence_package)
            if modified_field in {"source_title", "title"}:
                pkg["matched_content"]["title"] = f"{pkg['matched_content'].get('title', '')} - {tampered_value}"
            elif modified_field in {"source_url", "url"}:
                pkg["matched_content"]["source_url"] = "https://malicious-counterfeit-mirror.com/fake"
            elif modified_field in {"platform"}:
                pkg["matched_content"]["platform"] = tampered_value
            elif modified_field in {"similarity_score", "score"}:
                pkg["face_verification"]["similarity_score"] = 0.9999
            elif modified_field in {"input_image_hash"}:
                pkg["input"]["image_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
            else:
                pkg["matched_content"]["title"] = f"{pkg['matched_content'].get('title', '')} [MODIFIED]"

            tampered_record.evidence_package = pkg
            new_hash, new_bytes32 = hash_evidence_package(pkg)
            # Re-verify the tampered record against the original expected hash
            verif_result = self.verify_record(
                tampered_record,
                expected_original_hash=original_record.content_hash
            )
        else:
            if modified_field == "source_url":
                tampered_record.source_url = "https://malicious-counterfeit-mirror.com/fake"
            elif modified_field == "source_title":
                tampered_record.source_title = f"{original_record.source_title} - {tampered_value}"
            elif modified_field == "image_bytes":
                if tampered_record.image_bytes:
                    b = bytearray(tampered_record.image_bytes)
                    b[0] = (b[0] + 1) % 256
                    tampered_record.image_bytes = bytes(b)
            else:
                tampered_record.metadata = copy.deepcopy(original_record.metadata)
                tampered_record.metadata["tamper_test"] = tampered_value

            verif_result = self.verify_record(
                tampered_record,
                expected_original_hash=original_record.content_hash
            )

        return tampered_record, verif_result
