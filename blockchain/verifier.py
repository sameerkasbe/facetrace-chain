"""Cryptographic re-verification engine and tampering demonstration."""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import copy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.match_record import MatchRecord
from models.verification_record import VerificationRecord, VerificationStatus
from utils.hashing import create_content_fingerprint
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
        """Recalculates the SHA-256 fingerprint of the record and validates it on-chain."""
        # Step 1: Recalculate SHA-256
        recalculated_hex, _ = create_content_fingerprint(
            source_url=record.source_url,
            content_identifier=record.content_identifier,
            image_bytes=record.image_bytes or b"",
            metadata=record.metadata,
            title=record.source_title
        )

        expected = (expected_original_hash or record.content_hash).lower().replace("0x", "")
        recalculated_clean = recalculated_hex.lower()
        hashes_match = (recalculated_clean == expected)

        # Step 2: Query Blockchain
        # Query with the recalculated hash
        on_chain_data = self.client.query_verification(recalculated_clean)

        if not hashes_match:
            # The content fingerprint changed!
            return VerificationRecord(
                status=VerificationStatus.TAMPERING_DETECTED,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=False,
                on_chain_found=on_chain_data is not None,
                source_reference=record.source_url,
                details=(
                    "TAMPERING DETECTED: The recalculated content fingerprint does not match "
                    "the anchored fingerprint. Content, metadata, or source URL was modified."
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
                details="VERIFIED ✓: Content fingerprint perfectly matches the immutable on-chain record."
            )
        else:
            return VerificationRecord(
                status=VerificationStatus.NOT_FOUND,
                content_hash="0x" + expected,
                recalculated_hash="0x" + recalculated_clean,
                hashes_match=True,
                on_chain_found=False,
                source_reference=record.source_url,
                details="NOT FOUND: Content hash matches local fingerprint, but no record was found on-chain."
            )

    def simulate_tampering(
        self,
        original_record: MatchRecord,
        modified_field: str = "metadata_author",
        tampered_value: str = "Unauthorized Alteration [TAMPERED]"
    ) -> Tuple[MatchRecord, VerificationRecord]:
        """Creates a tampered version of a MatchRecord and demonstrates detection."""
        tampered_record = copy.deepcopy(original_record)
        
        if modified_field == "source_url":
            tampered_record.source_url = "https://malicious-counterfeit-mirror.com/fake"
        elif modified_field == "source_title":
            tampered_record.source_title = f"{original_record.source_title} - {tampered_value}"
        elif modified_field == "image_bytes":
            # Flip some bytes in the image
            if tampered_record.image_bytes:
                b = bytearray(tampered_record.image_bytes)
                b[0] = (b[0] + 1) % 256
                tampered_record.image_bytes = bytes(b)
        else:
            # Metadata modification
            tampered_record.metadata = copy.deepcopy(original_record.metadata)
            tampered_record.metadata["tamper_test"] = tampered_value

        # Re-verify the tampered record against the original expected hash
        verif_result = self.verify_record(
            tampered_record,
            expected_original_hash=original_record.content_hash
        )

        return tampered_record, verif_result
