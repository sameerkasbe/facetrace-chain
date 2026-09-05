"""Unit and integration tests for blockchain storage, re-verification, and tamper detection."""
import pytest
import hashlib
from datetime import datetime, timezone
from blockchain.client import BlockchainClient
from blockchain.verifier import BlockchainVerifier
from blockchain.deploy import deploy_contract
from models.match_record import MatchRecord
from models.evidence_package import EvidencePackage
from models.verification_record import VerificationStatus
from utils.hashing import create_content_fingerprint, hash_evidence_package

@pytest.fixture(scope="module")
def blockchain_setup():
    """Initializes blockchain client against Ganache or local test provider."""
    client = BlockchainClient()
    needs_deploy = client.contract is None
    if not needs_deploy:
        try:
            client.contract.functions.recordCount().call()
        except Exception:
            needs_deploy = True

    if needs_deploy:
        deploy_info = deploy_contract()
        client = BlockchainClient(contract_address=deploy_info["contract_address"])
    return client


def test_blockchain_connection(blockchain_setup):
    client = blockchain_setup
    assert client.w3.is_connected() is True
    assert client.contract is not None
    assert client.get_account().startswith("0x")


def test_blockchain_store_and_query(blockchain_setup):
    client = blockchain_setup
    dummy_payload = f"test_record_{datetime.now(timezone.utc).timestamp()}"
    content_hash = hashlib.sha256(dummy_payload.encode()).hexdigest()
    source_ref = "https://commons.wikimedia.org/wiki/File:Test_Portrait.jpg"

    # Store verification
    res = client.store_verification(content_hash, source_ref)
    assert res.success is True
    assert res.block_number > 0
    assert len(res.tx_hash) > 0

    # Query verification
    query_res = client.query_verification(content_hash)
    assert query_res is not None
    assert query_res["exists"] is True
    assert query_res["source_reference"] == source_ref
    assert query_res["timestamp"] > 0
    assert query_res["submitter"].lower() == client.get_account().lower()


def test_blockchain_re_verification_and_tamper_detection(blockchain_setup):
    client = blockchain_setup
    verifier = BlockchainVerifier(client=client)

    url = "https://example.org/verified_author.jpg"
    content_id = f"author_{datetime.now(timezone.utc).timestamp()}"
    img_bytes = b"author_sample_pixels_data"
    metadata = {"author": "Jane Doe", "organization": "Public Gallery"}

    title = "Jane Doe Portrait"
    hex_hash, bytes32_hash = create_content_fingerprint(url, content_id, img_bytes, metadata, title=title)

    # Store original record on blockchain
    store_res = client.store_verification(hex_hash, url)
    assert store_res.success is True

    record = MatchRecord(
        source_url=url,
        source_title="Jane Doe Portrait",
        image_url=url,
        similarity_score=0.94,
        similarity_status="High Similarity Match",
        discovery_timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash=hex_hash,
        content_bytes32=bytes32_hash,
        content_identifier=content_id,
        source_domain="example.org",
        metadata=metadata,
        image_bytes=img_bytes
    )

    # 1. Run re-verification on authentic record -> MUST BE VERIFIED
    verif_authentic = verifier.verify_record(record)
    assert verif_authentic.status == VerificationStatus.VERIFIED
    assert verif_authentic.hashes_match is True
    assert verif_authentic.on_chain_found is True

    # 2. Simulate tampering -> MUST BE TAMPERING DETECTED
    tampered_rec, verif_tampered = verifier.simulate_tampering(
        record,
        modified_field="source_title",
        tampered_value="Counterfeit Copy"
    )
    assert verif_tampered.status == VerificationStatus.TAMPERING_DETECTED
    assert verif_tampered.hashes_match is False

    # 3. Simulate tampering with altered metadata -> MUST BE TAMPERING DETECTED
    tampered_meta_rec, verif_tampered_meta = verifier.simulate_tampering(
        record,
        modified_field="metadata_author",
        tampered_value="Impostor Author"
    )
    assert verif_tampered_meta.status == VerificationStatus.TAMPERING_DETECTED
    assert verif_tampered_meta.hashes_match is False


def test_evidence_package_blockchain_verification(blockchain_setup):
    """Verifies that an EvidencePackage can be hashed, stored on-chain, and re-verified."""
    client = blockchain_setup
    verifier = BlockchainVerifier(client=client)

    evidence = EvidencePackage.create(
        input_image_hash="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        face_detected=True,
        search_provider="Google Lens via SerpAPI",
        search_timestamp=datetime.now(timezone.utc).isoformat(),
        result_count=8,
        matched_title="Verified Historical Portrait",
        matched_source_url="https://en.wikipedia.org/wiki/Historical_Figure",
        matched_image_url="https://upload.wikimedia.org/wikipedia/commons/test.jpg",
        matched_source_domain="en.wikipedia.org",
        matched_platform="Wikipedia",
        similarity_score=0.965,
        threshold=0.85,
        verification_result="MATCH"
    )
    hex_hash, bytes32_hash = evidence.compute_hash()

    # Anchor on blockchain
    store_res = client.store_verification(hex_hash, evidence.matched_content["source_url"])
    assert store_res.success is True

    record = MatchRecord(
        source_url=evidence.matched_content["source_url"],
        source_title=evidence.matched_content["title"],
        image_url=evidence.matched_content["image_url"],
        similarity_score=0.965,
        similarity_status="High Similarity Match",
        discovery_timestamp=evidence.verification_timestamp,
        content_hash=hex_hash,
        content_bytes32=bytes32_hash,
        content_identifier="wiki_test_1",
        source_domain="en.wikipedia.org",
        platform="Wikipedia",
        evidence_package=evidence.to_dict()
    )

    # 1. Authentic Re-verification
    verif = verifier.verify_record(record)
    assert verif.status == VerificationStatus.VERIFIED
    assert verif.hashes_match is True

    # 2. Raw evidence verification
    verif_raw = verifier.verify_raw_evidence(evidence)
    assert verif_raw.status == VerificationStatus.VERIFIED
    assert verif_raw.is_authentic is True
    assert verif_raw.stored_similarity == 0.965
    assert verif_raw.stored_url == evidence.matched_content["source_url"]

    # 3. Direct verify_evidence_package call
    verif_pkg = verifier.verify_evidence_package(evidence.to_dict())
    assert verif_pkg.is_authentic is True
    assert verif_pkg.hashes_match is True

    # 4. Tampering with platform field
    tampered_rec, verif_tamper = verifier.simulate_tampering(
        record,
        modified_field="platform",
        tampered_value="Counterfeit_Platform"
    )
    assert verif_tamper.status == VerificationStatus.TAMPERING_DETECTED
    assert verif_tamper.hashes_match is False
    assert verif_tamper.is_authentic is False


def test_record_face_verification_method(blockchain_setup):
    """Verifies that BlockchainClient.record_face_verification returns the expected receipt dictionary."""
    client = blockchain_setup
    dummy_hash = "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
    source_url = "https://example.org/verified_source.jpg"

    receipt = client.record_face_verification(
        content_hash=dummy_hash,
        source_url=source_url,
        similarity_score=0.92,
        metadata_json='{"provenance": "test"}'
    )
    assert isinstance(receipt, dict)
    assert receipt["success"] is True
    assert len(receipt["transaction_hash"]) > 0
    assert receipt["block_number"] > 0
    assert receipt["source_reference"] == source_url
    assert receipt["similarity_score"] == 0.92
