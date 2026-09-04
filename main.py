"""Main CLI entry point for FaceTrace Chain pipeline.

Supports:
- Live Reverse Image Search (Google Lens API via SerpAPI)
- Offline Demo Mode (Local Corpus Simulation for offline evaluation)
- Controlled Tampering Demonstration & On-Chain Re-Verification
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from face.detector import FaceDetector, FaceDetectionError
from face.encoder import FaceEncoder
from face.matcher import FaceMatcher, PROBABILISTIC_DISCLAIMER
from search.search_provider import get_search_provider, SearchProviderError
from search.candidate_collector import CandidateCollector
from blockchain.client import BlockchainClient
from blockchain.verifier import BlockchainVerifier
from blockchain.deploy import deploy_contract
from utils.config import get_config
from utils.logger import PipelineProgress


def run_pipeline(
    image_path: str,
    threshold: Optional[float] = None,
    mode: Optional[str] = None,
    api_key: Optional[str] = None,
    run_tamper_demo: bool = False
) -> int:
    """Executes the complete FaceTrace Chain pipeline.
    
    Returns 0 on success, 1 on failure.
    """
    config = get_config()
    match_threshold = threshold if threshold is not None else config.match_threshold
    active_mode = (mode or config.search_mode or "live").lower().strip()

    search_provider = get_search_provider(mode=active_mode, api_key=api_key)

    print("\n" + "=" * 65)
    print("  FaceTrace Chain — Face Discovery & Blockchain Pipeline")
    print("=" * 65)
    print(f"Target Image:      {image_path}")
    print(f"Discovery Mode:    {search_provider.mode_name}")
    print(f"Match Threshold:   {match_threshold:.2f}")
    print(f"Blockchain RPC:    {config.blockchain_rpc_url}")
    print("=" * 65)

    # STEP 1: Face Detection
    PipelineProgress.step(1, "Detecting face")
    try:
        detector = FaceDetector()
        face_result = detector.detect_primary_face(image_path, enforce_single=True)
        print("Face detected")
        PipelineProgress.detail(f"Face confidence: {face_result.confidence_pct}%")
        PipelineProgress.detail(f"Bounding box (x, y, w, h): {face_result.bbox}")
    except FaceDetectionError as e:
        PipelineProgress.error(str(e))
        PipelineProgress.status_banner("FAILED — FACE DETECTION ERROR")
        return 1
    except (FileNotFoundError, ValueError) as e:
        PipelineProgress.error(str(e))
        PipelineProgress.status_banner("FAILED — INVALID IMAGE INPUT")
        return 1
    except Exception as e:
        PipelineProgress.error(f"Unexpected image processing error: {e}")
        PipelineProgress.status_banner("FAILED")
        return 1

    # STEP 2: Facial Embedding Generation
    PipelineProgress.step(2, "Generating embedding")
    try:
        encoder = FaceEncoder()
        embedding = encoder.generate_embedding(face_result)
        print("Complete")
        PipelineProgress.detail("128-dimensional dense feature vector generated")
        PipelineProgress.detail("L2 normalization applied (Euclidean norm: 1.000)")
    except Exception as e:
        PipelineProgress.error(f"Failed to generate facial embedding: {e}")
        PipelineProgress.status_banner("FAILED — EMBEDDING ERROR")
        return 1

    # STEP 3: Genuine Public Content Discovery / Search
    PipelineProgress.step(3, f"Executing {search_provider.mode_name}")
    try:
        collector = CandidateCollector(
            search_provider=search_provider,
            detector=detector,
            encoder=encoder,
            threshold=match_threshold
        )
        candidates = collector.discover_candidates(image_path, max_results=5)
        if not candidates:
            PipelineProgress.warning("No candidate web results found from search provider.")
            PipelineProgress.status_banner("COMPLETED — NO CANDIDATES FOUND")
            return 0

        for idx, cand in enumerate(candidates, start=1):
            live_tag = "[LIVE]" if getattr(cand, "is_live", True) else "[OFFLINE DEMO]"
            PipelineProgress.detail(f"[{idx}] {live_tag} {cand.title[:45]} ({cand.source_domain})")
    except SearchProviderError as e:
        PipelineProgress.error(f"Search provider error: {e}")
        PipelineProgress.status_banner("FAILED — SEARCH PROVIDER ERROR")
        return 1
    except Exception as e:
        PipelineProgress.error(f"Candidate discovery failed: {e}")
        PipelineProgress.status_banner("FAILED — SEARCH ERROR")
        return 1

    # STEP 4: Candidate Face Verification
    PipelineProgress.step(4, "Verifying candidates")
    try:
        enriched_candidates, best_match = collector.verify_candidates(
            input_embedding=embedding,
            candidates=candidates,
            verbose=False
        )

        print("\nCandidate Verification Results:")
        print("-" * 68)
        print(f"{'Candidate':<30} {'Similarity':<12} {'Status'}")
        print("-" * 68)
        for i, cand in enumerate(enriched_candidates, start=1):
            name = cand.title[:28]
            sim_str = f"{int(round(cand.similarity_score * 100))}%"
            print(f"Candidate {i:<19} {sim_str:<12} {cand.similarity_status}")
        print("-" * 68)
        print(f"\n{PROBABILISTIC_DISCLAIMER}\n")

        if best_match:
            print(f"Best similarity: {best_match.similarity_score:.2f} ({best_match.similarity_status})")
            PipelineProgress.detail(f"Matched Source: {best_match.source_url}")
            PipelineProgress.detail(f"Content Fingerprint (SHA-256): {best_match.content_hash}")
        else:
            highest_score = enriched_candidates[0].similarity_score if enriched_candidates else 0.0
            print(f"Best similarity: {highest_score:.2f}")
            PipelineProgress.warning(
                f"No candidate met the match threshold of {match_threshold:.2f}."
            )
            PipelineProgress.status_banner("COMPLETED — NO MATCH ABOVE THRESHOLD")
            return 0
    except Exception as e:
        PipelineProgress.error(f"Candidate verification error: {e}")
        PipelineProgress.status_banner("FAILED — VERIFICATION ERROR")
        return 1

    # STEP 5: Blockchain Recording
    PipelineProgress.step(5, "Recording fingerprint on blockchain")
    try:
        client = BlockchainClient()
        if client.contract is None:
            print("Deploying smart contract to blockchain network...")
            deploy_contract()
            client = BlockchainClient()

        store_res = client.store_verification(
            content_hash_hex=best_match.content_hash,
            source_reference=best_match.source_url
        )

        if not store_res.success:
            PipelineProgress.error(f"Transaction rejected: {store_res.error_message}")
            PipelineProgress.status_banner("FAILED — BLOCKCHAIN REJECTION")
            return 1

        print("Transaction confirmed")
        PipelineProgress.detail(f"Contract Address: {client.contract_address}")
        PipelineProgress.detail(f"Transaction Hash: {store_res.tx_hash}")
        PipelineProgress.detail(f"Block Number:     {store_res.block_number}")
        PipelineProgress.detail(f"Block Timestamp:  {store_res.timestamp}")
        PipelineProgress.detail(f"Submitter:        {store_res.submitter}")
    except Exception as e:
        PipelineProgress.error(f"Blockchain recording failed: {e}")
        PipelineProgress.status_banner("FAILED — BLOCKCHAIN ERROR")
        return 1

    PipelineProgress.status_banner("SUCCESS")

    # STEP 6 (Optional / Demo): Re-Verification & Tampering Demonstration
    if run_tamper_demo and best_match:
        print("\n" + "=" * 65)
        print("  CONTROLLED TAMPERING & RE-VERIFICATION DEMONSTRATION")
        print("=" * 65)
        verifier = BlockchainVerifier(client=client)

        print("\n[Demo 1/2] Re-verifying Authentic On-Chain Record...")
        verif_clean = verifier.verify_record(best_match)
        print(f"On-chain Record: Found")
        print(f"Hash Match:      {'Yes' if verif_clean.hashes_match else 'No'}")
        print(f"Transaction:     {store_res.tx_hash[:20]}...")
        print(f"Block:           {store_res.block_number}")
        print(f"STATUS:          {verif_clean.status.value}")

        print("\n[Demo 2/2] Simulating Metadata / Title Modification (Tampering)...")
        tampered_rec, verif_tampered = verifier.simulate_tampering(
            original_record=best_match,
            modified_field="source_title",
            tampered_value="UNAUTHORIZED_FORGERY_ATTEMPT"
        )
        print(f"Original Hash A:   {best_match.content_hash}")
        print(f"Tampered Hash B:   {verif_tampered.recalculated_hash.replace('0x', '')}")
        print(f"Hash Comparison:   Hash A != Hash B")
        print(f"STATUS:            {verif_tampered.status.value}")
        print(f"Explanation:       {verif_tampered.details}")
        print("=" * 65 + "\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FaceTrace Chain — Face Discovery and Blockchain Verification Pipeline"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to the input face image (e.g. sample_data/sample_portrait_a.jpg)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="live",
        choices=["live", "offline"],
        help="Search mode: 'live' (Google Lens reverse search) or 'offline' (local demo corpus)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Legacy alias for --mode ('serpapi' -> 'live', 'local' -> 'offline')"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="SerpAPI API key for Google Lens live reverse image search"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Similarity match threshold (default: 0.85)"
    )
    parser.add_argument(
        "--tamper-demo",
        action="store_true",
        help="Run on-chain re-verification and controlled tampering demonstration"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the FaceVerification smart contract to blockchain network"
    )

    args = parser.parse_args()

    if args.deploy:
        deploy_contract()
        return

    if not args.image:
        print("Usage: python main.py --image <path_to_image> [--mode live|offline] [--tamper-demo]")
        print("Examples:")
        print("  python main.py --image sample_data/sample_portrait_a.jpg --mode offline --tamper-demo")
        print("  python main.py --image sample_data/sample_portrait_a.jpg --mode live --api-key <SERPAPI_KEY>")
        sys.exit(1)

    # Handle provider alias if given
    mode = args.mode
    if args.provider:
        if args.provider.lower() in {"local", "offline"}:
            mode = "offline"
        else:
            mode = "live"

    exit_code = run_pipeline(
        image_path=args.image,
        threshold=args.threshold,
        mode=mode,
        api_key=args.api_key,
        run_tamper_demo=args.tamper_demo
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
