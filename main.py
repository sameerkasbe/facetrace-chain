"""Main CLI entry point for FaceTrace Chain pipeline.

Supports:
- Live Reverse Image Search (PRIMARY: Google Lens API via SerpAPI)
- Public Web Search (SECONDARY: Wikimedia Commons API)
- Deep Face Recognition (ArcFace 512-d Primary, SFace 128-d Fallback)
- Biometric Face Quality Checks & Landmark Alignment
- Content Classification (Profiles, Posts/Images, Reels/Videos)
- Controlled Tampering Demonstration & On-Chain Re-Verification
- Standalone Evidence Package Verification
"""
import sys
import os
import json
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
from face.recognizer import get_face_recognizer, BaseFaceRecognizer
from face.quality import FaceQualityAnalyzer, FaceQualityReport
from face.matcher import FaceMatcher, PROBABILISTIC_DISCLAIMER
from search.search_provider import get_search_provider, SearchProviderError
from search.candidate_collector import CandidateCollector
from utils.classifier import ResultClassifier
from blockchain.client import BlockchainClient
from blockchain.verifier import BlockchainVerifier
from blockchain.deploy import deploy_contract
from utils.config import get_config
from utils.logger import PipelineProgress
from utils.hashing import hash_evidence_package, canonicalize_evidence_package


def verify_standalone_evidence_file(evidence_file_path: str) -> int:
    """Verifies a previously generated EvidencePackage JSON file against the blockchain."""
    path = Path(evidence_file_path)
    if not path.exists():
        print(f"[ERROR] Evidence file not found: {path}")
        return 1

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Unable to parse evidence JSON file: {e}")
        return 1

    client = BlockchainClient()
    verifier = BlockchainVerifier(client=client)

    print("\n" + "=" * 65)
    print("  STANDALONE BLOCKCHAIN EVIDENCE RE-VERIFICATION")
    print("=" * 65)
    print(f"Evidence File: {evidence_file_path}")
    print(f"Case ID:       {data.get('case_id', 'N/A')}")
    print(f"Timestamp:     {data.get('verification_timestamp', 'N/A')}")
    print(f"Matched URL:   {data.get('matched_content', {}).get('source_url', 'N/A')}")

    rec_hex, bytes32_hex = hash_evidence_package(data)
    print(f"Computed Hash: {rec_hex}")

    verif_res = verifier.verify_raw_evidence(data)

    print("-" * 65)
    if verif_res.status.value.startswith("VERIFIED"):
        print(f"STATUS:        🟢 {verif_res.status.value}")
        print("AUTHENTIC — DATA HAS NOT BEEN MODIFIED")
        print(f"On-Chain Record: Found (Block timestamp: {verif_res.on_chain_timestamp})")
        print(f"Submitter:       {verif_res.submitter}")
    elif "TAMPERING" in verif_res.status.value or not verif_res.hashes_match:
        print(f"STATUS:        🔴 {verif_res.status.value}")
        print(f"DETAILS:       {verif_res.details}")
    else:
        print(f"STATUS:        ⚠️ {verif_res.status.value}")
        print(f"DETAILS:       {verif_res.details}")
    print("=" * 65 + "\n")

    return 0 if verif_res.hashes_match and verif_res.on_chain_found else 1


def run_pipeline(
    image_path: str,
    threshold: Optional[float] = None,
    mode: Optional[str] = None,
    engine: Optional[str] = None,
    api_key: Optional[str] = None,
    top_k: int = 5,
    run_tamper_demo: bool = False,
    save_evidence_path: Optional[str] = None
) -> int:
    """Executes the complete FaceTrace Chain pipeline.
    
    Returns 0 on success, 1 on failure.
    """
    config = get_config()
    match_threshold = threshold if threshold is not None else config.face_match_high_threshold
    active_mode = (mode or config.search_mode or "live").lower().strip()
    active_engine = (engine or config.face_recognition_engine or "arcface").lower().strip()

    search_provider = get_search_provider(mode=active_mode, api_key=api_key)
    recognizer = get_face_recognizer(engine_name=active_engine)

    print("\n" + "=" * 65)
    print("  FaceTrace Chain — Face Discovery & Blockchain Pipeline")
    print("=" * 65)
    print(f"Target Image:        {image_path}")
    print(f"Recognition Engine:  {recognizer.name} ({recognizer.embedding_dim}-d)")
    print(f"Discovery Mode:      {search_provider.mode_name}")
    print(f"Match Threshold:     {match_threshold:.2f}")
    print(f"Blockchain RPC:      {config.blockchain_rpc_url}")
    print("=" * 65)

    # Read raw image bytes for hashing
    try:
        raw_image_bytes = Path(image_path).read_bytes()
    except Exception as e:
        PipelineProgress.error(f"Cannot read image file: {e}")
        return 1

    # STEP 1: Face Detection & Biometric Quality Check
    PipelineProgress.step(1, "Detecting face and evaluating image quality")
    try:
        detector = FaceDetector()
        face_result = detector.detect_primary_face(raw_image_bytes, enforce_single=True)
        print("Face detected")
        PipelineProgress.detail(f"Face confidence: {face_result.confidence_pct}%")
        PipelineProgress.detail(f"Bounding box (x, y, w, h): {face_result.bbox}")

        quality = FaceQualityAnalyzer.analyze(face_result.image_bgr, face_result.bbox)
        PipelineProgress.detail(f"Quality Assessment: {quality.overall_quality} ({quality.summary_text()})")
        if quality.warnings:
            for w in quality.warnings:
                PipelineProgress.warning(f"Quality notice: {w}")
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

    # STEP 2: Facial Embedding Generation & Landmark Alignment
    PipelineProgress.step(2, f"Generating {recognizer.embedding_dim}-d embedding ({recognizer.name})")
    try:
        aligned_face = recognizer.align_face(face_result)
        embedding = recognizer.generate_embedding(face_result)
        print("Complete")
        PipelineProgress.detail(f"{recognizer.embedding_dim}-dimensional normalized biometric vector generated")
        PipelineProgress.detail("L2 normalization verified (Euclidean norm: 1.000)")
    except Exception as e:
        PipelineProgress.error(f"Failed to generate facial embedding: {e}")
        PipelineProgress.status_banner("FAILED — EMBEDDING ERROR")
        return 1

    # STEP 3: Genuine Public Content Discovery / Search
    PipelineProgress.step(3, f"Executing {search_provider.mode_name}")
    try:
        matcher = FaceMatcher(match_threshold=match_threshold)
        collector = CandidateCollector(
            search_provider=search_provider,
            detector=detector,
            encoder=recognizer,
            matcher=matcher,
            top_k=top_k
        )
        candidates = collector.discover_candidates(raw_image_bytes, max_results=15)
        if not candidates:
            PipelineProgress.warning("No candidate web results found from search provider.")
            PipelineProgress.status_banner("COMPLETED — NO CANDIDATES FOUND")
            return 0

        for idx, cand in enumerate(candidates, start=1):
            PipelineProgress.detail(
                f"[{idx}] [{cand.platform}] [{cand.content_type.upper()}] {cand.title[:32]} ({cand.source_domain})"
            )
    except SearchProviderError as e:
        PipelineProgress.error(f"Search provider error: {e}")
        PipelineProgress.status_banner("FAILED — SEARCH PROVIDER ERROR")
        return 1
    except Exception as e:
        PipelineProgress.error(f"Candidate discovery failed: {e}")
        PipelineProgress.status_banner("FAILED — SEARCH ERROR")
        return 1

    # STEP 4: Two-Stage Candidate Verification
    PipelineProgress.step(4, f"Running two-stage candidate verification (Top-{top_k})")
    try:
        enriched_candidates, best_match = collector.verify_candidates(
            input_embedding=embedding,
            candidates=candidates,
            input_image_bytes=raw_image_bytes,
            verbose=False
        )

        categorized = CandidateCollector.separate_by_category(enriched_candidates)
        print("\nDiscovered Results Breakdown:")
        print(f"  • Profiles:        {len(categorized['profiles'])}")
        print(f"  • Posts & Images:  {len(categorized['posts_images'])}")
        print(f"  • Reels & Videos:  {len(categorized['reels_videos'])}")

        print("\nCandidate Verification Results:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Candidate':<24} {'Platform':<12} {'Type':<10} {'Sim':<8} {'Confidence':<12}")
        print("-" * 80)
        for i, cand in enumerate(enriched_candidates, start=1):
            name = cand.title[:22]
            sim_str = f"{int(round(cand.similarity_score * 100))}%"
            conf_str = f"{int(round(cand.overall_confidence * 100))}%"
            print(f"#{i:<5} {name:<24} {cand.platform:<12} {cand.content_type:<10} {sim_str:<8} {conf_str:<12} {cand.confidence_badge}")
        print("-" * 80)
        print(f"\n{PROBABILISTIC_DISCLAIMER}\n")

        if best_match:
            print(f"🏆 BEST VERIFIED MATCH: {best_match.similarity_score:.2f} ({best_match.similarity_status})")
            PipelineProgress.detail(f"Platform:      {best_match.platform}")
            PipelineProgress.detail(f"Content Type:  {best_match.content_type}")
            PipelineProgress.detail(f"Source URL:    {best_match.source_url}")
            PipelineProgress.detail(f"Evidence Hash: {best_match.content_hash}")

            if save_evidence_path and best_match.evidence_package:
                out_path = Path(save_evidence_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(best_match.evidence_package, indent=2), encoding="utf-8")
                PipelineProgress.detail(f"Evidence saved: {out_path.resolve()}")
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

    # STEP 6 (Optional): Re-Verification & Tampering Demonstration
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
        description="FaceTrace Chain — AI Face Discovery and Blockchain Verification Pipeline"
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
        choices=["live", "web"],
        help="Search mode: 'live' (Google Lens reverse search) or 'web' (Wikimedia dynamic public search)"
    )
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default=None,
        choices=["arcface", "sface"],
        help="Face recognition engine: 'arcface' (InsightFace 512-d Primary) or 'sface' (OpenCV SFace 128-d Fallback)"
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
        help="Similarity match threshold (e.g. 0.65 for ArcFace, 0.85 for SFace)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates for Stage 2 deep biometric verification (default: 5)"
    )
    parser.add_argument(
        "--tamper-demo",
        action="store_true",
        help="Run on-chain re-verification and controlled tampering demonstration"
    )
    parser.add_argument(
        "--save-evidence",
        type=str,
        default=None,
        help="Optional file path to save the generated Evidence Package JSON"
    )
    parser.add_argument(
        "--verify-file",
        type=str,
        default=None,
        help="Path to an existing Evidence Package JSON file to re-verify against blockchain"
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

    if args.verify_file:
        code = verify_standalone_evidence_file(args.verify_file)
        sys.exit(code)

    if not args.image:
        print("Usage: python main.py --image <path_to_image> [--mode live|web] [--engine arcface|sface] [--tamper-demo]")
        print("Examples:")
        print("  python main.py --image sample_data/sample_portrait_a.jpg --mode live --tamper-demo")
        print("  python main.py --image sample_data/sample_portrait_a.jpg --mode web")
        print("  python main.py --verify-file evidence.json")
        sys.exit(1)

    exit_code = run_pipeline(
        image_path=args.image,
        threshold=args.threshold,
        mode=args.mode,
        engine=args.engine,
        api_key=args.api_key,
        top_k=args.top_k,
        run_tamper_demo=args.tamper_demo,
        save_evidence_path=args.save_evidence
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
