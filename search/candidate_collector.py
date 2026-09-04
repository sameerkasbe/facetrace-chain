"""Collects, downloads, and algorithmically verifies candidate face matches."""
from typing import List, Tuple, Optional, Union
from pathlib import Path
import requests
import numpy as np

from models.match_record import CandidateResult, MatchRecord
from face.detector import FaceDetector, FaceDetectionError
from face.encoder import FaceEncoder
from face.matcher import FaceMatcher, MatchClassification
from utils.hashing import create_content_fingerprint
from utils.config import get_config
from .search_provider import BaseSearchProvider, get_search_provider

class CandidateCollector:
    """Orchestrates candidate retrieval, image acquisition, and face verification."""

    def __init__(
        self,
        search_provider: Optional[BaseSearchProvider] = None,
        detector: Optional[FaceDetector] = None,
        encoder: Optional[FaceEncoder] = None,
        matcher: Optional[FaceMatcher] = None,
        threshold: Optional[float] = None
    ):
        config = get_config()
        self.search_provider = search_provider or get_search_provider()
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()
        thresh = threshold if threshold is not None else config.match_threshold
        self.matcher = matcher or FaceMatcher(match_threshold=thresh)

    def download_image(self, url: str) -> Optional[bytes]:
        """Safely downloads candidate image with timeout and error handling."""
        if url.startswith("file:///"):
            local_path = Path(url.replace("file:///", ""))
            if local_path.exists():
                return local_path.read_bytes()
            return None

        headers = {
            "User-Agent": "FaceTraceChain/1.0 (Public Domain Content Discovery & Verification Pipeline; contact: security@example.org)"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 500:
                return resp.content
        except Exception:
            pass
        return None

    def discover_candidates(
        self,
        input_image: Union[str, Path, bytes],
        max_results: int = 5
    ) -> List[CandidateResult]:
        """Queries the search provider for candidate results."""
        return self.search_provider.search_by_image(input_image, max_results=max_results)

    def verify_candidates(
        self,
        input_embedding: np.ndarray,
        candidates: List[CandidateResult],
        verbose: bool = True
    ) -> Tuple[List[CandidateResult], Optional[MatchRecord]]:
        """Processes each candidate image, extracts face embedding, and computes similarity."""
        enriched_candidates: List[CandidateResult] = []
        best_candidate: Optional[CandidateResult] = None
        best_score = -1.0

        print("[VERIFY] Downloading candidate images")
        print("[VERIFY] Comparing facial embeddings")

        for i, cand in enumerate(candidates, start=1):
            cand_bytes = self.download_image(cand.image_url)
            if cand_bytes is None:
                cand.similarity_status = "Download Failed"
                enriched_candidates.append(cand)
                if verbose:
                    print(f"Candidate {i} [{cand.title[:30]}]: Download failed")
                continue

            cand.downloaded_image_bytes = cand_bytes

            # Attempt face detection on candidate image
            try:
                cand_face = self.detector.detect_primary_face(cand_bytes, enforce_single=False)
                cand.face_detected = True
                cand.face_confidence = cand_face.confidence

                # Generate embedding
                cand_emb = self.encoder.generate_embedding(cand_face)

                # Compute cosine similarity against query input face
                sim = self.matcher.compute_similarity(input_embedding, cand_emb)
                classification, score_str = self.matcher.classify(sim)

                cand.similarity_score = sim
                cand.similarity_status = classification.value

                if verbose:
                    print(f"Candidate {i} ({cand.title[:25]}...)")
                    print(f"  Similarity: {score_str}")
                    print(f"  Status: {classification.value}")

                if sim > best_score:
                    best_score = sim
                    best_candidate = cand

            except FaceDetectionError:
                cand.similarity_status = "No Face Detected"
                if verbose:
                    print(f"Candidate {i}: No face detected in image")
            except Exception as e:
                cand.similarity_status = f"Error ({type(e).__name__})"
                if verbose:
                    print(f"Candidate {i}: Error processing image ({e})")

            enriched_candidates.append(cand)

        # Sort candidates descending by similarity score
        enriched_candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        # Form MatchRecord if best candidate meets or exceeds threshold
        match_record: Optional[MatchRecord] = None
        if best_candidate and best_score >= self.matcher.match_threshold:
            hex_hash, bytes32_hash = create_content_fingerprint(
                source_url=best_candidate.source_url,
                content_identifier=best_candidate.content_identifier,
                image_bytes=best_candidate.downloaded_image_bytes or b"",
                metadata=best_candidate.metadata,
                title=best_candidate.title
            )

            match_record = MatchRecord(
                source_url=best_candidate.source_url,
                source_title=best_candidate.title,
                image_url=best_candidate.image_url,
                similarity_score=best_score,
                similarity_status=best_candidate.similarity_status,
                discovery_timestamp=best_candidate.discovery_timestamp,
                content_hash=hex_hash,
                content_bytes32=bytes32_hash,
                content_identifier=best_candidate.content_identifier,
                source_domain=best_candidate.source_domain,
                metadata=best_candidate.metadata,
                image_bytes=best_candidate.downloaded_image_bytes
            )

        return enriched_candidates, match_record
