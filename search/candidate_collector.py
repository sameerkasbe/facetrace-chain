"""Collects, downloads, and algorithmically verifies candidate face matches using multi-stage verification."""
import time
import hashlib
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import requests
import numpy as np

from models.match_record import CandidateResult, MatchRecord
from models.evidence_package import EvidencePackage
from face.detector import FaceDetector, FaceDetectionError, FaceDetectionResult
from face.encoder import FaceEncoder
from face.matcher import FaceMatcher, MatchClassification, MatchAssessment
from face.quality import FaceQualityAnalyzer, FaceQualityReport
from utils.classifier import ResultClassifier, ContentType, ResultCategory
from utils.hashing import create_content_fingerprint, hash_evidence_package
from utils.config import get_config
from .search_provider import BaseSearchProvider, get_search_provider


class CandidateCollector:
    """Orchestrates candidate retrieval, image acquisition, and multi-stage face verification."""

    def __init__(
        self,
        search_provider: Optional[BaseSearchProvider] = None,
        detector: Optional[FaceDetector] = None,
        encoder: Optional[FaceEncoder] = None,
        matcher: Optional[FaceMatcher] = None,
        top_k: Optional[int] = None
    ):
        config = get_config()
        self.search_provider = search_provider or get_search_provider()
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()
        self.matcher = matcher or FaceMatcher()
        self.top_k = top_k or getattr(config, "top_k_candidates", 5)

    def download_image(self, url: str) -> Optional[bytes]:
        """Safely downloads candidate image with timeout and error handling."""
        if url.startswith("file:///"):
            local_path = Path(url.replace("file:///", ""))
            if local_path.exists():
                return local_path.read_bytes()
            return None

        headers = {
            "User-Agent": "FaceTraceChain/2.0 (Reverse Search & Verification Pipeline; contact: security@example.org)"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200 and len(resp.content) > 200:
                return resp.content
        except Exception:
            pass
        return None

    def discover_candidates(
        self,
        input_image: Union[str, Path, bytes],
        max_results: int = 15
    ) -> List[CandidateResult]:
        """Queries the search provider for dynamic candidate results."""
        return self.search_provider.search_by_image(input_image, max_results=max_results)

    def verify_candidates(
        self,
        input_embedding: np.ndarray,
        candidates: List[CandidateResult],
        input_image_bytes: Optional[bytes] = None,
        input_image_hash: str = "",
        input_face_detected: bool = True,
        verbose: bool = True
    ) -> Tuple[List[CandidateResult], Optional[MatchRecord]]:
        """Executes a two-stage verification pipeline:
        
        Stage 1: Fast embedding comparison on all candidates to select Top-K.
        Stage 2: In-depth verification on Top-K (face quality checks, alignment, transparent multi-factor scoring).
        """
        if not candidates:
            return [], None

        if not input_image_hash and input_image_bytes:
            input_image_hash = hashlib.sha256(input_image_bytes).hexdigest()

        if verbose:
            print(f"[VERIFY] Starting Stage 1: Fast screening on {len(candidates)} candidates...")

        # -------------------------------------------------------------
        # STAGE 1: Fast Screening across all candidates
        # -------------------------------------------------------------
        stage1_results: List[Tuple[CandidateResult, Optional[FaceDetectionResult]]] = []

        for i, cand in enumerate(candidates, start=1):
            # Classify content type if not already populated
            if not cand.content_type or cand.content_type == "webpage":
                c_type = ResultClassifier.classify(cand.source_url, cand.metadata)
                cand.content_type = c_type.value
                cand.category = ResultClassifier.get_category(c_type).value

            cand_bytes = self.download_image(cand.image_url)
            if cand_bytes is None:
                cand.similarity_status = "Download Failed"
                stage1_results.append((cand, None))
                if verbose:
                    print(f"Candidate {i} [{cand.title[:25]}]: Download failed")
                continue

            cand.downloaded_image_bytes = cand_bytes

            # Fast face detection
            try:
                cand_face = self.detector.detect_primary_face(cand_bytes, enforce_single=False)
                cand.face_detected = True
                cand.face_confidence = cand_face.confidence

                # Fast embedding
                fast_emb = self.encoder.generate_embedding(cand_face)
                prelim_sim = self.matcher.compute_similarity(input_embedding, fast_emb)
                cand.similarity_score = prelim_sim
                stage1_results.append((cand, cand_face))

                if verbose:
                    print(f"Candidate {i} [{cand.title[:20]}]: Prelim similarity = {int(round(prelim_sim * 100))}%")
            except FaceDetectionError:
                cand.similarity_status = "No Face Detected"
                stage1_results.append((cand, None))
                if verbose:
                    print(f"Candidate {i}: No face detected")
            except Exception as e:
                cand.similarity_status = f"Error ({type(e).__name__})"
                stage1_results.append((cand, None))

        # Sort by preliminary similarity score
        stage1_results.sort(key=lambda item: item[0].similarity_score, reverse=True)

        # -------------------------------------------------------------
        # STAGE 2: Deep High-Accuracy Verification on Top-K
        # -------------------------------------------------------------
        top_k_items = stage1_results[:self.top_k]
        remaining_items = stage1_results[self.top_k:]

        if verbose:
            print(f"[VERIFY] Starting Stage 2: Deep verification on Top-{len(top_k_items)} candidates...")

        for cand, face_res in top_k_items:
            if face_res is None or not cand.downloaded_image_bytes:
                assessment = self.matcher.assess_match(0.0, 0.1, 0.0)
                cand.overall_confidence = assessment.overall_confidence
                cand.confidence_badge = assessment.badge_label
                cand.match_explanation = assessment.explanation
                cand.is_verified_match = False
                continue

            # 1. Face Quality Assessment
            quality_report = FaceQualityAnalyzer.analyze(face_res.image_bgr, face_res.bbox)
            cand.quality_label = quality_report.overall_quality

            # 2. Careful Landmark Alignment & Refined Embedding
            aligned_face = self.encoder.align_face(face_res)
            refined_emb = self.encoder.generate_embedding(face_res)

            # 3. Precision Similarity & Multi-Factor Confidence Assessment
            sim = self.matcher.compute_similarity(input_embedding, refined_emb)
            assessment = self.matcher.assess_match(
                similarity_score=sim,
                quality_score=quality_report.quality_score,
                detection_confidence=face_res.confidence
            )

            cand.similarity_score = assessment.face_similarity
            cand.overall_confidence = assessment.overall_confidence
            cand.similarity_status = assessment.classification.value
            cand.confidence_badge = assessment.badge_label
            cand.match_explanation = assessment.explanation
            cand.is_verified_match = assessment.is_match

            if verbose:
                print(f"Top Candidate [{cand.title[:25]}]:")
                print(f"  Face Similarity: {assessment.similarity_pct}%")
                print(f"  Overall Confidence: {assessment.confidence_pct}% ({assessment.badge_label})")
                print(f"  Quality: {quality_report.overall_quality}")

        # Update remaining non-top-k items with basic assessment
        for cand, face_res in remaining_items:
            assessment = self.matcher.assess_match(
                similarity_score=cand.similarity_score,
                quality_score=0.7,
                detection_confidence=cand.face_confidence
            )
            cand.overall_confidence = assessment.overall_confidence
            cand.confidence_badge = assessment.badge_label
            cand.match_explanation = assessment.explanation
            cand.is_verified_match = assessment.is_match

        # Combine and re-rank
        enriched_candidates = [item[0] for item in top_k_items] + [item[0] for item in remaining_items]
        # Sort primary by verified status and overall confidence, then similarity
        enriched_candidates.sort(
            key=lambda c: (c.is_verified_match, c.overall_confidence, c.similarity_score),
            reverse=True
        )

        # -------------------------------------------------------------
        # BLOCKCHAIN EVIDENCE ANCHORING
        # -------------------------------------------------------------
        best_candidate: Optional[CandidateResult] = None
        if enriched_candidates and enriched_candidates[0].is_verified_match:
            best_candidate = enriched_candidates[0]

        match_record: Optional[MatchRecord] = None
        min_threshold = min(self.matcher.match_threshold, self.matcher.possible_threshold)
        if best_candidate and best_candidate.similarity_score >= min_threshold:
            evidence_pkg = EvidencePackage.create(
                input_image_hash=input_image_hash or "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                face_detected=input_face_detected,
                search_provider=best_candidate.search_provider or self.search_provider.mode_name,
                search_timestamp=best_candidate.discovery_timestamp,
                result_count=len(candidates),
                matched_title=best_candidate.title,
                matched_source_url=best_candidate.source_url,
                matched_image_url=best_candidate.image_url,
                matched_source_domain=best_candidate.source_domain,
                matched_platform=best_candidate.platform,
                similarity_score=best_candidate.similarity_score,
                threshold=self.matcher.match_threshold,
                verification_result="MATCH" if best_candidate.similarity_score >= self.matcher.match_threshold else "POSSIBLE_MATCH",
                extra_content_meta={
                    "snippet": best_candidate.snippet,
                    "author": best_candidate.author,
                    "content_identifier": best_candidate.content_identifier,
                    "content_type": best_candidate.content_type,
                    "category": best_candidate.category,
                    "overall_confidence": best_candidate.overall_confidence,
                    "quality_label": best_candidate.quality_label
                }
            )
            hex_hash, bytes32_hash = evidence_pkg.compute_hash()

            match_record = MatchRecord(
                source_url=best_candidate.source_url,
                source_title=best_candidate.title,
                image_url=best_candidate.image_url,
                similarity_score=best_candidate.similarity_score,
                similarity_status=best_candidate.similarity_status,
                discovery_timestamp=best_candidate.discovery_timestamp,
                content_hash=hex_hash,
                content_bytes32=bytes32_hash,
                content_identifier=best_candidate.content_identifier,
                source_domain=best_candidate.source_domain,
                platform=best_candidate.platform,
                content_type=best_candidate.content_type,
                category=best_candidate.category,
                overall_confidence=best_candidate.overall_confidence,
                confidence_badge=best_candidate.confidence_badge,
                quality_label=best_candidate.quality_label,
                match_explanation=best_candidate.match_explanation,
                metadata=best_candidate.metadata,
                image_bytes=best_candidate.downloaded_image_bytes,
                evidence_package=evidence_pkg.to_dict()
            )

        return enriched_candidates, match_record

    @staticmethod
    def separate_by_category(candidates: List[CandidateResult]) -> Dict[str, List[CandidateResult]]:
        """Groups candidates into the 3 distinct presentation categories."""
        grouped: Dict[str, List[CandidateResult]] = {
            "profiles": [],
            "posts_images": [],
            "reels_videos": []
        }
        for c in candidates:
            cat = c.category or "Posts & Images"
            if cat == ResultCategory.PROFILES.value or c.content_type == ContentType.PROFILE.value:
                grouped["profiles"].append(c)
            elif cat == ResultCategory.REELS_VIDEOS.value or c.content_type in {ContentType.REEL.value, ContentType.VIDEO.value}:
                grouped["reels_videos"].append(c)
            else:
                grouped["posts_images"].append(c)
        return grouped

    @staticmethod
    def get_verified_matches(candidates: List[CandidateResult]) -> List[CandidateResult]:
        """Filters to only verified biometric face matches."""
        return [c for c in candidates if c.is_verified_match]
