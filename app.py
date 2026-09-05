"""FaceTrace Chain — Modern AI-Powered Face Discovery & Blockchain Verification Interface.

Pipeline:
1. Face Scan Input & Biometric Validation (Upload or Live Camera Capture)
2. Face Quality Analysis & Landmark-Based Alignment
3. Deep Feature Embedding (ArcFace / InsightFace 512-d Primary, SFace Fallback)
4. Genuine Live Reverse Image Search (Google Lens via SerpAPI + Public Web Discovery)
5. Content & URL Classification (Profiles vs Posts/Images vs Reels/Videos)
6. Two-Stage Candidate Verification & Multi-Factor Confidence Scoring
7. Canonical Evidence Package Creation & Deterministic SHA-256 Fingerprinting
8. Immutable Ethereum / EVM Blockchain Anchoring
9. Cryptographic Re-Verification & Interactive Tamper Demonstration
"""
import sys
import os
import io
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
import streamlit as st
import cv2
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face.detector import FaceDetector, FaceDetectionError, FaceDetectionResult
from face.recognizer import BaseFaceRecognizer, get_face_recognizer, InsightFaceRecognizer, SFaceRecognizer
from face.quality import FaceQualityAnalyzer, FaceQualityReport
from face.matcher import FaceMatcher, MatchClassification, PROBABILISTIC_DISCLAIMER
from search.search_provider import (
    get_search_provider,
    SearchProviderError,
    BaseSearchProvider,
    LiveReverseImageSearchProvider,
    PublicWebSearchProvider
)
from search.candidate_collector import CandidateCollector
from utils.classifier import ResultClassifier, ContentType, ResultCategory
from utils.ui_components import (
    load_css,
    render_header,
    render_pipeline_tracker,
    render_quality_card,
    render_candidate_card
)
from models.evidence_package import EvidencePackage
from models.match_record import MatchRecord, CandidateResult
from blockchain.client import BlockchainClient
from blockchain.verifier import BlockchainVerifier
from blockchain.deploy import deploy_contract
from utils.config import get_config
from utils.hashing import hash_evidence_package, canonicalize_evidence_package

# Streamlit page configuration
st.set_page_config(
    page_title="FaceTrace Chain — AI Face Discovery & Blockchain Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load modern cyberpunk/AI stylesheet
load_css()


@st.cache_resource
def load_ai_models(engine_name: str = "arcface") -> Tuple[FaceDetector, BaseFaceRecognizer]:
    """Caches the face detector and biometric face recognition engine."""
    detector = FaceDetector()
    recognizer = get_face_recognizer(engine_name)
    return detector, recognizer


@st.cache_resource
def load_blockchain_client(rpc_url: str) -> BlockchainClient:
    """Caches the Ethereum / EVM blockchain client."""
    return BlockchainClient(rpc_url=rpc_url)


def main():
    config = get_config()

    # Sidebar: System Configuration & Readiness
    with st.sidebar:
        st.markdown("### ⚙️ System Settings")
        
        # Recognition Engine Selector
        engine_choice = st.selectbox(
            "Face Recognition Engine",
            options=["arcface", "sface"],
            format_func=lambda e: "ArcFace / InsightFace (512-d) [PRIMARY]" if e == "arcface" else "OpenCV SFace (128-d) [FALLBACK]",
            index=0 if config.face_recognition_engine == "arcface" else 1,
            help="ArcFace provides higher accuracy across diverse poses, lighting, and web compression."
        )

        detector, recognizer = load_ai_models(engine_choice)

        st.markdown("---")
        has_serp_key = bool(config.serpapi_key or config.search_api_key or os.getenv("SERPAPI_KEY"))
        default_search_idx = 0 if (config.search_mode == "live" and has_serp_key) else (0 if config.search_mode == "live" else 1)
        if "search_engine_mode" not in st.session_state:
            st.session_state["search_engine_mode"] = "live" if (config.search_mode == "live" and has_serp_key) else "web"

        search_mode = st.radio(
            "Discovery Engine",
            options=["live", "web"],
            format_func=lambda m: "Google Lens via SerpAPI [PRIMARY]" if m == "live" else "Public Web Media (Wikimedia) [SECONDARY]",
            index=0 if st.session_state["search_engine_mode"] == "live" else 1,
            key="search_engine_radio",
            help="Live search dynamically stages the face and queries Google Lens reverse visual search engine."
        )
        st.session_state["search_engine_mode"] = search_mode

        if search_mode == "live" and not has_serp_key:
            st.warning("⚠️ **Google Lens search requires SERPAPI_KEY.**")
            st.caption(
                "**Setup Instructions:**\n"
                "1. Obtain an API key from [serpapi.com](https://serpapi.com)\n"
                "2. Add to your `.env` file: `SERPAPI_KEY=your_key_here`\n"
                "3. Restart Streamlit.\n\n"
                "Or switch to the zero-config secondary provider below:"
            )
            if st.button("🌐 Switch to Wikimedia (Zero-Config)", use_container_width=True):
                st.session_state["search_engine_mode"] = "web"
                st.rerun()

        top_k = st.slider(
            "Stage-2 Top-K Verification",
            min_value=3,
            max_value=10,
            value=config.top_k_candidates,
            step=1,
            help="Number of top candidates passed to deep biometric verification & alignment."
        )

        st.markdown("---")
        st.markdown("### 🎯 Match Thresholds")
        high_thresh = st.slider(
            "High Confidence Threshold",
            min_value=0.50,
            max_value=0.95,
            value=float(config.face_match_high_threshold),
            step=0.01,
            help="Minimum cosine similarity for High-Confidence Face Match."
        )
        med_thresh = st.slider(
            "Possible Match Cutoff",
            min_value=0.30,
            max_value=0.80,
            value=float(config.face_match_medium_threshold),
            step=0.01,
            help="Minimum cosine similarity for Possible Match."
        )

        st.markdown("---")
        st.markdown("### ⛓️ Blockchain Infrastructure")
        rpc_url = st.text_input("EVM RPC Endpoint", value=config.blockchain_rpc_url)
        client = load_blockchain_client(rpc_url)
        verifier = BlockchainVerifier(client=client)

        if st.button("Deploy / Reset Smart Contract", use_container_width=True):
            with st.spinner("Compiling Solidity contract & deploying to EVM..."):
                try:
                    deploy_info = deploy_contract(rpc_url=rpc_url)
                    st.success(f"Contract deployed: `{deploy_info['contract_address'][:12]}...`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Deployment error: {e}")

        # System Readiness Status Card
        st.markdown("---")
        st.markdown("### 📡 Live System Readiness")
        is_chain_online = client.w3.is_connected()
        has_serp_key = bool(config.search_api_key or config.serpapi_key)

        status_detector = "YuNet Online"
        status_recognizer = f"{recognizer.name.split()[0]} Active"
        status_search = "Google Lens Live" if has_serp_key else "SerpAPI Key Missing"
        status_chain = f"Connected (Block #{client.w3.eth.block_number})" if is_chain_online else "In-Memory EVM"

        st.markdown(
            f"""
            <div style="background: rgba(15,23,42,0.7); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.75rem; font-size: 0.8rem;">
                <div style="margin-bottom: 0.35rem;">🟢 <b>Face Detector:</b> {status_detector}</div>
                <div style="margin-bottom: 0.35rem;">🟣 <b>Biometrics:</b> {status_recognizer} ({recognizer.embedding_dim}-d)</div>
                <div style="margin-bottom: 0.35rem;">{'🟢' if has_serp_key else '🟡'} <b>Search Engine:</b> {status_search}</div>
                <div>🟢 <b>Blockchain:</b> {status_chain}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Render Hero Header with live status indicators
    render_header({
        "detector": "YuNet Detector Online",
        "recognition": f"{recognizer.name.split()[0]} Active",
        "search": "Google Lens Ready" if has_serp_key else "Public Web Ready",
        "blockchain": "EVM Connected"
    })

    # Main Application Navigation Tabs
    tab_pipeline, tab_reverify = st.tabs([
        "🚀 Live End-to-End Discovery Pipeline",
        "🔎 Cryptographic Re-Verification & Tamper Audit"
    ])

    # =========================================================================
    # TAB 1: Live End-to-End Discovery & Verification Pipeline
    # =========================================================================
    with tab_pipeline:
        # Initialize session state for pipeline
        if "pipeline_stages" not in st.session_state:
            st.session_state.pipeline_stages = {
                "detection": {"status": "waiting", "time": ""},
                "encoding": {"status": "waiting", "time": ""},
                "search": {"status": "waiting", "time": ""},
                "verification": {"status": "waiting", "time": ""},
                "evidence": {"status": "waiting", "time": ""},
                "blockchain": {"status": "waiting", "time": ""},
                "reverification": {"status": "waiting", "time": ""}
            }

        # ---------------------------------------------------------------------
        # STEP 1 — FACE INPUT SOURCE CARD
        # ---------------------------------------------------------------------
        st.markdown('<div class="card-title"><span>📥 Step 1 — Face Input Source</span></div>', unsafe_allow_html=True)
        
        input_tab_upload, input_tab_camera, input_tab_sample = st.tabs([
            "📁 Option 1: Upload Face Image",
            "📸 Option 2: Capture Using Camera",
            "🖼️ Option 3: Authorized Sample Portrait"
        ])

        image_bytes: Optional[bytes] = None
        input_filename = "face_input.jpg"

        with input_tab_upload:
            uploaded_file = st.file_uploader(
                "Upload a clear front-facing portrait image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Drag & drop a single clear portrait. Supported formats: JPG, PNG, WEBP."
            )
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                input_filename = uploaded_file.name

        with input_tab_camera:
            st.write("Align face within the frame and capture a live snapshot:")
            camera_img = st.camera_input("Capture Face")
            if camera_img is not None:
                image_bytes = camera_img.read()
                input_filename = "camera_capture.jpg"

        with input_tab_sample:
            sample_options = {
                "Abraham Lincoln (Historical Portrait)": config.sample_data_dir / "sample_portrait_a.jpg",
                "Albert Einstein (Single Face Portrait)": config.sample_data_dir / "sample_portrait_b.jpg",
            }
            chosen_sample = st.selectbox(
                "Select a benchmark sample portrait:",
                list(sample_options.keys())
            )
            sample_path = sample_options[chosen_sample]
            if sample_path.exists() and image_bytes is None:
                if st.button("Load Sample Image"):
                    image_bytes = sample_path.read_bytes()
                    input_filename = sample_path.name
                    st.session_state["loaded_sample"] = sample_path.name

            if st.session_state.get("loaded_sample") == sample_path.name and image_bytes is None:
                image_bytes = sample_path.read_bytes()
                input_filename = sample_path.name

        # If image provided, preview and analyze quality
        query_face_result: Optional[FaceDetectionResult] = None
        quality_report: Optional[FaceQualityReport] = None

        if image_bytes is not None:
            # Controlled State Reset: Detect when image changes and reset execution state
            current_img_hash = hashlib.sha256(image_bytes).hexdigest()
            if current_img_hash != st.session_state.get("active_image_hash"):
                st.session_state["active_image_hash"] = current_img_hash
                st.session_state["pipeline_executed"] = False
                st.session_state["pipeline_result"] = None
                st.session_state["executed_candidates"] = []
                st.session_state["match_record"] = None
                st.session_state["tx_receipt"] = None

            col_preview, col_quality = st.columns([1, 1.4])
            
            with col_preview:
                st.markdown("**Face Input Preview & Bounding Box**")
                try:
                    query_face_result = detector.detect_primary_face(image_bytes, enforce_single=True)
                    annotated = query_face_result.draw_visualization()
                    # Convert BGR to RGB for display
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True, caption=f"Primary Face Detected ({query_face_result.confidence_pct}% Confidence)")
                except FaceDetectionError as fde:
                    st.error(f"❌ {fde}")
                    # Try showing raw image without detection box
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    raw_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if raw_bgr is not None:
                        st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Raw Input (Detection Failed)")
                    query_face_result = None
                except Exception as ex:
                    st.error(f"❌ Face detection notice: {ex}")
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    raw_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if raw_bgr is not None:
                        st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Raw Input (Detection Failed)")
                    query_face_result = None

            with col_quality:
                if query_face_result is not None:
                    quality_report = FaceQualityAnalyzer.analyze(
                        query_face_result.image_bgr,
                        query_face_result.bbox
                    )
                    render_quality_card(quality_report)
                else:
                    st.info("A single clear face is required to compute biometric quality metrics and begin analysis.")

            # Confirmation and Execution Button
            if query_face_result is not None:
                st.markdown("---")
                btn_cols = st.columns([1, 2, 1])
                with btn_cols[1]:
                    start_btn = st.button(
                        "⚡ Start Full Analysis & Verification Pipeline",
                        type="primary",
                        use_container_width=True
                    )

                if start_btn:
                    logging.info(f"Pipeline started for image ({len(image_bytes)} bytes). Mode: {search_mode}, Engine: {recognizer.name}")
                    with st.status("Executing Face Discovery & Blockchain Pipeline...", expanded=True) as status_box:
                        matcher = FaceMatcher(match_threshold=high_thresh, possible_threshold=med_thresh)
                        search_prov = get_search_provider(mode=search_mode)
                        collector = CandidateCollector(
                            search_provider=search_prov,
                            detector=detector,
                            encoder=recognizer,
                            matcher=matcher,
                            top_k=top_k
                        )

                        # Stage 1: Face Detection
                        t0 = time.perf_counter()
                        t_det = (time.perf_counter() - t0) * 1000
                        logging.info(f"Face detected: confidence={query_face_result.confidence_pct}%, bbox={query_face_result.bbox}")
                        st.write(f"✓ **Face Detection:** Primary face located ({query_face_result.confidence_pct}% confidence)")

                        # Stage 2: Biometric Encoding & Alignment
                        t0 = time.perf_counter()
                        query_emb = recognizer.generate_embedding(query_face_result)
                        t_enc = (time.perf_counter() - t0) * 1000
                        logging.info(f"Biometric embedding generated: {recognizer.embedding_dim}-d ({recognizer.name})")
                        st.write(f"✓ **Face Encoding:** Extracted {recognizer.embedding_dim}-d normalized {recognizer.name.split()[0]} embedding in {int(t_enc)} ms")

                        # Stage 3: Live Reverse Image Search
                        t0 = time.perf_counter()
                        st.write(f"🔍 **Discovery Search:** Querying {search_prov.mode_name}...")
                        logging.info(f"Querying search provider: {search_prov.mode_name}")
                        
                        search_status = {
                            "success": False,
                            "error": None,
                            "candidates": [],
                            "duration": 0.0
                        }

                        # Validate SerpAPI key upfront in live mode
                        has_live_key = bool(config.serpapi_key or config.search_api_key or os.getenv("SERPAPI_KEY"))
                        if search_mode == "live" and not has_live_key:
                            t_search = time.perf_counter() - t0
                            search_status["duration"] = t_search
                            search_status["success"] = False
                            search_status["error"] = "SERPAPI_KEY is missing or invalid. Google Lens search requires a valid SerpAPI key in .env."
                            logging.warning("SerpAPI key is missing in live mode.")
                            st.error(f"❌ {search_status['error']}")
                        else:
                            try:
                                discovered_candidates = collector.discover_candidates(image_bytes, max_results=15)
                                t_search = time.perf_counter() - t0
                                search_status["duration"] = t_search
                                search_status["success"] = True
                                search_status["candidates"] = discovered_candidates
                                logging.info(f"Discovered {len(discovered_candidates)} candidates in {t_search:.2f}s")
                                st.write(f"✓ **Search Complete:** Discovered {len(discovered_candidates)} candidates in {t_search:.1f} s")
                            except Exception as e:
                                t_search = time.perf_counter() - t0
                                search_status["duration"] = t_search
                                search_status["success"] = False
                                search_status["error"] = str(e)
                                logging.error(f"Discovery search failed: {e}")
                                st.error(f"❌ Discovery search error: {e}")

                        # Stage 4: Multi-Stage Candidate Verification (FIX 6: Do not continue broken verification)
                        t0 = time.perf_counter()
                        verified_candidates = []
                        match_record = None
                        verification_status = ""

                        if search_status["success"] and search_status["candidates"]:
                            st.write(f"⚖️ **Candidate Verification:** Running two-stage verification on {len(search_status['candidates'])} candidates...")
                            logging.info(f"Running candidate verification on {len(search_status['candidates'])} candidates")
                            verified_candidates, match_record = collector.verify_candidates(
                                input_embedding=query_emb,
                                candidates=search_status["candidates"],
                                input_image_bytes=image_bytes,
                                input_face_detected=True,
                                verbose=False
                            )
                            t_ver = time.perf_counter() - t0
                            verification_status = f"Completed ({len(verified_candidates)} verified, Top-{top_k} in {t_ver:.1f} s)"
                            logging.info(f"Verification finished in {t_ver:.2f}s. Match: {match_record is not None}")
                            st.write(f"✓ **Verification Finished:** Processed Top-{top_k} in {t_ver:.1f} s")
                        elif search_status["success"] and not search_status["candidates"]:
                            verification_status = "Skipped (No candidate media returned from discovery search)"
                            logging.info("Verification skipped: no candidates returned")
                            st.write("ℹ️ **Verification Skipped:** No candidate records available for biometric comparison.")
                        else:
                            verification_status = f"Skipped (Search unavailable: {search_status['error']})"
                            logging.info(f"Verification skipped: search unavailable ({search_status['error']})")
                            st.write("⚠️ **Verification Skipped:** Search unavailable.")

                        # Stage 5: Evidence Package Generation
                        if match_record and match_record.evidence_package:
                            hex_hash = match_record.content_hash
                            logging.info(f"Evidence package generated: {hex_hash}")
                            st.write(f"✓ **Evidence Package:** Generated canonical SHA-256 fingerprint: `{hex_hash[:16]}...`")
                        elif search_status["candidates"]:
                            logging.info("Standard evidence package envelope created")
                            st.write("ℹ️ **Evidence Package:** No candidate exceeded match threshold; standard evidence envelope created.")
                        else:
                            logging.info("Evidence package skipped (no candidates)")
                            st.write("ℹ️ **Evidence Package:** Skipped (no candidates).")

                        # Stage 6: Blockchain Storage (FIX 8: Blockchain must not break pipeline)
                        tx_receipt = None
                        blockchain_status = "skipped"
                        blockchain_details = "Skipped (no verified candidate match)"

                        if match_record:
                            st.write("⛓️ **Blockchain Storage:** Anchoring verification record to smart contract...")
                            try:
                                tx_receipt = client.record_face_verification(
                                    content_hash=match_record.content_hash,
                                    source_url=match_record.source_url,
                                    similarity_score=match_record.similarity_score,
                                    metadata_json=json.dumps(match_record.metadata)
                                )
                                blockchain_status = "anchored"
                                blockchain_details = f"Anchored in Block #{tx_receipt.get('block_number', 'N/A')}"
                                logging.info(f"Blockchain anchored: tx={tx_receipt.get('transaction_hash')}, block={tx_receipt.get('block_number')}")
                                st.write(f"✓ **Blockchain Anchored:** Tx `{tx_receipt['transaction_hash'][:16]}...` (Block #{tx_receipt['block_number']})")
                            except Exception as e:
                                blockchain_status = "unavailable"
                                blockchain_details = f"Blockchain unavailable — evidence package generated locally ({e})"
                                logging.warning(f"Blockchain recording failed: {e}")
                                st.warning(f"⚠️ {blockchain_details}")
                        else:
                            st.write("ℹ️ **Blockchain Storage:** Skipped (no candidate met minimum verification threshold).")

                        status_box.update(label="✓ Pipeline Execution Finished!", state="complete", expanded=False)

                    # Save execution state to session
                    st.session_state["pipeline_result"] = {
                        "executed": True,
                        "status": "SUCCESS" if (search_status["success"] and len(search_status["candidates"]) > 0) else (
                            "COMPLETED_NO_CANDIDATES" if (search_status["success"] and len(search_status["candidates"]) == 0) else "SEARCH_UNAVAILABLE"
                        ),
                        "face_detected": True,
                        "face_confidence": query_face_result.confidence_pct,
                        "bbox": query_face_result.bbox,
                        "quality_report": quality_report,
                        "engine_name": recognizer.name,
                        "embedding_dim": recognizer.embedding_dim,
                        "search_mode": search_mode,
                        "search_provider_name": search_prov.mode_name,
                        "search_success": search_status["success"],
                        "search_error": search_status["error"],
                        "search_duration": search_status["duration"],
                        "candidate_count": len(search_status["candidates"]),
                        "verified_candidates": verified_candidates,
                        "match_record": match_record,
                        "tx_receipt": tx_receipt,
                        "verification_status": verification_status,
                        "blockchain_status": blockchain_status,
                        "blockchain_details": blockchain_details
                    }
                    st.session_state["pipeline_executed"] = True
                    st.session_state["executed_candidates"] = verified_candidates
                    st.session_state["match_record"] = match_record
                    st.session_state["tx_receipt"] = tx_receipt
                    logging.info("Pipeline execution completed successfully and saved to session state.")

        # ---------------------------------------------------------------------
        # RESULTS DISPLAY SECTION (ALWAYS VISIBLE WHEN PIPELINE EXECUTED)
        # ---------------------------------------------------------------------
        if st.session_state.get("pipeline_executed", False) and st.session_state.get("pipeline_result"):
            res = st.session_state["pipeline_result"]
            all_candidates: List[CandidateResult] = res.get("verified_candidates", [])
            match_record: Optional[MatchRecord] = res.get("match_record")
            tx_receipt: Optional[Dict[str, Any]] = res.get("tx_receipt")
            q_report: Optional[FaceQualityReport] = res.get("quality_report")

            st.markdown("---")

            # =================================================================
            # REQUIRED FIX 9: ALWAYS-VISIBLE PIPELINE EXECUTION SUMMARY CARD
            # =================================================================
            st.markdown("### 📊 Pipeline Execution Summary")
            
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 0.8rem; min-height: 140px;">
                        <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase;">Biometrics</div>
                        <div style="font-weight: 700; color: #10B981; font-size: 1rem; margin: 4px 0;">✓ Face Detected</div>
                        <div style="font-size: 0.8rem; color: var(--text-primary);">{res['face_confidence']}% Detection Confidence</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">{res['engine_name'].split()[0]} ({res['embedding_dim']}-d)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with summary_cols[1]:
                q_overall = q_report.overall_quality if q_report else "N/A"
                q_color = "#10B981" if q_overall == "HIGH" else ("#F59E0B" if q_overall == "MEDIUM" else "#EF4444")
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 0.8rem; min-height: 140px;">
                        <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase;">Quality Check</div>
                        <div style="font-weight: 700; color: {q_color}; font-size: 1rem; margin: 4px 0;">✓ {q_overall} Quality</div>
                        <div style="font-size: 0.8rem; color: var(--text-primary);">{q_report.resolution_status if q_report else 'Checked'} • {q_report.blur_status if q_report else 'Checked'}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Illumination: {q_report.brightness_status if q_report else 'Checked'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with summary_cols[2]:
                s_icon = "✓" if res['search_success'] else "⚠️"
                s_color = "#10B981" if res['search_success'] else "#EF4444"
                s_status_text = f"{res['candidate_count']} Candidates" if res['search_success'] else "Unavailable"
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 0.8rem; min-height: 140px;">
                        <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase;">Discovery Search</div>
                        <div style="font-weight: 700; color: {s_color}; font-size: 1rem; margin: 4px 0;">{s_icon} {s_status_text}</div>
                        <div style="font-size: 0.8rem; color: var(--text-primary);">{res['search_provider_name'].split('(')[0]}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Query Time: {res['search_duration']:.2f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with summary_cols[3]:
                b_color = "#10B981" if res['blockchain_status'] == "anchored" else ("#F59E0B" if res['blockchain_status'] == "unavailable" else "#64748B")
                b_badge = "✓ Anchored" if res['blockchain_status'] == "anchored" else ("⚠️ Local Proof" if res['blockchain_status'] == "unavailable" else "○ Skipped")
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 0.8rem; min-height: 140px;">
                        <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase;">Blockchain & Evidence</div>
                        <div style="font-weight: 700; color: {b_color}; font-size: 1rem; margin: 4px 0;">{b_badge}</div>
                        <div style="font-size: 0.8rem; color: var(--text-primary);">Verification: {res['verification_status'].split('(')[0]}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">{res['blockchain_details'][:30]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # =================================================================
            # BRANCH: THREE RESULT STATES
            # =================================================================

            # -----------------------------------------------------------------
            # STATE C: Search Unavailable / Failed
            # -----------------------------------------------------------------
            if not res["search_success"]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 4px solid #EF4444; padding: 1.25rem;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem;">⚠️</span>
                            <h3 style="margin: 0; color: #EF4444;">Discovery Search Unavailable</h3>
                        </div>
                        <p style="color: var(--text-primary); font-size: 0.95rem; margin-bottom: 0.5rem;">
                            <b>Reason:</b> {res['search_error']}
                        </p>
                        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">
                            Face analysis completed successfully (biometric embedding generated), but public candidate discovery could not be performed with the selected provider.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col_act1, col_act2 = st.columns([1, 1])
                with col_act1:
                    st.markdown("**How to configure Google Lens (SerpAPI):**")
                    st.markdown(
                        """
                        1. Sign up for a free key at [serpapi.com](https://serpapi.com).
                        2. Add key to `.env` in the project root:
                           ```env
                           SERPAPI_KEY=your_actual_key_here
                           ```
                        3. Restart Streamlit or refresh the application.
                        """
                    )
                with col_act2:
                    st.markdown("**Alternative: Zero-Config Secondary Provider:**")
                    st.write("You can switch immediately to Public Web Media (Wikimedia) without requiring any API key.")
                    if st.button("🌐 Switch to Wikimedia Discovery Mode", use_container_width=True, type="primary"):
                        st.session_state["search_engine_mode"] = "web"
                        st.session_state["pipeline_executed"] = False
                        st.rerun()

            # -----------------------------------------------------------------
            # STATE B: Search Completed But No Candidates Found
            # -----------------------------------------------------------------
            elif res["search_success"] and res["candidate_count"] == 0:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 4px solid #3B82F6; padding: 1.25rem;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem;">🔍</span>
                            <h3 style="margin: 0; color: #60A5FA;">Search Completed — No Candidate Matches Found</h3>
                        </div>
                        <p style="color: var(--text-primary); font-size: 0.95rem; margin-bottom: 0.5rem;">
                            No visually relevant candidates were returned by the configured discovery provider (<b>{res['search_provider_name']}</b>).
                        </p>
                        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">
                            <b>Important Notice:</b> This does <b>NOT</b> mean the person does not exist online. It only means no candidate media records were returned by the discovery search index for verification.
                        </p>
                        <div style="display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.8rem; color: var(--text-primary); background: rgba(15,23,42,0.4); padding: 0.5rem; border-radius: 6px;">
                            <span>✓ Face detected ({res['face_confidence']}%)</span>
                            <span>✓ Face quality analyzed ({q_report.overall_quality if q_report else 'Checked'})</span>
                            <span>✓ Biometric embedding generated ({res['embedding_dim']}-d)</span>
                            <span>✓ Search executed ({res['search_duration']:.2f}s)</span>
                            <span style="color: var(--text-secondary);">○ 0 candidate results returned</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(15,23,42,0.6); border: 1px solid var(--border-subtle); border-radius: 8px; font-size: 0.8rem;">
                        <b>Technical Diagnostics:</b><br/>
                        • Recognition Engine: {res['engine_name']} ({res['embedding_dim']}-dimensional normalized embedding)<br/>
                        • Detection Bounding Box: {res.get('bbox', 'N/A')}<br/>
                        • Biometric Quality: {q_report.summary_text() if q_report else 'Assessed'}<br/>
                        • Provider: {res['search_provider_name']}<br/>
                        • Query Latency: {res['search_duration']:.2f} seconds
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col_b1, col_b2, col_b3 = st.columns([1, 1, 1.5])
                with col_b1:
                    if st.button("🖼️ Try Another Image", use_container_width=True):
                        st.session_state["pipeline_executed"] = False
                        st.session_state["pipeline_result"] = None
                        st.rerun()
                with col_b2:
                    if st.button("🔄 Retry Search", use_container_width=True):
                        st.session_state["pipeline_executed"] = False
                        st.rerun()
                with col_b3:
                    if "google" in res["search_provider_name"].lower():
                        if st.button("🌐 Switch to Wikimedia Mode", use_container_width=True):
                            st.session_state["search_engine_mode"] = "web"
                            st.session_state["pipeline_executed"] = False
                            st.rerun()

            # -----------------------------------------------------------------
            # STATE A: Candidates Found (Deep Verification & Categorization)
            # -----------------------------------------------------------------
            else:
                # If Wikimedia mode, show disclaimer
                if "wikimedia" in res["search_provider_name"].lower():
                    st.markdown(
                        """
                        <div style="background: rgba(245,158,11,0.1); border: 1px solid #F59E0B; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #FCD34D;">
                            ⚠️ <b>Public Web Discovery Results</b>: This secondary mode queries open web media repositories via metadata and is not equivalent to neural reverse image search. Results represent public domain discovery media. Candidate biometric verification is performed independently on each downloaded item.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                categorized = CandidateCollector.separate_by_category(all_candidates)
                verified_matches = CandidateCollector.get_verified_matches(all_candidates)

                # Section 1: AI-Verified Face Matches
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                        <h3 style="margin: 0; color: #F8FAFC;">🎯 AI-Verified Face Matches</h3>
                        <span class="badge-tag badge-high">{len(verified_matches)} Verified</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if verified_matches:
                    v_cols = st.columns(min(len(verified_matches), 3))
                    for i, v_cand in enumerate(verified_matches):
                        with v_cols[i % len(v_cols)]:
                            render_candidate_card(v_cand, rank=i + 1, is_verified_section=True)
                else:
                    st.info(
                        "No discovered candidates met the active match threshold. "
                        "All discovered web resources are available below under Discovered Results."
                    )

                st.markdown("---")

                # Section 2: Discovered Results (Profiles / Posts & Images / Reels & Videos)
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                        <h3 style="margin: 0; color: #F8FAFC;">🌐 Discovered Web Results</h3>
                        <span class="badge-tag badge-cyan">{len(all_candidates)} Total Discovered</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                profiles = categorized["profiles"]
                posts_images = categorized["posts_images"]
                reels_videos = categorized["reels_videos"]

                res_tab_profiles, res_tab_posts, res_tab_reels = st.tabs([
                    f"👤 Profiles ({len(profiles)})",
                    f"📄 Posts & Images ({len(posts_images)})",
                    f"🎬 Reels & Videos ({len(reels_videos)})"
                ])

                with res_tab_profiles:
                    if profiles:
                        p_cols = st.columns(min(len(profiles), 3))
                        for i, p_cand in enumerate(profiles):
                            with p_cols[i % len(p_cols)]:
                                render_candidate_card(p_cand, rank=i + 1)
                    else:
                        st.write("No standalone social profile pages identified in this query.")

                with res_tab_posts:
                    if posts_images:
                        pi_cols = st.columns(min(len(posts_images), 3))
                        for i, pi_cand in enumerate(posts_images):
                            with pi_cols[i % len(pi_cols)]:
                                render_candidate_card(pi_cand, rank=i + 1)
                    else:
                        st.write("No web posts or standalone image articles found.")

                with res_tab_reels:
                    if reels_videos:
                        rv_cols = st.columns(min(len(reels_videos), 3))
                        for i, rv_cand in enumerate(reels_videos):
                            with rv_cols[i % len(rv_cols)]:
                                render_candidate_card(rv_cand, rank=i + 1)
                    else:
                        st.write("No short-form video reels or video content links identified in this query.")

                # Section 3: Blockchain Evidence & Immutable Anchor
                if match_record and match_record.evidence_package:
                    st.markdown("---")
                    st.markdown("### ⛓️ Cryptographic Evidence & Blockchain Anchor")

                    ev_pkg = match_record.evidence_package
                    canonical_str = canonicalize_evidence_package(ev_pkg)

                    col_meta, col_code = st.columns([1.2, 1])
                    with col_meta:
                        st.markdown(
                            f"""
                            <div class="glass-card">
                                <div class="card-title">Immutable On-Chain Proof</div>
                                <div class="evidence-timeline">
                                    <div class="timeline-item">
                                        <div class="timeline-badge">SHA-256</div>
                                        <div class="timeline-content">
                                            <div class="timeline-label">Evidence Fingerprint</div>
                                            <div class="timeline-detail">{match_record.content_hash}</div>
                                        </div>
                                    </div>
                                    <div class="timeline-item">
                                        <div class="timeline-badge">BYTES32</div>
                                        <div class="timeline-content">
                                            <div class="timeline-label">EVM Storage Key</div>
                                            <div class="timeline-detail">{match_record.content_bytes32}</div>
                                        </div>
                                    </div>
                                    <div class="timeline-item">
                                        <div class="timeline-badge">TX HASH</div>
                                        <div class="timeline-content">
                                            <div class="timeline-label">Transaction</div>
                                            <div class="timeline-detail">{tx_receipt['transaction_hash'] if tx_receipt else 'Local Proof (RPC Offline)'}</div>
                                        </div>
                                    </div>
                                    <div class="timeline-item">
                                        <div class="timeline-badge">STATUS</div>
                                        <div class="timeline-content">
                                            <div class="timeline-label">Blockchain Anchor Status</div>
                                            <div class="timeline-detail">{res['blockchain_details']}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col_code:
                        st.markdown("**Canonical Evidence Package JSON**")
                        st.code(canonical_str, language="json")
                        st.download_button(
                            label="📥 Download Evidence Package (.json)",
                            data=canonical_str,
                            file_name=f"evidence_{match_record.content_hash[:12]}.json",
                            mime="application/json",
                            use_container_width=True
                        )

    # =========================================================================
    # TAB 2: Cryptographic Re-Verification & Tamper Audit
    # =========================================================================
    with tab_reverify:
        st.markdown('<div class="card-title"><span>🔎 Cryptographic Evidence Re-Verification</span></div>', unsafe_allow_html=True)
        st.write(
            "Validate any previously generated Evidence Package against the immutable blockchain ledger. "
            "Because the SHA-256 hash is computed deterministically from canonical sorted JSON keys, "
            "even a single-character alteration will cause a hash divergence and fail verification."
        )

        subtab_verify, subtab_tamper = st.tabs([
            "🛡️ Verify Existing Record",
            "⚠️ Interactive Tamper Demonstration"
        ])

        with subtab_verify:
            # Default to the current pipeline's evidence package if available
            default_json = ""
            if "match_record" in st.session_state and st.session_state["match_record"]:
                default_json = canonicalize_evidence_package(st.session_state["match_record"].evidence_package)

            evidence_input = st.text_area(
                "Paste Evidence Package JSON:",
                value=default_json,
                height=220,
                help="Paste the canonical evidence JSON to re-verify against the smart contract."
            )

            if st.button("Verify Record on Blockchain", type="primary", use_container_width=True):
                if not evidence_input.strip():
                    st.warning("Please paste valid evidence JSON.")
                else:
                    try:
                        parsed_pkg = json.loads(evidence_input)
                        computed_hex, computed_bytes32 = hash_evidence_package(parsed_pkg)

                        st.write(f"**Recalculated SHA-256 Fingerprint:** `{computed_hex}`")
                        v_result = verifier.verify_evidence_package(parsed_pkg)

                        if v_result.is_authentic:
                            st.success(f"🟢 **AUTHENTIC RECORD CONFIRMED ON BLOCKCHAIN**\n\n"
                                       f"• Status: Authenticated on-chain\n"
                                       f"• Block Number: #{v_result.block_number}\n"
                                       f"• Timestamp: {v_result.timestamp}\n"
                                       f"• Stored URL: {v_result.stored_url}\n"
                                       f"• Recorded Similarity: {v_result.stored_similarity * 100:.1f}%")
                        else:
                            st.error(f"🔴 **VERIFICATION FAILED / RECORD NOT FOUND**\n\n"
                                     f"• Status: {v_result.status}\n"
                                     f"• Error: {v_result.error_message or 'Hash does not exist on blockchain ledger.'}")
                    except json.JSONDecodeError as je:
                        st.error(f"Invalid JSON format: {je}")
                    except Exception as e:
                        st.error(f"Re-verification error: {e}")

        with subtab_tamper:
            st.markdown("### 🧪 Tamper Resistance Proof")
            st.write(
                "Modify any field in the evidence package below to observe the SHA-256 cryptographic avalanche effect. "
                "The altered hash will not match the immutable blockchain record."
            )

            if "match_record" in st.session_state and st.session_state["match_record"]:
                orig_pkg = st.session_state["match_record"].evidence_package
                orig_hash, _ = hash_evidence_package(orig_pkg)

                col_orig, col_mod = st.columns(2)
                with col_orig:
                    st.markdown("**Original Valid Evidence**")
                    st.code(canonicalize_evidence_package(orig_pkg), language="json")
                    st.markdown(f"**Valid Hash:** `{orig_hash}`")

                with col_mod:
                    st.markdown("**Simulated Modified Evidence**")
                    tampered_pkg = json.loads(json.dumps(orig_pkg))
                    # Allow user to choose what to tamper
                    field_to_tamper = st.selectbox(
                        "Select field to alter:",
                        ["similarity_score", "matched_source_url", "search_timestamp", "threshold"]
                    )
                    if field_to_tamper == "similarity_score":
                        if "face_verification" in tampered_pkg:
                            tampered_pkg["face_verification"]["similarity_score"] = 0.9999
                        else:
                            tampered_pkg["similarity_score"] = 0.9999
                    elif field_to_tamper == "matched_source_url":
                        if "matched_content" in tampered_pkg:
                            tampered_pkg["matched_content"]["source_url"] = "https://fake-imposter-site.org/profile"
                        else:
                            tampered_pkg["matched_source_url"] = "https://fake-imposter-site.org/profile"
                    elif field_to_tamper == "search_timestamp":
                        if "search" in tampered_pkg:
                            tampered_pkg["search"]["search_timestamp"] = datetime.now(timezone.utc).isoformat()
                        else:
                            tampered_pkg["search_timestamp"] = datetime.now(timezone.utc).isoformat()
                    elif field_to_tamper == "threshold":
                        if "face_verification" in tampered_pkg:
                            tampered_pkg["face_verification"]["threshold"] = 0.50
                        else:
                            tampered_pkg["threshold"] = 0.50

                    tampered_hash, _ = hash_evidence_package(tampered_pkg)
                    st.code(canonicalize_evidence_package(tampered_pkg), language="json")
                    st.markdown(f"**Tampered Hash:** `{tampered_hash}`")

                st.markdown("---")
                if st.button("Audit Tampered Evidence Against Blockchain", use_container_width=True):
                    t_res = verifier.verify_evidence_package(tampered_pkg, expected_original_hash=orig_hash)
                    if t_res.is_authentic:
                        st.error("Unexpected: Tampered record matched.")
                    else:
                        st.error(
                            f"🔴 **TAMPERING DETECTED**\n\n"
                            f"• Original Hash: `{orig_hash}`\n"
                            f"• Tampered Hash: `{tampered_hash}`\n"
                            f"• Discrepancy: Altered `{field_to_tamper}` resulted in complete hash divergence.\n"
                            f"• Blockchain Verdict: {t_res.details or 'Hash does not match immutable on-chain record.'}"
                        )
            else:
                st.info("Run the pipeline on an image in Tab 1 first to generate a verified match record for this demonstration.")


if __name__ == "__main__":
    main()
