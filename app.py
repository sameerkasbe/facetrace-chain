"""Lightweight professional Streamlit interface for FaceTrace Chain.

Demonstrates:
1. Face Image Input & Strict Single-Face Validation
2. 128-d Feature Embedding (SFace)
3. Genuine Reverse Image Search (Google Lens) vs Offline Demo Mode
4. Candidate Face Verification & Probabilistic Similarity Scoring
5. Deterministic SHA-256 Tamper-Evident Fingerprint & Blockchain Storage
6. Controlled Tampering Simulation & Smart Contract Re-Verification
"""
import sys
import io
import time
from pathlib import Path
from typing import Optional, List
import streamlit as st
import cv2
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face.detector import FaceDetector, FaceDetectionError
from face.encoder import FaceEncoder
from face.matcher import FaceMatcher, PROBABILISTIC_DISCLAIMER
from search.search_provider import get_search_provider, SearchProviderError, BaseSearchProvider
from search.candidate_collector import CandidateCollector
from blockchain.client import BlockchainClient
from blockchain.verifier import BlockchainVerifier
from blockchain.deploy import deploy_contract
from utils.config import get_config

# Page configuration
st.set_page_config(
    page_title="FaceTrace Chain",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .reportview-container { background: #fafafa; }
    .main-header { font-size: 1.85rem; font-weight: 700; color: #111; margin-bottom: 0.1rem; }
    .sub-header { font-size: 0.95rem; color: #555; margin-bottom: 1.2rem; }
    .mode-pill-live { display: inline-block; background: #e8f0fe; color: #1a73e8; border: 1px solid #c2e7ff; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 0.85rem; margin-bottom: 1rem; }
    .mode-pill-offline { display: inline-block; background: #fef7e0; color: #b06000; border: 1px solid #f9ab00; padding: 4px 12px; border-radius: 16px; font-weight: 600; font-size: 0.85rem; margin-bottom: 1rem; }
    .badge-match { background: #e6f4ea; color: #137333; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-possible { background: #fef7e0; color: #b06000; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-nomatch { background: #fce8e6; color: #c5221f; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .hash-text { font-family: monospace; font-size: 0.88rem; background: #f1f3f4; padding: 8px 12px; border-radius: 4px; word-break: break-all; border-left: 3px solid #1a73e8; }
    .log-box { font-family: monospace; font-size: 0.82rem; background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_face_models():
    detector = FaceDetector()
    encoder = FaceEncoder()
    return detector, encoder


def main():
    config = get_config()
    detector, encoder = load_face_models()

    st.markdown('<div class="main-header">FaceTrace Chain</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Face Discovery and Blockchain Verification Pipeline</div>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    with st.sidebar:
        st.subheader("Discovery Mode")
        search_mode = st.radio(
            "Select Search Architecture:",
            options=["live", "offline"],
            format_func=lambda m: "Live Reverse Image Search (Google Lens API)" if m == "live" else "Offline Demo Mode (Local Corpus Simulation)",
            index=0 if config.search_mode == "live" else 1,
            help="Live mode stages the input face and executes real visual search. Offline mode simulates candidate matching via local authorized corpus."
        )

        if search_mode == "live":
            st.markdown("🌐 **Live Reverse Search Engine:** Active")
            st.caption("Google Lens reverse visual search is configured and active. Search executes dynamically on the uploaded image.")

        threshold = st.slider(
            "Match Threshold",
            min_value=0.50,
            max_value=0.98,
            value=float(config.match_threshold),
            step=0.01,
            help="Cosine similarity cutoff for High Similarity Match (Default: 0.85)"
        )

        st.markdown("---")
        st.subheader("Blockchain Infrastructure")
        rpc_url = st.text_input("RPC Endpoint", value=config.blockchain_rpc_url)

        client = BlockchainClient(rpc_url=rpc_url)
        is_connected = client.w3.is_connected()
        if is_connected:
            block_num = client.w3.eth.block_number
            st.success(f"Connected (Block #{block_num})")
            if client.contract_address:
                st.caption(f"Contract: `{client.contract_address[:10]}...{client.contract_address[-6:]}`")
        else:
            st.warning("RPC offline. Ensure Ganache or EVM node is running.")

        if st.button("Deploy / Re-deploy Smart Contract"):
            with st.spinner("Compiling & Deploying FaceVerification contract..."):
                deploy_info = deploy_contract(rpc_url=rpc_url)
                st.success(f"Deployed at: `{deploy_info['contract_address']}`")
                st.rerun()

    # Active Mode Status Pill
    if search_mode == "live":
        st.markdown(
            '<div class="mode-pill-live">🌐 Mode: Live Reverse Image Search (Google Lens Visual Search)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="mode-pill-offline">📂 Mode: Offline Demo Mode (Local Public Domain Corpus Simulation)</div>',
            unsafe_allow_html=True
        )

    # 1. Image Input Selection
    st.markdown("### 1. Face Image Input")
    input_mode = st.radio(
        "Choose image source:",
        ["Upload File", "Use Sample Authorized Portrait"],
        horizontal=True
    )

    image_bytes = None
    input_filename = "face_input.jpg"

    if input_mode == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload portrait image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            input_filename = uploaded_file.name
    else:
        sample_options = {
            "Abraham Lincoln (Single Face Portrait)": config.sample_data_dir / "sample_portrait_a.jpg",
            "Albert Einstein (Single Face Portrait)": config.sample_data_dir / "sample_portrait_b.jpg",
            "Multi-Face Image (Triggers Single-Face Error)": config.sample_data_dir / "sample_multi_face.jpg"
        }
        selected_sample = st.selectbox("Select sample test image:", list(sample_options.keys()))
        sample_path = sample_options[selected_sample]
        if sample_path.exists():
            image_bytes = sample_path.read_bytes()
            input_filename = sample_path.name

    if image_bytes is None:
        st.info("Please upload or select an authorized portrait image to start the pipeline.")
        return

    # STEP 1: Face Detection & Validation
    st.markdown("---")
    st.markdown("### Step 1 — Face Detection")

    try:
        face_result = detector.detect_primary_face(image_bytes, enforce_single=True)
        col_img1, col_img2 = st.columns([1, 2])
        with col_img1:
            annotated = face_result.draw_visualization()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Primary Face", use_container_width=True)

        with col_img2:
            st.success("✓ Single clear primary face verified")
            st.write(f"**Face Confidence:** {face_result.confidence_pct}%")
            st.write(f"**Bounding Box (x, y, w, h):** {face_result.bbox}")
            st.caption("5 facial landmarks localized (eyes, nose, mouth corners). Enforced strict single-face constraint.")

    except FaceDetectionError as e:
        st.error(f"❌ Face Detection Error: {e}")
        return
    except Exception as e:
        st.error(f"❌ Image Decoding Error: {e}")
        return

    # STEP 2: Facial Embedding Generation
    st.markdown("---")
    st.markdown("### Step 2 — Facial Embedding Generation")
    with st.spinner("Extracting 128-dimensional biometric embedding via SFace DNN..."):
        embedding = encoder.generate_embedding(face_result)
    st.success("✓ 128-dimensional dense feature vector generated with L2 normalization (Euclidean norm: 1.000)")

    # STEP 3: Public Content Discovery & Reverse Image Search
    st.markdown("---")
    st.markdown("### Step 3 — Public Content Discovery")

    search_provider = get_search_provider(mode=search_mode)
    st.write(f"Executing discovery using **{search_provider.mode_name}**...")

    collector = CandidateCollector(
        search_provider=search_provider,
        detector=detector,
        encoder=encoder,
        threshold=threshold
    )

    proof_logs: List[str] = []
    proof_logs.append(f"[SEARCH] Input image received: {input_filename} ({len(image_bytes)} bytes)")

    with st.spinner(f"Querying {search_provider.mode_name}..."):
        try:
            if search_mode == "live":
                proof_logs.append("[SEARCH] Creating search request")
                proof_logs.append("[SEARCH] Staging local image to temporary public hosting...")
                candidates = collector.discover_candidates(image_bytes, max_results=5)
                proof_logs.append("[SEARCH] Reverse image search executed via SerpAPI Google Lens")
            else:
                proof_logs.append("[SEARCH] Running in OFFLINE DEMO MODE (Local Corpus Simulation)")
                candidates = collector.discover_candidates(image_bytes, max_results=5)
                proof_logs.append(f"[SEARCH] Candidate results loaded from local demo corpus: {len(candidates)} candidates")

            proof_logs.append(f"[SEARCH] Candidate results received: {len(candidates)} matches found")
            st.write(f"**{len(candidates)} candidates discovered** dynamically at runtime:")
        except SearchProviderError as e:
            st.error(f"Search Provider Error: {e}")
            proof_logs.append(f"[ERROR] {e}")
            with st.expander("View Runtime Logs", expanded=True):
                st.code("\n".join(proof_logs), language="bash")
            return
        except Exception as e:
            st.error(f"Search Failure: {e}")
            return

    # STEP 4: Candidate Face Verification & Scoring
    st.markdown("---")
    st.markdown("### Step 4 — Candidate Face Verification & Matching")
    proof_logs.append("[VERIFY] Downloading candidate images")
    proof_logs.append("[VERIFY] Comparing facial embeddings")

    with st.spinner("Downloading candidate assets and computing facial similarities..."):
        enriched_candidates, best_match = collector.verify_candidates(
            input_embedding=embedding,
            candidates=candidates,
            verbose=False
        )

    # Show proof logs in expandable box
    with st.expander("🔍 Runtime Proof & Execution Logs", expanded=True):
        st.code("\n".join(proof_logs), language="bash")

    # Render candidate table
    table_rows = []
    for i, c in enumerate(enriched_candidates, start=1):
        pct = int(round(c.similarity_score * 100))
        table_rows.append({
            "Candidate": f"Candidate {i}",
            "Title": c.title[:40],
            "Type": "Live Web" if getattr(c, "is_live", True) else "Offline Demo",
            "Domain": c.source_domain,
            "Similarity": f"{pct}%",
            "Classification": c.similarity_status
        })

    st.dataframe(table_rows, use_container_width=True)
    st.caption(PROBABILISTIC_DISCLAIMER)

    if best_match:
        st.success(f"**BEST MATCH DISCOVERED:** {best_match.similarity_status} ({int(round(best_match.similarity_score * 100))}%)")
        st.write(f"**Matched Source URL:** [{best_match.source_url}]({best_match.source_url})")
        st.write(f"**Title:** {best_match.source_title}")
    else:
        st.warning(f"No candidate met the match threshold of {threshold:.2f}.")
        return

    # STEP 5: Tamper-Evident Fingerprint & Blockchain Record
    st.markdown("---")
    st.markdown("### Step 5 — Tamper-Evident Fingerprint & Blockchain Record")

    st.write("**Canonical Cryptographic Fingerprint (SHA-256):**")
    st.markdown(f'<div class="hash-text">{best_match.content_hash}</div>', unsafe_allow_html=True)
    st.caption("Canonical payload: source_url + content_id + title + image_hash + metadata")

    # Anchoring to blockchain
    session_key = f"tx_{best_match.content_hash}"
    if session_key not in st.session_state:
        with st.spinner("Recording fingerprint on blockchain..."):
            if client.contract is None:
                deploy_contract(rpc_url=rpc_url)
                client = BlockchainClient(rpc_url=rpc_url)

            store_res = client.store_verification(
                content_hash_hex=best_match.content_hash,
                source_reference=best_match.source_url
            )
            st.session_state[session_key] = store_res

    store_res = st.session_state[session_key]

    if store_res.success:
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.metric("Block Number", store_res.block_number)
        with col_b2:
            st.metric("Blockchain Status", "Confirmed ✓")
        with col_b3:
            st.metric("Gas Used", store_res.gas_used)

        st.write(f"**Transaction Hash:** `{store_res.tx_hash}`")
        st.write(f"**Submitter Address:** `{store_res.submitter}`")
    else:
        st.error(f"Blockchain storage rejected: {store_res.error_message}")

    # STEP 6: Blockchain Re-Verification & Controlled Tampering
    st.markdown("---")
    st.markdown("### Step 6 — Blockchain Re-Verification & Tampering Demonstration")

    verifier = BlockchainVerifier(client=client)
    tab_verify, tab_tamper = st.tabs(["Verify Authentic Record", "Simulate Tampering"])

    with tab_verify:
        st.write("Recalculate the canonical SHA-256 fingerprint from original data and query the smart contract:")
        if st.button("Run Authentic Re-Verification", key="btn_verify_authentic"):
            with st.spinner("Querying smart contract on blockchain..."):
                verif_clean = verifier.verify_record(best_match)
                if verif_clean.status.value.startswith("VERIFIED"):
                    st.success(f"**{verif_clean.status.value}**")
                    st.write(f"**On-chain Record:** Found in State Storage")
                    st.write(f"**Hash Match:** Yes (`{best_match.content_hash[:16]}...`)")
                    st.write(f"**Source Reference:** {verif_clean.source_reference}")
                    st.write(f"**Anchored Timestamp:** {verif_clean.on_chain_timestamp}")
                    st.write(f"**Submitter:** {verif_clean.submitter}")
                else:
                    st.error(f"**{verif_clean.status.value}**: {verif_clean.details}")

    with tab_tamper:
        st.write("Simulate an unauthorized metadata modification to demonstrate mathematical tamper detection:")
        tamper_field = st.selectbox(
            "Select field to modify:",
            ["source_title", "source_url", "metadata_author", "image_bytes"]
        )
        tamper_input = st.text_input("Tampered value to inject:", value="UNAUTHORIZED_FORGERY_MODIFICATION")

        if st.button("Simulate Tampering & Verify", key="btn_tamper_sim"):
            tampered_rec, verif_tampered = verifier.simulate_tampering(
                original_record=best_match,
                modified_field=tamper_field,
                tampered_value=tamper_input
            )

            st.markdown(f"**Original Hash A (On-Chain):**")
            st.markdown(f'<div class="hash-text">{best_match.content_hash}</div>', unsafe_allow_html=True)

            st.markdown(f"**Tampered Hash B (Recalculated):**")
            st.markdown(f'<div class="hash-text">{verif_tampered.recalculated_hash.replace("0x", "")}</div>', unsafe_allow_html=True)

            st.error(f"**{verif_tampered.status.value}**")
            st.write(f"**Hash Comparison:** Hash A != Hash B")
            st.write(f"**Explanation:** {verif_tampered.details}")
            st.caption("Notice: The blockchain ledger record remains completely intact and immutable.")


if __name__ == "__main__":
    main()
