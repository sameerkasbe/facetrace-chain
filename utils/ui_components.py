"""Reusable UI components and layout renderers for the modernized Streamlit interface."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

from models.match_record import CandidateResult, MatchRecord
from face.quality import FaceQualityReport
from utils.classifier import ContentType, ResultCategory


def load_css():
    """Injects custom stylesheet from assets/styles.css into Streamlit."""
    css_path = Path(__file__).resolve().parent.parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header(system_status: Optional[Dict[str, str]] = None):
    """Renders the cyber-AI gradient application header with live system readiness indicators."""
    status = system_status or {
        "detector": "YuNet Face Detector Active",
        "recognition": "ArcFace / InsightFace Active",
        "search": "Google Lens Live",
        "blockchain": "EVM Connected"
    }

    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-title-row">
                <div class="hero-brand">
                    <div class="hero-icon">🛡️</div>
                    <div>
                        <h1 class="hero-title">FACETRACE CHAIN</h1>
                        <p class="hero-subtitle">AI-Powered Reverse Face Discovery & Cryptographic Blockchain Verification</p>
                    </div>
                </div>
                <div class="system-status-pills">
                    <div class="status-pill">
                        <span class="status-dot dot-green"></span>
                        <span>{status.get('detector', 'Detector Online')}</span>
                    </div>
                    <div class="status-pill">
                        <span class="status-dot dot-purple"></span>
                        <span>{status.get('recognition', 'ArcFace Active')}</span>
                    </div>
                    <div class="status-pill">
                        <span class="status-dot dot-cyan"></span>
                        <span>{status.get('search', 'Search Ready')}</span>
                    </div>
                    <div class="status-pill">
                        <span class="status-dot dot-green"></span>
                        <span>{status.get('blockchain', 'Blockchain Ready')}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_tracker(stage_states: Dict[str, Dict[str, Any]]):
    """Renders visual 7-stage interactive pipeline tracker with real timing metrics.
    
    stage_states format:
    {
        "detection": {"status": "complete", "time": "32 ms"},
        "encoding": {"status": "complete", "time": "58 ms"},
        "search": {"status": "processing", "time": "1.2 s"},
        "verification": {"status": "waiting", "time": ""},
        "evidence": {"status": "waiting", "time": ""},
        "blockchain": {"status": "waiting", "time": ""},
        "reverification": {"status": "waiting", "time": ""}
    }
    """
    stages = [
        ("detection", "1. Face Detection", "🔍"),
        ("encoding", "2. ArcFace Encoding", "🧬"),
        ("search", "3. Reverse Image Search", "🌐"),
        ("verification", "4. Candidate Verification", "⚖️"),
        ("evidence", "5. Evidence Package", "📦"),
        ("blockchain", "6. Blockchain Storage", "⛓️"),
        ("reverification", "7. Re-Verification", "🛡️"),
    ]

    steps_html = []
    for key, label, icon in stages:
        data = stage_states.get(key, {"status": "waiting", "time": ""})
        status = data.get("status", "waiting")
        elapsed = data.get("time", "")

        if status == "complete":
            step_class = "step-complete"
            badge = "✓ Complete"
        elif status == "processing":
            step_class = "step-active"
            badge = "◉ Processing"
        else:
            step_class = "step-waiting"
            badge = "○ Waiting"

        time_html = f'<div class="step-time">{elapsed}</div>' if elapsed else ""

        steps_html.append(f"""
        <div class="pipeline-step {step_class}">
            <div class="step-icon">{icon}</div>
            <div class="step-label">{label}</div>
            <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 2px;">{badge}</div>
            {time_html}
        </div>
        """)

    tracker_html = f"""
    <div class="pipeline-wrapper">
        <div class="pipeline-steps">
            {''.join(steps_html)}
        </div>
    </div>
    """
    st.markdown(tracker_html, unsafe_allow_html=True)


def render_quality_card(quality: FaceQualityReport):
    """Renders the face quality breakdown card."""
    qual_color = "var(--status-verified)" if quality.overall_quality == "HIGH" else ("var(--status-warning)" if quality.overall_quality == "MEDIUM" else "var(--status-tampered)")

    warnings_html = ""
    if quality.warnings:
        warn_items = "".join([f"<li>⚠️ {w}</li>" for w in quality.warnings])
        warnings_html = f"""
        <div style="margin-top: 0.6rem; padding: 0.5rem; background: rgba(245, 158, 11, 0.1); border-left: 3px solid var(--status-warning); border-radius: 4px; font-size: 0.78rem;">
            <ul style="margin: 0; padding-left: 1.2rem; color: #FDE68A;">{warn_items}</ul>
        </div>
        """

    st.markdown(
        f"""
        <div class="glass-card" style="margin-top: 0.5rem;">
            <div class="card-title">
                <span>Biometric Input Face Quality</span>
                <span style="font-size: 0.8rem; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(255,255,255,0.06); color: {qual_color}; font-weight: 700;">
                    {quality.overall_quality} QUALITY ({int(quality.quality_score * 100)}%)
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 0.6rem; font-size: 0.8rem; margin-top: 0.4rem;">
                <div style="background: rgba(15,23,42,0.6); padding: 0.4rem 0.6rem; border-radius: 6px;">
                    <div style="color: var(--text-secondary); font-size: 0.7rem;">Resolution</div>
                    <div style="font-weight: 600;">{quality.resolution[0]}x{quality.resolution[1]} px</div>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 0.4rem 0.6rem; border-radius: 6px;">
                    <div style="color: var(--text-secondary); font-size: 0.7rem;">Focus / Blur</div>
                    <div style="font-weight: 600;">{quality.blur_status} ({quality.blur_score})</div>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 0.4rem 0.6rem; border-radius: 6px;">
                    <div style="color: var(--text-secondary); font-size: 0.7rem;">Illumination</div>
                    <div style="font-weight: 600;">{quality.brightness_status} ({quality.brightness_score})</div>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 0.4rem 0.6rem; border-radius: 6px;">
                    <div style="color: var(--text-secondary); font-size: 0.7rem;">Face Dimensions</div>
                    <div style="font-weight: 600;">{quality.face_size_status}</div>
                </div>
            </div>
            {warnings_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_candidate_card(cand: CandidateResult, rank: int, is_verified_section: bool = False):
    """Renders modern responsive candidate card with platform, content badge, and similarity."""
    sim_pct = int(round(cand.similarity_score * 100))
    conf_pct = int(round(cand.overall_confidence * 100)) if cand.overall_confidence > 0 else sim_pct

    badge_class = "badge-high" if cand.similarity_score >= 0.65 else ("badge-medium" if cand.similarity_score >= 0.45 else "badge-low")
    
    # Platform badge color
    plat_class = "badge-purple" if cand.platform in {"Instagram", "Facebook", "TikTok", "YouTube"} else "badge-cyan"

    # Category icon
    c_icon = "👤" if cand.category == ResultCategory.PROFILES.value else ("🎬" if cand.category == ResultCategory.REELS_VIDEOS.value else "📄")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-card-header">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-weight: 700; color: var(--accent-cyan);">#{rank}</span>
                    <span class="badge-tag {plat_class}">{cand.platform}</span>
                    <span class="badge-tag" style="background: rgba(255,255,255,0.06); color: var(--text-secondary);">{c_icon} {cand.content_type.upper()}</span>
                </div>
                <div>
                    <span class="badge-tag {badge_class}">{cand.confidence_badge or cand.similarity_status}</span>
                </div>
            </div>
            <div class="result-card-body">
                <div class="result-card-title" title="{cand.title}">{cand.title}</div>
                <div class="result-meta-row" style="font-size: 0.75rem; color: var(--text-muted);">
                    <span>🌐 {cand.source_domain}</span>
                    {f"<span>• By: {cand.author[:20]}</span>" if cand.author else ""}
                </div>
                <div class="similarity-box">
                    <div>
                        <div style="font-size: 0.68rem; color: var(--text-secondary);">FACE SIMILARITY</div>
                        <div class="sim-val">{sim_pct}%</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: var(--text-secondary);">CONFIDENCE</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #E2E8F0;">{conf_pct}%</div>
                    </div>
                </div>
                {f'<div class="result-explanation">{cand.match_explanation}</div>' if cand.match_explanation else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Action button underneath card
    col1, col2 = st.columns([1, 1])
    with col1:
        if cand.source_url:
            st.link_button("🌐 Open Source", cand.source_url, use_container_width=True)
    with col2:
        if cand.image_url:
            st.link_button("🖼️ View Media", cand.image_url, use_container_width=True)
