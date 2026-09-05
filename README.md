# FaceTrace Chain — AI Face Discovery & Blockchain Verification Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace%20512--d-purple.svg)](https://github.com/deepinsight/insightface)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SFace-green.svg)](https://opencv.org/)
[![Solidity](https://img.shields.io/badge/Solidity-0.8.20-363636.svg)](https://soliditylang.org/)
[![Web3.py](https://img.shields.io/badge/Web3.py-8.0-orange.svg)](https://web3py.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62+-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-34%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **FaceTrace Chain** is a production-grade biometric discovery and decentralized verification application. It takes an input face image (via file upload or live camera capture), enforces strict single-face detection (YuNet), evaluates biometric image quality (focus/blur, illumination, resolution), generates normalized deep facial embeddings using **ArcFace / InsightFace (Primary: 512-d)** with **OpenCV SFace (Fallback: 128-d)**, executes a **genuine dynamic reverse image search** (Google Lens via SerpAPI), classifies discovered content into **Profiles**, **Posts & Images**, and **Reels & Videos**, performs **two-stage candidate verification** with multi-factor confidence scoring, generates a canonical **Search Evidence Package**, computes a deterministic SHA-256 fingerprint, anchors the fingerprint to an Ethereum smart contract, and provides a dedicated **Cryptographic Re-Verification & Tampering Audit** system.

---

## Table of Contents

1. [Problem Statement & Design Principles](#1-problem-statement--design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Biometric Face Recognition Engine (ArcFace & SFace)](#3-biometric-face-recognition-engine-arcface--sface)
4. [Face Quality Checks & Landmark Alignment](#4-face-quality-checks--landmark-alignment)
5. [Genuine Dynamic Reverse Image Search](#5-genuine-dynamic-reverse-image-search)
6. [Content & URL Classification](#6-content--url-classification)
7. [Two-Stage Candidate Verification & Multi-Threshold Confidence](#7-two-stage-candidate-verification--multi-threshold-confidence)
8. [Search Evidence Package & Deterministic Hashing](#8-search-evidence-package--deterministic-hashing)
9. [Blockchain Architecture & Smart Contract](#9-blockchain-architecture--smart-contract)
10. [Dedicated Re-Verification & Tampering Demonstration](#10-dedicated-re-verification--tampering-demonstration)
11. [Modern Reactive Gradient UI](#11-modern-reactive-gradient-ui)
12. [Environment Variables & Configuration](#12-environment-variables--configuration)
13. [Installation & Setup](#13-installation--setup)
14. [How to Run (Streamlit & CLI)](#14-how-to-run-streamlit--cli)
15. [Automated Test Suite](#15-automated-test-suite)
16. [Privacy, Legal & Ethical Considerations](#16-privacy-legal--ethical-considerations)

---

## 1. Problem Statement & Design Principles

In the era of synthetic media and rapid online distribution, verifying the authenticity, original discovery context, and provenance of images found online is a major challenge:

- **Fragility of Metadata**: Exif data, image URLs, and web page titles are easily forged, altered, or deleted.
- **Biometric Privacy Risks**: Storing high-dimensional biometric facial embeddings or raw face photos on public distributed ledgers directly violates data protection regulations (GDPR Art. 9, BIPA). FaceTrace Chain only stores non-invertible SHA-256 hashes of canonical evidence packages.
- **Elimination of Offline / Mock Modes**: The system performs **strictly genuine dynamic discovery**. Offline demo modes and local corpora have been completely removed from the primary search flow.
- **Probabilistic Verification**: The system clearly distinguishes between pure facial embedding similarity and composite verification confidence, avoiding unscientific claims of absolute identity certainty.

---

## 2. Architecture Overview

```text
               INPUT SOURCE (Upload or Live Camera Capture)
                                    ↓
       [STEP 1] FACE DETECTION & VALIDATION (OpenCV YuNet DNN)
                • Single-face constraint validation (0 faces -> error, >1 faces -> error)
                • 5 facial landmark localization (eyes, nose, mouth corners)
                                    ↓
       [STEP 2] BIOMETRIC QUALITY ANALYSIS & NORMALIZED ALIGNMENT
                • Resolution check (minimum width/height metrics)
                • Laplacian variance blur detection (sharp vs blurry)
                • Illumination / brightness assessment (dark vs overexposed)
                • Similarity transform landmark alignment
                                    ↓
       [STEP 3] DEEP EMBEDDING GENERATION (BaseFaceRecognizer)
                • Primary: ArcFace / InsightFace (512-d normalized vector)
                • Fallback: OpenCV SFace MobileFaceNet (128-d normalized vector)
                                    ↓
       [STEP 4] GENUINE REVERSE IMAGE SEARCH
                • Primary: Google Lens via SerpAPI (automatic staging via FreeImage/TmpFiles/ImgBB/Cloudinary)
                • Secondary: Dynamic Public Web Media API (Wikimedia Commons)
                                    ↓
       [STEP 5] CONTENT & URL CLASSIFICATION (ResultClassifier)
                • Profiles (Instagram, Facebook, X, YouTube channels, LinkedIn)
                • Posts & Images (web articles, feed posts, standalone photos)
                • Reels & Videos (Instagram Reels, YouTube Shorts/videos, TikToks)
                                    ↓
       [STEP 6] TWO-STAGE CANDIDATE VERIFICATION
                • Stage 1: Fast embedding screening across all candidates
                • Stage 2: Deep verification on Top-K with face alignment & quality scoring
                • Separation of Discovered Results vs AI-Verified Matches
                                    ↓
       [STEP 7] EVIDENCE PACKAGE & DETERMINISTIC SHA-256 HASHING
                • Canonical JSON serialization (RFC 8785 key ordering, no whitespace)
                • Deterministic SHA-256 fingerprint generation
                                    ↓
       [STEP 8] IMMUTABLE BLOCKCHAIN ANCHORING
                • Smart contract storage on Ethereum / EVM testnet
                • Emits VerificationRecorded event with timestamp and block number
                                    ↓
       [STEP 9] CRYPTOGRAPHIC RE-VERIFICATION & TAMPER AUDIT
                • Re-verify record against immutable on-chain state (🟢 AUTHENTIC)
                • Interactive tamper demonstration with hash divergence proof (🔴 TAMPERING DETECTED)
```

---

## 3. Biometric Face Recognition Engine (ArcFace & SFace)

FaceTrace Chain implements a clean abstraction layer (`BaseFaceRecognizer`) supporting multiple neural architectures:

```text
               BaseFaceRecognizer (Interface)
                        ├── InsightFaceRecognizer (PRIMARY: ArcFace 512-d)
                        └── SFaceRecognizer (FALLBACK: OpenCV SFace 128-d)
```

### Primary Engine: ArcFace / InsightFace
- **Architecture**: Deep additive angular margin loss (ArcFace) utilizing the `buffalo_s` / `buffalo_l` pretrained models.
- **Embedding Dimensions**: 512-dimensional L2-normalized feature vectors.
- **Advantages**: State-of-the-art real-world verification accuracy under variable lighting, non-frontal poses, slight rotation, heavy web compression artifacts, and low-resolution thumbnails.
- **Default Calibrated Thresholds**: High Confidence Match $\ge 0.65$; Possible Match $\ge 0.45$.

### Fallback Engine: OpenCV SFace
- **Architecture**: Lightweight MobileFaceNet trained with CosFace/ArcFace margin loss (`FaceRecognizerSF`).
- **Embedding Dimensions**: 128-dimensional L2-normalized feature vectors.
- **Advantages**: Built directly into OpenCV without external dependencies.
- **Default Calibrated Thresholds**: High Confidence Match $\ge 0.80$; Possible Match $\ge 0.60$.

The factory function `get_face_recognizer()` automatically loads the primary ArcFace engine and seamlessly falls back to SFace if dependencies or models are unavailable.

---

## 4. Face Quality Checks & Landmark Alignment

Before generating embeddings or matching candidates, `FaceQualityAnalyzer` analyzes the input face and candidate crops:

- **Resolution**: Evaluates bounding box dimensions ($w \times h$). Minimum recommended: $100 \times 100$ px; crops below $60 \times 60$ px trigger resolution warnings.
- **Blur / Focus**: Computes Laplacian variance on grayscale pixel luminance:
  - $\ge 100$: **Sharp**
  - $45 - 99$: **Moderate Focus** (warning issued)
  - $< 45$: **Blurry** (warning issued)
- **Illumination / Brightness**: Evaluates mean grayscale luminance:
  - $< 45$: **Underexposed / Dark**
  - $> 215$: **Overexposed / Harsh Highlights**
  - $45 - 215$: **Good**
- **Composite Quality Score**: Normalized composite score ($0.0$ to $1.0$) categorized as `HIGH`, `MEDIUM`, or `LOW`.
- **Landmark Alignment**: 5-point facial landmarks (eyes, nose tip, mouth corners) are aligned via similarity transform to standard coordinate orientation prior to embedding extraction.

---

## 5. Genuine Dynamic Reverse Image Search

Offline demo mode and preselected local corpora have been **completely removed**. The search flow performs 100% genuine dynamic discovery:

1. **Temporary Image Staging**: `TempImageUploader` stages local image files or raw bytes to a temporary public URL:
   - **ImgBB API**: If `IMGBB_API_KEY` is configured.
   - **Cloudinary API**: If `CLOUDINARY_*` credentials are configured.
   - **FreeImage.host / TmpFiles.org**: Zero-config automatic fallback.
2. **Primary Provider: Google Lens via SerpAPI**: Executes real neural visual similarity searches across indexed web pages, returning direct links, platform metadata, titles, and high-resolution media URLs.
3. **Secondary Provider: Dynamic Public Web Search**: Queries Wikimedia Commons open repository via dynamic media API.

---

## 6. Content & URL Classification

Discovered search results are classified by `ResultClassifier` based on URL patterns and metadata:

- **Profiles**: Social profile pages and channels.
  - Instagram: `instagram.com/<username>/`
  - YouTube: `youtube.com/@<channel>`, `/c/`, `/channel/`
  - TikTok: `tiktok.com/@<username>`
  - Facebook: `facebook.com/<username>`, `/pages/`, `/people/`
  - X/Twitter: `x.com/<username>`, `twitter.com/<username>`
  - LinkedIn: `linkedin.com/in/<name>`, `/company/<name>`
- **Posts & Images**: Feed posts, standalone articles, and direct image files.
  - Instagram: `/p/<id>`
  - X/Twitter: `/status/<id>`
  - Facebook: `/posts/`, `/photos/`
  - Direct Images: `.jpg`, `.png`, `.webp`, `.bmp`
- **Reels & Videos**: Short-form videos and video streams.
  - Instagram: `/reel/<id>`, `/reels/<id>`, `/tv/<id>`
  - YouTube: `/watch?v=<id>`, `/shorts/<id>`, `youtu.be/<id>`
  - TikTok: `/video/<id>`
  - Facebook: `/watch`, `/reel/`, `/videos/`

In the Streamlit interface, results are presented across 3 dedicated tabs:
- `[👤 Profiles]`
- `[📄 Posts & Images]`
- `[🎬 Reels & Videos]`

---

## 7. Two-Stage Candidate Verification & Multi-Threshold Confidence

Rather than performing expensive alignment and deep quality checks on hundreds of raw candidates, `CandidateCollector` executes a two-stage verification pipeline:

1. **Stage 1 (Fast Screening)**: Downloads candidate thumbnails, runs fast face detection, and computes preliminary embedding cosine similarity across all discovered candidates.
2. **Stage 2 (Deep Verification)**: Selects Top-K candidates (configurable via `TOP_K_CANDIDATES`, default 5):
   - Computes comprehensive face quality metrics (`FaceQualityAnalyzer`).
   - Applies landmark alignment and recomputes refined embeddings.
   - Computes multi-factor composite confidence:
     $$\text{Overall Confidence} = 0.75 \times \text{Similarity} + 0.15 \times \text{Quality} + 0.10 \times \text{Detection}$$
   - Re-ranks candidates based on verification status and composite confidence.

### Configurable Match Thresholds
- $\text{Similarity} \ge \text{High Threshold}$: `🟢 HIGH CONFIDENCE MATCH`
- $\text{Similarity} \ge \text{Medium Threshold}$: `🟡 POSSIBLE MATCH`
- $\text{Similarity} < \text{Medium Threshold}$: `🔴 NO RELIABLE MATCH`

---

## 8. Search Evidence Package & Deterministic Hashing

When a candidate meets the verification threshold, a structured `EvidencePackage` is compiled:

```json
{
  "case_id": "case_20260904_123456_a1b2c3d4",
  "evidence_version": "2.0",
  "verification_timestamp": "2026-09-04T12:34:56.789012+00:00",
  "input_face": {
    "image_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "face_detected": true
  },
  "search_metadata": {
    "provider": "Live Reverse Image Search (Google Lens via SerpAPI)",
    "timestamp": "2026-09-04T12:34:57.123456+00:00",
    "result_count": 15
  },
  "matched_content": {
    "title": "Historical Portrait Record",
    "source_url": "https://example.org/verified_portrait",
    "image_url": "https://example.org/portrait.jpg",
    "domain": "example.org",
    "platform": "Web",
    "content_type": "post",
    "category": "Posts & Images"
  },
  "algorithmic_verification": {
    "similarity_score": 0.924,
    "overall_confidence": 0.941,
    "threshold": 0.65,
    "result": "MATCH"
  }
}
```

- **Deterministic Serialization**: Keys are sorted lexicographically, and whitespace is stripped (RFC 8785).
- **Cryptographic Fingerprint**: SHA-256 produces a unique 64-character hex string (and a corresponding `bytes32` for Solidity).

---

## 9. Blockchain Architecture & Smart Contract

The `FaceVerification.sol` smart contract records the evidence package hash on-chain:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FaceVerification {
    struct VerificationRecord {
        bytes32 contentHash;
        string sourceReference;
        uint256 timestamp;
        address submitter;
        bool exists;
    }

    uint256 public recordCount;

    event ContentStored(
        bytes32 indexed contentHash,
        string sourceReference,
        uint256 timestamp,
        address indexed submitter
    );

    function storeVerification(bytes32 contentHash, string calldata sourceReference) external returns (bool success);
    function verifyContent(bytes32 contentHash) external view returns (bool exists, string memory sourceReference, uint256 timestamp, address submitter);
    function isContentVerified(bytes32 contentHash) external view returns (bool);
}
```

The application automatically connects to a local Ganache/Anvil node or falls back seamlessly to an in-memory EVM tester provider (`EthereumTesterProvider`).

---

## 10. Dedicated Re-Verification & Tampering Demonstration

The system includes a dedicated re-verification interface:

1. **Verify Record**: Upload or paste any Evidence Package JSON. The system canonically serializes the JSON, computes the SHA-256 fingerprint, and queries the smart contract.
   - 🟢 **AUTHENTIC RECORD**: On-chain timestamp, block number, submitter, and stored URL confirmed.
   - 🔴 **RECORD NOT FOUND / TAMPERED**: Hash does not exist on-chain.
2. **Interactive Tamper Demonstration**: Allows users to alter individual fields (such as `similarity_score`, `source_url`, or `timestamp`) and observe the cryptographic avalanche effect:
   - Shows original hash vs tampered hash.
   - Highlights the exact altered key.
   - Confirms that the tampered record fails on-chain verification.

---

## 11. Modern Reactive Gradient UI

The Streamlit UI has been completely redesigned with a modern cybersecurity + AI SaaS aesthetic:

- **Color System**: Deep navy background (`#07090E`), subtle border glows (`#06B6D4`), blue-to-purple gradient accents, verified green (`#10B981`), and warning red (`#EF4444`).
- **Styles Organization**: External stylesheet located at `assets/styles.css` with modular UI helpers in `utils/ui_components.py`.
- **Hero Header**: Live system readiness status pills for Face Detector, ArcFace Engine, Search Provider, and Blockchain.
- **Input Source Card**: Clearly separated options for **Upload Image** (drag-and-drop) and **Capture Using Camera** (with live preview and retake).
- **Interactive Visual Pipeline Tracker**: Real-time progress updates across 7 stages with actual measured millisecond and second timings.
- **Categorized Results**: Discovered candidates separated from AI-verified matches, organized into Profiles, Posts/Images, and Reels/Videos.

---

## 12. Environment Variables & Configuration

Create a `.env` file in the project root based on `.env.example`:

```env
# Reverse Image Search Configuration
# PRIMARY: 'live' (Google Lens via SerpAPI); SECONDARY: 'web' (Wikimedia API)
SEARCH_MODE=live
SERPAPI_KEY=your_serpapi_key_here
SEARCH_API_KEY=your_serpapi_key_here

# Optional Image Staging Providers (for Google Lens staging)
# If left blank, zero-config staging (Freeimage / TmpFiles) is used automatically.
IMGBB_API_KEY=
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

# Face Recognition Engine Architecture
# Options: 'arcface' (PRIMARY: InsightFace ArcFace 512-d) or 'sface' (FALLBACK: OpenCV SFace 128-d)
FACE_RECOGNITION_ENGINE=arcface

# Multi-Level Match Thresholds (ArcFace defaults: 0.65 / 0.45; SFace defaults: 0.80 / 0.60)
FACE_MATCH_HIGH_THRESHOLD=0.65
FACE_MATCH_MEDIUM_THRESHOLD=0.45
TOP_K_CANDIDATES=5

# Blockchain Configuration (Local Ganache, Anvil, Hardhat, or In-Memory EVM)
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
BLOCKCHAIN_PRIVATE_KEY=
BLOCKCHAIN_CONTRACT_ADDRESS=
```

---

## 13. Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Internet connection (for model downloads and reverse search)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup & Diagnostics
Run the automated environment verifier:
```bash
python verify_setup.py
```

---

## 14. How to Run (Streamlit & CLI)

### Running the Web Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Running the CLI Pipeline

**Live Reverse Image Search (Google Lens)**:
```bash
python main.py --image sample_data/sample_portrait_a.jpg --mode live --engine arcface --tamper-demo
```

**Dynamic Public Web Media Search**:
```bash
python main.py --image sample_data/sample_portrait_a.jpg --mode web --engine arcface --top-k 5
```

**OpenCV SFace Fallback Engine**:
```bash
python main.py --image sample_data/sample_portrait_a.jpg --mode web --engine sface
```

**Standalone Evidence File Verification**:
```bash
python main.py --verify-file evidence.json
```

**Deploy Smart Contract**:
```bash
python main.py --deploy
```

---

## 15. Automated Test Suite

Run the full pytest suite:
```bash
pytest -q
```
Expected output:
```text
..................................                                       [100%]
34 passed in ~24s
```

The test suite covers:
- URL and content classification (profiles, posts, images, reels, videos)
- Category mapping
- Biometric face quality assessment (resolution, focus/blur, illumination)
- ArcFace 512-d embedding generation and L2 normalization
- SFace 128-d embedding generation and fallback
- Cosine similarity and multi-factor confidence scoring
- Single-face enforcement and error validation
- Two-stage candidate screening and ranking
- Smart contract deployment, evidence hashing, on-chain recording, and tamper detection

---

## 16. Privacy, Legal & Ethical Considerations

> **IMPORTANT NOTICE**: Facial similarity scores generated by this system are **probabilistic mathematical metrics** computed from high-dimensional deep learning embeddings. They indicate algorithmic visual correlation rather than absolute biological or legal identity confirmation.

- **Biometric Data Protection**: Raw biometric vectors, raw photographs, and sensitive biometric markers are **never** stored on the public blockchain. Only one-way SHA-256 fingerprints of evidence packages are anchored on-chain.
- **Search Scope**: Reverse search is restricted to publicly indexed web content and open media repositories.
- **Responsible Use**: This system is designed for provenance verification, research, and synthetic media auditing. It should not be utilized for unauthorized mass surveillance or profiling.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
