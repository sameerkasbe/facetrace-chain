# FaceTrace Chain — Face Discovery and Blockchain Verification Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green.svg)](https://opencv.org/)
[![Solidity](https://img.shields.io/badge/Solidity-0.8.20-363636.svg)](https://soliditylang.org/)
[![Web3.py](https://img.shields.io/badge/Web3.py-7.8+-orange.svg)](https://web3py.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-21%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **FaceTrace Chain** is an end-to-end computer vision and decentralized verification pipeline. It takes an input face image, performs single-face detection and 128-dimensional deep feature encoding, executes genuine reverse image search across public web sources, verifies visual similarity algorithmically, creates a canonical tamper-evident cryptographic fingerprint, and anchors the proof on an Ethereum-compatible blockchain for immutable provenance and tamper detection.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [The Solution](#3-the-solution)
4. [Pipeline Architecture Diagram](#4-pipeline-architecture-diagram)
5. [Key Features](#5-key-features)
6. [Technology Stack](#6-technology-stack)
7. [Face Detection Methodology](#7-face-detection-methodology)
8. [Search Methodology & Mode Separation](#8-search-methodology--mode-separation)
9. [Blockchain Architecture](#9-blockchain-architecture)
10. [Smart Contract Explanation](#10-smart-contract-explanation)
11. [Installation & Setup Verification](#11-installation--setup-verification)
12. [Environment Setup](#12-environment-setup)
13. [How to Run CLI](#13-how-to-run-cli)
14. [How to Run UI](#14-how-to-run-ui)
15. [Blockchain Deployment Instructions](#15-blockchain-deployment-instructions)
16. [Verification & Tamper Demonstration](#16-verification--tamper-demonstration)
17. [Automated Test Suite](#17-automated-test-suite)
18. [Known Limitations](#18-known-limitations)
19. [Privacy and Ethical Considerations](#19-privacy-and-ethical-considerations)

---

## 1. Project Overview

In digital media, unauthorized image re-uploading, deepfakes, copyright infringement, and digital impersonation pose significant threats. Traditional digital rights management often relies on centralized registries or fragile watermarks that can be easily stripped.

**FaceTrace Chain** bridges deep learning computer vision and decentralized smart contracts to establish verifiable, tamper-evident provenance for publicly discovered portrait imagery without ever exposing raw biometric data on-chain.

---

## 2. Problem Statement

1. **Ephemeral Digital Provenance**: Content distributed across the web lacks immutable records tying discoveries to cryptographic timestamps and author references.
2. **Fragility of Metadata**: Exif data, image URLs, and page titles are trivial to edit, tamper with, or forge.
3. **Biometric Privacy Dilemma**: Storing face biometric vectors or high-resolution facial images in public databases or distributed ledgers violates privacy laws (e.g. GDPR, BIPA).
4. **Mocked Demonstrations**: Many prototype systems hardcode search outcomes or use generic text queries instead of genuine reverse image search using the input face.

---

## 3. The Solution

FaceTrace Chain resolves these challenges through a strict, transparent architecture:

- **Zero Biometrics on Blockchain**: Only canonical 32-byte SHA-256 fingerprints of verifiable content are stored on-chain.
- **Genuine Reverse Image Search**: Production mode takes the uploaded local face image, automatically stages it to temporary public hosting, and runs genuine visual search via SerpAPI Google Lens.
- **Explicit Mode Separation**: The system explicitly differentiates between `LIVE SEARCH` (real runtime external search using the input image) and `OFFLINE DEMO MODE` (local public domain corpus simulation).
- **Strict Face Input Validation**: Validates image integrity and enforces that exactly one primary face is present, preventing ambiguous multi-face selection.
- **Algorithmic Probabilistic Matching**: Uses 128-dimensional dense neural embeddings (SFace/MobileFaceNet architecture) and cosine similarity to grade candidate matches.
- **Tamper-Evident Fingerprint**: Deterministic hashing binds the canonical URL, image byte hash, title, and normalized metadata into a tamper-evident digest.
- **Cryptographic Re-Verification & Tamper Detection**: Proves integrity by comparing re-calculated fingerprints against the immutable blockchain ledger.

---

## 4. Pipeline Architecture Diagram

```
                        [ FACE IMAGE INPUT ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          [0 Faces Found]                [>1 Faces Found]
       "Unable to detect face"       "Multiple faces detected.
                                      Provide 1 primary face"
                                 │
                                 ▼ (Exactly 1 Face Validated)
                   [ FACE DETECTION & CROPPING ]
               OpenCV YuNet DNN / Haar Cascade Fallback
                                 │
                                 ▼
                     [ 128-D FACE ENCODING ]
             OpenCV SFace (MobileFaceNet/ArcFace)
               L2-Normalized Dense Feature Vector
                                 │
                                 ▼
         [ REVERSE IMAGE SEARCH / DISCOVERY SELECTION ]
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
[LIVE REVERSE SEARCH]                          [OFFLINE DEMO MODE]
SerpAPI Google Lens Engine                     Local Public Corpus Simulation
• Auto-stages local image to public URL        • Evaluates sample domain portraits
• Real visual search on input face             • Transparent provenance tagging
• Real-time candidate extraction               • Zero external API required
         └───────────────────────┬───────────────────────┘
                                 ▼
                  [ CANDIDATE FACE MATCHING ]
              Download Image ──> Detect Face Crop
              ──> SFace Embedding ──> Cosine Sim
              ──> Classify: High / Possible / None
                                 │
                                 ▼
                [ CANONICAL SHA-256 FINGERPRINT ]
         SHA256(canon_url + id + title + img_hash + meta)
                                 │
                                 ▼
                    [ BLOCKCHAIN STORAGE ]
          Solidity FaceVerification.sol on Ganache
            storeVerification(bytes32, sourceRef)
                                 │
                                 ▼
                 [ RE-VERIFICATION & TAMPER DEMO ]
            Recalculate Hash ──> Query Smart Contract
            Match   ──> VERIFIED ✓
            Altered ──> TAMPERING DETECTED ✗
```

---

## 5. Key Features

- **Strict Validation Gate**: Automatically blocks images with 0 faces or multiple faces with helpful, descriptive messages.
- **Genuine Reverse Image Search**: Uses the input image itself to perform Google Lens visual search via temporary staging (`TempImageUploader`).
- **Runtime Execution Proof Logs**:
  ```
  [SEARCH] Input image received
  [SEARCH] Creating search request
  [SEARCH] Reverse image search executed
  [SEARCH] Candidate results received
  [VERIFY] Downloading candidate images
  [VERIFY] Comparing facial embeddings
  ```
- **Transparent Probabilistic Metric**:
  - `≥ 85%`: High Similarity Match
  - `65% – 84%`: Possible Match
  - `< 65%`: No Match
  - Accompanied by probabilistic algorithmic disclaimer (denoting visual correlation, not absolute biological certainty).
- **Canonical Hashing Protocol**: Eliminates URL tracking parameters (`utm_*`) and canonicalizes JSON metadata for byte-level determinism.
- **Solidity Smart Contract**: `FaceVerification.sol` written in Solidity `^0.8.20` featuring re-entrancy prevention, error reversions, and event logging.
- **Dual Interface**:
  - Full CLI pipeline (`python main.py --image <path> --mode <live|offline>`).
  - Professional Streamlit web app (`python -m streamlit run app.py`).
- **One-Click Tampering Demonstration**: Demonstrates the security value of the blockchain ledger by altering title or metadata and detecting the cryptographic mismatch.

---

## 6. Technology Stack

| Layer | Component | Version / Library | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11 / 3.12 / 3.13 | Core runtime |
| **Face Detection** | OpenCV YuNet | `cv2.FaceDetectorYN` | Deep learning face detection |
| **Face Recognition** | OpenCV SFace | `cv2.FaceRecognizerSF` | 128-d ArcFace/MobileFaceNet embeddings |
| **Similarity** | Cosine Similarity | `numpy`, `scipy` | Algorithmic vector comparison |
| **Reverse Image Search** | Google Lens / SerpAPI | `requests` | Genuine visual similarity search |
| **Temporary Staging** | Freeimage / TmpFiles | `requests` | Public URL staging for reverse search |
| **Blockchain** | Local Ethereum Node | Ganache v7.9 (port 8545) | Local EVM network |
| **Smart Contract** | Solidity | `^0.8.20` | Immutable record storage |
| **Compiler** | py-solc-x | `solcx 2.0+` | Automatic compilation & ABI extraction |
| **Web3 Client** | Web3.py | `web3 7.8+` | EVM contract interaction & event tracking |
| **Hashing** | Cryptographic SHA-256 | `hashlib` | Tamper-evident content fingerprinting |
| **User Interface** | Streamlit | `streamlit 1.40+` | Clean, lightweight inspection GUI |
| **Testing** | Pytest | `pytest 8.0+` | 21 automated regression tests |

---

## 7. Face Detection Methodology

FaceTrace Chain employs OpenCV's **YuNet** (a lightweight convolutional deep neural network) backed by an automatic **Haar Cascade** fallback:

1. **Resolution Normalization**: Input images are dynamically scaled to an optimal receptive field (`max(h, w) <= 800`) while preserving aspect ratio.
2. **Single Primary Face Enforcement**:
   - If 0 faces are detected, the system immediately returns:  
     `"Unable to detect a face in this image."`
   - If >1 faces are detected, the system immediately returns:  
     `"Multiple faces detected. Please provide an image containing one clear primary face."`
3. **Face Alignment**: 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner) are detected and passed to `recognizer.alignCrop()` for standard 112×112 alignment.
4. **Normalized Embedding**: `FaceRecognizerSF` outputs a 128-dimensional dense feature representation normalized with the Euclidean L2 norm:
   $$\mathbf{e}_{\text{norm}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$

---

## 8. Search Methodology & Mode Separation

The system clearly separates two distinct operational modes:

### Mode 1: Live Reverse Image Search (`--mode live`)
- **Primary Production Provider**: `LiveReverseImageSearchProvider`
- **Workflow**:
  1. Accepts input image (local file path, bytes, or URL).
  2. If local, `TempImageUploader` stages the image to temporary public hosting (Freeimage.host / TmpFiles.org) and obtains a live direct image URL.
  3. Queries SerpAPI Google Lens endpoint (`https://serpapi.com/search.json?engine=google_lens&url=<public_url>`).
  4. Parses genuine visual matches from across the web.
  5. Returns candidates tagged with `is_live=True`.

### Mode 2: Offline Demo Mode (`--mode offline`)
- **Simulation Provider**: `OfflineDemoCorpusProvider`
- **Workflow**:
  1. Scans `sample_data/public_corpus/` containing authorized public domain portraits.
  2. Clearly labels each candidate as `(Local Demo Corpus)` and tags `is_live=False`, `source_domain="local.public_corpus"`.
  3. Never pretends to be a live web search.
  4. Requires zero API keys and functions in completely air-gapped environments.

---

## 9. Blockchain Architecture

FaceTrace Chain anchors fingerprints to an Ethereum-compatible blockchain:

- **Local Network**: Ganache development node running on `http://127.0.0.1:8545` (Chain ID 1337).
- **Deterministic Accounts**: Uses Ganache deterministic keys for consistent demo deployment.
- **Fail-Safe Fallback**: If an external RPC daemon is unavailable, the client automatically falls back to an in-process `EthereumTesterProvider` so evaluations never crash.

---

## 10. Smart Contract Explanation

Located in `blockchain/FaceVerification.sol`:

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

    mapping(bytes32 => VerificationRecord) private _records;
    uint256 public recordCount;

    event ContentStored(bytes32 indexed contentHash, string sourceReference, uint256 timestamp, address indexed submitter);

    function storeVerification(bytes32 contentHash, string calldata sourceReference) external returns (bool);
    function verifyContent(bytes32 contentHash) external view returns (bool exists, string memory sourceReference, uint256 timestamp, address submitter);
    function isContentVerified(bytes32 contentHash) external view returns (bool);
}
```

- **Gas Optimized**: The fingerprint is stored as a native `bytes32` value (32 bytes).
- **Duplicate Prevention**: Prevents overwriting previously confirmed fingerprints.
- **Provenance Attribution**: Stores the canonical public URL, block confirmation timestamp, and submitter address.

---

## 11. Installation & Setup Verification

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Node.js (for Ganache local blockchain)

### Step 1: Clone and Activate Environment
```powershell
cd "d:\goa h task 3"
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Setup Verification
A standalone verification script is provided to validate your entire environment in seconds:
```powershell
python verify_setup.py
```

Sample verification output:
```
====================================================================
  FaceTrace Chain — System & Environment Setup Verification
====================================================================

Category                 Status   Details
--------------------------------------------------------------------
Python Environment       ✓ PASS   Python 3.13.3
Dependencies             ✓ PASS   OpenCV is installed
Dependencies             ✓ PASS   NumPy is installed
Dependencies             ✓ PASS   Web3.py is installed
Dependencies             ✓ PASS   py-solc-x is installed
Dependencies             ✓ PASS   Requests is installed
Dependencies             ✓ PASS   Streamlit is installed
Dependencies             ✓ PASS   Pytest is installed
Dependencies             ✓ PASS   python-dotenv is installed
AI Models                ✓ PASS   YuNet ONNX detector found (yunet.onnx)
AI Models                ✓ PASS   SFace ONNX recognition model found (sface.onnx)
Blockchain RPC           ✓ PASS   Connected to RPC at http://127.0.0.1:8545 (Block #6)
Smart Contract           ✓ PASS   FaceVerification contract active at 0x... (Total records: 3)
Search: Offline Demo     ✓ PASS   Demo corpus ready (2 images in public_corpus)
Search: Live Search      ✓ PASS   SerpAPI Google Lens key configured
Sample Test Data         ✓ PASS   All standard test portraits present
--------------------------------------------------------------------

[RESULT] ✓ System verified successfully and ready for operation!
```

---

## 12. Environment Setup

Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```

Configuration in `.env`:
```ini
# Search Mode: 'live' (Google Lens reverse search) or 'offline' (Local corpus simulation)
SEARCH_MODE=live

# SerpAPI API key for live Google Lens reverse search
SEARCH_API_KEY=your_serpapi_key_here

# Blockchain Configuration (Default local Ganache)
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
BLOCKCHAIN_PRIVATE_KEY=
BLOCKCHAIN_CONTRACT_ADDRESS=

# Face matching similarity threshold (0.0 to 1.0)
MATCH_THRESHOLD=0.85
```

---

## 13. How to Run CLI

### Start Ganache (if not already running)
```powershell
npx -y ganache --port 8545 --deterministic --networkId 1337
```

### Option A: Offline Demo Mode with Tamper Demonstration
```powershell
python main.py --image sample_data/sample_portrait_a.jpg --mode offline --tamper-demo
```

### Option B: Live Reverse Image Search (Google Lens)
```powershell
python main.py --image sample_data/sample_portrait_a.jpg --mode live --api-key <YOUR_SERPAPI_KEY>
```

### Option C: Test Multi-Face Input Rejection (Error Handling Gate)
```powershell
python main.py --image sample_data/sample_multi_face.jpg
```

---

## 14. How to Run UI

Launch the Streamlit interface:
```powershell
python -m streamlit run app.py
```

Open your browser to `http://localhost:8501` (or indicated port).

### UI Features:
1. **Search Mode Switcher**: Select between `Live Reverse Image Search (Google Lens API)` and `Offline Demo Mode (Local Corpus Simulation)`.
2. **SerpAPI Key Input**: Easily input or override your API key directly in the sidebar.
3. **Interactive Image Selector**: Switch between custom upload and pre-loaded test portraits.
4. **Visual Inspection**: Visualizes detected bounding box, 5 landmarks, and confidence.
5. **Runtime Execution Proof Logs**: Real-time terminal log display of all search and verification steps.
6. **Candidate Scoring Matrix**: Displays candidates, domain, similarity %, and classification badge.
7. **On-Chain Dashboard**: Live transaction hashes, block numbers, gas used, and submitter address.
8. **Interactive Tamper Demo**: Two-tab panel allowing authentic re-verification and controlled metadata tampering.

---

## 15. Blockchain Deployment Instructions

To start Ganache:
```powershell
npx -y ganache --port 8545 --deterministic --networkId 1337
```

To compile and deploy `FaceVerification.sol`:
```powershell
python blockchain/deploy.py
```

The script compiles the contract using `solcx` 0.8.20, deploys it to Ganache, updates `BLOCKCHAIN_CONTRACT_ADDRESS` in `.env`, and saves the ABI artifact to `blockchain/build/FaceVerification.json`.

---

## 16. Verification & Tamper Demonstration

### 1. Authentic On-Chain Re-Verification
The canonical SHA-256 fingerprint is recalculated from the discovered source metadata and image bytes:
$$\text{Hash}_{\text{recalc}} = \text{SHA256}(\text{canon\_url} \,\|\, \text{content\_id} \,\|\, \text{title} \,\|\, \text{SHA256}(\text{image}) \,\|\, \text{canon\_meta})$$
The client queries `verifyContent(Hash_recalc)`:
$$\rightarrow \mathbf{VERIFIED \ \checkmark}$$

### 2. Controlled Tampering Demonstration
To demonstrate tamper detection, an unauthorized change is made to the title or author metadata:
- Original Hash A: `be370df5f7eeffab98b2d75d6e1a20357873e0452226c810feb74f63aec6d9f7`
- Tampered Hash B: `68360259271a669e66dee0416e368876f1cf3cf24d765a177809a74f37b42456`
- Result: **Hash A != Hash B**
$$\rightarrow \mathbf{TAMPERING \ DETECTED \ \times}$$

The immutable on-chain record remains untouched, proving that unauthorized alteration invalidates the cryptographic proof.

---

## 17. Automated Test Suite

Run the full pytest suite (21 unit and integration tests):
```powershell
pytest -q
```

Output:
```
.....................                                                    [100%]
21 passed in 1.82s
```

Test coverage includes:
- Face detection, single-face gate, and multi-face rejection (`tests/test_matching.py`)
- SFace 128-d cosine similarity and threshold classification (`tests/test_matching.py`)
- Deterministic canonical SHA-256 fingerprinting (`tests/test_hashing.py`)
- Smart contract deployment, storage, and tamper detection (`tests/test_blockchain.py`)
- `TempImageUploader` staging and fallback (`tests/test_search.py`)
- `OfflineDemoCorpusProvider` simulation provenance (`tests/test_search.py`)
- `LiveReverseImageSearchProvider` and SerpAPI Google Lens parsing (`tests/test_search.py`)

---

## 18. Known Limitations

- **Search Coverage**: Web discovery depends on indexed public search engines and reachable public image URLs. Content behind authentication walls or CAPTCHAs cannot be indexed.
- **Extreme Pose Variation**: Severe side profiles (>60 degrees) or heavy facial occlusions may reduce embedding similarity below the configured threshold.
- **Gas Costs**: Local Ganache chains use default gas estimation; public mainnet deployment would require EIP-1559 gas price monitoring.

---

## 19. Privacy and Ethical Considerations

> [!IMPORTANT]
> **Biometric Privacy Safeguard**: Biometric face vectors and raw face imagery are **never** committed to the blockchain. Only canonical SHA-256 cryptographic fingerprints are anchored on-chain.
>
> **Authorized Demonstration Scope**: This project is designed strictly for authorized demonstrations, digital rights tracking, and content attribution. It is **not** intended for mass surveillance, unauthorized profiling, or non-consensual tracking.
>
> **Probabilistic Classification**: Facial recognition is algorithmic and probabilistic. Matching results denote statistical visual correlation and must not be interpreted as absolute biological identity certainty.

---

## License

This project is open-source and released under the [MIT License](LICENSE).
