"""Genuine Reverse Image Search and Public Content Discovery Providers.

Architecturally separates:
1. Live Reverse Image Search (Production Provider via SerpAPI Google Lens + Temporary Image Staging)
2. Offline Demo Mode (Explicit Local Corpus Simulation for offline/air-gapped evaluation)
"""
import os
import io
import time
import base64
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timezone
import requests
import cv2
import numpy as np

from models.match_record import CandidateResult
from utils.config import get_config

USER_AGENT = "FaceTraceChain/2.0 (Reverse Image Search & Blockchain Verification Pipeline)"


class SearchProviderError(Exception):
    """Custom exception raised when search provider fails or is misconfigured."""
    pass


class TempImageUploader:
    """Stages local image files or raw bytes to a temporary public URL for reverse search engines."""

    FREEIMAGE_API = "https://freeimage.host/api/1/upload"
    FREEIMAGE_KEY = "6d207e02198a847aa98d0a2a901485a5"
    TMPFILES_API = "https://tmpfiles.org/api/v1/upload"

    @staticmethod
    def _prepare_image_bytes(image_input: Union[str, Path, bytes]) -> Tuple[bytes, str]:
        """Loads and optionally resizes image to ensure rapid, reliable temporary upload."""
        raw_bytes: bytes
        filename = "search_face.jpg"

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            filename = p.name or "search_face.jpg"
            raw_bytes = p.read_bytes()
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Optimize image if larger than 2MB or large dimensions for fast upload
        try:
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                max_dim = 1200
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                # Encode as high-quality JPEG
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return buf.tobytes(), "search_face.jpg"
        except Exception:
            pass

        return raw_bytes, filename

    def upload_image(self, image_input: Union[str, Path, bytes]) -> str:
        """Uploads image to temporary public hosting and returns direct image URL."""
        # 1. If already a public web URL, return directly
        if isinstance(image_input, (str, Path)):
            s = str(image_input).strip()
            if s.startswith("http://") or s.startswith("https://"):
                return s

        img_bytes, filename = self._prepare_image_bytes(image_input)

        # 2. Try Primary: Freeimage.host API
        try:
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
            data = {
                "key": self.FREEIMAGE_KEY,
                "action": "upload",
                "source": b64_data,
                "format": "json"
            }
            headers = {"User-Agent": USER_AGENT}
            resp = requests.post(self.FREEIMAGE_API, data=data, headers=headers, timeout=12)
            if resp.status_code == 200:
                res_json = resp.json()
                img_url = res_json.get("image", {}).get("url")
                if img_url:
                    return str(img_url)
        except Exception as e:
            # Fall through to fallback provider
            pass

        # 3. Try Fallback: TmpFiles.org API
        try:
            files = {"file": (filename, io.BytesIO(img_bytes), "image/jpeg")}
            resp = requests.post(self.TMPFILES_API, files=files, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                orig_url = data.get("data", {}).get("url", "")
                if orig_url:
                    # Convert to direct raw download URL
                    direct_url = orig_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    return direct_url
        except Exception as e:
            raise SearchProviderError(
                f"Failed to stage image to temporary hosting: {e}. "
                "Ensure internet connectivity or switch to '--mode offline' for local evaluation."
            )

        raise SearchProviderError(
            "Could not stage image to temporary hosting service. "
            "Please check internet access or use '--mode offline'."
        )


class BaseSearchProvider(ABC):
    """Abstract base class for all search adapters."""

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Human-readable name of the discovery mode."""
        pass

    @abstractmethod
    def search_by_image(
        self,
        image_input: Union[str, Path, bytes],
        max_results: int = 5
    ) -> List[CandidateResult]:
        """Dynamically search public sources at runtime for candidate matches."""
        pass


class LiveReverseImageSearchProvider(BaseSearchProvider):
    """Production provider: Genuine Reverse Image Search via Google Lens (SerpAPI).
    
    Accepts local images or public URLs, automatically stages local files via 
    TempImageUploader, and performs a real visual similarity query.
    """

    SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else get_config().search_api_key
        self.uploader = TempImageUploader()

    @property
    def mode_name(self) -> str:
        return "Live Reverse Image Search (Google Lens API)"

    def search_by_image(
        self,
        image_input: Union[str, Path, bytes],
        max_results: int = 5
    ) -> List[CandidateResult]:
        # Required runtime proof logs
        input_desc = str(image_input) if isinstance(image_input, (str, Path)) else f"bytes({len(image_input)} B)"
        print(f"[SEARCH] Input image received: {input_desc}")

        if not self.api_key:
            raise SearchProviderError(
                "SerpAPI key not found. Please set SEARCH_API_KEY in your .env file or "
                "supply it via the interface.\n"
                "-> To run offline without an API key, use '--mode offline' for the local demo corpus."
            )

        print("[SEARCH] Creating search request")
        public_url = self.uploader.upload_image(image_input)

        params = {
            "engine": "google_lens",
            "url": public_url,
            "api_key": self.api_key,
        }

        try:
            resp = requests.get(self.SERPAPI_ENDPOINT, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            err_msg = str(e)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[PROTECTED_KEY]")
            raise SearchProviderError(f"SerpAPI Google Lens query failed: {err_msg}")

        print("[SEARCH] Reverse image search executed via SerpAPI Google Lens")

        visual_matches = data.get("visual_matches", [])
        candidates: List[CandidateResult] = []

        for i, match in enumerate(visual_matches[:max_results]):
            title = match.get("title", f"Web Candidate #{i+1}")
            link = match.get("link", "")
            thumb = match.get("thumbnail", "")
            domain = urllib.parse.urlparse(link).netloc or match.get("source", "web")

            candidates.append(CandidateResult(
                title=title,
                source_url=link or f"https://{domain}",
                image_url=thumb or link,
                source_domain=domain,
                discovery_timestamp=datetime.now(timezone.utc).isoformat(),
                content_identifier=f"google_lens_{i+1}",
                is_live=True,
                metadata={
                    "source": match.get("source", ""),
                    "position": match.get("position", i + 1),
                    "search_engine": "google_lens"
                }
            ))

        print(f"[SEARCH] Candidate results received: {len(candidates)} matches found")
        return candidates


class OfflineDemoCorpusProvider(BaseSearchProvider):
    """Explicit Local Public Corpus Simulation for offline, air-gapped, or demonstration runs.
    
    Clearly demarcated as an offline simulation — never claimed to be a live web search.
    """

    def __init__(self, corpus_dir: Optional[Union[str, Path]] = None):
        self.corpus_dir = Path(corpus_dir or get_config().sample_data_dir / "public_corpus")
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

    @property
    def mode_name(self) -> str:
        return "Offline Demo Mode (Local Corpus Simulation)"

    def search_by_image(
        self,
        image_input: Union[str, Path, bytes],
        max_results: int = 5
    ) -> List[CandidateResult]:
        input_desc = str(image_input) if isinstance(image_input, (str, Path)) else f"bytes({len(image_input)} B)"
        print("[SEARCH] Running in OFFLINE DEMO MODE (Local Corpus Simulation)")
        print(f"[SEARCH] Input image received: {input_desc}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [p for p in self.corpus_dir.iterdir() if p.suffix.lower() in valid_exts]

        candidates: List[CandidateResult] = []
        for i, p in enumerate(files[:max_results], start=1):
            title = p.stem.replace("_", " ").title()
            file_url = f"file:///{p.resolve().as_posix()}"
            candidates.append(CandidateResult(
                title=f"{title} (Local Demo Corpus)",
                source_url=file_url,
                image_url=file_url,
                source_domain="local.public_corpus",
                discovery_timestamp=datetime.now(timezone.utc).isoformat(),
                content_identifier=f"offline_demo_{p.stem}",
                is_live=False,
                metadata={
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                    "provenance": "offline_simulation_corpus"
                }
            ))

        print(f"[SEARCH] Candidate results loaded from local demo corpus: {len(candidates)} candidates")
        return candidates


def get_search_provider(
    mode: Optional[str] = None,
    api_key: Optional[str] = None,
    corpus_dir: Optional[Union[str, Path]] = None
) -> BaseSearchProvider:
    """Factory to instantiate search provider based on explicit mode."""
    active_mode = (mode or get_config().search_provider or "live").lower().strip()

    if active_mode in {"live", "serpapi", "google_lens", "lens", "online"}:
        return LiveReverseImageSearchProvider(api_key=api_key)
    elif active_mode in {"offline", "local", "demo", "simulation"}:
        return OfflineDemoCorpusProvider(corpus_dir=corpus_dir)
    else:
        # Default to live reverse image search
        print(f"[WARNING] Unrecognized search mode '{active_mode}'. Defaulting to 'live'.")
        return LiveReverseImageSearchProvider(api_key=api_key)
