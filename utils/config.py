"""Configuration management for FaceTrace Chain."""
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass
class AppConfig:
    """Application configuration parameters."""
    root_dir: Path
    models_dir: Path
    sample_data_dir: Path
    
    # Face Recognition Engine
    face_recognition_engine: str = "arcface"  # "arcface" (primary) or "sface" (fallback)
    face_match_high_threshold: float = 0.65
    face_match_medium_threshold: float = 0.45
    top_k_candidates: int = 5

    # Search Configuration
    search_provider: str = "live"
    search_mode: str = "live"
    search_api_key: str = ""
    serpapi_key: str = ""
    nvidia_api_key: str = ""
    imgbb_api_key: str = ""
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    gemini_api_key: str = ""
    
    # Blockchain
    blockchain_rpc_url: str = "http://127.0.0.1:8545"
    blockchain_private_key: str = ""
    blockchain_contract_address: str = ""
    
    # Legacy threshold alias
    match_threshold: float = 0.65
    
    # Model URLs for SFace fallback and YuNet
    yunet_model_url: str = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    sface_model_url: str = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    )

    @property
    def yunet_path(self) -> Path:
        return self.models_dir / "yunet.onnx"

    @property
    def sface_path(self) -> Path:
        return self.models_dir / "sface.onnx"


def get_config() -> AppConfig:
    """Retrieve validated application configuration."""
    load_dotenv(ROOT_DIR / ".env", override=True)
    
    models_dir = ROOT_DIR / "models_cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data_dir = ROOT_DIR / "sample_data"
    sample_data_dir.mkdir(parents=True, exist_ok=True)

    engine = os.getenv("FACE_RECOGNITION_ENGINE", "arcface").lower().strip()
    
    # Default thresholds calibrated to recognition engine
    default_high = 0.65 if engine in {"arcface", "insightface"} else 0.80
    default_med = 0.45 if engine in {"arcface", "insightface"} else 0.60

    try:
        high_thresh = float(os.getenv("FACE_MATCH_HIGH_THRESHOLD", os.getenv("MATCH_THRESHOLD", str(default_high))))
    except ValueError:
        high_thresh = default_high

    try:
        med_thresh = float(os.getenv("FACE_MATCH_MEDIUM_THRESHOLD", str(default_med)))
    except ValueError:
        med_thresh = default_med

    try:
        top_k = int(os.getenv("TOP_K_CANDIDATES", "5"))
    except ValueError:
        top_k = 5

    raw_contract = os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS", "").strip()
    if raw_contract.lower() in {"none", "null", "false", ""}:
        contract_addr = ""
    else:
        contract_addr = raw_contract

    serp_key = os.getenv("SERPAPI_KEY", os.getenv("SEARCH_API_KEY", "")).strip()

    return AppConfig(
        root_dir=ROOT_DIR,
        models_dir=models_dir,
        sample_data_dir=sample_data_dir,
        face_recognition_engine=engine,
        face_match_high_threshold=high_thresh,
        face_match_medium_threshold=med_thresh,
        top_k_candidates=top_k,
        search_provider=os.getenv("SEARCH_MODE", os.getenv("SEARCH_PROVIDER", "live")).lower().strip(),
        search_mode=os.getenv("SEARCH_MODE", os.getenv("SEARCH_PROVIDER", "live")).lower().strip(),
        search_api_key=serp_key,
        serpapi_key=serp_key,
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
        imgbb_api_key=os.getenv("IMGBB_API_KEY", "").strip(),
        cloudinary_cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "").strip(),
        cloudinary_api_key=os.getenv("CLOUDINARY_API_KEY", "").strip(),
        cloudinary_api_secret=os.getenv("CLOUDINARY_API_SECRET", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        blockchain_rpc_url=os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545").strip(),
        blockchain_private_key=os.getenv("BLOCKCHAIN_PRIVATE_KEY", "").strip(),
        blockchain_contract_address=contract_addr,
        match_threshold=high_thresh,
    )
