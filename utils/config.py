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
    
    # Search
    search_provider: str
    search_mode: str
    search_api_key: str
    nvidia_api_key: str
    
    # Blockchain
    blockchain_rpc_url: str
    blockchain_private_key: str
    blockchain_contract_address: str
    
    # Matching
    match_threshold: float
    
    # Model URLs
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
    models_dir = ROOT_DIR / "models_cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data_dir = ROOT_DIR / "sample_data"
    sample_data_dir.mkdir(parents=True, exist_ok=True)

    match_thresh_str = os.getenv("MATCH_THRESHOLD", "0.85")
    try:
        match_threshold = float(match_thresh_str)
    except ValueError:
        match_threshold = 0.85

    return AppConfig(
        root_dir=ROOT_DIR,
        models_dir=models_dir,
        sample_data_dir=sample_data_dir,
        search_provider=os.getenv("SEARCH_MODE", os.getenv("SEARCH_PROVIDER", "live")).lower().strip(),
        search_mode=os.getenv("SEARCH_MODE", os.getenv("SEARCH_PROVIDER", "live")).lower().strip(),
        search_api_key=os.getenv("SEARCH_API_KEY", "").strip(),
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
        blockchain_rpc_url=os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545").strip(),
        blockchain_private_key=os.getenv("BLOCKCHAIN_PRIVATE_KEY", "").strip(),
        blockchain_contract_address=os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS", "").strip(),
        match_threshold=match_threshold,
    )
