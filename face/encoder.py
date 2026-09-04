"""Face feature extraction and 128-dimensional normalized embedding generation."""
import urllib.request
from typing import Optional
import cv2
import numpy as np
from utils.config import get_config
from .detector import FaceDetectionResult

class FaceEncoder:
    """Extracts 128-dimensional normalized facial embeddings using OpenCV SFace (MobileFaceNet/ArcFace architecture)."""

    def __init__(self):
        self.config = get_config()
        self._ensure_model_exists()
        self._init_encoder()

    def _ensure_model_exists(self):
        sface_path = self.config.sface_path
        if not sface_path.exists() or sface_path.stat().st_size < 1000:
            sface_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading SFace feature extraction model to {sface_path}...")
            urllib.request.urlretrieve(self.config.sface_model_url, str(sface_path))

    def _init_encoder(self):
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.config.sface_path),
                config=""
            )
            self.is_ready = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SFace recognizer: {e}")

    def align_face(self, face_result: FaceDetectionResult) -> np.ndarray:
        """Aligns and crops the face to standard 112x112 input dimensions."""
        if face_result.raw_face_entry is not None:
            aligned = self.recognizer.alignCrop(
                face_result.image_bgr,
                face_result.raw_face_entry
            )
            return aligned
        else:
            crop = face_result.cropped_face
            return cv2.resize(crop, (112, 112))

    def generate_embedding(self, face_result: FaceDetectionResult, verbose: bool = False) -> np.ndarray:
        """Generates a 128-dimensional L2-normalized embedding vector.
        
        Outputs formatted progress:
          Face detected successfully
          Face confidence: XX%
          Generating facial embedding...
          Embedding generated
        """
        if verbose:
            print("Face detected successfully")
            print(f"Face confidence: {face_result.confidence_pct}%")
            print("Generating facial embedding...")

        aligned = self.align_face(face_result)
        raw_emb = self.recognizer.feature(aligned)
        
        # Flatten to 1D array
        emb = raw_emb.flatten()
        
        # Explicit L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        else:
            emb = np.zeros_like(emb)

        if verbose:
            print("Embedding generated")

        return emb
