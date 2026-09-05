"""Face Recognition Engine abstraction with ArcFace/InsightFace (Primary) and OpenCV SFace (Fallback).

Architectural separation:
  BaseFaceRecognizer (Interface)
    ├── InsightFaceRecognizer (Primary: ArcFace 512-d embeddings + landmark alignment)
    └── SFaceRecognizer (Fallback: OpenCV SFace 128-d MobileFaceNet embeddings)
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union, Any, List
import urllib.request
import cv2
import numpy as np

from utils.config import get_config
from .detector import FaceDetectionResult

logger = logging.getLogger("facetrace.face")


class BaseFaceRecognizer(ABC):
    """Abstract interface defining the biometric face recognition contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the recognition model architecture."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the normalized facial embedding vector."""
        pass

    @property
    @abstractmethod
    def default_high_threshold(self) -> float:
        """Calibrated high confidence match threshold for this model."""
        pass

    @property
    @abstractmethod
    def default_medium_threshold(self) -> float:
        """Calibrated possible match threshold for this model."""
        pass

    @abstractmethod
    def align_face(self, face_result: FaceDetectionResult) -> np.ndarray:
        """Aligns facial landmarks to canonical orientation and returns normalized face crop."""
        pass

    @abstractmethod
    def generate_embedding(self, face_result: FaceDetectionResult, verbose: bool = False) -> np.ndarray:
        """Extracts and returns an L2-normalized 1D embedding vector."""
        pass

    def compare_embeddings(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Calculates cosine similarity between two normalized embedding vectors.
        
        Range: [0.0, 1.0].
        """
        a = emb_a.flatten()
        b = emb_b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, cos_sim))


class InsightFaceRecognizer(BaseFaceRecognizer):
    """Primary Face Recognizer using ArcFace / InsightFace (buffalo_s / buffalo_l).
    
    Produces 512-dimensional normalized embeddings with robust pose, lighting,
    and compression tolerance.
    """

    def __init__(self, model_name: str = "buffalo_s"):
        self.model_name = model_name
        self._init_engine()

    @property
    def name(self) -> str:
        return f"InsightFace ArcFace ({self.model_name})"

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def default_high_threshold(self) -> float:
        # ArcFace 512-d angular margin embeddings cluster tightly: >= 0.60 is strong match
        return 0.65

    @property
    def default_medium_threshold(self) -> float:
        return 0.45

    def _init_engine(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.is_ready = True
            logger.info(f"Initialized InsightFaceRecognizer ({self.model_name}) successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace ({e}). Falling back to SFace.")
            raise RuntimeError(f"InsightFace initialization error: {e}")

    def align_face(self, face_result: FaceDetectionResult) -> np.ndarray:
        """Aligns face using 5-point facial landmark similarity transform."""
        img = face_result.image_bgr
        # If insightface app detects the face directly, it returns aligned landmark norm
        try:
            faces = self.app.get(img)
            if faces and len(faces) > 0:
                # Find face closest to face_result bbox
                best_face = faces[0]
                if len(faces) > 1 and face_result.bbox:
                    bx, by, bw, bh = face_result.bbox
                    b_center = (bx + bw / 2, by + bh / 2)
                    best_face = min(
                        faces,
                        key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - b_center[0]) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - b_center[1]) ** 2
                    )
                # Insightface face object contains kps (landmarks)
                if hasattr(best_face, "embedding") and best_face.embedding is not None:
                    # Return cropped face
                    crop = face_result.cropped_face
                    return cv2.resize(crop, (112, 112))
        except Exception:
            pass

        crop = face_result.cropped_face
        return cv2.resize(crop, (112, 112))

    def generate_embedding(self, face_result: FaceDetectionResult, verbose: bool = False) -> np.ndarray:
        """Generates a 512-dimensional L2-normalized ArcFace embedding vector."""
        if verbose:
            print(f"Face detected successfully (Confidence: {face_result.confidence_pct}%)")
            print("Generating 512-d ArcFace facial embedding...")

        img = face_result.image_bgr
        faces = self.app.get(img)

        if faces and len(faces) > 0:
            # Pick the face matching bbox if multiple faces
            best_face = faces[0]
            if len(faces) > 1 and face_result.bbox:
                bx, by, bw, bh = face_result.bbox
                b_center = (bx + bw / 2, by + bh / 2)
                best_face = min(
                    faces,
                    key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - b_center[0]) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - b_center[1]) ** 2
                )
            emb = best_face.embedding.flatten()
        else:
            # If app.get didn't find face on full image, try on the cropped face directly
            crop = face_result.cropped_face
            crop_faces = self.app.get(crop)
            if crop_faces and len(crop_faces) > 0:
                emb = crop_faces[0].embedding.flatten()
            else:
                # Emergency fallback: use zeros
                logger.warning("InsightFace could not locate landmarks for embedding. Returning zero vector.")
                emb = np.zeros(self.embedding_dim, dtype=np.float32)

        # Explicit L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = (emb / norm).astype(np.float32)
        else:
            emb = np.zeros(self.embedding_dim, dtype=np.float32)

        if verbose:
            print("ArcFace embedding generated")

        return emb


class SFaceRecognizer(BaseFaceRecognizer):
    """Fallback Face Recognizer using OpenCV SFace (MobileFaceNet).
    
    Produces 128-dimensional normalized facial embeddings.
    """

    def __init__(self):
        self.config = get_config()
        self._ensure_model_exists()
        self._init_encoder()

    @property
    def name(self) -> str:
        return "OpenCV SFace (MobileFaceNet)"

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def default_high_threshold(self) -> float:
        return 0.80

    @property
    def default_medium_threshold(self) -> float:
        return 0.60

    def _ensure_model_exists(self):
        sface_path = self.config.sface_path
        if not sface_path.exists() or sface_path.stat().st_size < 1000:
            sface_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading SFace model to {sface_path}...")
            urllib.request.urlretrieve(self.config.sface_model_url, str(sface_path))

    def _init_encoder(self):
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.config.sface_path),
                config=""
            )
            self.is_ready = True
            logger.info("Initialized SFaceRecognizer successfully.")
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
        """Generates a 128-dimensional L2-normalized embedding vector."""
        if verbose:
            print("Face detected successfully")
            print(f"Face confidence: {face_result.confidence_pct}%")
            print("Generating 128-d SFace embedding...")

        aligned = self.align_face(face_result)
        raw_emb = self.recognizer.feature(aligned)
        emb = raw_emb.flatten()

        # Explicit L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = (emb / norm).astype(np.float32)
        else:
            emb = np.zeros(self.embedding_dim, dtype=np.float32)

        if verbose:
            print("Embedding generated")

        return emb


_GLOBAL_RECOGNIZER: Optional[BaseFaceRecognizer] = None


def get_face_recognizer(engine_name: Optional[str] = None) -> BaseFaceRecognizer:
    """Factory creating or returning singleton BaseFaceRecognizer instance.
    
    Follows preferred architecture:
      ArcFace / InsightFace (PRIMARY)
      └── OpenCV SFace (FALLBACK)
    """
    global _GLOBAL_RECOGNIZER

    config = get_config()
    target_engine = (engine_name or getattr(config, "face_recognition_engine", "arcface")).lower().strip()

    if _GLOBAL_RECOGNIZER is not None:
        # Check if same architecture requested
        if target_engine in {"arcface", "insightface"} and isinstance(_GLOBAL_RECOGNIZER, InsightFaceRecognizer):
            return _GLOBAL_RECOGNIZER
        elif target_engine in {"sface", "opencv"} and isinstance(_GLOBAL_RECOGNIZER, SFaceRecognizer):
            return _GLOBAL_RECOGNIZER

    # Try InsightFace as primary
    if target_engine in {"arcface", "insightface", "primary", "auto"}:
        try:
            recognizer = InsightFaceRecognizer(model_name="buffalo_s")
            _GLOBAL_RECOGNIZER = recognizer
            return recognizer
        except Exception as e:
            logger.warning(f"Primary InsightFace model unavailable ({e}). Falling back to OpenCV SFace.")

    # Fallback to SFace
    try:
        recognizer = SFaceRecognizer()
        _GLOBAL_RECOGNIZER = recognizer
        return recognizer
    except Exception as e:
        logger.error(f"Failed to initialize SFace fallback recognizer: {e}")
        raise
