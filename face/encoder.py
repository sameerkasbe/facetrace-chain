"""Face feature extraction and normalized embedding generation.

Provides backward-compatible FaceEncoder delegating to the modular
BaseFaceRecognizer engine (ArcFace primary, SFace fallback).
"""
import numpy as np
from typing import Optional
from utils.config import get_config
from .detector import FaceDetectionResult
from .recognizer import get_face_recognizer, BaseFaceRecognizer


class FaceEncoder:
    """Extracts normalized facial embeddings using the active recognition engine (ArcFace / SFace)."""

    def __init__(self, engine_name: Optional[str] = None):
        self.config = get_config()
        self.engine: BaseFaceRecognizer = get_face_recognizer(engine_name)
        self.is_ready = True
        # Expose underlying recognizer if SFace for legacy compatibility
        if hasattr(self.engine, "recognizer"):
            self.recognizer = self.engine.recognizer

    @property
    def embedding_dim(self) -> int:
        return self.engine.embedding_dim

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def align_face(self, face_result: FaceDetectionResult) -> np.ndarray:
        """Aligns and crops the face using the active recognition engine's alignment logic."""
        return self.engine.align_face(face_result)

    def generate_embedding(self, face_result: FaceDetectionResult, verbose: bool = False) -> np.ndarray:
        """Generates a normalized embedding vector using the active recognition engine."""
        return self.engine.generate_embedding(face_result, verbose=verbose)
