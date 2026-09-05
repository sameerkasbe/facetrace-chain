"""Face image quality assessment and validation metrics.

Evaluates:
- Face resolution (width & height)
- Focus / blur (Laplacian variance)
- Illumination / brightness (grayscale mean)
- Overall composite quality score and actionable warnings
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np


@dataclass
class FaceQualityReport:
    """Structured report containing detailed facial quality metrics."""
    overall_quality: str  # "HIGH", "MEDIUM", "LOW"
    quality_score: float  # Normalized 0.0 to 1.0
    resolution: Tuple[int, int]  # (width, height)
    resolution_status: str  # "Good", "Acceptable", "Low Resolution"
    blur_score: float  # Laplacian variance
    blur_status: str  # "Sharp", "Moderate Focus", "Blurry"
    brightness_score: float  # Mean pixel luminance (0-255)
    brightness_status: str  # "Good", "Dark", "Overexposed"
    face_size_status: str  # "Good", "Small Face"
    warnings: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Returns True if quality is sufficient for reliable verification."""
        return self.overall_quality in {"HIGH", "MEDIUM"}

    def summary_text(self) -> str:
        """User-friendly summary text."""
        parts = [
            f"Quality: {self.overall_quality}",
            f"Resolution: {self.resolution[0]}x{self.resolution[1]} ({self.resolution_status})",
            f"Sharpness: {self.blur_status}",
            f"Illumination: {self.brightness_status}"
        ]
        return " | ".join(parts)


class FaceQualityAnalyzer:
    """Analyzes face crops and full images to evaluate biometric quality."""

    # Threshold constants
    MIN_FACE_DIM_GOOD = 100
    MIN_FACE_DIM_ACCEPTABLE = 60

    BLUR_SHARP_THRESHOLD = 100.0
    BLUR_ACCEPTABLE_THRESHOLD = 45.0

    BRIGHTNESS_LOW = 45.0
    BRIGHTNESS_HIGH = 215.0

    @classmethod
    def analyze(
        cls,
        image_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> FaceQualityReport:
        """Computes comprehensive biometric quality metrics on the detected face region."""
        img_h, img_w = image_bgr.shape[:2]

        if bbox is not None:
            x, y, w, h = bbox
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            face_crop = image_bgr[y1:y2, x1:x2]
        else:
            face_crop = image_bgr
            w, h = img_w, img_h

        # Fallback if crop is invalid or empty
        if face_crop is None or face_crop.size == 0 or w <= 0 or h <= 0:
            return FaceQualityReport(
                overall_quality="LOW",
                quality_score=0.1,
                resolution=(w, h),
                resolution_status="Invalid",
                blur_score=0.0,
                blur_status="Blurry",
                brightness_score=0.0,
                brightness_status="Dark",
                face_size_status="Small Face",
                warnings=["Invalid or empty face region detected."]
            )

        # Convert to grayscale for metrics
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        warnings: List[str] = []
        score_penalty = 0.0

        # 1. Resolution & Face Size Check
        min_dim = min(w, h)
        if min_dim >= cls.MIN_FACE_DIM_GOOD:
            res_status = "Good"
            face_size_status = "Good"
        elif min_dim >= cls.MIN_FACE_DIM_ACCEPTABLE:
            res_status = "Acceptable"
            face_size_status = "Acceptable"
            score_penalty += 0.15
            warnings.append(f"Moderate face size ({w}x{h}px). Standard verification is optimal above 100x100px.")
        else:
            res_status = "Low Resolution"
            face_size_status = "Small Face"
            score_penalty += 0.40
            warnings.append(f"Small face crop ({w}x{h}px). Sub-60px crops may reduce embedding precision.")

        # 2. Blur / Sharpness Check (Laplacian Variance)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_val >= cls.BLUR_SHARP_THRESHOLD:
            blur_status = "Sharp"
        elif blur_val >= cls.BLUR_ACCEPTABLE_THRESHOLD:
            blur_status = "Moderate Focus"
            score_penalty += 0.15
            warnings.append("Mild blur or motion softness detected in face region.")
        else:
            blur_status = "Blurry"
            score_penalty += 0.35
            warnings.append("Significant blur detected in face. High-frequency features may be degraded.")

        # 3. Brightness / Illumination Check
        brightness_val = float(np.mean(gray))
        if brightness_val < cls.BRIGHTNESS_LOW:
            brightness_status = "Dark"
            score_penalty += 0.25
            warnings.append("Low lighting / underexposed face. Facial contours may be obscured.")
        elif brightness_val > cls.BRIGHTNESS_HIGH:
            brightness_status = "Overexposed"
            score_penalty += 0.20
            warnings.append("Harsh highlights or overexposed lighting detected.")
        else:
            brightness_status = "Good"

        # Overall composite score: 1.0 minus penalties (clamped to [0.1, 1.0])
        quality_score = max(0.1, min(1.0, 1.0 - score_penalty))

        # Overall quality label
        if quality_score >= 0.80:
            overall_quality = "HIGH"
        elif quality_score >= 0.50:
            overall_quality = "MEDIUM"
        else:
            overall_quality = "LOW"

        return FaceQualityReport(
            overall_quality=overall_quality,
            quality_score=round(quality_score, 2),
            resolution=(int(w), int(h)),
            resolution_status=res_status,
            blur_score=round(blur_val, 1),
            blur_status=blur_status,
            brightness_score=round(brightness_val, 1),
            brightness_status=brightness_status,
            face_size_status=face_size_status,
            warnings=warnings
        )
