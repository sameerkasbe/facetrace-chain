"""Face detection with single-face validation and bounding box extraction."""
import os
import urllib.request
from pathlib import Path
from typing import Tuple, List, Optional, Union
import cv2
import numpy as np
from utils.config import get_config

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class FaceDetectionError(Exception):
    """Custom exception raised when face detection criteria are violated."""
    pass


class FaceDetectionResult:
    """Encapsulates the detected face bounding box, crop, and metadata."""
    def __init__(
        self,
        image_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],  # (x, y, w, h)
        confidence: float,
        raw_face_entry: Optional[np.ndarray] = None
    ):
        self.image_bgr = image_bgr
        self.bbox = bbox
        self.confidence = confidence
        self.raw_face_entry = raw_face_entry

        x, y, w, h = bbox
        img_h, img_w = image_bgr.shape[:2]
        x1 = max(0, min(img_w - 1, int(x)))
        y1 = max(0, min(img_h - 1, int(y)))
        x2 = max(x1 + 1, min(img_w, int(x + w)))
        y2 = max(y1 + 1, min(img_h, int(y + h)))
        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            crop = cv2.resize(image_bgr, (112, 112))
        self.cropped_face = crop

    @property
    def confidence_pct(self) -> int:
        return int(round(self.confidence * 100))

    def draw_visualization(self) -> np.ndarray:
        """Returns a copy of the image with the face bounding box and label drawn."""
        annotated = self.image_bgr.copy()
        x, y, w, h = self.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"Face: {self.confidence_pct}%"
        cv2.putText(
            annotated, label, (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
        )
        return annotated


class FaceDetector:
    """OpenCV YuNet Deep Neural Network Face Detector with Haar Cascade fallback."""

    def __init__(self, score_threshold: float = 0.6):
        self.config = get_config()
        self.score_threshold = score_threshold
        self._ensure_model_exists()
        self._init_detector()

    def _ensure_model_exists(self):
        yunet_path = self.config.yunet_path
        if not yunet_path.exists() or yunet_path.stat().st_size < 1000:
            yunet_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading YuNet face detection model to {yunet_path}...")
            urllib.request.urlretrieve(self.config.yunet_model_url, str(yunet_path))

    def _init_detector(self):
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.haar_detector = cv2.CascadeClassifier(haar_path)

        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=str(self.config.yunet_path),
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=0.3,
                top_k=5000
            )
            self.use_yunet = True
        except Exception as e:
            print(f"YuNet initialization failed ({e}). Falling back to Haar Cascade.")
            self.use_yunet = False

    def load_image(self, image_input: Union[str, Path, bytes]) -> np.ndarray:
        """Loads and validates an image from a filepath or raw bytes."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image format '{path.suffix}'. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Unable to decode image file: {path}")
            return img
        elif isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode raw image byte buffer.")
            return img
        else:
            raise TypeError(f"Invalid image input type: {type(image_input)}")

    def detect_all_faces(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]]:
        """Returns all detected faces as a list of ((x, y, w, h), confidence, raw_entry)."""
        orig_h, orig_w = image.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return []
        
        # Scale to optimal neural network detection dimension (max dim 800)
        max_dim = max(orig_h, orig_w)
        scale = min(1.0, 800.0 / max_dim) if max_dim > 800 else 1.0
        if scale <= 1e-6:
            scale = 1.0

        if scale < 1.0:
            det_w, det_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
            det_img = cv2.resize(image, (det_w, det_h))
        else:
            det_w, det_h = orig_w, orig_h
            det_img = image

        results = []
        if self.use_yunet:
            self.detector.setInputSize((det_w, det_h))
            _, faces = self.detector.detect(det_img)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    if face is None or len(face) < 15:
                        continue
                    if not np.all(np.isfinite(face[:15])):
                        continue

                    scaled_face = face.copy()
                    if scale < 1.0:
                        # Scale bounding box and 5 facial landmark points back to original coordinates
                        scaled_face[0:14] = np.nan_to_num(
                            scaled_face[0:14] / scale, nan=0.0, posinf=0.0, neginf=0.0
                        )

                    try:
                        fx = float(scaled_face[0])
                        fy = float(scaled_face[1])
                        fw = float(scaled_face[2])
                        fh = float(scaled_face[3])
                        conf = float(scaled_face[-1])

                        if not (np.isfinite(fx) and np.isfinite(fy) and np.isfinite(fw) and np.isfinite(fh) and np.isfinite(conf)):
                            continue

                        if conf < self.score_threshold or fw <= 0 or fh <= 0:
                            continue

                        x = max(0, min(orig_w - 1, int(round(fx))))
                        y = max(0, min(orig_h - 1, int(round(fy))))
                        w_box = max(1, min(orig_w - x, int(round(fw))))
                        h_box = max(1, min(orig_h - y, int(round(fh))))

                        box = (x, y, w_box, h_box)
                        results.append((box, conf, scaled_face))
                    except (OverflowError, ValueError):
                        continue

        # If YuNet returned no faces, fallback to Haar Cascade
        if not results:
            gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
            haar_faces = self.haar_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (hx, hy, hw, hh) in haar_faces:
                try:
                    fx, fy, fw, fh = float(hx), float(hy), float(hw), float(hh)
                    if scale < 1.0 and scale > 0:
                        fx /= scale
                        fy /= scale
                        fw /= scale
                        fh /= scale

                    if not (np.isfinite(fx) and np.isfinite(fy) and np.isfinite(fw) and np.isfinite(fh)):
                        continue

                    x = max(0, min(orig_w - 1, int(round(fx))))
                    y = max(0, min(orig_h - 1, int(round(fy))))
                    w_f = max(1, min(orig_w - x, int(round(fw))))
                    h_f = max(1, min(orig_h - y, int(round(fh))))
                    results.append(((x, y, w_f, h_f), 0.85, None))
                except (OverflowError, ValueError):
                    continue

        return results

    def detect_primary_face(
        self,
        image_input: Union[str, Path, bytes],
        enforce_single: bool = True
    ) -> FaceDetectionResult:
        """Detects and validates exactly one primary face.
        
        Raises FaceDetectionError with explicit messages if no face or multiple
        faces are found.
        """
        img = self.load_image(image_input)
        detected = self.detect_all_faces(img)

        if len(detected) == 0:
            raise FaceDetectionError("Unable to detect a face in this image.")

        if enforce_single and len(detected) > 1:
            raise FaceDetectionError(
                "Multiple faces detected. Please provide an image containing one clear primary face."
            )

        # Use the highest confidence detection
        best_box, best_conf, raw_entry = max(detected, key=lambda item: item[1])
        return FaceDetectionResult(
            image_bgr=img,
            bbox=best_box,
            confidence=best_conf,
            raw_face_entry=raw_entry
        )
