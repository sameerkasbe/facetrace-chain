"""Standardized logging and CLI step formatting for FaceTrace Chain."""
import logging
import sys

def setup_logger(name: str = "FaceTraceChain", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

class PipelineProgress:
    """Helper for formatted CLI step reporting."""
    TOTAL_STEPS = 5

    @staticmethod
    def step(step_num: int, title: str):
        print(f"\n[{step_num}/{PipelineProgress.TOTAL_STEPS}] {title}...")

    @staticmethod
    def detail(msg: str):
        print(f"  -> {msg}")

    @staticmethod
    def success(msg: str):
        print(f"[OK] {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"[WARN] {msg}")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {msg}")

    @staticmethod
    def status_banner(status: str):
        print("\n" + "=" * 50)
        print(f"STATUS: {status}")
        print("=" * 50 + "\n")
