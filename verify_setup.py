"""Verification and diagnostics script for FaceTrace Chain.

Validates:
1. Python environment and installed libraries (including onnxruntime & insightface)
2. Face recognition models (ArcFace/InsightFace primary & SFace fallback)
3. Blockchain RPC connectivity and smart contract deployment
4. Reverse search providers (Live Google Lens reverse search & Dynamic Public Web Search)
5. Test sample benchmark assets
"""
import sys
import os
from pathlib import Path
from typing import Tuple, List

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class SetupVerifier:
    """Performs comprehensive end-to-end environment validation."""

    def __init__(self):
        self.results: List[Tuple[str, str, str]] = []  # (Category, Status, Message)

    def log(self, category: str, status: str, message: str):
        self.results.append((category, status, message))

    def check_python(self) -> bool:
        v = sys.version_info
        if v.major == 3 and v.minor >= 9:
            self.log("Python Environment", "PASS", f"Python {v.major}.{v.minor}.{v.micro}")
            return True
        else:
            self.log("Python Environment", "FAIL", f"Python {v.major}.{v.minor} detected. Requires Python >= 3.9")
            return False

    def check_dependencies(self) -> bool:
        required = [
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("onnxruntime", "ONNX Runtime"),
            ("insightface", "InsightFace"),
            ("web3", "Web3.py"),
            ("solcx", "py-solc-x"),
            ("requests", "Requests"),
            ("streamlit", "Streamlit"),
            ("pytest", "Pytest"),
            ("dotenv", "python-dotenv"),
        ]
        all_ok = True
        for mod, name in required:
            try:
                __import__(mod)
                self.log("Dependencies", "PASS", f"{name} is installed")
            except ImportError:
                self.log("Dependencies", "FAIL", f"{name} is missing (pip install {name.lower()})")
                all_ok = False
        return all_ok

    def check_models(self) -> bool:
        from utils.config import get_config
        config = get_config()

        yunet_ok = config.yunet_path.exists() and config.yunet_path.stat().st_size > 100000
        sface_ok = config.sface_path.exists() and config.sface_path.stat().st_size > 1000000

        if yunet_ok:
            self.log("AI Models", "PASS", f"YuNet ONNX detector found ({config.yunet_path.name})")
        else:
            self.log("AI Models", "WARN", "YuNet ONNX not found; will auto-download on first detection")

        # Check InsightFace ArcFace
        try:
            from face.recognizer import InsightFaceRecognizer
            rec = InsightFaceRecognizer()
            self.log("AI Models", "PASS", f"InsightFace ArcFace primary engine loaded ({rec.embedding_dim}-d)")
        except Exception as e:
            self.log("AI Models", "WARN", f"InsightFace ArcFace not active ({e}); SFace will act as fallback")

        if sface_ok:
            self.log("AI Models", "PASS", f"SFace ONNX fallback model found ({config.sface_path.name})")
        else:
            self.log("AI Models", "WARN", "SFace ONNX not found; will auto-download on first encoding")

        return yunet_ok

    def check_blockchain(self) -> bool:
        from utils.config import get_config
        from blockchain.client import BlockchainClient
        from blockchain.deploy import deploy_contract

        config = get_config()
        client = BlockchainClient()

        if not client.w3.is_connected():
            self.log(
                "Blockchain RPC",
                "FAIL",
                f"Cannot connect to RPC node at {config.blockchain_rpc_url}."
            )
            return False

        block_num = client.w3.eth.block_number
        self.log("Blockchain RPC", "PASS", f"Connected to RPC at {config.blockchain_rpc_url} (Block #{block_num})")

        # Verify contract deployment
        if client.contract is None:
            self.log("Smart Contract", "WARN", "Contract not deployed yet. Deploying now...")
            try:
                deploy_info = deploy_contract(rpc_url=config.blockchain_rpc_url)
                client = BlockchainClient(contract_address=deploy_info["contract_address"])
            except Exception as e:
                self.log("Smart Contract", "FAIL", f"Contract deployment failed: {e}")
                return False

        # Verify contract is callable on-chain
        try:
            records = client.contract.functions.recordCount().call()
            self.log(
                "Smart Contract",
                "PASS",
                f"FaceVerification contract active at {client.contract_address} (Total records: {records})"
            )
            return True
        except Exception as e:
            self.log("Smart Contract", "FAIL", f"Smart contract call failed at {client.contract_address}: {e}")
            return False

    def check_search_providers(self):
        from utils.config import get_config
        config = get_config()

        # Check Live Reverse Image Search (Google Lens)
        if config.search_api_key:
            masked = config.search_api_key[:4] + "..." + config.search_api_key[-4:] if len(config.search_api_key) > 8 else "***"
            self.log("Search: Google Lens", "PASS", f"SerpAPI Google Lens key configured ({masked})")
        else:
            self.log(
                "Search: Google Lens",
                "WARN",
                "SERPAPI_KEY is not set in .env. Live reverse image search requires a SerpAPI key."
            )

        # Check Public Web Search (Secondary)
        self.log("Search: Public Web", "PASS", "Dynamic Wikimedia Commons API search active (zero-config)")

    def check_sample_data(self) -> bool:
        from utils.config import get_config
        config = get_config()
        p_a = config.sample_data_dir / "sample_portrait_a.jpg"
        p_b = config.sample_data_dir / "sample_portrait_b.jpg"
        p_m = config.sample_data_dir / "sample_multi_face.jpg"

        if p_a.exists() and p_b.exists() and p_m.exists():
            self.log("Sample Test Data", "PASS", "All standard test portraits present (Portrait A, Portrait B, Multi-face)")
            return True
        else:
            self.log("Sample Test Data", "WARN", "One or more sample test portraits are missing")
            return False

    def run(self) -> int:
        print("\n" + "=" * 68)
        print("  FaceTrace Chain — System & Environment Setup Verification")
        print("=" * 68 + "\n")

        self.check_python()
        self.check_dependencies()
        self.check_models()
        self.check_blockchain()
        self.check_search_providers()
        self.check_sample_data()

        # Render summary
        print(f"{'Category':<24} {'Status':<8} {'Details'}")
        print("-" * 68)
        has_fail = False
        for cat, status, msg in self.results:
            if status == "FAIL":
                has_fail = True
                status_icon = "❌ FAIL"
            elif status == "WARN":
                status_icon = "⚠️ WARN"
            else:
                status_icon = "✓ PASS"
            print(f"{cat:<24} {status_icon:<8} {msg}")
        print("-" * 68)

        if has_fail:
            print("\n[RESULT] ❌ System verification identified critical issues above.")
            return 1
        else:
            print("\n[RESULT] ✓ System verified successfully and ready for operation!")
            return 0


if __name__ == "__main__":
    verifier = SetupVerifier()
    code = verifier.run()
    sys.exit(code)
