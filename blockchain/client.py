"""Web3 client interface for FaceVerification smart contract."""
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from web3 import Web3, EthereumTesterProvider
from hexbytes import HexBytes

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.verification_record import BlockchainStoreResult
from utils.config import get_config
from blockchain.deploy import ARTIFACT_PATH, compile_contract

class BlockchainClient:
    """Interacts with the FaceVerification smart contract on Ethereum / Ganache."""

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None
    ):
        config = get_config()
        self.rpc_url = rpc_url or config.blockchain_rpc_url
        self.contract_address = contract_address or config.blockchain_contract_address
        self.private_key = private_key or config.blockchain_private_key
        
        self.w3 = self._init_web3()
        self.abi = self._load_abi()
        self.contract = None
        
        if self.contract_address:
            self._init_contract(self.contract_address)

    def _init_web3(self) -> Web3:
        """Connects to Ethereum RPC or falls back to in-process EthereumTesterProvider."""
        if self.rpc_url:
            try:
                http_w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 5}))
                if http_w3.is_connected():
                    return http_w3
            except Exception:
                pass

        print("[WARN] Local RPC unavailable. Using in-memory EthereumTesterProvider.")
        return Web3(EthereumTesterProvider())

    def _load_abi(self) -> list:
        if ARTIFACT_PATH.exists():
            with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["abi"]
        abi, _ = compile_contract()
        return abi

    def _init_contract(self, address: str):
        checksum_address = Web3.to_checksum_address(address)
        self.contract = self.w3.eth.contract(address=checksum_address, abi=self.abi)
        self.contract_address = checksum_address

    def get_account(self) -> str:
        """Returns the default submitting account."""
        if self.private_key:
            return self.w3.eth.account.from_key(self.private_key).address
        accounts = self.w3.eth.accounts
        if accounts:
            return accounts[0]
        raise RuntimeError("No accounts available on blockchain provider.")

    def _to_bytes32(self, hash_hex: str) -> bytes:
        clean = hash_hex.strip()
        if clean.startswith("0x"):
            clean = clean[2:]
        if len(clean) != 64:
            raise ValueError(f"Content hash must be 32 bytes (64 hex characters), got {len(clean)} chars.")
        return bytes.fromhex(clean)

    def store_verification(
        self,
        content_hash_hex: str,
        source_reference: str
    ) -> BlockchainStoreResult:
        """Anchors a content fingerprint on-chain."""
        if self.contract is None:
            raise RuntimeError("Smart contract not initialized. Deploy contract first.")

        content_bytes32 = self._to_bytes32(content_hash_hex)
        submitter = self.get_account()

        # If already registered on-chain, return existing confirmation
        if self.is_verified(content_hash_hex):
            existing = self.query_verification(content_hash_hex)
            return BlockchainStoreResult(
                success=True,
                content_hash="0x" + content_hash_hex.replace("0x", ""),
                tx_hash="0x[ALREADY_CONFIRMED_ON_CHAIN]",
                block_number=self.w3.eth.block_number,
                timestamp=existing["timestamp"] if existing else 0,
                submitter=existing["submitter"] if existing else submitter,
                source_reference=existing["source_reference"] if existing else source_reference,
                gas_used=0
            )

        try:
            if self.private_key:
                nonce = self.w3.eth.get_transaction_count(submitter)
                tx = self.contract.functions.storeVerification(
                    content_bytes32,
                    source_reference
                ).build_transaction({
                    "from": submitter,
                    "nonce": nonce,
                    "gas": 300000,
                    "gasPrice": self.w3.eth.gas_price
                })
                signed = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            else:
                tx_hash = self.contract.functions.storeVerification(
                    content_bytes32,
                    source_reference
                ).transact({
                    "from": submitter,
                    "gas": 300000
                })

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            block = self.w3.eth.get_block(receipt.blockNumber)
            timestamp = block.get("timestamp", 0)

            return BlockchainStoreResult(
                success=receipt.status == 1,
                content_hash="0x" + content_hash_hex.replace("0x", ""),
                tx_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                timestamp=timestamp,
                submitter=submitter,
                source_reference=source_reference,
                gas_used=receipt.gasUsed
            )
        except Exception as e:
            return BlockchainStoreResult(
                success=False,
                content_hash="0x" + content_hash_hex.replace("0x", ""),
                tx_hash="",
                block_number=0,
                timestamp=0,
                submitter=submitter,
                source_reference=source_reference,
                error_message=str(e)
            )

    def query_verification(self, content_hash_hex: str) -> Optional[Dict[str, Any]]:
        """Queries on-chain record for a content hash."""
        if self.contract is None:
            raise RuntimeError("Smart contract not initialized.")

        content_bytes32 = self._to_bytes32(content_hash_hex)
        try:
            exists, source_ref, timestamp, submitter = self.contract.functions.verifyContent(
                content_bytes32
            ).call()
            if not exists:
                return None
            return {
                "exists": exists,
                "source_reference": source_ref,
                "timestamp": timestamp,
                "submitter": submitter,
                "content_hash": "0x" + content_hash_hex.replace("0x", "")
            }
        except Exception as e:
            print(f"Error querying blockchain: {e}")
            return None

    def is_verified(self, content_hash_hex: str) -> bool:
        """Checks if a hash is stored on-chain."""
        if self.contract is None:
            return False
        try:
            content_bytes32 = self._to_bytes32(content_hash_hex)
            return self.contract.functions.isContentVerified(content_bytes32).call()
        except Exception:
            return False
