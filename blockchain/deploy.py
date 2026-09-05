import sys
import os
import re
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from web3 import Web3, EthereumTesterProvider
import solcx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import get_config

BUILD_DIR = Path(__file__).resolve().parent / "build"
CONTRACT_PATH = Path(__file__).resolve().parent / "FaceVerification.sol"
ARTIFACT_PATH = BUILD_DIR / "FaceVerification.json"

def compile_contract(force: bool = False) -> Tuple[Dict[str, Any], str]:
    """Compiles FaceVerification.sol and saves the JSON artifact (ABI + Bytecode)."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not force and ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["abi"], data["bytecode"]

    print("Compiling FaceVerification.sol via solcx...")
    installed = solcx.get_installed_solc_versions()
    if not any(v.major == 0 and v.minor == 8 for v in installed):
        solcx.install_solc("0.8.20")
    solcx.set_solc_version("0.8.20")

    compiled = solcx.compile_files(
        [str(CONTRACT_PATH)],
        output_values=["abi", "bin"]
    )

    contract_data = None
    for k, v in compiled.items():
        if k.endswith(":FaceVerification"):
            contract_data = v
            break
    if not contract_data:
        contract_data = list(compiled.values())[0]

    abi = contract_data["abi"]
    bytecode = contract_data["bin"]

    artifact = {
        "contractName": "FaceVerification",
        "abi": abi,
        "bytecode": bytecode
    }
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Contract compiled and artifact saved to {ARTIFACT_PATH}")
    return abi, bytecode


def update_env_file(key: str, value: str):
    """Updates or adds an environment variable in the root .env file."""
    config = get_config()
    env_file = config.root_dir / ".env"
    if not env_file.exists():
        env_file.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    content = env_file.read_text(encoding="utf-8")
    pattern = rf"^{key}=.*$"
    if re.search(pattern, content, flags=re.MULTILINE):
        new_content = re.sub(pattern, f"{key}={value}", content, flags=re.MULTILINE)
    else:
        new_content = content.rstrip() + f"\n{key}={value}\n"
    env_file.write_text(new_content, encoding="utf-8")


def deploy_contract(
    rpc_url: str = "http://127.0.0.1:8545",
    private_key: str = "",
    force_recompile: bool = False
) -> Dict[str, Any]:
    """Deploys FaceVerification contract to local Ganache or testnet."""
    abi, bytecode = compile_contract(force=force_recompile)

    # Establish Web3 connection
    w3 = None
    if rpc_url:
        try:
            http_w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))
            if http_w3.is_connected():
                w3 = http_w3
                print(f"Connected to Ethereum RPC at {rpc_url} (Chain ID: {w3.eth.chain_id})")
        except Exception:
            pass

    if w3 is None:
        print("Local RPC not reachable. Falling back to in-memory EthereumTesterProvider.")
        from blockchain.client import get_shared_tester_provider
        w3 = Web3(get_shared_tester_provider())

    accounts = w3.eth.accounts
    if not accounts and not private_key:
        raise RuntimeError("No accounts available on blockchain provider.")

    deployer_address = ""
    if private_key:
        account_obj = w3.eth.account.from_key(private_key)
        deployer_address = account_obj.address
    else:
        deployer_address = accounts[0]

    print(f"Deploying from account: {deployer_address}")
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    if private_key:
        # Build signed transaction
        nonce = w3.eth.get_transaction_count(deployer_address)
        tx = Contract.constructor().build_transaction({
            "from": deployer_address,
            "nonce": nonce,
            "gas": 3000000,
            "gasPrice": w3.eth.gas_price
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    else:
        tx_hash = Contract.constructor().transact({
            "from": deployer_address,
            "gas": 3000000
        })

    print(f"Deployment transaction sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    contract_address = receipt.contractAddress

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print(f"[OK] FaceVerification deployed at: {contract_address}")
    print(f"  Block number: {receipt.blockNumber}")
    print(f"  Gas used: {receipt.gasUsed}")

    # Persist address in .env only if valid string
    if contract_address and str(contract_address).lower() not in {"none", "null", "false", ""}:
        update_env_file("BLOCKCHAIN_CONTRACT_ADDRESS", contract_address)

    return {
        "contract_address": contract_address,
        "tx_hash": tx_hash.hex(),
        "block_number": receipt.blockNumber,
        "deployer": deployer_address,
        "gas_used": receipt.gasUsed,
        "abi": abi
    }


if __name__ == "__main__":
    cfg = get_config()
    res = deploy_contract(
        rpc_url=cfg.blockchain_rpc_url,
        private_key=cfg.blockchain_private_key
    )
    print("\nDeployment complete:", json.dumps(res, indent=2))
