// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceVerification
 * @notice Immutable blockchain storage for tamper-evident face discovery fingerprints.
 * @dev Stores cryptographic content fingerprints (SHA-256 hashes as bytes32) and public source references.
 */
contract FaceVerification {

    struct VerificationRecord {
        bytes32 contentHash;
        string sourceReference;
        uint256 timestamp;
        address submitter;
        bool exists;
    }

    // Mapping from contentHash (bytes32) to VerificationRecord
    mapping(bytes32 => VerificationRecord) private _records;
    
    // Total count of verified records anchored on-chain
    uint256 public recordCount;

    // Emitted when a new content verification is recorded
    event ContentStored(
        bytes32 indexed contentHash,
        string sourceReference,
        uint256 timestamp,
        address indexed submitter
    );

    error ContentAlreadyRegistered(bytes32 contentHash);
    error InvalidContentHash();

    /**
     * @notice Store a tamper-evident content fingerprint on the blockchain.
     * @param contentHash The 32-byte cryptographic SHA-256 hash of the canonical content.
     * @param sourceReference The public canonical URL or identifier for provenance attribution.
     * @return success Boolean indicating successful on-chain anchoring.
     */
    function storeVerification(
        bytes32 contentHash,
        string calldata sourceReference
    ) external returns (bool success) {
        if (contentHash == bytes32(0)) {
            revert InvalidContentHash();
        }
        if (_records[contentHash].exists) {
            revert ContentAlreadyRegistered(contentHash);
        }

        _records[contentHash] = VerificationRecord({
            contentHash: contentHash,
            sourceReference: sourceReference,
            timestamp: block.timestamp,
            submitter: msg.sender,
            exists: true
        });

        recordCount += 1;

        emit ContentStored(contentHash, sourceReference, block.timestamp, msg.sender);
        return true;
    }

    /**
     * @notice Query verification status and provenance metadata for a given content hash.
     * @param contentHash The 32-byte content hash to look up.
     * @return exists True if the content hash is anchored on-chain.
     * @return sourceReference The associated source URL or attribution string.
     * @return timestamp The block timestamp when the record was confirmed.
     * @return submitter The Ethereum account that submitted the record.
     */
    function verifyContent(
        bytes32 contentHash
    ) external view returns (
        bool exists,
        string memory sourceReference,
        uint256 timestamp,
        address submitter
    ) {
        VerificationRecord memory rec = _records[contentHash];
        return (rec.exists, rec.sourceReference, rec.timestamp, rec.submitter);
    }

    /**
     * @notice Lightweight existence check for a content hash.
     * @param contentHash The 32-byte content hash.
     */
    function isContentVerified(bytes32 contentHash) external view returns (bool) {
        return _records[contentHash].exists;
    }
}
