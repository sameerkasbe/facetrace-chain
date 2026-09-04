# Sample Data & Demonstration Assets

This directory contains test assets intended exclusively for authorized demonstration and validation of the **FaceTrace Chain** pipeline.

## Files Included

1. **`sample_portrait_a.jpg`**:
   - Single clear primary portrait of Albert Einstein (Public Domain via Wikimedia Commons).
   - Use this to test normal end-to-end pipeline execution: face detection, 128-d embedding extraction, runtime public web search, verification, and on-chain record keeping.

2. **`sample_portrait_b.jpg`**:
   - Single clear primary portrait of Abraham Lincoln (Public Domain via Wikimedia Commons).
   - Use this to demonstrate cross-person non-matching (`No Match` status).

3. **`sample_multi_face.jpg`**:
   - Stitched image containing two distinct faces.
   - Used to test the strict input validation rule:
     > *"Multiple faces detected. Please provide an image containing one clear primary face."*

4. **`public_corpus/`**:
   - Local directory of authorized public domain portraits.
   - Used when configuring `SEARCH_PROVIDER=local` for completely air-gapped or offline hackathon demonstrations.

## Ethical & Privacy Notice

These samples utilize historical public domain photographs to ensure complete compliance with privacy standards and copyright laws. Do not use unconsented private individuals' photos.
