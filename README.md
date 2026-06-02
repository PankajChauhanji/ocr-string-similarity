# ocr-string-similarity

A visually-grounded weighted Levenshtein distance for OCR error correction.

Standard Levenshtein treats every substitution equally — replacing `O` with `0` costs the same as replacing `O` with `X`. But OCR errors are not random; they happen because characters **look alike**. This library builds a substitution cost matrix from actual glyph similarity, so visually similar pairs get a lower edit cost.

```
'hello'   ↔ 'he1lo'    →  weighted_dist = 0.12   (l → 1, visually close)
'hello'   ↔ 'hexlo'    →  weighted_dist = 0.29   (l → x, visually far)

'INVOICE' ↔ 'INV0ICE'  →  weighted_dist = 0.18   (O → 0, classic OCR error)
'INVOICE' ↔ 'INVXICE'  →  weighted_dist = 0.37   (O → X, random error)
```

---

## How it works

```
Render chars (multi-font)
    → SSIM pairwise distances
    → Normalize to [0, 1]
    → Symmetric N×N substitution matrix
    → weighted_levenshtein(s1, s2, matrix, idx)
```

Each character is rendered as a 32×32 grayscale glyph across multiple fonts. SSIM (Structural Similarity Index) measures perceptual distance between every pair — capturing luminance, contrast, and structure, unlike pixel histograms. Distances are averaged across fonts for OCR-realistic robustness, then stored in a symmetric matrix for O(1) lookup at runtime.

---

## Installation

```bash
pip install pillow scikit-image numpy
```

TrueType fonts are read from system font paths defined in `FONT_PATHS` at the top of the file. The defaults cover most Linux systems (DejaVu, Liberation). Edit the list to match your OS or add custom fonts.

---

## Quick start

```python
from ocr_substitution_matrix import (
    build_substitution_matrix,
    save_matrix, load_matrix,
    substitution_cost,
    weighted_levenshtein,
)

# Build once, save to disk
matrix, idx = build_substitution_matrix()
save_matrix(matrix, idx, "sub_matrix")

# Reload on subsequent runs
matrix, idx = load_matrix("sub_matrix")

# Single pair cost
print(substitution_cost(matrix, idx, "O", "0"))   # ~0.18
print(substitution_cost(matrix, idx, "O", "X"))   # ~0.37

# Weighted edit distance
dist = weighted_levenshtein("INV0ICE", "INVOICE", matrix, idx)
print(dist)   # 0.18
```
---

## Configuration

```python
IMAGE_SIZE  = (32, 32)   # Glyph render size — increase for finer detail
FONT_SIZE   = 24         # Font size in pixels
FONT_PATHS  = [...]      # Paths to TrueType fonts to average over
DEFAULT_CHARS = "ABC..."  # Character set to include
```

Adding more fonts to `FONT_PATHS` improves robustness at the cost of build time. The matrix is computed once and cached, so build time does not affect runtime.

---

## Use cases

- Post-OCR string matching and fuzzy search
- Confidence scoring for OCR output validation
- Document digitisation pipelines (invoices, forms, receipts)
- Dataset cleaning where OCR noise needs to be distinguished from genuine differences

---

## License

MIT © Pankaj Chauhan 2026