import os
import json
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

IMAGE_SIZE   = (32, 32)         # Render size for each character glyph
FONT_SIZE    = 24               # Font size in pixels
PADDING      = 4                # Pixels of padding around glyph

# Fonts to average over — improves robustness across OCR inputs.
# Add / remove paths to match fonts available on your system.
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

# Default character set — feel free to expand
DEFAULT_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "!@#$%&(){}[]|?><"
)


# ---------------------------------------------------------------------------
# STEP 1 — Render character glyphs
# ---------------------------------------------------------------------------

def render_char(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """
    Render a single character to a normalised grayscale numpy array.
    The glyph is centred inside IMAGE_SIZE with PADDING on each side.

    Returns:
        np.ndarray of shape IMAGE_SIZE, dtype float32, values in [0, 1].
    """
    img  = Image.new("L", IMAGE_SIZE, color=255)   # white background
    draw = ImageDraw.Draw(img)

    # Centre the glyph
    bbox = draw.textbbox((0, 0), char, font=font)
    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]
    x = (IMAGE_SIZE[0] - glyph_w) // 2 - bbox[0]
    y = (IMAGE_SIZE[1] - glyph_h) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)     # black text

    return np.array(img, dtype=np.float32) / 255.0


def load_fonts() -> list[ImageFont.FreeTypeFont]:
    """Load all available fonts from FONT_PATHS; fall back to default if none found."""
    fonts = []
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                fonts.append(ImageFont.truetype(path, FONT_SIZE))
            except Exception:
                pass

    if not fonts:
        print("[warn] No TrueType fonts found — using PIL default (lower quality).")
        fonts = [ImageFont.load_default()]

    print(f"[info] Loaded {len(fonts)} font(s): {[os.path.basename(p) for p in FONT_PATHS if os.path.exists(p)]}")
    return fonts


# ---------------------------------------------------------------------------
# STEP 2 — SSIM-based distance (single font)
# ---------------------------------------------------------------------------

def ssim_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute perceptual distance between two rendered glyphs using SSIM.

    SSIM ∈ [-1, 1] where 1 = identical.
    We return  distance = (1 - ssim) / 2  so the result is in [0, 1].
    """
    score = ssim(img1, img2, data_range=1.0)
    return float((1.0 - score) / 2.0)


# ---------------------------------------------------------------------------
# STEP 3 — Multi-font averaging
# ---------------------------------------------------------------------------

def multi_font_distance(char1: str, char2: str,
                        fonts: list[ImageFont.FreeTypeFont]) -> float:
    """
    Average SSIM distance across all loaded fonts.
    Averaging over fonts makes scores more OCR-realistic because
    real OCR engines encounter many typefaces.
    """
    distances = []
    for font in fonts:
        img1 = render_char(char1, font)
        img2 = render_char(char2, font)
        distances.append(ssim_distance(img1, img2))
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# STEP 4 — Build symmetric N×N substitution matrix
# ---------------------------------------------------------------------------

def build_substitution_matrix(
    chars: str = DEFAULT_CHARS,
    fonts: list[ImageFont.FreeTypeFont] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build a symmetric substitution-cost matrix for all character pairs.

    Args:
        chars:   String of characters to include (each char used once).
        fonts:   Pre-loaded font list. Loads defaults if None.
        verbose: Print progress.

    Returns:
        matrix : np.ndarray, shape (N, N), dtype float32.
                 matrix[i][j] = visual distance between chars[i] and chars[j].
                 0.0 = identical, 1.0 = maximally different.
        idx    : dict mapping each character to its row/column index.
    """
    chars = list(dict.fromkeys(chars))          # deduplicate, preserve order
    n     = len(chars)
    idx   = {c: i for i, c in enumerate(chars)}

    if fonts is None:
        fonts = load_fonts()

    matrix = np.zeros((n, n), dtype=np.float32)

    total_pairs = n * (n - 1) // 2
    computed    = 0

    for i in range(n):
        for j in range(i + 1, n):             # upper triangle only
            dist = multi_font_distance(chars[i], chars[j], fonts)
            matrix[i][j] = dist
            matrix[j][i] = dist               # symmetric

            computed += 1
            if verbose and computed % 200 == 0:
                pct = 100 * computed / total_pairs
                print(f"  [{pct:5.1f}%] {computed}/{total_pairs} pairs done …")

    if verbose:
        print(f"  [100.0%] {total_pairs}/{total_pairs} pairs done.")

    return matrix, idx


# ---------------------------------------------------------------------------
# STEP 5 — Persist & reload (so you don't recompute every run)
# ---------------------------------------------------------------------------

def save_matrix(matrix: np.ndarray, idx: dict[str, int], path: str = "sub_matrix") -> None:
    """Save matrix (.npy) and character index (.json)."""
    np.save(f"{path}.npy", matrix)
    with open(f"{path}_idx.json", "w") as f:
        json.dump(idx, f, ensure_ascii=False)
    print(f"[info] Saved matrix → {path}.npy  |  index → {path}_idx.json")


def load_matrix(path: str = "sub_matrix") -> tuple[np.ndarray, dict[str, int]]:
    """Load a previously saved matrix and index."""
    matrix = np.load(f"{path}.npy")
    with open(f"{path}_idx.json") as f:
        idx = json.load(f)
    print(f"[info] Loaded matrix {matrix.shape} from {path}.npy")
    return matrix, idx


# ---------------------------------------------------------------------------
# STEP 6 — Lookup helper
# ---------------------------------------------------------------------------

def substitution_cost(
    matrix: np.ndarray,
    idx: dict[str, int],
    c1: str,
    c2: str,
    default_cost: float = 1.0,
) -> float:
    """
    Return the visual substitution cost for replacing c1 with c2.

    Args:
        default_cost: Returned when either character is not in the matrix.
    """
    if c1 == c2:
        return 0.0
    i = idx.get(c1)
    j = idx.get(c2)
    if i is None or j is None:
        return default_cost
    return float(matrix[i][j])


# ---------------------------------------------------------------------------
# STEP 7 — Weighted Levenshtein using the substitution matrix
# ---------------------------------------------------------------------------

def weighted_levenshtein(
    s1: str,
    s2: str,
    matrix: np.ndarray,
    idx: dict[str, int],
    insert_cost: float  = 1.0,
    delete_cost: float  = 1.0,
) -> float:
    """
    Levenshtein distance where substitution cost comes from the visual matrix.

    A visually similar pair (e.g. 'O'↔'0') gets a low substitution cost,
    so the overall distance between strings differing only in such pairs is
    smaller than between strings differing in unrelated characters.

    Args:
        s1, s2       : Strings to compare.
        insert_cost  : Cost of inserting a character (default 1.0).
        delete_cost  : Cost of deleting a character (default 1.0).

    Returns:
        float — weighted edit distance.
    """
    m, n = len(s1), len(s2)

    # dp[i][j] = weighted edit distance between s1[:i] and s2[:j]
    dp = np.zeros((m + 1, n + 1), dtype=np.float32)

    dp[:, 0] = np.arange(m + 1) * delete_cost
    dp[0, :] = np.arange(n + 1) * insert_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = substitution_cost(matrix, idx, s1[i - 1], s2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j]     + delete_cost,   # delete from s1
                dp[i][j - 1]     + insert_cost,   # insert into s1
                dp[i - 1][j - 1] + sub_cost,      # substitute
            )

    return float(dp[m][n])


# ---------------------------------------------------------------------------
# STEP 8 — Diagnostics: most-similar character pairs
# ---------------------------------------------------------------------------

def most_similar_pairs(
    matrix: np.ndarray,
    idx: dict[str, int],
    top_k: int = 20,
) -> list[tuple[str, str, float]]:
    """
    Return the top_k most visually similar (lowest-cost) character pairs.
    Useful for sanity-checking the matrix — you should see pairs like
    (O, 0), (l, 1), (I, l), (rn, m) near the top.
    """
    chars  = {v: k for k, v in idx.items()}   # reverse idx
    pairs  = []
    n      = matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((chars[i], chars[j], float(matrix[i][j])))

    pairs.sort(key=lambda x: x[2])
    return pairs[:top_k]

if __name__ == "__main__":
    MATRIX_PATH = "sub_matrix"

    if os.path.exists(f"{MATRIX_PATH}.npy"):
        print("Loading existing matrix …")
        matrix, idx = load_matrix(MATRIX_PATH)
    else:
        print("Building substitution matrix (this takes a minute) …")
        fonts  = load_fonts()
        matrix, idx = build_substitution_matrix(
            chars   = DEFAULT_CHARS,
            fonts   = fonts,
            verbose = True,
        )
        save_matrix(matrix, idx, MATRIX_PATH)

    # --- Spot-check individual pairs ---
    print("\n── Spot-check substitution costs ──")
    classic_pairs = [
        ("O", "0"), ("l", "1"), ("I", "l"), ("I", "1"),
        ("5", "S"), ("2", "Z"), ("8", "B"), ("G", "6"),
        ("A", "B"), ("X", "Y"),
    ]
    for c1, c2 in classic_pairs:
        cost = substitution_cost(matrix, idx, c1, c2)
        bar  = "█" * int(cost * 20)
        print(f"  {c1!r} ↔ {c2!r}  cost={cost:.4f}  |{bar:<20}|")

    # --- Top 20 most visually similar pairs ---
    print("\n── Top 20 most similar pairs ──")
    for c1, c2, cost in most_similar_pairs(matrix, idx, top_k=20):
        print(f"  {c1!r} ↔ {c2!r}  {cost:.4f}")

    # --- Weighted Levenshtein demo ---
    print("\n── Weighted Levenshtein demo ──")
    test_cases = [
        ("hello",   "he1lo"),    # l → 1  (visually close)
        ("hello",   "hexlo"),    # l → x  (visually far)
        ("INVOICE",  "INV0ICE"), # O → 0  (classic OCR error)
        ("INVOICE",  "INVXICE"), # O → X  (random error)
        ("cat",     "cut"),
        ("cat",     "bat"),
    ]
    for s1, s2 in test_cases:
        w_dist = weighted_levenshtein(s1, s2, matrix, idx)
        print(f"  {s1!r:12} ↔ {s2!r:12}  weighted_dist={w_dist:.4f}")