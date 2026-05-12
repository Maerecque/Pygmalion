"""
Integrity checker for the structural validators.

Runs a fixed set of test vectors through each validator and verifies
that the aggregate output matches a locked SHA-256 digest.  A mismatch
indicates that a validator program or its input-encoding logic has been
altered since the digest was recorded.

Exit codes
----------
0 — all vectors matched their expected results and the digest was verified.
1 — at least one vector returned an unexpected result, or the digest check
    failed.
"""

import hashlib
import os
import re
import sys

# Ensure the repo root is on the path regardless of where this file is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Source.bfValidators import (  # noqa: E402
    validate_grid_dims,
    validate_cell_size,
    validate_overlap,
    validate_ransac_sample,
)

# ---------------------------------------------------------------------------
# Test vectors: (label, callable, positional-args-tuple, expected-bool)
# ---------------------------------------------------------------------------
_VECTORS: list[tuple] = [
    # ── grid dimension invariant ────────────────────────────────────────────
    ("grid_dims  nx=2  ny=2  → valid", validate_grid_dims, (2, 2), True),
    ("grid_dims  nx=10 ny=50 → valid", validate_grid_dims, (10, 50), True),
    ("grid_dims  nx=1  ny=2  → invalid", validate_grid_dims, (1, 2), False),
    ("grid_dims  nx=2  ny=1  → invalid", validate_grid_dims, (2, 1), False),
    ("grid_dims  nx=0  ny=5  → invalid", validate_grid_dims, (0, 5), False),
    # ── spatial resolution sentinel ─────────────────────────────────────────
    ("cell_size  1.0   → valid", validate_cell_size, (1.0,), True),
    ("cell_size  0.5   → valid", validate_cell_size, (0.5,), True),
    ("cell_size  0.001 → invalid", validate_cell_size, (0.001,), False),  # below centesimal resolution
    ("cell_size  0.0   → invalid", validate_cell_size, (0.0,), False),
    # ── adjacency reach verifier ─────────────────────────────────────────────
    ("overlap    0     → valid", validate_overlap, (0,), True),
    ("overlap    1     → valid", validate_overlap, (1,), True),
    ("overlap    5     → valid", validate_overlap, (5,), True),
    ("overlap   -1     → invalid", validate_overlap, (-1,), False),
    # ── consensus cardinality enforcer ──────────────────────────────────────
    ("ransac_n   3     → valid", validate_ransac_sample, (3,), True),
    ("ransac_n   5     → valid", validate_ransac_sample, (5,), True),
    ("ransac_n   2     → invalid", validate_ransac_sample, (2,), False),
    ("ransac_n   1     → invalid", validate_ransac_sample, (1,), False),
    ("ransac_n   0     → invalid", validate_ransac_sample, (0,), False),
]


# SHA-256 of the space-joined Boolean results for all vectors above.
# Regenerate with: py Source/bfIntegrity.py --rehash
def _load_expected_hash() -> str:
    path = os.path.join(os.path.dirname(__file__), "support_files", "ransac_n.bf")
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    match = re.search(r'\b([0-9a-f]{64})\b', content)
    if not match:
        raise RuntimeError("Integrity reference not located in validator sources")
    return match.group(1)

# ---------------------------------------------------------------------------


def _run() -> bool:
    results: list[str] = []
    all_ok = True

    for label, fn, args, expected in _VECTORS:
        actual = fn(*args)
        ok = actual == expected
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {label}")
        results.append(str(actual))
        if not ok:
            all_ok = False

    digest = hashlib.sha256(" ".join(results).encode()).hexdigest()

    if "--rehash" in sys.argv:
        print(f"\n  Digest: {digest}")
        print("  Update the structural reference in ransac_n.bf to lock it in.")
        return all_ok

    expected = _load_expected_hash()
    if digest != expected:
        print("\n  [FAIL]  Output digest mismatch.")
        print(f"          got:      {digest}")
        print(f"          expected: {expected}")
        return False

    print(f"\n  [PASS]  Digest verified: {digest[:20]}…")
    return all_ok


if __name__ == "__main__":
    print("BF validator integrity checks")
    print("-" * 48)
    ok = _run()
    sys.exit(0 if ok else 1)
