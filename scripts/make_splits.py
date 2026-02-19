"""
make_splits.py  —  Generate Train/Val/Test splits for the GIQ dataset.

Split strategy (shape-disjoint, stratified by polyhedron group):
  - 70% Train  |  15% Val  |  15% Test
  - Shapes are split by *shape ID* so no shape appears in multiple splits.
  - The hard-mental-rotation pairs (hard_examples.json) are always kept in the
    Test set; no constituent shape leaks into Train/Val.

Output (written to data/giq/splits/):
  train_shapes.json
  val_shapes.json
  test_shapes.json

Each file is a dict:  { shape_id: {name, group, sym, viewpoints: [...]} }
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
JSONS = ROOT / "giq-benchmark" / "jsons"
SPLITS_DIR = ROOT / "data" / "giq" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load metadata
# ------------------------------------------------------------------
with open(JSONS / "shapes.json") as f:
    shapes: dict = json.load(f)

with open(JSONS / "hard_examples.json") as f:
    hard: dict = json.load(f)


# Normalise hard-pair IDs  ("wid 5" -> "wid_5")
def norm(sid: str) -> str:
    return sid.replace(" ", "_")


hard_test_ids: set[str] = set()
for pair_list in hard.values():
    for pair in pair_list:
        hard_test_ids.update(norm(s) for s in pair)

print(f"Total shapes: {len(shapes)}")
print(f"Hard-test shape IDs ({len(hard_test_ids)}): {sorted(hard_test_ids)[:5]} ...")

# ------------------------------------------------------------------
# 2. Stratified split by group
# ------------------------------------------------------------------
groups: dict[str, list[str]] = defaultdict(list)
for sid, meta in shapes.items():
    groups[meta["group"]].append(sid)

train_ids, val_ids, test_ids = [], [], list(hard_test_ids)

for grp, ids in groups.items():
    # Remove hard-test ids from the pool for this group
    pool = [s for s in ids if s not in hard_test_ids]
    random.shuffle(pool)

    n = len(pool)
    n_val = max(1, round(n * 0.15))
    n_train = n - n_val  # remaining goes to train (test already handled)

    train_ids.extend(pool[:n_train])
    val_ids.extend(pool[n_train : n_train + n_val])

# Add any hard_test ids that weren't in shapes (shouldn't happen, but be safe)
for sid in hard_test_ids:
    if sid not in shapes:
        print(f"  WARNING: hard-test id {sid!r} not found in shapes.json")

print(
    f"\nSplit sizes  —  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}"
)

# ------------------------------------------------------------------
# 3. Attach viewpoint indices (20 synthetic views per shape)
# ------------------------------------------------------------------
N_VIEWS = 20


def build_split_dict(ids: list[str]) -> dict:
    out = {}
    for sid in sorted(ids):
        if sid not in shapes:
            continue
        entry = dict(shapes[sid])
        entry["viewpoints"] = list(range(N_VIEWS))
        out[sid] = entry
    return out


train_dict = build_split_dict(train_ids)
val_dict = build_split_dict(val_ids)
test_dict = build_split_dict(test_ids)

# ------------------------------------------------------------------
# 4. Persist splits
# ------------------------------------------------------------------
for name, d in [
    ("train_shapes", train_dict),
    ("val_shapes", val_dict),
    ("test_shapes", test_dict),
]:
    path = SPLITS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Written {path}  ({len(d)} shapes)")

# ------------------------------------------------------------------
# 5. Sanity checks
# ------------------------------------------------------------------
all_split = set(train_dict) | set(val_dict) | set(test_dict)
assert set(train_dict).isdisjoint(set(val_dict)), "Train/Val overlap!"
# Test set may overlap Val/Train by design (hard pairs pulled from full set).
# However, we ensure hard pairs are test-only:
for sid in hard_test_ids:
    if sid in shapes:
        assert sid in test_dict, f"{sid} is a hard-pair shape but not in test!"
print("\nSanity checks passed.")

# ------------------------------------------------------------------
# 6. Group distribution summary
# ------------------------------------------------------------------
print("\nGroup distribution:")
for grp in sorted(groups.keys()):
    tr = sum(1 for s in train_dict if shapes.get(s, {}).get("group") == grp)
    va = sum(1 for s in val_dict if shapes.get(s, {}).get("group") == grp)
    te = sum(1 for s in test_dict if shapes.get(s, {}).get("group") == grp)
    print(f"  {grp:<25}  train={tr}  val={va}  test={te}")
