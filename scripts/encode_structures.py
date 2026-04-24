"""
Pre-encode PT-BPE Tokenizer pkl files → flat integer token arrays.

Reads all Tokenizer pkl files, calls bpe.quantize(), and saves each protein's
token sequence to a single .npz archive for fast loading during training.

Output: /data/steven/ProteinChamaleon/encoded/warmup.npz
  - token_ids: object array of int32 arrays, one per protein
  - fnames:    object array of source filenames

Usage:
    python scripts/encode_structures.py
    python scripts/encode_structures.py --workers 16 --limit 500
"""

import argparse
import logging
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/steven/PT-BPE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("encode")

BPE_CKPT = Path("/data/steven/PT-BPE/ckpts/swissprot_michael/bpe_post_init.pkl")
PKL_DIR  = Path("/data/steven/PT-BPE/ckpts/1772514222.7787716/init_glue_opt")
OUT_DIR  = Path("/data/steven/ProteinChamaleon/encoded")
OUT_FILE = OUT_DIR / "warmup.npz"


def main(workers: int, limit: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(PKL_DIR.glob("*.pkl"))
    if limit:
        pkl_files = pkl_files[:limit]
    logger.info("Found %d tokenizer pkl files", len(pkl_files))

    logger.info("Loading BPE checkpoint...")
    with open(BPE_CKPT, "rb") as f:
        bpe = pickle.load(f)

    all_ids: list[np.ndarray] = []
    all_fnames: list[str] = []
    failed = 0

    for pkl_path in tqdm(pkl_files, desc="encoding"):
        try:
            with open(pkl_path, "rb") as f:
                tokenizer = pickle.load(f)
            ids = bpe.quantize(tokenizer)
            all_ids.append(np.array(ids, dtype=np.int32))
            all_fnames.append(str(pkl_path))
        except Exception as e:
            logger.warning("Failed %s: %s", pkl_path, e)
            failed += 1

    logger.info("Encoded %d proteins, %d failed", len(all_ids), failed)

    id_arr = np.empty(len(all_ids), dtype=object)
    for i, arr in enumerate(all_ids):
        id_arr[i] = arr
    fname_arr = np.array(all_fnames, dtype=object)

    np.savez(OUT_FILE, token_ids=id_arr, fnames=fname_arr)
    logger.info("Saved to %s", OUT_FILE)

    lengths = [len(a) for a in all_ids]
    logger.info(
        "Token length stats: min=%d  median=%d  max=%d  mean=%.0f",
        min(lengths), int(np.median(lengths)), max(lengths), np.mean(lengths),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()
    main(args.workers, args.limit)
