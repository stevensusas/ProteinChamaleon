"""
Download AlphaFold PDB structures for all proteins in proteins.csv.

For each UniProt accession, tries:
  1. AlphaFold DB  (covers ~100% of reviewed UniProt proteins)
  2. RCSB PDB      (fallback using first PDB ID from proteins.csv)

Output: /data/steven/ProteinChamaleon/structures/{accession}.pdb

Usage:
    python scripts/download_structures.py
    python scripts/download_structures.py --workers 50 --limit 1000
"""

import argparse
import asyncio
import csv
import logging
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download")

PROTEINS_CSV    = Path("/home/steven/ProteinChamaleon-Dataset/output/proteins.csv")
OUT_DIR         = Path("/data/steven/ProteinChamaleon/structures")
FAILED_LOG      = Path("/data/steven/ProteinChamaleon/structures/failed.txt")
AF_VERSIONS     = ["v6", "v5", "v4", "v3"]
AF_URL          = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{ver}.pdb"
PDB_URL         = "https://files.rcsb.org/download/{pdb_id}.pdb"


def load_proteins(csv_path: Path, limit: int | None) -> list[dict]:
    proteins = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pdb_ids = [x for x in row["pdb_ids"].split("|") if x]
            proteins.append({
                "accession": row["accession"],
                "pdb_id": pdb_ids[0] if pdb_ids else None,
            })
            if limit and len(proteins) >= limit:
                break
    return proteins


async def download_one(
    client: httpx.AsyncClient,
    accession: str,
    pdb_id: str | None,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """Returns (accession, status) where status is 'ok', 'skip', or 'failed'."""
    dest = out_dir / f"{accession}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return accession, "skip"

    async with semaphore:
        # Try AlphaFold versions first, then fall back to PDB
        af_urls = [AF_URL.format(acc=accession, ver=ver) for ver in AF_VERSIONS]
        pdb_urls = [PDB_URL.format(pdb_id=pdb_id)] if pdb_id else []
        for url in af_urls + pdb_urls:
            try:
                resp = await client.get(url, timeout=30)
                if resp.status_code == 200:
                    dest.write_bytes(resp.content)
                    return accession, "ok"
            except Exception:
                continue

        return accession, "failed"


async def main(workers: int, limit: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    proteins = load_proteins(PROTEINS_CSV, limit)
    logger.info("Loaded %d proteins", len(proteins))

    semaphore = asyncio.Semaphore(workers)
    failed: list[str] = []
    ok = skip = 0
    t0 = time.monotonic()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [
            download_one(client, p["accession"], p["pdb_id"], OUT_DIR, semaphore)
            for p in proteins
        ]

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            accession, status = await coro
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                failed.append(accession)

            if i % 1000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / elapsed
                remaining = (len(proteins) - i) / rate
                logger.info(
                    "%d/%d  ok=%d skip=%d failed=%d  %.0f/s  ~%.0f min left",
                    i, len(proteins), ok, skip, len(failed), rate, remaining / 60,
                )

    # Write failed accessions for retry
    if failed:
        FAILED_LOG.write_text("\n".join(failed))
        logger.warning("%d failed — saved to %s", len(failed), FAILED_LOG)

    elapsed = time.monotonic() - t0
    logger.info(
        "Done: %d ok, %d skipped, %d failed in %.1f min",
        ok, skip, len(failed), elapsed / 60,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=50,
                        help="Concurrent downloads (default: 50)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of proteins (for testing)")
    args = parser.parse_args()

    asyncio.run(main(args.workers, args.limit))
