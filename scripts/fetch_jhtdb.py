"""
Download and preprocess JHTDB snapshots for training.

Usage:
    # Use token from .env / environment variable:
    python scripts/fetch_jhtdb.py

    # Or pass explicitly:
    python scripts/fetch_jhtdb.py --token edu.jhu.pha.turbulence.testing-201406

    # Download multiple time snapshots:
    python scripts/fetch_jhtdb.py --time-indices 0 1 2 3 4

Output:
    jhtdb_cache/isotropic1024_t{idx:04d}.h5   raw DNS velocities (1024^3)
    jhtdb_cache/les_t{idx:04d}.h5             filtered LES fields (64^3)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.jhtdb import JHTDBLoader, compute_sgs_stress


def fetch_and_preprocess(
    token: str,
    cache_dir: Path,
    time_indices: list[int],
) -> None:
    loader = JHTDBLoader(token=token, cache_dir=str(cache_dir))

    for t_idx in time_indices:
        les_path = cache_dir / f"les_t{t_idx:04d}.h5"
        if les_path.exists():
            print(f"[t={t_idx:04d}] LES cache already exists, skipping.")
            continue

        print(f"[t={t_idx:04d}] Downloading DNS snapshot …")
        u, v, w = loader.load_snapshot(time_idx=t_idx)
        print(f"[t={t_idx:04d}] DNS loaded {u.shape}. Filtering to LES …")

        u_les, v_les, w_les, tau, grad_u = compute_sgs_stress(
            u, v, w,
            filter_width=loader.FILTER_WIDTH,
            dx_dns=loader.DX_DNS,
        )
        print(f"[t={t_idx:04d}] LES grid {u_les.shape}. Saving …")

        with h5py.File(les_path, "w") as f:
            f.create_dataset("u_les",   data=u_les,   compression="gzip")
            f.create_dataset("v_les",   data=v_les,   compression="gzip")
            f.create_dataset("w_les",   data=w_les,   compression="gzip")
            f.create_dataset("tau",     data=tau,     compression="gzip")
            f.create_dataset("grad_u",  data=grad_u,  compression="gzip")
            f.attrs["time_idx"]     = t_idx
            f.attrs["filter_width"] = loader.FILTER_WIDTH
            f.attrs["les_n"]        = u_les.shape[0]

        print(f"[t={t_idx:04d}] Saved → {les_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch JHTDB data and compute LES fields.")
    parser.add_argument(
        "--token", default="",
        help="JHTDB token (defaults to JHTDB_TOKEN env var or .env file)",
    )
    parser.add_argument(
        "--cache-dir", default="./jhtdb_cache",
        help="Directory for raw DNS and processed LES cache files",
    )
    parser.add_argument(
        "--time-indices", nargs="+", type=int, default=[0],
        help="Time snapshot indices to download (default: 0)",
    )
    args = parser.parse_args()

    # Load .env if present
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    token = args.token or os.environ.get("JHTDB_TOKEN", "")
    if not token:
        print("ERROR: No JHTDB token found. Set JHTDB_TOKEN in .env or pass --token.")
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Token   : {token[:12]}…")
    print(f"Cache   : {cache_dir.resolve()}")
    print(f"Indices : {args.time_indices}")
    fetch_and_preprocess(token, cache_dir, args.time_indices)
    print("Done.")


if __name__ == "__main__":
    main()
