#!/usr/bin/env python3
"""
Package the current pictos.db, pictogram_phrases.db, and PNG pictograms
into a versioned zip that mobile clients can download for offline use.

Produces:
    bundles/<version>/aropi-bundle-<version>.zip
    bundles/latest.json        (pointer to the newest bundle)

The zip layout is:
    manifest.json
    pictos.db
    pictogram_phrases.db
    png/<picto>.png ...

The version is a UTC timestamp ("YYYYMMDD-HHMM") unless overridden via
--version. `latest.json` always points at the most recently produced
bundle.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PICTOS_DB = os.path.join(REPO_ROOT, "project", "output", "pictos.db")
DEFAULT_PHRASES_DB = os.path.join(REPO_ROOT, "pictogram_phrases.db")
DEFAULT_IMAGES_DIR = os.path.join(REPO_ROOT, "project", "output", "png")
DEFAULT_BUNDLES_DIR = os.path.join(REPO_ROOT, "bundles")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(version: str, files: dict[str, str]) -> dict:
    """`files` maps archive-path -> filesystem path."""
    entries = []
    for arcname, src in files.items():
        entries.append({
            "path": arcname,
            "size": os.path.getsize(src),
            "sha256": _sha256(src),
        })
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
    }


def build_bundle(
    pictos_db: str,
    phrases_db: str,
    images_dir: str,
    bundles_dir: str,
    version: str | None = None,
) -> dict:
    for path in (pictos_db, phrases_db):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

    out_dir = os.path.join(bundles_dir, version)
    os.makedirs(out_dir, exist_ok=True)
    zip_name = f"aropi-bundle-{version}.zip"
    zip_path = os.path.join(out_dir, zip_name)

    # Collect file map -> archive names.
    files: dict[str, str] = {
        "pictos.db": pictos_db,
        "pictogram_phrases.db": phrases_db,
    }
    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith(".png"):
            files[f"png/{fname}"] = os.path.join(images_dir, fname)

    manifest = _build_manifest(version, files)

    print(f"Packaging bundle {version}")
    print(f"  files: {len(files)}  output: {zip_path}")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for arcname, src in files.items():
            zf.write(src, arcname)
    elapsed = time.time() - t0

    size = os.path.getsize(zip_path)
    print(f"  wrote {size/1024:.1f} KiB in {elapsed:.2f}s")

    pointer = {
        "version": version,
        "filename": zip_name,
        "url": f"/bundle/{version}/{zip_name}",
        "size": size,
        "sha256": _sha256(zip_path),
        "created_at": manifest["created_at"],
    }
    pointer_path = os.path.join(bundles_dir, "latest.json")
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)
    print(f"  updated {pointer_path}")
    return pointer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pictos-db", default=DEFAULT_PICTOS_DB)
    parser.add_argument("--phrases-db", default=DEFAULT_PHRASES_DB)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--bundles-dir", default=DEFAULT_BUNDLES_DIR)
    parser.add_argument("--version", default=None,
                        help="Override auto-generated version (UTC timestamp).")
    args = parser.parse_args()

    try:
        pointer = build_bundle(
            args.pictos_db, args.phrases_db, args.images_dir,
            args.bundles_dir, version=args.version,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone:")
    print(json.dumps(pointer, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
