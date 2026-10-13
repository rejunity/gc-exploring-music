"""Download, verify and extract the corpora listed in data/manifest.json.

    uv run python -m evomusic.data list
    uv run python -m evomusic.data download            # all "core" datasets
    uv run python -m evomusic.data download pop909 nottingham
    uv run python -m evomusic.data download --tier large
    uv run python -m evomusic.data download --record   # (re)compute hashes into the manifest
    uv run python -m evomusic.data verify              # re-hash what is on disk vs manifest

Layout: data/raw/<name>/<filename> plus, for archives, data/raw/<name>/extracted/.
Nothing under data/raw is committed; the manifest with hashes is.
Stdlib only, so it works before `uv sync`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
RAW = ROOT / "data" / "raw"
CHUNK = 1 << 20


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash of (relative path, file hash) over every file, sorted. Independent of
    archive compression, mtimes and permissions."""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "evomusic-data/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            if not sys.stderr.isatty():
                continue
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
            else:
                print(f"\r  {done / 1e6:7.1f} MB", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    tmp.replace(dest)


def _safe_members(root: Path, names: list[str]) -> None:
    for n in names:
        if not (root / n).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"archive member escapes target dir: {n}")


def extract(archive: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            _safe_members(out, z.namelist())
            z.extractall(out)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            _safe_members(out, t.getnames())
            t.extractall(out)  # members checked above; Python 3.12+ could use filter="data"
    else:
        raise RuntimeError(f"don't know how to extract {archive.name}")


def status_line(name: str, d: dict, ok: bool | None) -> str:
    mark = {True: "ok ", False: "BAD", None: "-- "}[ok]
    size = f"{d['size'] / 1e6:7.1f} MB" if d.get("size") else "      ? MB"
    return f"{mark} {name:18s} {size}  {d['tier']:5s} {d['style']}"


def fetch(name: str, d: dict, record: bool, force: bool) -> bool:
    folder = RAW / name
    dest = folder / d["filename"]
    if dest.exists() and not force:
        print(f"[{name}] have {dest.relative_to(ROOT)}")
    else:
        print(f"[{name}] downloading {d['url']}")
        download(d["url"], dest)

    digest = sha256_file(dest)
    size = dest.stat().st_size
    ok = True
    if d.get("sha256") and digest != d["sha256"]:
        print(f"[{name}] WARNING archive sha256 mismatch\n   manifest {d['sha256']}\n   on disk  {digest}")
        ok = False

    tree = None
    if d.get("extract"):
        out = folder / "extracted"
        if force or not out.exists():
            print(f"[{name}] extracting")
            extract(dest, out)
        tree = sha256_tree(out)
        if d.get("tree_sha256") and tree != d["tree_sha256"]:
            print(f"[{name}] WARNING extracted tree sha256 mismatch: content changed")
            ok = False
        elif not ok and d.get("tree_sha256") == tree:
            print(f"[{name}] note: archive bytes differ but extracted content is identical (host recompressed)")
            ok = True

    if record or not d.get("sha256"):
        d["sha256"], d["size"], d["tree_sha256"] = digest, size, tree
        print(f"[{name}] recorded sha256={digest[:12]}… size={size}")
    return ok


def verify(name: str, d: dict) -> bool | None:
    dest = RAW / name / d["filename"]
    if not dest.exists():
        return None
    if not d.get("sha256"):
        return None
    ok = sha256_file(dest) == d["sha256"]
    if d.get("extract") and d.get("tree_sha256") and (RAW / name / "extracted").exists():
        ok = ok and sha256_tree(RAW / name / "extracted") == d["tree_sha256"]
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    dl = sub.add_parser("download")
    dl.add_argument("names", nargs="*")
    dl.add_argument("--tier", choices=["core", "large", "all"], default="core")
    dl.add_argument("--record", action="store_true", help="write fresh hashes into the manifest")
    dl.add_argument("--force", action="store_true", help="re-download and re-extract")
    sub.add_parser("verify")
    a = ap.parse_args(argv)

    m = load_manifest()
    ds = m["datasets"]

    if a.cmd == "list":
        for n, d in ds.items():
            print(status_line(n, d, verify(n, d)))
        return 0

    if a.cmd == "verify":
        bad = 0
        for n, d in ds.items():
            ok = verify(n, d)
            print(status_line(n, d, ok))
            bad += ok is False
        return 1 if bad else 0

    names = a.names or [n for n, d in ds.items() if a.tier == "all" or d["tier"] == a.tier]
    unknown = [n for n in names if n not in ds]
    if unknown:
        sys.exit(f"unknown dataset(s): {unknown}; see `list`")
    all_ok = True
    for n in names:
        try:
            all_ok &= fetch(n, ds[n], a.record, a.force)
        except Exception as e:  # keep going; report at the end
            print(f"[{n}] FAILED: {e}")
            all_ok = False
    save_manifest(m)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
