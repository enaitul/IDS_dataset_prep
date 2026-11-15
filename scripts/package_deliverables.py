"""Build the Member 1 handoff archives and their checksum manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
LIMIT = 2 * 1024**3
DATASETS = ("NSL-KDD", "InSDN", "CICIDS2017")


def files_under(*roots: Path) -> list[Path]:
    result = []
    for base in roots:
        if base.is_file():
            result.append(base)
        elif base.exists():
            result.extend(p for p in base.rglob("*") if p.is_file())
    return sorted(set(result))


def add_tree(zf: zipfile.ZipFile, paths: list[Path]) -> None:
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts or "data/raw" in rel or "data/processed/cleaned" in rel:
            continue
        zf.write(path, rel, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def make_zip(name: str, paths: list[Path]) -> Path:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", allowZip64=True) as zf:
        add_tree(zf, paths)
    return target


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_model(ds: str) -> list[Path]:
    tree = ROOT / "data/processed" / ds
    model_paths = files_under(tree)
    model_paths = [p for p in model_paths if "data/processed/cleaned" not in p.as_posix()]
    split_paths = files_under(ROOT / "data/splits")
    split_paths = [p for p in split_paths if p.name.startswith(ds + "_") and p.suffix == ".json"]
    combined = model_paths + split_paths
    # Parquet files are already compressed, so uncompressed total is a close
    # estimate and avoids spending several minutes creating an oversized archive.
    if sum(p.stat().st_size for p in combined) <= LIMIT:
        target = make_zip(f"Member1_for_models_{ds}.zip", combined)
        if target.stat().st_size <= LIMIT:
            return [target]
        target.unlink()
    # Large datasets are divided by feature configuration and representation;
    # shared artifacts and split indices go into a separate archive.
    outputs = [make_zip(f"Member1_for_models_{ds}_shared.zip", files_under(tree / "artifacts", tree / "labels", tree / "external_test_binary") + split_paths)]
    for cfg in ("all_features", "info_gain", "rfe", "info_gain_rfe"):
        for rep in ("tree", "mlp"):
            paths = files_under(tree / rep / cfg)
            if not paths:
                continue
            archive = make_zip(f"Member1_for_models_{ds}_{cfg}_{rep}.zip", paths)
            if archive.stat().st_size > LIMIT:
                raise RuntimeError(f"{archive.name} exceeds 2 GiB after feature-set packaging")
            outputs.append(archive)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-result", required=True, help="Final pytest summary, e.g. 'passed (N tests)'")
    args = parser.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    paper = files_under(ROOT / "outputs", ROOT / "data/metadata/label_mapping.csv", ROOT / "configs", ROOT / "CHANGELOG_member1.md")
    archives = [make_zip("Member1_for_paper.zip", paper)]
    for ds in DATASETS:
        archives.extend(package_model(ds))
    code_paths = files_under(*(ROOT / p for p in (
        "src", "tests", "examples", "configs", "scripts", "run_pipeline.py", "check_raw.py",
        "requirements.txt", "README.md", "Makefile", "CHANGELOG_member1.md")))
    archives.append(make_zip("Member1_code.zip", code_paths))

    versions = {
        "python": platform.python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "imbalanced-learn": importlib.metadata.version("imbalanced-learn"),
        "xgboost": importlib.metadata.version("xgboost"),
    }
    manifest = ["# Member 1 Deliverables Manifest", "", "## Archives", "",
                "| Archive | Size (bytes) | Files | SHA-256 |", "|---|---:|---:|---|"]
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            count = sum(1 for item in zf.infolist() if not item.is_dir())
        manifest.append(f"| `{archive.name}` | {archive.stat().st_size} | {count} | `{sha256(archive)}` |")
    manifest += ["", "## Run details", "", f"- Final pytest result: {args.pytest_result}",
                 f"- Seed: read from `configs/config.yaml` = {__import__('yaml').safe_load((ROOT / 'configs/config.yaml').read_text())['seed']}",
                 "- Library versions:"]
    manifest += [f"  - {name}: {version}" for name, version in versions.items()]
    manifest += ["", "Archives above 2 GiB are split by feature set and representation, listed as separate archives." , ""]
    (OUT / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    print("\n".join(manifest))
    print("\nArchive sizes:")
    for archive in archives:
        print(f"{archive.name}: {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
