"""Snapshot, check and roll back HYDRA releases so a retrain can never silently replace a better model.

    python scripts/publish_release.py snapshot            # BEFORE retraining: keep the currently published files
    ... retrain, rebuild replay, hydra_post.evaluate (both leads), hydra_post.apply ...
    python scripts/publish_release.py check               # gate the new model, including "no worse than the snapshot"
    python scripts/publish_release.py rollback            # restore the last snapshot if the check fails
    python scripts/publish_release.py list
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
RELEASES = RUNTIME / "releases"
FILES = ["hydra_rolling_rainfall_replay.json", "hydra_v3_1_postprocessed.json", "hydra_v3_1_evaluation.json",
         "hydra_v3_1_evaluation_imd.json", "hydra_v3_1_evaluation_lead2.json", "hydra_v3_1_evaluation_lead2_imd.json",
         "hydra_v3_1_oos_predictions.csv", "hydra_v3_1_oos_predictions_imd.csv", "hydra_release_gate.json"]
DIRS = ["hydra_v3_1", "hydra_rain_v3"]


def snapshots() -> list[Path]:
    return sorted(p for p in RELEASES.glob("*") if p.is_dir()) if RELEASES.exists() else []


def snapshot(note: str) -> Path:
    dest = RELEASES / datetime.now().strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True)
    saved = []
    for name in FILES:
        if (RUNTIME / name).exists():
            shutil.copy2(RUNTIME / name, dest / name)
            saved.append(name)
    for name in DIRS:
        if (RUNTIME / name).is_dir():
            shutil.copytree(RUNTIME / name, dest / name)
            saved.append(name + "/")
    (dest / "release.json").write_text(json.dumps({"created": dest.name, "note": note, "files": saved}, indent=1))
    print(f"snapshot {dest.name}: {len(saved)} items")
    return dest


def check() -> int:
    snaps = snapshots()
    baseline = None
    if snaps:
        for name in ("hydra_v3_1_evaluation_imd.json", "hydra_v3_1_evaluation.json"):
            if (snaps[-1] / name).exists():
                baseline = snaps[-1] / name
                break
    cmd = [sys.executable, "-m", "hydra_post.release"] + (["--baseline", str(baseline)] if baseline else [])
    print("baseline:", baseline or "none (first release)")
    if baseline and baseline.name.endswith("_imd.json") != (RUNTIME / "hydra_v3_1_evaluation_imd.json").exists():
        print("WARNING: the snapshot and the new evaluation use different truths (IMD vs ERA5); the comparison is not like for like.")
    return subprocess.run(cmd, cwd=ROOT).returncode


def rollback() -> None:
    snaps = snapshots()
    if not snaps:
        raise SystemExit("no snapshot to roll back to")
    src = snaps[-1]
    for name in FILES:
        if (src / name).exists():
            shutil.copy2(src / name, RUNTIME / name)
        elif (RUNTIME / name).exists():
            (RUNTIME / name).unlink()           # file did not exist in the release being restored
    for name in DIRS:
        if (src / name).is_dir():
            shutil.rmtree(RUNTIME / name, ignore_errors=True)
            shutil.copytree(src / name, RUNTIME / name)
    print(f"restored release {src.name}; restart the API to reload")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["snapshot", "check", "rollback", "list"])
    p.add_argument("--note", default="")
    a = p.parse_args()
    if a.action == "snapshot":
        snapshot(a.note)
    elif a.action == "check":
        sys.exit(check())
    elif a.action == "rollback":
        rollback()
    else:
        for s in snapshots():
            meta = json.loads((s / "release.json").read_text())
            print(s.name, meta.get("note", ""), f"({len(meta['files'])} items)")


if __name__ == "__main__":
    main()
