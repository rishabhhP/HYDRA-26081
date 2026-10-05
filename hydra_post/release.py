"""Release gate: may the post-processed HYDRA be presented as validated?

    python -m hydra_post.release                                   # gate on the newest evaluation (IMD truth preferred)
    python -m hydra_post.release --baseline runtime/previous/hydra_v3_1_evaluation_imd.json   # also require no regression vs the old model

Exit code 0 = validated, 1 = provisional. The backend shows the same verdict next to every number it serves.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
FINAL, RAW = "FINAL (history-selected)", "raw HYDRA v3"


def evaluation_path(lead: int = 1) -> Path | None:
    suffix = "" if lead == 1 else f"_lead{lead}"
    for truth in ("_imd", ""):
        p = RUNTIME / f"hydra_v3_1_evaluation{suffix}{truth}.json"
        if p.exists():
            return p
    return None


def _check(name, ok, detail, required=True):
    return {"check": name, "passed": bool(ok), "required": required, "detail": detail}


def gate(evaluation: dict, baseline: dict | None = None, lead2: dict | None = None, stale: bool = False) -> dict:
    final_iv = evaluation.get("interval_final", "asymmetric split conformal")
    a, iv, pr = evaluation["amount"], evaluation["interval"][final_iv], evaluation["probability"]
    f, r = a[FINAL], a[RAW]
    lose = evaluation["states_nonpositive_mae_skill_vs_persistence"]
    checks = [
        _check("MAE not worse than raw v3", f["mae"] <= r["mae"], f"{f['mae']:.3f} vs {r['mae']:.3f} mm/day"),
        _check("RMSE not worse than raw v3", f["rmse"] <= r["rmse"], f"{f['rmse']:.3f} vs {r['rmse']:.3f} mm/day"),
        _check("No more states losing to persistence", lose["final"] <= lose["raw"], f"{lose['final']} vs {lose['raw']} of 36"),
        _check("80% interval coverage within 75-85%", 0.75 <= iv["coverage"] <= 0.85, f"{iv['coverage']:.1%} ({final_iv})"),
        _check("Interval tails balanced (within 4 points)", abs(iv["above"] - iv["below"]) <= 0.04,
               f"{iv['above']:.1%} above, {iv['below']:.1%} below"),
        _check("P(>=10 mm) calibrated (Brier skill > 0)", pr["state_10"]["FINAL calibrated"]["bss"] > 0, f"{pr['state_10']['FINAL calibrated']['bss']:.3f}"),
        _check("P(>=20 mm) calibrated (Brier skill > 0)", pr["state_20"]["FINAL calibrated"]["bss"] > 0, f"{pr['state_20']['FINAL calibrated']['bss']:.3f}"),
        _check("Post-processor rebuilt after the latest replay", not stale, "up to date" if not stale else "replay is newer: rerun python -m hydra_post.apply"),
        _check("Lead 2 (+48h) evaluated", lead2 is not None, "evaluated" if lead2 else "run python -m hydra_post.evaluate --lead 2", required=False),
    ]
    if baseline:
        bf = baseline["amount"][FINAL]
        bl = baseline["states_nonpositive_mae_skill_vs_persistence"]["final"]
        checks += [_check("MAE not worse than the previous release", f["mae"] <= bf["mae"] + 1e-9, f"{f['mae']:.3f} vs {bf['mae']:.3f}"),
                   _check("RMSE not worse than the previous release", f["rmse"] <= bf["rmse"] + 1e-9, f"{f['rmse']:.3f} vs {bf['rmse']:.3f}"),
                   _check("No more losing states than the previous release", lose["final"] <= bl, f"{lose['final']} vs {bl}")]
    failed = [c["check"] for c in checks if c["required"] and not c["passed"]]
    return {"status": "validated" if not failed else "provisional", "failed": failed, "checks": checks,
            "truth": evaluation.get("design", {}).get("truth", "era5"), "test_period": evaluation.get("design", {}).get("test_period"),
            "test_state_days": evaluation.get("design", {}).get("test_state_days")}


def is_stale() -> bool:
    replay, post = RUNTIME / "hydra_rolling_rainfall_replay.json", RUNTIME / "hydra_v3_1_postprocessed.json"
    return replay.exists() and post.exists() and replay.stat().st_mtime > post.stat().st_mtime + 1


def current(baseline: Path | None = None) -> dict:
    p1 = evaluation_path(1)
    if not p1:
        return {"status": "not_evaluated", "failed": ["no evaluation found: run python -m hydra_post.evaluate"], "checks": []}
    p2 = evaluation_path(2)
    out = gate(json.loads(p1.read_text()), json.loads(baseline.read_text()) if baseline else None,
               json.loads(p2.read_text()) if p2 else None, is_stale())
    out["evaluation_file"] = p1.name
    return out


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, help="evaluation JSON of the release currently published")
    a = ap.parse_args()
    g = current(a.baseline)
    (RUNTIME / "hydra_release_gate.json").write_text(json.dumps(g, indent=1))
    for c in g["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL' if c['required'] else 'warn'}] {c['check']}: {c['detail']}")
    print(f"\nRelease status: {g['status'].upper()}" + (f"  (failed: {', '.join(g['failed'])})" if g["failed"] else ""))
    sys.exit(0 if g["status"] == "validated" else 1)
