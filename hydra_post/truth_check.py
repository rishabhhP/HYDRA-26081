"""Is ERA5 itself damping the peaks?  Compare the replay's ERA5 'observed' state rainfall with IMD gauge-based rainfall.

    python -m hydra_post.truth_check            -> reports/ERA5_VS_IMD_TRUTH_CHECK.md + runtime/era5_vs_imd.json

If ERA5 under-reports heavy days relative to IMD, part of HYDRA's peak error comes from its training target,
and retraining / verifying on IMD is the fix. If they agree, the problem is in the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M

ROOT = Path(__file__).resolve().parents[1]
BINS = ((0, 1), (1, 10), (10, 20), (20, 64.5), (64.5, np.inf))


def compare(df: pd.DataFrame) -> dict:
    imd, era, hyd = df["y"].to_numpy(), df["y_era5"].to_numpy(), df["hydra"].to_numpy()
    out = {"state_days": len(df), "states": int(df["state"].nunique()),
           "correlation_era5_imd": float(np.corrcoef(era, imd)[0, 1]),
           "mean_imd": float(imd.mean()), "mean_era5": float(era.mean()), "era5_minus_imd_bias": float((era - imd).mean()),
           "bins": []}
    for lo, hi in BINS:
        m = (imd >= lo) & (imd < hi)
        if m.any():
            out["bins"].append({"imd_bin": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}", "n": int(m.sum()),
                                "imd_mean": float(imd[m].mean()), "era5_mean": float(era[m].mean()), "hydra_mean": float(hyd[m].mean()),
                                "era5_ratio": float(era[m].mean() / max(imd[m].mean(), 1e-9)), "hydra_ratio": float(hyd[m].mean() / max(imd[m].mean(), 1e-9))})
    top = np.argsort(imd)[::-1][:20]
    out["top20_imd_days"] = {"imd_sum": float(imd[top].sum()), "era5_ratio": float(era[top].sum() / imd[top].sum()),
                             "hydra_ratio": float(hyd[top].sum() / imd[top].sum())}
    for t in (20.0, 64.5):
        out[f"events_ge{t:g}"] = {"imd": int((imd >= t).sum()), "era5": int((era >= t).sum()),
                                  "era5_detects_imd_events": M.contingency(imd >= t, era >= t)}
    out["hydra_vs_imd"] = {k: float(v) for k, v in M.point(imd, hyd).items() if k in ("mae", "rmse", "bias", "recall_20", "csi_20", "top5_ratio")}
    out["hydra_vs_era5"] = {k: float(v) for k, v in M.point(era, hyd).items() if k in ("mae", "rmse", "bias", "recall_20", "csi_20", "top5_ratio")}
    r20 = next((b for b in out["bins"] if b["imd_bin"] == "20-64.5"), None)
    if r20:
        share = (1 - r20["era5_ratio"]) / max(1 - r20["hydra_ratio"], 1e-9)
        out["share_of_heavy_shortfall_explained_by_era5"] = float(np.clip(share, 0, 1))
    return out


def markdown(o: dict) -> str:
    L = ["# ERA5 vs IMD: is the training target damping the peaks?", "",
         f"{o['state_days']:,} state-days across {o['states']} states. Correlation {o['correlation_era5_imd']:.2f}; "
         f"mean ERA5 − IMD {o['era5_minus_imd_bias']:+.2f} mm/day.", "",
         "| IMD bin (mm/day) | Days | IMD mean | ERA5 mean | HYDRA mean | ERA5 / IMD | HYDRA / IMD |", "|---|---|---|---|---|---|---|"]
    for b in o["bins"]:
        L.append(f"| {b['imd_bin']} | {b['n']} | {b['imd_mean']:.1f} | {b['era5_mean']:.1f} | {b['hydra_mean']:.1f} | {b['era5_ratio']:.2f} | {b['hydra_ratio']:.2f} |")
    t = o["top20_imd_days"]
    L += ["", f"20 wettest IMD state-days: ERA5 captured {t['era5_ratio']:.0%} of the rain, HYDRA {t['hydra_ratio']:.0%}.",
          f"Events ≥20 mm: IMD {o['events_ge20']['imd']}, ERA5 {o['events_ge20']['era5']}; ≥64.5 mm: IMD {o['events_ge64.5']['imd']}, ERA5 {o['events_ge64.5']['era5']}."]
    if "share_of_heavy_shortfall_explained_by_era5" in o:
        L.append(f"On 20-64.5 mm days, about {o['share_of_heavy_shortfall_explained_by_era5']:.0%} of HYDRA's shortfall against IMD is already present in ERA5 itself.")
    L += ["", "HYDRA scored against each truth:", "", "| Truth | MAE | RMSE | Bias | ≥20 recall | ≥20 CSI | Top-5 ratio |", "|---|---|---|---|---|---|---|"]
    for name, k in (("IMD", "hydra_vs_imd"), ("ERA5", "hydra_vs_era5")):
        m = o[k]
        L.append(f"| {name} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['bias']:+.2f} | {m['recall_20']:.1%} | {m['csi_20']:.3f} | {m['top5_ratio']:.2f} |")
    L += ["", "Reading: if ERA5 / IMD is well below 1 on heavy bins, retrain and verify HYDRA on IMD truth. If it is near 1, the damping is in the model."]
    return "\n".join(L) + "\n"


def main(imd_csv: Path | None = None) -> dict:
    df, payload = D.load(truth="imd", imd_csv=imd_csv)
    if (payload.get("truth") or {}).get("source") == "imd":
        raise SystemExit("This replay was trained on IMD rainfall (--target imd), so its 'observed' values are already IMD and "
                         "there is no ERA5 rainfall to compare. Keep the ERA5-trained replay's report for this check.")
    o = compare(df)
    (ROOT / "runtime").mkdir(exist_ok=True)
    (ROOT / "runtime" / "era5_vs_imd.json").write_text(json.dumps(o, indent=1))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "ERA5_VS_IMD_TRUTH_CHECK.md").write_text(markdown(o), encoding="utf-8")
    print(markdown(o))
    return o


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--imd-csv", type=Path)
    main(ap.parse_args().imd_csv)
