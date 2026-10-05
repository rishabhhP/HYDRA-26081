"""Turn downloaded IMD files into runtime/imd_state_daily.csv (used as truth by hydra_post) plus an extremes summary.

    python -m hydra_imd.build --dir data/raw/imd
    python -m hydra_imd.build --dir data/raw/imd --realtime 2025-07-01:2025-12-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import open_dir
from .states import state_daily

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "imd_state_daily.csv"


def extremes_summary(df: pd.DataFrame) -> dict:
    """How much heavy-rain evidence the IMD record provides (the diagnostic's 'too few cases' problem)."""
    df = df.assign(year=df["date"].dt.year)
    per_year = df.groupby("year").agg(state_days=("imd_mm", "size"), ge20=("imd_mm", lambda s: int((s >= 20).sum())),
                                      ge64_5=("imd_mm", lambda s: int((s >= 64.5).sum())),
                                      any_cell_ge64_5=("imd_local_max_mm", lambda s: int((s >= 64.5).sum())))
    per_state = df.groupby("state").agg(p95=("imd_mm", lambda s: float(s.quantile(0.95))),
                                        p99=("imd_mm", lambda s: float(s.quantile(0.99))),
                                        max=("imd_mm", "max"), ge20=("imd_mm", lambda s: int((s >= 20).sum())))
    return {"years": per_year.reset_index().to_dict("records"), "totals": {k: int(per_year[k].sum()) for k in per_year},
            "states": per_state.round(2).to_dict("index")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=ROOT / "data" / "raw" / "imd")
    p.add_argument("--realtime", help="also read IMD real-time files START:END")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    imd = open_dir(a.dir, realtime=tuple(a.realtime.split(":")) if a.realtime else None)
    print(f"IMD grid {imd.rain.shape[1]}x{imd.rain.shape[2]}, {len(imd.dates)} days ({imd.dates[0].date()} to {imd.dates[-1].date()})")
    df = state_daily(imd)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summary = extremes_summary(df)
    (a.out.parent / "imd_extremes_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    t = summary["totals"]
    print(f"wrote {a.out}: {len(df):,} state-days, {df['state'].nunique()} states")
    print(f"heavy-rain evidence: {t['ge20']} state-days >=20 mm, {t['ge64_5']} >=64.5 mm, {t['any_cell_ge64_5']} days with a cell >=64.5 mm")


if __name__ == "__main__":
    main()
