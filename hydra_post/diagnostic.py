"""Re-create the HYDRA_V3_DIAGNOSTIC.md headline table from any replay, so before/after numbers use identical code."""
from __future__ import annotations

import numpy as np

from . import data as D
from . import metrics as M


def headline(df) -> dict:
    y, p = df["y"].to_numpy(), df["hydra"].to_numpy()
    pt = M.point(y, p)
    occ = M.contingency(y >= 1, df["p_rain"].to_numpy() >= 0.5)
    occ_amount = M.contingency(y >= 1, p >= 1)
    iv = M.interval(y, df["lo80"], df["hi80"])
    sk = M.skill(y, p, df["e_persistence"])
    return {"mae": pt["mae"], "rmse": pt["rmse"], "bias": pt["bias"],
            "occurrence_recall_prain": occ["recall"], "occurrence_precision_prain": occ["precision"],
            "occurrence_recall_amount": occ_amount["recall"], "occurrence_precision_amount": occ_amount["precision"],
            "recall_10": pt["recall_10"], "recall_20": pt["recall_20"], "csi_20": pt["csi_20"], "coverage": iv["coverage"],
            "above": iv["above"], "below": iv["below"], "peak_obs": pt["peak_obs"], "peak_pred": pt["peak_pred"],
            "top5_ratio": pt["top5_ratio"], "mae_skill_persistence": sk["mae_skill"], "rmse_skill_persistence": sk["rmse_skill"],
            "events_64.5": pt["events_64.5"], "monsoon_weight": float(df["w_monsoon"].mean()) if "w_monsoon" in df else np.nan}


if __name__ == "__main__":
    import json
    df, _ = D.load()
    print(json.dumps({k: round(float(v), 4) for k, v in headline(df).items()}, indent=1))
