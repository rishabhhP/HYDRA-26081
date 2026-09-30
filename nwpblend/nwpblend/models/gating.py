"""(b) Neural gating network (mixture-of-experts blender).

    context x  --MLP-->  logits (K)  --softmax-->  w (K, sums to 1)
    forecast   =  sum_k  w_k * expert_k

Trained end-to-end on squared forecast error (normalised by the target's train variance),
so the weights are learned *through* the blend rather than fitted post-hoc per source.
Output is always a convex combination of the experts -> every forecast is explainable as
"x% persistence, y% climatology, ...".

``static=True`` builds the non-adaptive comparison: the only inputs are season x lead
one-hots, so it learns one fixed weight vector per (season, lead).

Weights are exported to a plain .npz + .json so a backend can run inference with numpy
only (see ``numpy_forward``); torch is needed only for training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import config as C
from ..features import CONTEXT_COLS

GATE_CATS = ["season_id", "regime_id", "region_id", "lead_bucket_id"]
EXPERT_COLS = [f"exp_{e}" for e in C.EXPERTS]


class _Net(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], hidden: list[int], k: int, emb_dim: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, min(emb_dim, n)) for n in cat_sizes])
        d = n_num + sum(e.embedding_dim for e in self.embs)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
            d = h
        layers.append(nn.Linear(d, k))
        self.mlp = nn.Sequential(*layers)

        self.anchor = None

    def add_anchor(self, n_anchor: int, k: int):
        """Static (season x lead) logit table; the MLP learns a zero-initialised correction on top."""
        self.anchor = nn.Embedding(n_anchor, k)
        nn.init.zeros_(self.anchor.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, xnum, xcat, xanchor=None):
        z = [xnum] + [e(xcat[:, i]) for i, e in enumerate(self.embs)]
        out = self.mlp(torch.cat(z, 1))
        if self.anchor is not None:
            out = out + self.anchor(xanchor)
        return out


class GatingBlender:
    def __init__(self, static: bool = False, hidden: list[int] | None = None,
                 exclude: tuple[str, ...] = C.GATING_EXCLUDE, dropout: float = C.GATING_DROPOUT,
                 weight_decay: float = C.GATING_WEIGHT_DECAY, anchored: bool = C.GATING_ANCHORED,
                 lr: float = C.GATING_LR):
        self.static = static
        self.hidden = hidden if hidden is not None else ([] if static else C.GATING_HIDDEN)
        self.dropout, self.weight_decay = (0.0 if static else dropout), weight_decay
        self.anchored, self.lr = (anchored and not static), lr
        if static:
            self.num_cols, self.cat_cols = [], ["season_id", "lead_days"]
        else:
            self.num_cols = [c for c in CONTEXT_COLS + EXPERT_COLS if c not in GATE_CATS and c not in exclude]
            self.cat_cols = [c for c in GATE_CATS if c not in exclude]
        self.vocab: dict[str, list[int]] = {}
        self.med = self.mu = self.sd = None
        self.y_scale = 1.0
        self.net: _Net | None = None
        self.history: list[dict] = []

    # ---------------- preprocessing
    def _fit_prep(self, df: pd.DataFrame):
        for c in self.cat_cols:
            self.vocab[c] = sorted(int(v) for v in pd.unique(df[c]))
        if self.num_cols:
            X = df[self.num_cols].to_numpy(np.float64)
            self.med = np.nanmedian(X, 0)
            X = np.where(np.isfinite(X), X, self.med)
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def _prep(self, df: pd.DataFrame):
        if self.num_cols:
            X = df[self.num_cols].to_numpy(np.float64)
            X = np.where(np.isfinite(X), X, self.med)
            X = ((X - self.mu) / self.sd).astype(np.float32)
        else:
            X = np.zeros((len(df), 0), np.float32)
        cats = []
        for c in self.cat_cols:
            lut = {v: i + 1 for i, v in enumerate(self.vocab[c])}  # 0 = unseen
            cats.append(pd.Series(df[c].to_numpy()).map(lut).fillna(0).to_numpy(np.int64))
        Xc = np.stack(cats, 1) if cats else np.zeros((len(df), 0), np.int64)
        E = df[EXPERT_COLS].to_numpy(np.float32)
        A = (df["season_id"].to_numpy(np.int64) * len(C.LEADS)
             + np.searchsorted(C.LEADS, df["lead_days"].to_numpy())).astype(np.int64)
        return X, Xc, E, A

    # ---------------- training
    def static_logits(self) -> np.ndarray:
        """(n_seasons*n_leads, K) log-weights of a fitted static blend, in anchor-index order."""
        probe = pd.DataFrame({"season_id": np.repeat(np.arange(len(C.SEASONS)), len(C.LEADS)),
                              "lead_days": np.tile(C.LEADS, len(C.SEASONS)),
                              **{c: 0.0 for c in EXPERT_COLS}})
        return np.log(np.clip(self.weights(probe), 1e-6, 1)).astype(np.float32)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, seed: int = C.SEED,
            anchor_init: np.ndarray | None = None) -> "GatingBlender":
        torch.manual_seed(seed)
        torch.set_num_threads(16)
        if len(train) > C.GATING_MAX_TRAIN_ROWS:
            train = train.sample(n=C.GATING_MAX_TRAIN_ROWS, random_state=seed)
        self._fit_prep(train)
        self.y_scale = float(train["target"].std())
        Xtr, Ctr, Etr, Atr = (torch.from_numpy(a) for a in self._prep(train))
        ytr = torch.from_numpy(train["target"].to_numpy(np.float32))
        Xva, Cva, Eva, Ava = (torch.from_numpy(a) for a in self._prep(val))
        yva = torch.from_numpy(val["target"].to_numpy(np.float32))
        self.net = _Net(Xtr.shape[1], [len(self.vocab[c]) + 1 for c in self.cat_cols], self.hidden, len(C.EXPERTS),
                        dropout=self.dropout)
        if self.anchored:
            self.net.add_anchor(len(C.SEASONS) * len(C.LEADS), len(C.EXPERTS))
            if anchor_init is not None:  # start exactly at the fitted static blend
                with torch.no_grad():
                    self.net.anchor.weight.copy_(torch.from_numpy(anchor_init))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
        best, best_state, bad = np.inf, None, 0
        n = len(ytr)
        for ep in range(C.GATING_EPOCHS):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for b in range(0, n, C.GATING_BATCH):
                ix = perm[b:b + C.GATING_BATCH]
                w = torch.softmax(self.net(Xtr[ix], Ctr[ix], Atr[ix]), 1)
                pred = (w * Etr[ix]).sum(1)
                loss = (((pred - ytr[ix]) / self.y_scale) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(ix)
            vloss = self._loss(Xva, Cva, Eva, Ava, yva)
            sched.step(vloss)
            self.history.append({"epoch": ep, "train_mse_norm": tot / n, "val_mse_norm": vloss})
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= C.GATING_PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _loss(self, X, Cc, E, A, y) -> float:
        self.net.eval()
        tot = 0.0
        for b in range(0, len(y), 65536):
            w = torch.softmax(self.net(X[b:b + 65536], Cc[b:b + 65536], A[b:b + 65536]), 1)
            tot += (((w * E[b:b + 65536]).sum(1) - y[b:b + 65536]) / self.y_scale).pow(2).sum().item()
        return tot / len(y)

    # ---------------- inference
    @torch.no_grad()
    def weights(self, df: pd.DataFrame) -> np.ndarray:
        X, Xc, _, A = self._prep(df)
        self.net.eval()
        out = []
        B = 131072
        for b in range(0, len(df), B):
            out.append(torch.softmax(self.net(torch.from_numpy(X[b:b + B]), torch.from_numpy(Xc[b:b + B]),
                                              torch.from_numpy(A[b:b + B])), 1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(C.EXPERTS)))

    def predict(self, df: pd.DataFrame, return_weights: bool = False):
        w = self.weights(df)
        p = (w * df[EXPERT_COLS].to_numpy(np.float32)).sum(1)
        return (p, w) if return_weights else p

    # ---------------- portable export
    def save(self, d: Path, name: str) -> None:
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), d / f"{name}.pt")
        arrays = {k: v.numpy() for k, v in self.net.state_dict().items()}
        if self.num_cols:
            arrays.update(med=self.med, mu=self.mu, sd=self.sd)
        np.savez(d / f"{name}.npz", **arrays)
        (d / f"{name}.json").write_text(json.dumps({
            "static": self.static, "hidden": self.hidden, "num_cols": self.num_cols, "cat_cols": self.cat_cols,
            "vocab": self.vocab, "experts": C.EXPERTS, "expert_cols": EXPERT_COLS, "y_scale": self.y_scale,
            "activation": "gelu", "dropout": self.dropout, "anchored": self.anchored, "leads": C.LEADS,
            "anchor_index": "season_id * n_leads + index(lead_days in leads)", "history": self.history}, indent=1))

    @classmethod
    def load(cls, d: Path, name: str) -> "GatingBlender":
        meta = json.loads((d / f"{name}.json").read_text())
        obj = cls(static=meta["static"], hidden=meta["hidden"], dropout=meta.get("dropout", 0.0),
                  anchored=meta.get("anchored", False))
        obj.num_cols, obj.cat_cols = meta["num_cols"], meta["cat_cols"]
        obj.vocab = {k: v for k, v in meta["vocab"].items()}
        obj.y_scale = meta["y_scale"]
        z = np.load(d / f"{name}.npz")
        if obj.num_cols:
            obj.med, obj.mu, obj.sd = z["med"], z["mu"], z["sd"]
        obj.net = _Net(len(obj.num_cols), [len(obj.vocab[c]) + 1 for c in obj.cat_cols], obj.hidden, len(C.EXPERTS),
                       dropout=obj.dropout)
        if obj.anchored:
            obj.net.add_anchor(len(C.SEASONS) * len(C.LEADS), len(C.EXPERTS))
        obj.net.load_state_dict(torch.load(d / f"{name}.pt"))
        return obj


def numpy_forward(npz: dict, meta: dict, df: pd.DataFrame) -> np.ndarray:
    """Torch-free gating inference: returns softmax weights (n, K). GELU is the exact erf form."""
    from scipy.special import erf as verf

    parts = []
    if meta["num_cols"]:
        X = df[meta["num_cols"]].to_numpy(np.float64)
        X = np.where(np.isfinite(X), X, npz["med"])
        parts.append((X - npz["mu"]) / npz["sd"])
    for i, c in enumerate(meta["cat_cols"]):
        lut = {int(v): j + 1 for j, v in enumerate(meta["vocab"][c])}
        idx = pd.Series(df[c].to_numpy()).map(lut).fillna(0).to_numpy(int)
        parts.append(npz[f"embs.{i}.weight"][idx])
    h = np.concatenate(parts, 1)
    lin = sorted({int(k.split(".")[1]) for k in npz if k.startswith("mlp.") and k.endswith(".weight")})
    n_lin = len(lin)
    for j, li in enumerate(lin):
        W, b = npz[f"mlp.{li}.weight"], npz[f"mlp.{li}.bias"]
        h = h @ W.T + b
        if j < n_lin - 1:
            h = 0.5 * h * (1 + verf(h / np.sqrt(2)))
    if meta.get("anchored"):
        a = df["season_id"].to_numpy(int) * len(meta["leads"]) + np.searchsorted(meta["leads"], df["lead_days"].to_numpy())
        h = h + npz["anchor.weight"][a]
    h = h - h.max(1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(1, keepdims=True)
