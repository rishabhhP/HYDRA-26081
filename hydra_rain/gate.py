"""HYDRA v3 neural gate with occurrence, heavy-rain, quantile and wet-amount heads.

    context (ERA5 state, rainfall history, neighbourhood, regime, out-of-fold experts)
        --MLP trunk-->  h
    h --> gate logits (K) --softmax--> w ;  blend = sum_k w_k * expert_k       (mm/day)
    h --> P(rain >= 1 mm)
    h --> P(rain >= 20), P(rain >= 64.5), P(rain >= state wet-day p95)
    h --> lower / upper offsets around the blend (pinball at 0.10 / 0.90, conformalised later)
    h --> amount if wet (mm/day)

Why v2's weights barely moved: the gate saw state-mean context, every expert was a rainfall
persistence variant, the correction layer was zero-initialised on top of a static anchor and
regularised (weight decay 1e-3, 12 epochs). v3 removes the anchor, feeds the atmospheric state
and the trained experts, lowers weight decay, trains longer with early stopping, adds a small
entropy penalty, and reports weight dynamics so a static gate is visible.

The point loss is extreme-aware: squared error is weighted by 1 + alpha * min(y / 20 mm, cap),
so the network cannot buy a low average loss by always forecasting moderate rain.

Training uses torch. Inference is pure numpy from the exported .npz/.json (backend friendly).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from . import config as C

GATE_CATS = ("season_id", "regime_id", "state_id", "lead_days")


def _gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def _softplus(x):
    return np.logaddexp(0, x)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gate_inputs(rows: pd.DataFrame, experts: pd.DataFrame, numeric: list[str], expert_names, aux_names):
    """Numeric gate matrix: features + log1p expert values + aux + expert disagreement."""
    feats = rows[numeric].to_numpy(np.float64)
    ex = np.log1p(np.clip(experts[list(expert_names)].to_numpy(np.float64), 0, None))
    # Pandas may return a read-only view here.  The wet-amount transform below
    # is in-place, so take an owned array before modifying it.
    aux = experts[list(aux_names)].to_numpy(np.float64, copy=True)
    aux[:, 1:] = np.log1p(np.clip(aux[:, 1:], 0, None))
    spread = ex.std(1, keepdims=True)
    return np.concatenate([feats, ex, aux, spread], 1)


class RainGate:
    def __init__(self, numeric: list[str], expert_names, aux_names, heavy_keys=C.HEAVY_KEYS,
                 hidden=C.GATE_HIDDEN, n_states: int = 40):
        self.numeric, self.experts, self.aux = list(numeric), list(expert_names), list(aux_names)
        self.heavy_keys = list(heavy_keys)
        self.hidden = list(hidden)
        self.input_names = self.numeric + [f"log1p_{e}" for e in self.experts] + self.aux + ["expert_log_spread"]
        self.vocab = {"season_id": list(range(len(C.SEASONS))), "regime_id": list(range(len(C.REGIMES))),
                      "state_id": list(range(n_states)), "lead_days": list(C.LEADS)}
        self.med = self.mu = self.sd = None
        self.y_scale = 1.0
        self.params: dict[str, np.ndarray] = {}
        self.history: list[dict] = []

    # -------------------------------------------------- preprocessing
    @property
    def emb_dims(self):
        return [min(4, len(self.vocab[c]) + 1) for c in GATE_CATS]

    def _cats(self, rows: pd.DataFrame) -> np.ndarray:
        out = []
        for c in GATE_CATS:
            lut = {int(v): i + 1 for i, v in enumerate(self.vocab[c])}
            out.append(pd.Series(rows[c].to_numpy()).map(lut).fillna(0).to_numpy(np.int64))
        return np.stack(out, 1)

    def _fit_scaler(self, X: np.ndarray):
        self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        X = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isfinite(X), X, self.med)
        return ((X - self.mu) / self.sd).astype(np.float32)

    # -------------------------------------------------- numpy inference
    def forward(self, rows: pd.DataFrame, experts: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._scale(gate_inputs(rows, experts, self.numeric, self.experts, self.aux))
        idx = self._cats(rows)
        parts = [X] + [self.params[f"embs.{i}.weight"][idx[:, i]] for i in range(len(GATE_CATS))]
        h = np.concatenate(parts, 1)
        for j in range(len(self.hidden)):
            h = _gelu(h @ self.params[f"trunk.{j}.weight"].T + self.params[f"trunk.{j}.bias"])
        head = lambda name: h @ self.params[f"{name}.weight"].T + self.params[f"{name}.bias"]  # noqa: E731
        logits = head("gate")
        logits -= logits.max(1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(1, keepdims=True)
        E = experts[self.experts].to_numpy(np.float64)
        blend = (w * E).sum(1)
        q = head("quant")
        lower = np.clip(blend - _softplus(q[:, 0]) * self.y_scale, 0, None)
        upper = blend + _softplus(q[:, 1]) * self.y_scale
        heavy = _sigmoid(head("heavy"))
        return {
            "weights": w, "blend": np.clip(blend, 0, None), "p_rain": _sigmoid(head("occ"))[:, 0],
            "heavy": {k: heavy[:, i] for i, k in enumerate(self.heavy_keys)},
            "raw_lower": lower, "raw_upper": upper,
            "wet_amount": np.expm1(_softplus(head("wet"))[:, 0]),
        }

    # -------------------------------------------------- torch training
    def fit(self, train: pd.DataFrame, train_experts: pd.DataFrame, val: pd.DataFrame, val_experts: pd.DataFrame,
            heavy_train: dict[str, np.ndarray], heavy_val: dict[str, np.ndarray], seed: int = C.SEED) -> "RainGate":
        import torch
        from torch import nn

        torch.manual_seed(seed)
        Xtr_raw = gate_inputs(train, train_experts, self.numeric, self.experts, self.aux)
        self._fit_scaler(Xtr_raw)
        y_tr = train["target"].to_numpy(np.float32)
        self.y_scale = float(np.std(y_tr) + 1e-3)

        def tensors(rows, ex, heavy_labels):
            return dict(
                X=torch.from_numpy(self._scale(gate_inputs(rows, ex, self.numeric, self.experts, self.aux))),
                cat=torch.from_numpy(self._cats(rows)),
                E=torch.from_numpy(ex[self.experts].to_numpy(np.float32, copy=True)),
                y=torch.from_numpy(rows["target"].to_numpy(np.float32)),
                heavy=torch.from_numpy(np.stack([heavy_labels[k] for k in self.heavy_keys], 1).astype(np.float32)),
            )

        tr, va = tensors(train, train_experts, heavy_train), tensors(val, val_experts, heavy_val)
        pos = tr["heavy"].mean(0).clamp_min(1e-4)
        pos_weight = ((1 - pos) / pos).clamp(1.0, 50.0)
        gate = self

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embs = nn.ModuleList([nn.Embedding(len(gate.vocab[c]) + 1, d)
                                           for c, d in zip(GATE_CATS, gate.emb_dims)])
                d = tr["X"].shape[1] + sum(gate.emb_dims)
                self.trunk = nn.ModuleList()
                for width in gate.hidden:
                    self.trunk.append(nn.Linear(d, width))
                    d = width
                self.drop = nn.Dropout(C.GATE_DROPOUT)
                self.gate = nn.Linear(d, len(gate.experts))
                self.occ = nn.Linear(d, 1)
                self.heavy = nn.Linear(d, len(gate.heavy_keys))
                self.quant = nn.Linear(d, 2)
                self.wet = nn.Linear(d, 1)

            def forward(self, X, cat):
                h = torch.cat([X] + [e(cat[:, i]) for i, e in enumerate(self.embs)], 1)
                for layer in self.trunk:
                    h = self.drop(nn.functional.gelu(layer(h)))
                return self.gate(h), self.occ(h)[:, 0], self.heavy(h), self.quant(h), self.wet(h)[:, 0]

        net = Net()
        bce = nn.functional.binary_cross_entropy_with_logits
        s = self.y_scale
        lo_q, hi_q = C.INTERVAL_QUANTILES

        def losses(batch, ix=None):
            b = {k: v if ix is None else v[ix] for k, v in batch.items()}
            g, occ, heavy, quant, wet = net(b["X"], b["cat"])
            w = torch.softmax(g, 1)
            blend = (w * b["E"]).sum(1)
            y = b["y"]
            sw = 1 + C.EXTREME_ALPHA * torch.clamp(y / C.EXTREME_REF_MM, max=C.EXTREME_CAP)
            point = (sw * ((blend - y) / s) ** 2).sum() / sw.sum()
            occ_l = bce(occ, (y >= C.WET_MM).float())
            heavy_l = bce(heavy, b["heavy"], pos_weight=pos_weight)
            lower = blend - nn.functional.softplus(quant[:, 0]) * s
            upper = blend + nn.functional.softplus(quant[:, 1]) * s
            pin = lambda q, tau: torch.maximum(tau * (y - q), (tau - 1) * (y - q)).mean() / s  # noqa: E731
            quant_l = pin(lower, lo_q) + pin(upper, hi_q)
            wet_mask = y >= C.WET_MM
            wet_l = (((nn.functional.softplus(wet) - torch.log1p(y)) ** 2)[wet_mask].mean()
                     if wet_mask.any() else torch.zeros(()))
            entropy = -(w * torch.log(w.clamp_min(1e-9))).sum(1).mean() / np.log(len(gate.experts))
            total = (point + C.LOSS_OCC * occ_l + C.LOSS_HEAVY * heavy_l + C.LOSS_QUANTILE * quant_l
                     + C.LOSS_WET_AMOUNT * wet_l + C.ENTROPY_PENALTY * entropy)
            return total, {"point": point.item(), "occ": occ_l.item(), "heavy": heavy_l.item(),
            "quantile": quant_l.item(), "wet": wet_l.item(), "entropy": entropy.item()}

        opt = torch.optim.AdamW(net.parameters(), lr=C.GATE_LR, weight_decay=C.GATE_WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        best, best_state, bad = np.inf, None, 0
        n = len(tr["y"])
        for epoch in range(C.GATE_EPOCHS):
            net.train()
            perm = torch.randperm(n)
            for start in range(0, n, C.GATE_BATCH):
                loss, _ = losses(tr, perm[start:start + C.GATE_BATCH])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss, parts = losses(va)
            sched.step(vloss.item())
            self.history.append({"epoch": epoch, "val_total": round(vloss.item(), 5), **{k: round(v, 5) for k, v in parts.items()}})
            print(f"    gate epoch {epoch:02d} val {vloss.item():.4f} point {parts['point']:.4f} "
                  f"heavy {parts['heavy']:.4f} entropy {parts['entropy']:.3f}", flush=True)
            if vloss.item() < best - 1e-5:
                best, bad = vloss.item(), 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= C.GATE_PATIENCE:
                    break
        net.load_state_dict(best_state)
        self.params = {k: v.numpy().astype(np.float64) for k, v in net.state_dict().items()}
        return self

    def random_init(self, seed: int = 0) -> "RainGate":
        """Shape-correct untrained parameters (tests and dry runs without torch only)."""
        rng = np.random.default_rng(seed)
        n_in = len(self.input_names)
        self.med, self.mu, self.sd = np.zeros(n_in), np.zeros(n_in), np.ones(n_in)
        for i, (c, d) in enumerate(zip(GATE_CATS, self.emb_dims)):
            self.params[f"embs.{i}.weight"] = rng.normal(0, .1, (len(self.vocab[c]) + 1, d))
        d = n_in + sum(self.emb_dims)
        for j, width in enumerate(self.hidden):
            self.params[f"trunk.{j}.weight"] = rng.normal(0, 1 / np.sqrt(d), (width, d))
            self.params[f"trunk.{j}.bias"] = np.zeros(width)
            d = width
        for name, k in (("gate", len(self.experts)), ("occ", 1), ("heavy", len(self.heavy_keys)), ("quant", 2), ("wet", 1)):
            self.params[f"{name}.weight"] = rng.normal(0, .1, (k, d))
            self.params[f"{name}.bias"] = np.zeros(k)
        return self

    # -------------------------------------------------- export
    def save(self, directory: Path, name: str = "gate") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / f"{name}.npz", med=self.med, mu=self.mu, sd=self.sd, **self.params)
        (directory / f"{name}.json").write_text(json.dumps({
            "numeric": self.numeric, "experts": self.experts, "aux": self.aux, "heavy_keys": self.heavy_keys,
            "hidden": self.hidden, "vocab": self.vocab, "y_scale": self.y_scale, "input_names": self.input_names,
            "categoricals": list(GATE_CATS), "history": self.history, "activation": "gelu(erf)",
            "loss": {"extreme_alpha": C.EXTREME_ALPHA, "extreme_ref_mm": C.EXTREME_REF_MM, "extreme_cap": C.EXTREME_CAP,
                     "occ": C.LOSS_OCC, "heavy": C.LOSS_HEAVY, "quantile": C.LOSS_QUANTILE,
                     "wet_amount": C.LOSS_WET_AMOUNT, "entropy_penalty": C.ENTROPY_PENALTY},
        }, indent=1))

    @classmethod
    def load(cls, directory: Path, name: str = "gate") -> "RainGate":
        meta = json.loads((directory / f"{name}.json").read_text())
        gate = cls(meta["numeric"], meta["experts"], meta["aux"], meta["heavy_keys"], meta["hidden"])
        gate.vocab = {k: list(v) for k, v in meta["vocab"].items()}
        gate.y_scale, gate.history = meta["y_scale"], meta.get("history", [])
        arrays = np.load(directory / f"{name}.npz")
        gate.med, gate.mu, gate.sd = arrays["med"], arrays["mu"], arrays["sd"]
        gate.params = {k: arrays[k] for k in arrays.files if k not in ("med", "mu", "sd")}
        return gate
