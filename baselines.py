"""
Centralised baselines trained on EXACTLY the same data as CT-HTTPS: same extraction, same seeded
patient-level train/validation/test split, same train-only normalisation, same early-stopping rule
(validation MSE, patience 10, max 60 epochs, best weights restored), metrics in original units.

  mean         : predicts the training-set label mean (lower bound, gives R2 = 0 reference)
  linear       : linear regression on the flattened window + mask
  mlp          : 2-hidden-layer MLP on the flattened window + mask
  gru / lstm   : 2-layer recurrent network, hidden state at the last valid event
  transformer  : plain Transformer encoder on linearly projected inputs (no AutoEncoders, no
                 capsules, no split), masked mean pooling + linear head

The learning rate of every trainable baseline is chosen on the VALIDATION set from LR_GRID
(CT-HTTPS itself uses a fixed lr), so the comparison does not favour the proposed model.
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn

from new_dataset import data_preparing, N_CHANNELS
from Train_simulation import regression_metrics, set_seed, DEFAULTS

BASELINES = ("mean", "linear", "mlp", "gru", "lstm", "transformer")
LR_GRID = (1e-3, 1e-2)


class FlatNet(nn.Module):
    def __init__(self, W, N, hidden=(256, 128)):
        super().__init__()
        d, layers = W * N + W, []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, m):
        return self.net(torch.cat([x.flatten(1), m], 1)).squeeze(-1)


class RNNNet(nn.Module):
    def __init__(self, N, kind="gru", hidden=96, layers=2):
        super().__init__()
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(N + 1, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, m):
        out, _ = self.rnn(torch.cat([x, m.unsqueeze(-1)], -1))
        last = (m.sum(1).long() - 1).clamp(min=0)          # windows are left-aligned
        h = out[torch.arange(x.size(0), device=x.device), last]
        return self.head(h).squeeze(-1)


class PlainTransformer(nn.Module):
    def __init__(self, N, W, d=64, nhead=4, layers=3, d_ff=128):
        super().__init__()
        self.inp = nn.Linear(N, d)
        self.pos = nn.Parameter(torch.zeros(1, W, d))
        layer = nn.TransformerEncoderLayer(d, nhead, d_ff, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x, m):
        h = self.enc(self.inp(x) + self.pos[:, : x.size(1)], src_key_padding_mask=m < 0.5)
        h = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(h).squeeze(-1)


def build(name, W, N):
    if name == "linear":
        return FlatNet(W, N, hidden=())
    if name == "mlp":
        return FlatNet(W, N)
    if name in ("gru", "lstm"):
        return RNNNet(N, name)
    if name == "transformer":
        return PlainTransformer(N, W)
    raise ValueError(name)


class BaselineRunner:
    def __init__(self, name, chartevents_path, dataset_name, target, seed=0, device=None,
                 cache_dir="./cache", df=None, test_size=0.2, val_size=0.1, **overrides):
        self.cfg = {**DEFAULTS, **overrides}
        self.name, self.seed = name, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        c = self.cfg
        self.data = data_preparing(chartevents_path, dataset_name, c["W"], target, c["batch_size"],
                                   seed=seed, test_size=test_size, val_size=val_size, device=self.device,
                                   cache_dir=cache_dir, df=df, filter_ranges=c["filter_ranges"],
                                   strict_past=c["strict_past"], history=c["history"],
                                   max_per_patient=c["max_per_patient"])
        for s in (self.data.val, self.data.test, self.data.train_eval):
            s.bs = 4096
        self.W, self.N = c["W"], N_CHANNELS[dataset_name]
        self.model = None

    @torch.no_grad()
    def predict(self, split, max_samples=None):
        preds, tgts, n = [], [], 0
        if self.model is not None:
            self.model.eval()
        for x, _, m, y_raw in split:
            if self.model is None:
                p = torch.full_like(y_raw, self.data.y_mean)
            else:
                p = self.model(x, m) * self.data.y_std + self.data.y_mean
            preds.append(p); tgts.append(y_raw); n += x.shape[0]
            if max_samples and n >= max_samples:
                break
        return torch.cat(preds).float(), torch.cat(tgts).float()

    def evaluate(self, split, max_samples=None):
        return regression_metrics(*self.predict(split, max_samples))

    def _train(self, lr):
        c = self.cfg
        set_seed(self.seed)
        model = build(self.name, self.W, self.N).to(self.device)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        keys = ("MSE", "RMSE", "MAE", "R2", "MAPE")
        hist = {f"{s}_{k}": [] for s in ("train", "val", "test") for k in keys}
        best, best_state, best_ep, bad = float("inf"), None, 0, 0
        for ep in range(1, c["max_epochs"] + 1):
            model.train()
            for x, y, m, _ in self.data.train:
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x, m), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
                opt.step()
            tr = self.evaluate(self.data.train_eval, c["eval_train_max"])
            va = self.evaluate(self.data.val)
            te = self.evaluate(self.data.test)        # logged only
            for s, r in (("train", tr), ("val", va), ("test", te)):
                for k in keys:
                    hist[f"{s}_{k}"].append(r[k])
            sch.step(va["MSE"])
            if va["MSE"] < best - c["min_delta"]:
                best, best_ep, bad = va["MSE"], ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if bad >= c["patience"]:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return best, best_ep, ep, hist, copy.deepcopy(model.state_dict())

    def fit(self, verbose=False):
        t0 = time.time()
        if self.name == "mean":
            hist, best_ep, ep, lr, params = {}, 0, 0, None, 0
        else:
            trials = []
            for lr in LR_GRID:
                best, best_ep, ep, hist, state = self._train(lr)
                trials.append((best, lr, best_ep, ep, hist, state))
                if verbose:
                    print(f"   {self.name} lr={lr:g}: best val MSE {best:.4f} (epoch {best_ep})")
            best, lr, best_ep, ep, hist, state = min(trials, key=lambda t: t[0])
            self.model = build(self.name, self.W, self.N).to(self.device)
            self.model.load_state_dict(state)
            params = sum(p.numel() for p in self.model.parameters())
        final = {s: self.evaluate(sp) for s, sp in (("val", self.data.val), ("test", self.data.test))}
        final["train"] = self.evaluate(self.data.train_eval, self.cfg["eval_train_max"])
        return {"history": hist, "best_epoch": best_ep, "epochs_run": ep, "final": final,
                "seconds": time.time() - t0, "params": {"total": params}, "lr_selected": lr,
                "data_stats": self.data.stats}
