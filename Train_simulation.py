"""
CT-HTTPS: three-segment client-server-client split learning, simulated in one process.

  client front : per-feature AutoEncoders            (raw values stay here)
  server core  : Transformer encoder (+[CLS]) + Capsule Network
  client back  : regression layer + loss             (labels stay here)

Only detached tensors cross the boundaries (see transmitter_simulation.SplitChannel).
Note: 'labels stay on the client' means labels are not transmitted. It is NOT a privacy guarantee:
the gradient sent back to the server is a function of the label (see privacy_attack.py).
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn

from client_net import ClientFront, ClientBack, FeatureAutoEncoder
from server_net import ServerCore
from new_dataset import data_preparing, N_CHANNELS
from transmitter_simulation import SplitChannel

DEFAULTS = dict(
    W=40, batch_size=128,
    d_model=16, nhead=4, d_ff=32, num_cells=3, dropout=0.1,
    fc_hidden1=128, fc_hidden2=64,
    num_primary_caps=8, primary_dim=8, num_output_caps=4, output_dim=16, num_routing=3,
    lr=0.01, lambda_ae=0.1, ae_lr_factor=0.1, grad_clip=1.0,
    max_epochs=60, patience=10, min_delta=1e-4, eval_train_max=50000,
    use_transformer=True, use_cls=True, use_pos=True, use_capsule=True,
    filter_ranges=True, strict_past=True, history="sliding",   # data extraction (new_dataset v3)
    max_per_patient=None,                   # cap on samples (label times) per patient, None = all
    z_noise=0.0,                            # std of Gaussian noise added on the client to the latents
)                                           # before they are sent (privacy-utility experiment)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(pred, target):
    err = pred - target
    mse = torch.mean(err ** 2)
    mae = torch.mean(err.abs())
    ss_res = torch.sum(err ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    nz = target.abs() > 1e-6
    mape = torch.mean((err[nz] / target[nz]).abs()) * 100.0 if nz.any() else torch.tensor(float("nan"))
    return {"MSE": mse.item(), "RMSE": mse.sqrt().item(), "MAE": mae.item(),
            "R2": (1.0 - ss_res / (ss_tot + 1e-8)).item(), "MAPE": mape.item()}


class CT_HTTPS(nn.Module):
    def __init__(self, chartevents_path, dataset_name, target, seed=0, device=None,
                 cache_dir="./cache", df=None, test_size=0.2, val_size=0.1, **overrides):
        super().__init__()
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown hyper-parameters: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **overrides}
        c = self.cfg
        self.dataset_name, self.target, self.seed = dataset_name, target, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_seed(seed)

        N = N_CHANNELS[dataset_name]
        self.N = N
        D = N * c["d_model"]
        self.front = ClientFront(N, c["d_model"])
        self.server = ServerCore(
            D, c["W"], nhead=c["nhead"], d_ff=c["d_ff"], num_cells=c["num_cells"], dropout=c["dropout"],
            fc_hidden1=c["fc_hidden1"], fc_hidden2=c["fc_hidden2"],
            num_primary_caps=c["num_primary_caps"], primary_dim=c["primary_dim"],
            num_output_caps=c["num_output_caps"], output_dim=c["output_dim"], num_routing=c["num_routing"],
            use_transformer=c["use_transformer"], use_cls=c["use_cls"],
            use_pos=c["use_pos"], use_capsule=c["use_capsule"])
        self.back = ClientBack(self.server.out_dim)
        self.to(self.device)

        self.data = data_preparing(chartevents_path, dataset_name, c["W"], target, c["batch_size"],
                                   seed=seed, test_size=test_size, val_size=val_size,
                                   device=self.device, cache_dir=cache_dir, df=df,
                                   filter_ranges=c["filter_ranges"], strict_past=c["strict_past"],
                                   history=c["history"], max_per_patient=c["max_per_patient"])
        for s in (self.data.val, self.data.test, self.data.train_eval):
            s.bs = 4096
        self.up, self.down = SplitChannel("up"), SplitChannel("down")
        self.loss_fn = nn.MSELoss()
        self.warm_started = False
        self._build_optimizer()

    # ------------------------------------------------------------------ optimiser
    def _build_optimizer(self):
        c = self.cfg
        ae_lr = c["lr"] * (c["ae_lr_factor"] if self.warm_started else 1.0)   # asymmetric LR (Eq. 21)
        groups = [{"params": list(self.front.parameters()), "lr": ae_lr},
                  {"params": list(self.server.parameters()) + list(self.back.parameters()), "lr": c["lr"]}]
        self.optimizer = torch.optim.Adam(groups)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3)

    # ------------------------------------------------------------------ parameters
    def count_params(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        cls = self.server.cls_token.numel() if hasattr(self.server, "cls_token") else 0
        trans = n(self.server.cells) + cls
        total = n(self)
        return {"ae_bank": n(self.front), "transformer_cls": trans,
                "capsule_head": total - n(self.front) - trans, "total": total}

    # ------------------------------------------------------------------ one split-learning step
    def _client_front(self, x, m):
        z, rec = self.front(x, m)
        if self.cfg["z_noise"] > 0:                  # client-side perturbation of what is sent
            z = z + self.cfg["z_noise"] * torch.randn_like(z)
        return z, rec

    def _forward_split(self, x, m):
        z, rec = self._client_front(x, m)            # client front
        z_srv = self.up.send(z, "up")                # client -> server (detached)
        h = self.server(z_srv, m)                    # server core
        h_cli = self.down.send(h, "down")            # server -> client (detached)
        return z, rec, z_srv, h, h_cli

    def train_one_epoch(self):
        self.train()
        clip = self.cfg["grad_clip"]
        for x, y, m, _ in self.data.train:
            self.optimizer.zero_grad(set_to_none=True)
            z, rec, z_srv, h, h_cli = self._forward_split(x, m)
            pred = self.back(h_cli)                                        # client back
            task = self.loss_fn(pred, y)                                   # labels used only here
            task.backward()                                                # -> back layer, grad wrt h_cli
            h.backward(self.down.send_grad_back())                         # gradient crosses to the server
            torch.autograd.backward([z, self.cfg["lambda_ae"] * rec],      # ... and back to the client front,
                                    [self.up.send_grad_back(), None])      #     together with the AE loss
            if clip:
                nn.utils.clip_grad_norm_(self.parameters(), clip)
            self.optimizer.step()

    # ------------------------------------------------------------------ evaluation (original units)
    @torch.no_grad()
    def predict(self, split, max_samples=None):
        self.eval()
        preds, tgts, n = [], [], 0
        for x, _, m, y_raw in split:
            z, _ = self._client_front(x, m)
            p = self.back(self.server(z, m)) * self.data.y_std + self.data.y_mean
            preds.append(p); tgts.append(y_raw)
            n += x.shape[0]
            if max_samples and n >= max_samples:
                break
        return torch.cat(preds).float(), torch.cat(tgts).float()

    def evaluate(self, split, max_samples=None):
        return regression_metrics(*self.predict(split, max_samples))

    # ------------------------------------------------------------------ training loop
    def fit(self, verbose=False):
        c = self.cfg
        keys = ("MSE", "RMSE", "MAE", "R2", "MAPE")
        hist = {f"{s}_{k}": [] for s in ("train", "val", "test") for k in keys}
        best, best_state, best_ep, bad = float("inf"), None, 0, 0
        self.up.reset(); self.down.reset()
        t0 = time.time()
        for ep in range(1, c["max_epochs"] + 1):
            self.train_one_epoch()
            tr = self.evaluate(self.data.train_eval, c["eval_train_max"])
            va = self.evaluate(self.data.val)
            te = self.evaluate(self.data.test)       # logged for convergence plots ONLY, never used for selection
            for s, r in (("train", tr), ("val", va), ("test", te)):
                for k in keys:
                    hist[f"{s}_{k}"].append(r[k])
            self.scheduler.step(va["MSE"])
            if va["MSE"] < best - c["min_delta"]:
                best, best_ep, bad = va["MSE"], ep, 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                bad += 1
            if verbose:
                print(f"ep {ep:3d} | train {tr['MSE']:.3f} | val {va['MSE']:.3f} | test {te['MSE']:.3f}")
            if bad >= c["patience"]:
                break
        if best_state is not None:
            self.load_state_dict(best_state)         # restore best-validation weights
        final = {s: self.evaluate(sp) for s, sp in (("val", self.data.val), ("test", self.data.test))}
        final["train"] = self.evaluate(self.data.train_eval, c["eval_train_max"])
        c2s = self.up.fwd_bytes + self.down.bwd_bytes
        s2c = self.up.bwd_bytes + self.down.fwd_bytes
        return {"history": hist, "best_epoch": best_ep, "epochs_run": ep, "final": final,
                "seconds": time.time() - t0, "params": self.count_params(),
                "data_stats": self.data.stats,
                "comm": {"client_to_server_MB_per_epoch": c2s / 1e6 / ep,
                         "server_to_client_MB_per_epoch": s2c / 1e6 / ep,
                         "train_samples": len(self.data.train)}}

    # ------------------------------------------------------------------ knowledge transfer
    @torch.no_grad()
    def _ae_l1(self, ae, idx, max_samples=20000):
        ae = ae.to(self.device).eval()
        tot, cnt, n = 0.0, 0.0, 0
        for x, _, m, _ in self.data.train_eval:
            xi = x[:, :, idx:idx + 1]
            _, xh = ae(xi)
            mm = m.unsqueeze(-1)
            tot += (((xh - xi) * mm).abs()).sum().item(); cnt += mm.sum().item()
            n += x.shape[0]
            if n >= max_samples:
                break
        return tot / max(cnt, 1.0)

    def get_knowledge(self, source_aes, strategy="min_l1", seed=0, verbose=True):
        """Warm-start the target AEs from a list of source FeatureAutoEncoders (or their state dicts).
        strategy: 'min_l1' (paper) picks, for every target feature, the source AE with the lowest
        L1 reconstruction error on the target train data; 'random' picks one at random (control)."""
        aes = []
        for a in source_aes:
            if isinstance(a, dict):
                m = FeatureAutoEncoder(self.cfg["d_model"]); m.load_state_dict(a); a = m
            aes.append(a)
        rng = np.random.RandomState(seed)
        n_shared = min(self.N, len(aes))
        choice = []
        for i in range(n_shared):
            if strategy == "min_l1":
                j = int(np.argmin([self._ae_l1(a, i) for a in aes]))
            elif strategy == "random":
                j = int(rng.randint(len(aes)))
            else:
                raise ValueError(strategy)
            self.front.feature_aes[i].load_state_dict(aes[j].state_dict())
            choice.append(j)
            if verbose:
                print(f"target feature {i} <- source AE {j}")
        self.warm_started = True
        self._build_optimizer()
        return choice

    def ae_state_dicts(self):
        return [copy.deepcopy(a.state_dict()) for a in self.front.feature_aes]
