"""
Empirical privacy evaluation of the three-segment split (honest-but-curious server).

Threat model (state it in the manuscript): the server follows the protocol, observes everything
it legitimately receives (latent series z and the validity mask on the way up; the gradient with
respect to its output h on the way back) and owns a small AUXILIARY set of samples from the same
distribution for which it knows the raw inputs / labels (here: the validation patients, which are
disjoint from the test patients). Attacks are evaluated on the TEST patients.

  A1  input reconstruction : server learns z_t -> x_t on the auxiliary set (per event position)
                             and reconstructs the raw vital signs of test patients.
  A2  label recovery       : with a linear client-side regression layer, the returned gradient is
                             g_i = 2 (pred_i - y_i) w, i.e. every g_i is parallel to w. The server
                             projects g_i and h_i on that direction and fits y on the two scalars
                             with k auxiliary labels (k = 10 and 100).
Both attacks report R2 / MAE in ORIGINAL units. High R2 means the corresponding claim
("raw inputs" / "labels never leave the client") does not imply confidentiality.
"""
import numpy as np
import torch
import torch.nn as nn


def _r2(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    return float(1.0 - ((p - t) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-12))


@torch.no_grad()
def _latents(model, split, max_samples=None):
    model.eval()
    Z, X, M, n = [], [], [], 0
    for x, _, m, _ in split:
        z, _ = model._client_front(x, m)
        Z.append(z); X.append(x); M.append(m); n += x.shape[0]
        if max_samples and n >= max_samples:
            break
    return torch.cat(Z), torch.cat(X), torch.cat(M)


def input_reconstruction(model, epochs=30, max_aux=20000, max_test=20000, seed=0):
    torch.manual_seed(seed)
    dev = model.device
    za, xa, ma = _latents(model, model.data.val, max_aux)
    zt, xt, mt = _latents(model, model.data.test, max_test)
    D, N = za.shape[-1], xa.shape[-1]
    def flat(z, x, m):
        keep = m.reshape(-1) > 0
        z, x = z.reshape(-1, D)[keep], x.reshape(-1, N)[keep]
        return z, x, (x != 0).float()
    za, xa, oa = flat(za, xa, ma)
    zt, xt, ot = flat(zt, xt, mt)
    att = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, N)).to(dev)
    opt = torch.optim.Adam(att.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(za.shape[0], device=dev)
        for i in range(0, za.shape[0], 1024):
            j = perm[i:i + 1024]
            loss = (((att(za[j]) - xa[j]) * oa[j]) ** 2).sum() / oa[j].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = att(zt)
    mean, std = model.data.norm.mean, model.data.norm.std
    out = {}
    for c in range(N):
        sel = ot[:, c] > 0
        if sel.sum() < 10:
            continue
        p = pr[sel, c].cpu().numpy() * std[c] + mean[c]
        t = xt[sel, c].cpu().numpy() * std[c] + mean[c]
        out[f"ch{c}"] = {"R2": _r2(p, t), "MAE": float(np.abs(p - t).mean()), "n": int(sel.sum())}
    r2s = [v["R2"] for v in out.values()]
    return {"per_channel": out, "mean_R2": float(np.mean(r2s)) if r2s else float("nan")}


def _grad_features(model, split, max_samples=None):
    """What the server observes per sample on the way back: its own output h and dL/dh."""
    model.eval()
    H, G, Y, n = [], [], [], 0
    for x, y, m, y_raw in split:
        with torch.no_grad():
            z, _ = model._client_front(x, m)
            h = model.server(z, m)
        h_cli = h.detach().clone().requires_grad_(True)       # client receives h
        loss = ((model.back(h_cli) - y) ** 2).sum()           # client loss (per-sample scale)
        g, = torch.autograd.grad(loss, h_cli)                 # sent back to the server
        H.append(h.cpu()); G.append(g.cpu()); Y.append(y_raw.cpu()); n += x.shape[0]
        if max_samples and n >= max_samples:
            break
    return torch.cat(H).double(), torch.cat(G).double(), torch.cat(Y).double()


def label_recovery(model, ks=(10, 100), max_test=20000, seed=0):
    Ha, Ga, Ya = _grad_features(model, model.data.val, 20000)
    Ht, Gt, Yt = _grad_features(model, model.data.test, max_test)
    # direction shared by all gradients (= w / |w|), estimated without labels
    _, _, V = torch.linalg.svd(torch.cat([Ga, Gt])[:20000], full_matrices=False)
    u = V[0]
    feats = lambda H, G: torch.stack([H @ u, G @ u, torch.ones(len(H), dtype=H.dtype)], 1)
    Fa, Ft = feats(Ha, Ga), feats(Ht, Gt)
    rng = np.random.RandomState(seed)
    out = {}
    for k in ks:
        idx = torch.as_tensor(rng.choice(len(Fa), size=min(k, len(Fa)), replace=False))
        coef = torch.linalg.lstsq(Fa[idx], Ya[idx].unsqueeze(1)).solution
        p = (Ft @ coef).squeeze(1).numpy(); t = Yt.numpy()
        out[f"k{k}"] = {"R2": _r2(p, t), "MAE": float(np.abs(p - t).mean())}
    return out


def run_all(model, seed=0):
    return {"input_reconstruction": input_reconstruction(model, seed=seed),
            "label_recovery": label_recovery(model, seed=seed),
            "z_noise": model.cfg["z_noise"]}
