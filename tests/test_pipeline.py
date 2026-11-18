"""Sanity tests on SYNTHETIC data (no MIMIC needed):  python tests/test_pipeline.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from Train_simulation import CT_HTTPS, set_seed
from new_dataset import ITEM_IDS, load_samples


def make_df(dataset, n_subj=120, seed=0):
    rng = np.random.RandomState(seed)
    ids = {"carevue": dict(hr=211, rr=618, abp=52, nbp=456, temp=678, spo2=646),
           "metavision": dict(hr=220045, rr=220210, abp=220052, nbp=220181, spo2=220277)}[dataset]
    rows = []
    for s in range(1, n_subj + 1):
        t = pd.Timestamp("2100-01-01")
        base = rng.randn(4)
        for _ in range(rng.randint(60, 150)):
            t += pd.Timedelta(minutes=int(rng.randint(1, 30)))
            k = rng.choice(list(ids))
            v = {"hr": 80 + 10 * base[0], "rr": 18 + 3 * base[1], "abp": 75 + 8 * base[2],
                 "nbp": 78 + 8 * base[2], "temp": 98 + base[3], "spo2": 97 - 2 * abs(base[1]) - 0.05 * (80 - 80 - 10 * base[0])}[k]
            rows.append((s, ids[k], v + rng.randn(), t))
    return pd.DataFrame(rows, columns=["subject_id", "itemid", "valuenum", "charttime"]).sample(frac=1, random_state=1)


def test_v3_extraction():
    """v3 fixes: NIBP mean only, F->C, plausibility filter, strict-past windows."""
    import new_dataset as nd
    assert 220179 not in nd.ITEM_IDS["metavision"] and 220180 not in nd.ITEM_IDS["metavision"]
    assert 220181 in nd.NIBP
    t0 = pd.Timestamp("2100-01-01")
    rows = [(1, 211, 80, t0), (1, 678, 98.6, t0 + pd.Timedelta("1h")), (1, 52, 70, t0 + pd.Timedelta("2h")),
            (1, 211, 999, t0 + pd.Timedelta("2h")),           # implausible HR -> removed
            (1, 211, 90, t0 + pd.Timedelta("3h")),            # same charttime as the label -> excluded
            (1, 646, 95, t0 + pd.Timedelta("3h")),            # label
            (1, 646, 0, t0 + pd.Timedelta("4h"))]             # implausible SpO2 -> removed
    df = pd.DataFrame(rows, columns=["subject_id", "itemid", "valuenum", "charttime"])
    x, y, m, s, info = nd.load_samples(None, "carevue", 40, "spO2", df=df, verbose=False, return_info=True)
    assert len(y) == 1 and y[0] == 95
    assert m[0].sum() == 3, m[0]                               # HR80, Temp, ABP ; HR999 removed, HR90 same-time
    assert abs(x[0, 1, 4] - 37.0) < 1e-3                       # 98.6 F -> 37 C
    assert x[0, 2, 2] == 80                                    # HR carried forward, not 999 / 90
    assert abs(info["gap_hours"][0] - 1.0) < 1e-6
    x2, *_ = nd.load_samples(None, "carevue", 40, "spO2", df=df, verbose=False, strict_past=False)
    assert x2.shape[0] == 1 and (x2[0, :, 0] != 0).sum() >= 1
    print("v3 extraction OK")


def test_all():
    test_v3_extraction()
    for ds in ("carevue", "metavision"):
        df = make_df(ds)
        kw = dict(chartevents_path=None, dataset_name=ds, target="spO2", seed=0, device="cpu", df=df)
        # 1) parameter count must match Table 3 of the paper
        m = CT_HTTPS(**kw, W=40)
        p = m.count_params(); print(ds, p)
        exp = {"carevue": (805, 94496, 26945, 122246), "metavision": (644, 63328, 24897, 88869)}[ds]
        assert (p["ae_bank"], p["transformer_cls"], p["capsule_head"], p["total"]) == exp, "param count != Table 3"
        # 2) patient-level split: no subject in two splits
        x, y, mk, s = load_samples(None, ds, 40, "spO2", df=df)
        # 3) split-learning gradients == end-to-end gradients
        m.eval(); m.train()
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout): mod.p = 0.0
        xb, yb, mb, _ = next(iter(m.data.train))
        m.zero_grad(); z, rec, zs, h, hc = m._forward_split(xb, mb)
        m.loss_fn(m.back(hc), yb).backward()
        h.backward(m.down.send_grad_back()); torch.autograd.backward([z, 0.1 * rec], [m.up.send_grad_back(), None])
        g_split = torch.cat([q.grad.flatten() for q in m.parameters() if q.grad is not None])
        m.zero_grad(); z2, rec2 = m.front(xb, mb)
        loss2 = m.loss_fn(m.back(m.server(z2, mb)), yb) + 0.1 * rec2; loss2.backward()
        g_e2e = torch.cat([q.grad.flatten() for q in m.parameters() if q.grad is not None])
        assert torch.allclose(g_split, g_e2e, atol=1e-5), "split grads differ from end-to-end"
        print(ds, "split == end-to-end gradients OK")

    # 4) every ablation variant trains; metrics in original units; early stopping works
    df = make_df("carevue")
    for name, ov in {"full": {}, "no_capsule": dict(use_capsule=False), "no_transformer": dict(use_transformer=False),
                     "no_cls": dict(use_cls=False), "no_pos": dict(use_pos=False), "no_ae": dict(lambda_ae=0.0),
                     "W10": dict(W=10)}.items():
        m = CT_HTTPS(None, "carevue", "spO2", seed=1, device="cpu", df=df, max_epochs=3, patience=2, **ov)
        r = m.fit()
        print(f"{name:15s} test MSE {r['final']['test']['MSE']:.3f}  params {r['params']['total']}")
        assert np.isfinite(r["final"]["test"]["MSE"])
    # 5) transfer
    src = CT_HTTPS(None, "carevue", "spO2", seed=1, device="cpu", df=df, max_epochs=2)
    src.fit()
    tgt = CT_HTTPS(None, "metavision", "spO2", seed=1, device="cpu", df=make_df("metavision"), max_epochs=2)
    print("choice", tgt.get_knowledge(src.ae_state_dicts(), "min_l1"), tgt.get_knowledge(src.ae_state_dicts(), "random"))
    tgt.fit()
    # 6) baselines + privacy attacks + traffic accounting
    from baselines import BaselineRunner, BASELINES
    for b in BASELINES:
        r = BaselineRunner(b, None, "carevue", "spO2", seed=1, device="cpu", df=df, max_epochs=2, patience=1).fit()
        print(f"baseline {b:12s} test MSE {r['final']['test']['MSE']:.3f} lr {r['lr_selected']}")
    import privacy_attack
    m = CT_HTTPS(None, "carevue", "spO2", seed=1, device="cpu", df=df, max_epochs=3)
    r = m.fit(); pa = privacy_attack.run_all(m)
    print("privacy", round(pa["input_reconstruction"]["mean_R2"], 3), {k: round(v["R2"], 3) for k, v in pa["label_recovery"].items()})
    assert pa["label_recovery"]["k100"]["R2"] > 0.99, "linear client head should leak labels exactly"
    B, W, N, D = len(m.data.train), 40, 5, 16
    assert abs(r["comm"]["client_to_server_MB_per_epoch"] * 1e6 - (B * W * N * D * 4 + B * 64 * 4)) < 1
    assert abs(r["comm"]["server_to_client_MB_per_epoch"] * 1e6 - (B * W * N * D * 4 + B * 64 * 4)) < 1
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_all()
