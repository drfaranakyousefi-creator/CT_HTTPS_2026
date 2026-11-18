"""
Experiment runner for CT-HTTPS (v3). Resumable: finished runs are skipped (one JSON per run).

  python ablation.py --csv CHARTEVENTS.csv --out results --seeds 0 1 2
  python ablation.py --csv CHARTEVENTS.csv --families A E          # subset of families
  python ablation.py --csv CHARTEVENTS.csv --list                  # print the run plan and exit

Families (default: A S B E P C)
  A  component ablation   full | no_capsule | no_transformer | no_cls | no_pos | no_ae
                          (6 variants x 6 cohort/target pairs x seeds); the 'full' runs are the main
                          CT-HTTPS results and also run the privacy attacks
  S  sensitivity          full model without the plausibility filter (no_filter) and with
                          HTTPS-style windows that are cleared after every label (reset_history)
  B  window length        W in 3, 10, 20 for --w_targets (W = 40 is A/full)
  E  baselines            mean | linear | mlp | gru | lstm | transformer on the SAME splits
  P  privacy-utility      full model with client-side latent noise z_noise in --noise_levels,
                          SpO2 only, + privacy attacks (z_noise = 0 is A/full)
  C  knowledge transfer   warm_minl1 (paper) | warm_random (control); source = A/full AEs of the
                          OTHER cohort, same target, same seed (cold start = A/full)
  D  cross-task transfer  (optional) source = A/full AEs of the other cohort, OTHER target

Per-sample test predictions are written to <out>/preds/*.npz for the paired bootstrap. They contain
patient identifiers and vital-sign values: keep them on the licensed machine, never share them.
"""
import argparse, glob, hashlib, json, os, platform, socket, sys, time
import numpy as np
import torch
from Train_simulation import CT_HTTPS
from baselines import BaselineRunner, BASELINES
from new_dataset import ITEM_IDS, read_chartevents
import privacy_attack

COHORTS = ("carevue", "metavision")
TARGETS = ("spO2", "BP", "RR")
A_VARIANTS = {
    "full": {},
    "no_capsule": dict(use_capsule=False),
    "no_transformer": dict(use_transformer=False),
    "no_cls": dict(use_cls=False),
    "no_pos": dict(use_pos=False),
    "no_ae": dict(lambda_ae=0.0),
}
B_WINDOWS = (3, 10, 20)
S_VARIANTS = {"no_filter": dict(filter_ranges=False), "reset_history": dict(history="reset")}
C_VARIANTS = ("warm_minl1", "warm_random")
DEFAULT_FAMILIES = ["A", "S", "B", "E", "P", "C"]


def code_fingerprint():
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.md5()
    for f in sorted(glob.glob(os.path.join(here, "*.py"))):
        h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()


def plan(args):
    runs = []
    for seed in args.seeds:
        for co in args.cohorts:
            for tg in args.targets:
                if "A" in args.families:
                    for v in A_VARIANTS:
                        runs.append(dict(family="A", variant=v, cohort=co, target=tg, seed=seed))
                if "S" in args.families:
                    for v in S_VARIANTS:
                        runs.append(dict(family="S", variant=v, cohort=co, target=tg, seed=seed))
                if "E" in args.families:
                    for b in BASELINES:
                        runs.append(dict(family="E", variant=b, cohort=co, target=tg, seed=seed))
            if "B" in args.families:
                for tg in args.w_targets:
                    for w in B_WINDOWS:
                        runs.append(dict(family="B", variant=f"W{w}", cohort=co, target=tg, seed=seed, W=w))
            if "P" in args.families:
                for tg in args.p_targets:
                    for nz in args.noise_levels:
                        runs.append(dict(family="P", variant=f"noise{nz:g}", cohort=co, target=tg, seed=seed, z_noise=nz))
    for seed in args.seeds:                       # transfer families need A/full of both cohorts
        for co in args.cohorts:
            for tg in args.targets:
                if "C" in args.families:
                    for v in C_VARIANTS:
                        runs.append(dict(family="C", variant=v, cohort=co, target=tg, seed=seed, src_target=tg))
                if "D" in args.families:
                    for st in TARGETS:
                        if st != tg:
                            runs.append(dict(family="D", variant=f"from_{st}", cohort=co, target=tg, seed=seed, src_target=st))
    return runs


def run_name(r):
    return f"{r['family']}__{r['cohort']}__{r['target']}__{r['variant']}__seed{r['seed']}"


def overrides_for(r, quick, mpp=None):
    ov = {}
    if r["family"] == "A":
        ov.update(A_VARIANTS[r["variant"]])
    elif r["family"] == "S":
        ov.update(S_VARIANTS[r["variant"]])
    elif r["family"] == "B":
        ov["W"] = r["W"]
    elif r["family"] == "P":
        ov["z_noise"] = r["z_noise"]
    if quick:
        ov.update(max_epochs=2, patience=1)
    if mpp:
        ov["max_per_patient"] = mpp
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--w_targets", nargs="+", default=["spO2"])
    ap.add_argument("--p_targets", nargs="+", default=["spO2"])
    ap.add_argument("--noise_levels", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="2 epochs per run: smoke test only, NOT for the paper")
    ap.add_argument("--max_per_patient", type=int, default=None,
                    help="cap on samples per patient (fixed random subset); use the SAME value for every run")
    ap.add_argument("--ct_lr_grid", type=float, nargs="+", default=[0.001, 0.01],
                    help="CT-HTTPS learning rates; the one with the lowest validation MSE is kept")
    ap.add_argument("--variants", nargs="+", default=None, help="restrict to these variant names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    runs = plan(args)
    if args.variants:
        runs = [r for r in runs if r["variant"] in args.variants]
    for d in ("runs", "ckpt", "preds", "logs"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    todo = [r for r in runs if not os.path.isfile(os.path.join(args.out, "runs", run_name(r) + ".json"))]
    print(f"{len(runs)} runs planned, {len(todo)} to do")
    if args.list:
        for r in todo: print(run_name(r))
        return

    env = {"code_md5": code_fingerprint(), "torch": torch.__version__, "python": sys.version.split()[0],
           "host": socket.gethostname(), "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "argv": sys.argv, "quick": args.quick}
    df = None
    t_all = time.time()
    for k, r in enumerate(todo, 1):
        out = os.path.join(args.out, "runs", run_name(r) + ".json")
        src_ae = None
        if r["family"] in ("C", "D"):
            other = "metavision" if r["cohort"] == "carevue" else "carevue"
            ck = os.path.join(args.out, "ckpt", f"{other}__{r['src_target']}__seed{r['seed']}.pt")
            if not os.path.isfile(ck):
                print(f"skip {run_name(r)}: needs {ck} (run family A first)")
                continue
            src_ae = torch.load(ck, map_location="cpu")
        if df is None:
            df = read_chartevents(args.csv, cache_dir=args.cache)
        print(f"[{k}/{len(todo)}] {run_name(r)}", flush=True)
        ov = overrides_for(r, args.quick, args.max_per_patient)
        t_start = time.strftime("%Y-%m-%dT%H:%M:%S")
        if r["family"] == "E":
            m = BaselineRunner(r["variant"], args.csv, r["cohort"], r["target"], seed=r["seed"],
                               device=args.device, cache_dir=args.cache, df=df, **ov)
            res = m.fit()
        else:
            # same tuning budget as the baselines: lr selected on the VALIDATION set
            best = None
            for lr in args.ct_lr_grid:
                mm = CT_HTTPS(args.csv, r["cohort"], r["target"], seed=r["seed"], device=args.device,
                              cache_dir=args.cache, df=df, **{**ov, "lr": lr})
                choice = None
                if src_ae is not None:
                    choice = mm.get_knowledge(src_ae, strategy="random" if r["variant"] == "warm_random" else "min_l1",
                                              seed=r["seed"], verbose=False)
                rr = mm.fit()
                rr["transfer_choice"] = choice
                rr["lr_trials"] = {}
                if best is None or rr["final"]["val"]["MSE"] < best[1]["final"]["val"]["MSE"]:
                    if best is not None:
                        rr["lr_trials"] = best[1]["lr_trials"]
                    best = (mm, rr)
                best[1]["lr_trials"][str(lr)] = rr["final"]["val"]["MSE"]
                if best[0] is not mm:
                    del mm
            m, res = best
            res["lr_selected"] = m.cfg["lr"]
            if r["family"] == "A" and r["variant"] == "full":
                torch.save(m.ae_state_dicts(), os.path.join(args.out, "ckpt", f"{r['cohort']}__{r['target']}__seed{r['seed']}.pt"))
            if (r["family"] == "A" and r["variant"] == "full") or r["family"] == "P":
                res["privacy"] = privacy_attack.run_all(m, seed=r["seed"])
        p, t = m.predict(m.data.test)
        np.savez_compressed(os.path.join(args.out, "preds", run_name(r) + ".npz"),
                            pred=p.cpu().numpy(), true=t.cpu().numpy(), subject=m.data.test.subject)
        res["run"] = r; res["config"] = m.cfg; res["env"] = env
        res["started"] = t_start; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(out, "w") as f:
            json.dump(res, f)
        te = res["final"]["test"]
        line = (f"{run_name(r)} | test MSE {te['MSE']:.4f} MAE {te['MAE']:.4f} R2 {te['R2']:.3f} | "
                f"best epoch {res['best_epoch']} | {res['seconds']/60:.1f} min")
        print("    " + line, flush=True)
        with open(os.path.join(args.out, "logs", "run_log.txt"), "a") as f:
            f.write(res["finished"] + " " + line + "\n")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"done in {(time.time()-t_all)/3600:.2f} h")


if __name__ == "__main__":
    main()
