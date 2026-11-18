"""
Build EVERY table, figure and quoted number of the manuscript from the run logs (v3).
Nothing is typed by hand: if a number is in the paper, it must come from this script.

  python make_report.py --results results --audit audit --out report

Writes
  report/REPORT.md          all tables (markdown) + provenance + completeness check
  report/numbers.json       every number quoted in the text (computed from the DISPLAYED values)
  report/tables/*.csv       the same tables as CSV
  report/fig_*.png          figures
  results_summary.zip       report/ + audit/ + runs/*.json (aggregate results only; NO per-patient
                            predictions, checkpoints or caches) -> the only thing to share
"""
import argparse, glob, json, os, zipfile
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMBOS = [("carevue", "spO2"), ("carevue", "BP"), ("carevue", "RR"),
          ("metavision", "spO2"), ("metavision", "BP"), ("metavision", "RR")]
CN = {"carevue": "CareVue", "metavision": "MetaVision"}
TN = {"spO2": "SpO2", "BP": "BP", "RR": "RR"}
UNIT = {"spO2": "%", "BP": "mmHg", "RR": "breaths/min"}
A_ORDER = ["full", "no_capsule", "no_transformer", "no_cls", "no_pos", "no_ae"]
A_LABEL = {"full": "Full CT-HTTPS", "no_capsule": "w/o Capsule head", "no_transformer": "w/o Transformer",
           "no_cls": "w/o [CLS] token", "no_pos": "w/o positional enc.", "no_ae": "w/o AE loss"}
E_ORDER = ["mean", "linear", "mlp", "gru", "lstm", "transformer"]
E_LABEL = {"mean": "Training mean", "linear": "Linear", "mlp": "MLP", "gru": "GRU", "lstm": "LSTM",
           "transformer": "Transformer (centralised)"}
C_LABEL = {"cold": "Cold start", "warm_minl1": "Warm start (min-L1 matching)", "warm_random": "Warm start (random matching)"}
# Numbers REPORTED in the HTTPS paper (Syu et al., Inf. Fusion 2025, Figs. 4-5 text). Different
# extraction, split (60/20/20), window (W=3), 10 repetitions: for context only, NOT comparable.
HTTPS_REPORTED = {("carevue", "spO2"): 12.38, ("carevue", "BP"): 329.44, ("carevue", "RR"): 44.26,
                  ("metavision", "spO2"): 29.72, ("metavision", "BP"): 2959.16, ("metavision", "RR"): 35.28}
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def fmt(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    return f"{x:.1f}" if a >= 1000 else f"{x:.2f}" if a >= 1 else f"{x:.3f}"


def disp(x):                      # value as displayed in the tables
    return float(fmt(x)) if fmt(x) != "n/a" else float("nan")


def ms(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


class Runs:
    def __init__(self, results):
        self.by = {}
        self.files = sorted(glob.glob(os.path.join(results, "runs", "*.json")))
        for f in self.files:
            r = json.load(open(f))
            d = r["run"]
            self.by[(d["family"], d["variant"], d["cohort"], d["target"], d["seed"])] = r
        self.preds_dir = os.path.join(results, "preds")

    def get(self, fam, var, co, tg):
        return {k[4]: v for k, v in self.by.items() if k[:4] == (fam, var, co, tg)}

    def metric(self, fam, var, co, tg, m="MSE"):
        rs = self.get(fam, var, co, tg)
        return ms([r["final"]["test"][m] for r in rs.values()]) if rs else None

    def preds(self, fam, var, co, tg, seed):
        f = os.path.join(self.preds_dir, f"{fam}__{co}__{tg}__{var}__seed{seed}.npz")
        return np.load(f) if os.path.isfile(f) else None


def cell(t):
    return "n/a" if t is None else f"{fmt(t[0])} ± {fmt(t[1])}"


def table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(L)


def save_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def paired_bootstrap(R, var_a, fam_a, var_b, fam_b, co, tg, B=2000, seed=0):
    """Patient-level bootstrap of MSE(a) - MSE(b) on the identical test set, averaged over seeds."""
    seeds = sorted(set(R.get(fam_a, var_a, co, tg)) & set(R.get(fam_b, var_b, co, tg)))
    rng = np.random.RandomState(seed)
    diffs, per_seed = np.zeros(B), []
    used = 0
    for s in seeds:
        a, b = R.preds(fam_a, var_a, co, tg, s), R.preds(fam_b, var_b, co, tg, s)
        if a is None or b is None:
            continue
        if len(a["true"]) != len(b["true"]) or not np.allclose(a["true"], b["true"]) \
                or not np.array_equal(a["subject"], b["subject"]):
            raise RuntimeError(f"test sets differ for {co}/{tg} seed {s}: {var_a} vs {var_b}")
        ea, eb = (a["pred"] - a["true"]) ** 2, (b["pred"] - b["true"]) ** 2
        per_seed.append(100 * (ea.mean() - eb.mean()) / eb.mean())
        subj = a["subject"]; u, inv = np.unique(subj, return_inverse=True)
        sa, sb = np.bincount(inv, ea), np.bincount(inv, eb); cnt = np.bincount(inv)
        for i in range(B):
            k = rng.randint(len(u), size=len(u))
            w = np.bincount(k, minlength=len(u))
            diffs[i] += 100 * ((w * sa).sum() / (w * cnt).sum() - (w * sb).sum() / (w * cnt).sum()) / ((w * sb).sum() / (w * cnt).sum())
        used += 1
    if not used:
        return None
    diffs /= used
    return {"delta_pct_mean_over_seeds": float(np.mean(per_seed)), "per_seed": [float(x) for x in per_seed],
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "n_seeds": used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--audit", default="audit")
    ap.add_argument("--out", default="report"); ap.add_argument("--zip", default="results_summary.zip")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    R = Runs(a.results)
    if not R.by:
        raise SystemExit(f"no runs found in {a.results}/runs")
    N, MD = {}, []          # numbers.json, markdown parts

    # ---------------- provenance ----------------
    envs = [r.get("env", {}) for r in R.by.values()]
    md5s = sorted({e.get("code_md5") for e in envs})
    quick = any(e.get("quick") for e in envs)
    seeds = sorted({k[4] for k in R.by})
    MD.append("# CT-HTTPS results report (generated by make_report.py)\n")
    if quick:
        MD.append("> **SMOKE TEST (--quick, 2 epochs). These numbers must NOT be used in the manuscript.**\n")
    MD.append(f"- runs found: {len(R.by)}; seeds: {seeds}; code versions (md5): {md5s}"
              + ("  **WARNING: runs come from different code versions**" if len(md5s) > 1 else ""))
    mpps = sorted({str(r.get("config", {}).get("max_per_patient")) for r in R.by.values()})
    if len(mpps) > 1:
        MD.append(f"- **WARNING: runs use different max_per_patient values {mpps}; results are not comparable**")
    N["provenance_max_per_patient"] = mpps
    MD.append(f"- GPU(s): {sorted({str(e.get('gpu')) for e in envs})}; torch: {sorted({str(e.get('torch')) for e in envs})}")
    fam_count = defaultdict(int)
    for k in R.by: fam_count[k[0]] += 1
    MD.append(f"- runs per family: {dict(sorted(fam_count.items()))}\n")
    missing = []
    for co, tg in COMBOS:
        for s in seeds:
            for v in A_ORDER:
                if ("A", v, co, tg, s) not in R.by: missing.append(f"A/{v}/{co}/{tg}/seed{s}")
            for v in E_ORDER:
                if ("E", v, co, tg, s) not in R.by: missing.append(f"E/{v}/{co}/{tg}/seed{s}")
    MD.append(f"- missing core runs (A, E): {len(missing)}" + (f" e.g. {missing[:5]}" if missing else "") + "\n")
    N["provenance"] = {"n_runs": len(R.by), "seeds": seeds, "code_md5": md5s, "quick": quick, "missing_core": missing}

    # ---------------- dataset table ----------------
    ap_ = os.path.join(a.audit, "audit.json")
    if os.path.isfile(ap_):
        MD.append("## Table 2 – Dataset (from dataset_audit.py)\n")
        MD.append(open(os.path.join(a.audit, "audit.md")).read().split("\n", 1)[1])
    rows = []
    for co, tg in COMBOS:
        r = R.get("A", "full", co, tg)
        if r:
            d = r[min(r)]["data_stats"]
            rows.append([CN[co], TN[tg], d["patients_total"], f"{d['patients_train']}/{d['patients_val']}/{d['patients_test']}",
                         f"{d['samples_train']}/{d['samples_val']}/{d['samples_test']}", d["samples_total"],
                         fmt(d.get("gap_hours_median") or float('nan'))])
    hdr = ["Cohort", "Target", "Patients", "Patients train/val/test", "Samples train/val/test", "Samples", "Median gap (h)"]
    MD.append(f"\n## Table 2b – Split actually used (seed {seeds[0]})\n\n" + table(hdr, rows) + "\n")
    save_csv(os.path.join(a.out, "tables", "table2_split.csv"), hdr, rows)

    # ---------------- main comparison ----------------
    hdr = ["Model"] + [f"{CN[c]} {TN[t]}" for c, t in COMBOS]
    for metric in ("MSE", "MAE", "R2"):
        rows, best = [], {}
        entries = [("E", v, E_LABEL[v]) for v in E_ORDER] + [("A", "full", "CT-HTTPS (proposed)")]
        vals = {(f, v): [R.metric(f, v, c, t, metric) for c, t in COMBOS] for f, v, _ in entries}
        for j in range(len(COMBOS)):
            cand = [(vals[(f, v)][j][0], (f, v)) for f, v, _ in entries if vals[(f, v)][j]]
            if cand:
                best[j] = (max if metric == "R2" else min)(cand)[1]
        for f, v, lab in entries:
            rows.append([lab] + [cell(vals[(f, v)][j]) + (" †" if best.get(j) == (f, v) else "") for j in range(len(COMBOS))])
            N.setdefault("main", {}).setdefault(metric, {})[v] = [disp(x[0]) if x else None for x in vals[(f, v)]]
        MD.append(f"\n## Table 4 – Test {metric} (mean ± SD over seeds; † best mean). Same patients, same split for all models.\n\n"
                  + table(hdr, rows) + "\n")
        save_csv(os.path.join(a.out, "tables", f"table4_{metric}.csv"), hdr, rows)
    rows = [["HTTPS, as reported in [10]"] + [fmt(HTTPS_REPORTED[c]) for c in COMBOS]]
    MD.append("\nReference only – values reported in the HTTPS paper under a DIFFERENT protocol (other extraction, "
              "60/20/20 split, W=3, 10 runs). Not a like-for-like comparison:\n\n" + table(hdr, rows) + "\n")

    # paired comparison with bootstrap
    rows = []
    for v in E_ORDER[1:]:
        row = [f"CT-HTTPS vs {E_LABEL[v]}"]
        for co, tg in COMBOS:
            b = paired_bootstrap(R, "full", "A", v, "E", co, tg, B=a.bootstrap)
            N.setdefault("paired", {}).setdefault(v, {})[f"{co}|{tg}"] = b
            row.append("n/a" if b is None else f"{b['delta_pct_mean_over_seeds']:+.1f}% [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}]")
        rows.append(row)
    MD.append("\n## Table 4b – Relative change in test MSE of CT-HTTPS vs each baseline (negative = CT-HTTPS better). "
              "Mean over seeds, 95% patient-level bootstrap CI.\n\n" + table(["Comparison"] + hdr[1:], rows) + "\n")
    save_csv(os.path.join(a.out, "tables", "table4b_paired.csv"), ["Comparison"] + hdr[1:], rows)

    # params + comm
    rows = []
    for f, v, lab in [("E", x, E_LABEL[x]) for x in E_ORDER[1:]] + [("A", x, A_LABEL[x]) for x in A_ORDER]:
        pr = [R.get(f, v, c, "spO2") for c in ("carevue", "metavision")]
        rows.append([lab] + [str(list(p.values())[0]["params"]["total"]) if p else "n/a" for p in pr])
    MD.append("\n## Table 3/7 – Trainable parameters\n\n" + table(["Model", "CareVue", "MetaVision"], rows) + "\n")
    rows = []
    for co, tg in COMBOS:
        r = R.get("A", "full", co, tg)
        if r and "comm" in list(r.values())[0]:
            cm = list(r.values())[0]["comm"]
            rows.append([CN[co], TN[tg], fmt(cm["client_to_server_MB_per_epoch"]), fmt(cm["server_to_client_MB_per_epoch"]), cm["train_samples"]])
    MD.append("\n## Communication per training epoch (simulated, float32)\n\n"
              + table(["Cohort", "Target", "Client→server MB", "Server→client MB", "Train samples"], rows) + "\n")

    # ---------------- ablation A ----------------
    rows = []
    for v in A_ORDER:
        rows.append([A_LABEL[v]] + [cell(R.metric("A", v, c, t)) for c, t in COMBOS])
    MD.append("\n## Table 6 – Component ablation, test MSE (mean ± SD)\n\n" + table(hdr, rows) + "\n")
    save_csv(os.path.join(a.out, "tables", "table6_ablation.csv"), hdr, rows)
    for v in A_ORDER[1:]:
        N.setdefault("ablation_pct", {})[v] = {}
        for c, t in COMBOS:
            f_, x_ = R.metric("A", "full", c, t), R.metric("A", v, c, t)
            if f_ and x_:
                N["ablation_pct"][v][f"{c}|{t}"] = round(100 * (disp(x_[0]) - disp(f_[0])) / disp(f_[0]), 1)

    # ---------------- window B ----------------
    rows = []
    for W in (3, 10, 20, 40):
        rows.append([f"W = {W}"] + [cell(R.metric("A", "full", c, "spO2") if W == 40 else R.metric("B", f"W{W}", c, "spO2"))
                                    for c in ("carevue", "metavision")])
    MD.append("\n## Table 8 – Window length, SpO2 test MSE\n\n" + table(["Window", "CareVue SpO2", "MetaVision SpO2"], rows) + "\n")
    for c in ("carevue", "metavision"):
        w3, w40 = R.metric("B", "W3", c, "spO2"), R.metric("A", "full", c, "spO2")
        if w3 and w40:
            N.setdefault("window_pct_W3_vs_W40", {})[c] = round(100 * (disp(w3[0]) - disp(w40[0])) / disp(w40[0]), 1)

    # ---------------- transfer C (+D) ----------------
    rows = []
    for v in ("cold", "warm_minl1", "warm_random"):
        rows.append([C_LABEL[v]] + [cell(R.metric("A", "full", c, t) if v == "cold" else R.metric("C", v, c, t)) for c, t in COMBOS])
    MD.append("\n## Table 9 – Knowledge transfer, test MSE\n\n" + table(hdr, rows) + "\n")
    for v in ("warm_minl1", "warm_random"):
        N.setdefault("transfer_pct", {})[v] = {}
        for c, t in COMBOS:
            f_, x_ = R.metric("A", "full", c, t), R.metric("C", v, c, t)
            if f_ and x_:
                N["transfer_pct"][v][f"{c}|{t}"] = round(100 * (disp(x_[0]) - disp(f_[0])) / disp(f_[0]), 1)
    dvars = sorted({k[1] for k in R.by if k[0] == "D"})
    if dvars:
        rows = [[v] + [cell(R.metric("D", v, c, t)) for c, t in COMBOS] for v in dvars]
        MD.append("\n## Cross-task transfer (family D), test MSE\n\n" + table(hdr, rows) + "\n")

    # ---------------- sensitivity S ----------------
    rows = [[lab] + [cell(R.metric(f, v, c, t)) for c, t in COMBOS]
            for f, v, lab in (("A", "full", "Main setting (filter on, sliding history)"), ("S", "no_filter", "Without plausibility filter"),
                              ("S", "reset_history", "HTTPS-style windows (cleared after each label)"))]
    MD.append("\n## Sensitivity analyses, CT-HTTPS test MSE (NOTE: sample sets differ between rows, compare with care)\n\n"
              + table(hdr, rows) + "\n")

    # ---------------- privacy ----------------
    rows = []
    for co, tg in COMBOS:
        for fam, var, nz in [("A", "full", 0.0)] + [("P", v, float(v[5:])) for v in sorted({k[1] for k in R.by if k[0] == "P"})]:
            rs = R.get(fam, var, co, tg)
            rs = {s: r for s, r in rs.items() if "privacy" in r}
            if not rs:
                continue
            mse = ms([r["final"]["test"]["MSE"] for r in rs.values()])
            a1 = ms([r["privacy"]["input_reconstruction"]["mean_R2"] for r in rs.values()])
            a2 = {k: ms([r["privacy"]["label_recovery"][k]["R2"] for r in rs.values()]) for k in list(rs.values())[0]["privacy"]["label_recovery"]}
            rows.append([CN[co], TN[tg], f"{nz:g}", cell(mse), cell(a1)] + [cell(a2[k]) for k in sorted(a2)])
            N.setdefault("privacy", {})[f"{co}|{tg}|noise{nz:g}"] = {"MSE": disp(mse[0]), "A1_R2": disp(a1[0]),
                                                                     **{f"A2_R2_{k}": disp(a2[k][0]) for k in a2}}
    if rows:
        ks = sorted(list(list(R.by.values())[0].get("privacy", {"label_recovery": {"k10": 0, "k100": 0}})["label_recovery"]))
        MD.append("\n## Table 10 – Privacy evaluation (honest-but-curious server, auxiliary = validation patients, "
                  "attacked = test patients). R2 close to 1 = near-perfect recovery.\n\n"
                  + table(["Cohort", "Target", "Latent noise", "Test MSE", "A1 input reconstruction R2"]
                          + [f"A2 label recovery R2 ({k})" for k in ks], rows) + "\n")
        save_csv(os.path.join(a.out, "tables", "table10_privacy.csv"),
                 ["Cohort", "Target", "Noise", "MSE", "A1_R2"] + [f"A2_{k}" for k in ks], rows)

    # ---------------- convergence ----------------
    conv = []
    for co in ("carevue", "metavision"):
        r = R.get("A", "full", co, "spO2")
        if r:
            rr = r[min(r)]
            conv.append((co, rr["history"], rr["best_epoch"], rr["epochs_run"]))
            fin = rr["final"]
            N.setdefault("convergence", {})[co] = {"best_epoch": rr["best_epoch"], "epochs_run": rr["epochs_run"],
                                                   "train_MAE": disp(fin["train"]["MAE"]), "test_MAE": disp(fin["test"]["MAE"])}
    figures(R, conv, a.out)

    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(MD) + "\n")
    json.dump(N, open(os.path.join(a.out, "numbers.json"), "w"), indent=1)
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join(a.out, "**", "*"), recursive=True):
            if os.path.isfile(f): z.write(f)
        for f in glob.glob(os.path.join(a.audit, "*")):
            if os.path.isfile(f): z.write(f)
        for f in R.files: z.write(f)
        lg = os.path.join(a.results, "logs", "run_log.txt")
        if os.path.isfile(lg): z.write(lg)
    print(f"report -> {a.out}/REPORT.md ; share ONLY {a.zip}")


def figures(R, conv, out):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # main comparison
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ents = [("E", v, E_LABEL[v]) for v in E_ORDER[1:]] + [("A", "full", "CT-HTTPS")]
    w = 0.8 / len(ents)
    for i, (f, v, lab) in enumerate(ents):
        rel, err = [], []
        for c, t in COMBOS:
            m0, x = R.metric("E", "mean", c, t), R.metric(f, v, c, t)
            rel.append(x[0] / m0[0] if (x and m0) else np.nan); err.append(x[1] / m0[0] if (x and m0) else 0)
        ax.bar(np.arange(6) + (i - (len(ents) - 1) / 2) * w, rel, w, yerr=err, capsize=2, color=PAL[i], label=lab)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{CN[c]}\n{TN[t]}" for c, t in COMBOS])
    ax.set_ylabel("Test MSE / MSE of training-mean predictor")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_main_comparison.png"), dpi=200); plt.close(fig)
    # ablation
    fig, ax = plt.subplots(figsize=(9.5, 3.8)); vs = A_ORDER[1:]; w = 0.8 / len(vs)
    for i, v in enumerate(vs):
        d, e = [], []
        for c, t in COMBOS:
            f_, x = R.metric("A", "full", c, t), R.metric("A", v, c, t)
            d.append(100 * (x[0] - f_[0]) / f_[0] if (f_ and x) else np.nan); e.append(100 * x[1] / f_[0] if (f_ and x) else 0)
        ax.bar(np.arange(6) + (i - (len(vs) - 1) / 2) * w, d, w, yerr=e, capsize=2, color=PAL[i], label=A_LABEL[v])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(6)); ax.set_xticklabels([f"{CN[c]}\n{TN[t]}" for c, t in COMBOS])
    ax.set_ylabel("Change in test MSE vs full model (%)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_ablation_components.png"), dpi=200); plt.close(fig)
    # window
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, c, col in zip(axes, ("carevue", "metavision"), PAL):
        Ws, m, s = [], [], []
        for W in (3, 10, 20, 40):
            x = R.metric("A", "full", c, "spO2") if W == 40 else R.metric("B", f"W{W}", c, "spO2")
            if x: Ws.append(W); m.append(x[0]); s.append(x[1])
        if Ws:
            m, s = np.array(m), np.array(s)
            ax.plot(Ws, m, color=col, lw=2, marker="o"); ax.fill_between(Ws, m - s, m + s, color=col, alpha=0.2, lw=0)
            ax.set_xscale("log"); ax.set_xticks(Ws); ax.set_xticklabels([str(x) for x in Ws])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Window length W (events)"); ax.set_ylabel("Test MSE"); ax.set_title(f"{CN[c]} SpO2")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_window.png"), dpi=200); plt.close(fig)
    # transfer
    fig, ax = plt.subplots(figsize=(9.5, 3.5)); vs = ("warm_minl1", "warm_random"); w = 0.35
    for i, v in enumerate(vs):
        d, e = [], []
        for c, t in COMBOS:
            f_, x = R.metric("A", "full", c, t), R.metric("C", v, c, t)
            d.append(100 * (x[0] - f_[0]) / f_[0] if (f_ and x) else np.nan); e.append(100 * x[1] / f_[0] if (f_ and x) else 0)
        ax.bar(np.arange(6) + (i - 0.5) * w, d, w, yerr=e, capsize=2, color=PAL[i + 1], label=C_LABEL[v])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(6)); ax.set_xticklabels([f"{CN[c]}\n{TN[t]}" for c, t in COMBOS])
    ax.set_ylabel("Change in test MSE vs cold start (%)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_transfer.png"), dpi=200); plt.close(fig)
    # convergence
    if conv:
        fig, axes = plt.subplots(1, len(conv), figsize=(4.4 * len(conv), 3.2))
        axes = np.atleast_1d(axes)
        for ax, (co, h, be, er) in zip(axes, conv):
            ep = np.arange(1, len(h["train_MAE"]) + 1)
            for sp, col, ls in (("train", "#2563EB", "-"), ("val", "#16A34A", "-"), ("test", "#DC2626", "--")):
                ax.plot(ep, h[f"{sp}_MAE"], color=col, ls=ls, lw=1.8, label=sp)
            ax.axvline(be, color="grey", lw=0.8, ls=":"); ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (%)")
            ax.set_title(f"{CN[co]} SpO2 (best epoch {be}, stopped at {er})", fontsize=9); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_convergence.png"), dpi=200); plt.close(fig)
    # privacy-utility
    pts = defaultdict(list)
    for k, r in R.by.items():
        if "privacy" in r and k[3] == "spO2":
            pts[k[2]].append((r["config"]["z_noise"], r["final"]["test"]["MSE"], r["privacy"]["input_reconstruction"]["mean_R2"]))
    if pts:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for (co, P), col in zip(sorted(pts.items()), PAL):
            P = np.array(P); nz = sorted(set(P[:, 0]))
            mx = [P[P[:, 0] == n, 1].mean() for n in nz]; my = [P[P[:, 0] == n, 2].mean() for n in nz]
            ax.plot(mx, my, marker="o", color=col, label=CN[co])
            for n, x, y in zip(nz, mx, my): ax.annotate(f"σ={n:g}", (x, y), fontsize=7)
        ax.set_xlabel("Test MSE (SpO2)"); ax.set_ylabel("Input reconstruction R²"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_privacy_utility.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
