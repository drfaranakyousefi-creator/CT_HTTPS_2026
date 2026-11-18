"""
Dataset audit (v3): run BEFORE the experiments and report the output in the manuscript.

  python dataset_audit.py --csv CHARTEVENTS.csv --out audit        # all cohorts/targets, W=40

For every cohort/target it writes (aggregate statistics only, no patient-level data):
  audit/audit.json      cleaning report, sample/patient counts, label distribution, padding,
                        horizon (gap between last input event and label), per-channel stats
  audit/audit.md        the same as readable tables
  audit/label_hist_<cohort>_<target>.png
(v2 imported a non-existent function `extract_data` and crashed.)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from new_dataset import load_samples, read_chartevents, patient_split

CH_NAMES = {"spO2": ["ABP mean", "RR", "HR", "NIBP mean", "Temp (C)"],
            "BP": ["SpO2", "RR", "HR", "NIBP mean", "Temp (C)"],
            "RR": ["ABP mean", "SpO2", "HR", "NIBP mean", "Temp (C)"]}


def pct(a, q):
    return float(np.percentile(a, q))


def audit_one(x, y, m, s, info, target, seed=0):
    tr, va, te = patient_split(s, seed)
    obs = (x != 0) & (m[:, :, None] > 0)
    ch = []
    for c in range(x.shape[2]):
        v = x[:, :, c][obs[:, :, c]]
        ch.append({"channel": CH_NAMES[target][c], "observed_pct_of_valid_steps": float(obs[:, :, c].sum() / max(m.sum(), 1) * 100),
                   "mean": float(v.mean()) if v.size else None, "std": float(v.std()) if v.size else None,
                   "p1": pct(v, 1) if v.size else None, "p99": pct(v, 99) if v.size else None})
    q1, q3 = np.percentile(y, [25, 75]); iqr = q3 - q1
    gap = info["gap_hours"]; gap = gap[np.isfinite(gap)]
    return {
        "clean_report": info["clean_report"],
        "patients": int(len(np.unique(s))), "samples": int(len(y)),
        "split_patients": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "split_samples": {k: int(np.isin(s, v).sum()) for k, v in (("train", tr), ("val", va), ("test", te))},
        "samples_per_patient_median": float(np.median(np.unique(s, return_counts=True)[1])),
        "label": {"mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()), "max": float(y.max()),
                  "p1": pct(y, 1), "p50": pct(y, 50), "p99": pct(y, 99),
                  "iqr_outlier_pct": float(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).mean() * 100)},
        "events_per_window": {"median": float(np.median(m.sum(1))), "p25": pct(m.sum(1), 25), "p75": pct(m.sum(1), 75)},
        "padding_pct": float((1 - m.mean()) * 100),
        "gap_hours": {"median": pct(gap, 50), "p25": pct(gap, 25), "p75": pct(gap, 75), "p95": pct(gap, 95)} if gap.size else None,
        "channels": ch,
    }


def to_md(res):
    L = ["# Dataset audit (W=40, seed-0 split)\n",
         "| Cohort | Target | Patients | Samples | Train/Val/Test patients | Train/Val/Test samples | Label mean ± sd | Label p1–p99 | Gap h median [IQR] | Events/window median | Padding % |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        co, tg = k.split("|")
        sp, ss, lb, g = r["split_patients"], r["split_samples"], r["label"], r["gap_hours"]
        gs = f"{g['median']:.2f} [{g['p25']:.2f}–{g['p75']:.2f}]" if g else "n/a"
        L.append(f"| {co} | {tg} | {r['patients']:,} | {r['samples']:,} | {sp['train']:,}/{sp['val']:,}/{sp['test']:,} | "
                 f"{ss['train']:,}/{ss['val']:,}/{ss['test']:,} | {lb['mean']:.2f} ± {lb['std']:.2f} | {lb['p1']:.1f}–{lb['p99']:.1f} | "
                 f"{gs} | {r['events_per_window']['median']:.0f} | {r['padding_pct']:.1f} |")
    L.append("\n## Cleaning report\n")
    for k, r in res.items():
        L.append(f"- **{k}**: `{json.dumps(r['clean_report'])}`")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="audit")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--max_per_patient", type=int, default=None)
    ap.add_argument("--cohorts", nargs="+", default=["carevue", "metavision"])
    ap.add_argument("--targets", nargs="+", default=["spO2", "BP", "RR"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = read_chartevents(a.csv, cache_dir=a.cache)
    res = {}
    for co in a.cohorts:
        for tg in a.targets:
            x, y, m, s, info = load_samples(a.csv, co, a.window, tg, a.cache, df=df, return_info=True,
                                         max_per_patient=a.max_per_patient)
            res[f"{co}|{tg}"] = audit_one(x, y, m, s, info, tg)
            plt.figure(figsize=(6, 3.5)); plt.hist(y, bins=80); plt.title(f"{co} {tg} label"); plt.tight_layout()
            plt.savefig(os.path.join(a.out, f"label_hist_{co}_{tg}.png"), dpi=150); plt.close()
    json.dump(res, open(os.path.join(a.out, "audit.json"), "w"), indent=1)
    open(os.path.join(a.out, "audit.md"), "w").write(to_md(res))
    print(to_md(res))


if __name__ == "__main__":
    main()
