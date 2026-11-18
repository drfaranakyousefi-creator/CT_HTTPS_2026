"""Train one configuration and save curves/metrics:  python run_main.py --csv CHARTEVENTS.csv --cohort carevue --target spO2"""
import argparse, json
from Train_simulation import CT_HTTPS
import plot

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--cohort", choices=["carevue", "metavision"], required=True)
ap.add_argument("--target", choices=["spO2", "BP", "RR"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="main_out")
a = ap.parse_args()
m = CT_HTTPS(a.csv, a.cohort, a.target, seed=a.seed)
res = m.fit(verbose=True)
print(json.dumps({"params": res["params"], "data_stats": res["data_stats"], "test": res["final"]["test"]}, indent=1))
plot.plot_history(res["history"], a.out)
json.dump(res, open(f"{a.out}/result_{a.cohort}_{a.target}_seed{a.seed}.json", "w"))
