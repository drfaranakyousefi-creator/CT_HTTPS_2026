"""End-to-end smoke test on SYNTHETIC data: CSV -> audit -> every experiment family (--quick) -> report.
   python tests/test_end_to_end.py            (~10-20 min on CPU)"""
import os, subprocess, sys, tempfile, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tests"))
import pandas as pd
from test_pipeline import make_df


def run(*a):
    print("$", " ".join(a), flush=True)
    subprocess.run([sys.executable, *a], check=True, cwd=HERE)


def main():
    d = tempfile.mkdtemp()
    csv = os.path.join(d, "CHARTEVENTS.csv")
    df = pd.concat([make_df("carevue", 80, 0), make_df("metavision", 80, 1).assign(subject_id=lambda x: x.subject_id + 1000)])
    df.to_csv(csv, index=False)
    out, aud, rep = (os.path.join(d, x) for x in ("results", "audit", "report"))
    run("dataset_audit.py", "--csv", csv, "--out", aud, "--cache", os.path.join(d, "cache"), "--max_per_patient", "20")
    run("ablation.py", "--csv", csv, "--out", out, "--cache", os.path.join(d, "cache"), "--seeds", "0", "1",
        "--families", "A", "S", "B", "E", "P", "C", "D", "--quick", "--max_per_patient", "20", "--device", "cpu",
        "--ct_lr_grid", "0.01")
    run("make_report.py", "--results", out, "--audit", aud, "--out", rep, "--zip", os.path.join(d, "summary.zip"),
        "--bootstrap", "200")
    md = open(os.path.join(rep, "REPORT.md")).read()
    assert "SMOKE TEST" in md and "Table 10" in md and "Table 4b" in md
    import zipfile
    names = zipfile.ZipFile(os.path.join(d, "summary.zip")).namelist()
    assert not any("/preds/" in n or "/ckpt/" in n or "/cache/" in n for n in names), "patient-level files in summary"
    print("END-TO-END OK", d)


if __name__ == "__main__":
    main()
