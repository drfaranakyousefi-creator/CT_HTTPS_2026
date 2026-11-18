# CT-HTTPS v3 (Transformer + Capsule split learning on MIMIC-III)

Architecture (Fig. 1 of the paper)
- client front : per-feature AutoEncoders (`client_net.ClientFront`)
- server core  : Transformer encoder + [CLS] + Capsule Network (`server_net.ServerCore`)
- client back  : regression layer + loss (`client_net.ClientBack`); labels are not transmitted
- `transmitter_simulation.SplitChannel`: only detached tensors cross client/server boundaries

## Run everything (recommended): `run_colab.ipynb`
Upload `ct_https_v3.zip` + your licensed `CHARTEVENTS.csv` to your own Google Drive, open the
notebook in Colab (GPU runtime), set the two paths, Run all. Share only `results_summary.zip`.

## Command line (on the machine that holds the licensed MIMIC-III data)
    pip install -r requirements.txt
    python tests/test_pipeline.py                                   # synthetic sanity tests
    python dataset_audit.py --csv CHARTEVENTS.csv --out audit --max_per_patient 20
    python ablation.py      --csv CHARTEVENTS.csv --out results --seeds 0 1 2 --max_per_patient 20
    python make_report.py   --results results --audit audit --out report

| File | Purpose |
|---|---|
| `new_dataset.py` | extraction v3: item mapping, F->C, plausibility filter, strict-past sliding windows, patient-level split, train-only normalisation |
| `Train_simulation.py` | CT-HTTPS model, split-learning training loop, early stopping, transfer |
| `baselines.py` | mean / linear / MLP / GRU / LSTM / centralised Transformer on the same splits |
| `privacy_attack.py` | input-reconstruction and label-recovery attacks (honest-but-curious server) |
| `ablation.py` | resumable runner for all experiment families (A, S, B, E, P, C, D) |
| `make_report.py` | every table / figure / quoted number from the logs + paired bootstrap + `results_summary.zip` |
| `dataset_audit.py` | dataset statistics for Table 2 and the Methods section |

Rules that keep the paper defensible
1. Every number in the manuscript comes from `report/REPORT.md` or `report/numbers.json`.
2. All runs of one paper use the same code version (the report flags mixed `code_md5`) and the same
   `--max_per_patient`.
3. `--quick` runs are smoke tests; the report marks them and they must never be used.
4. `results/preds`, `results/ckpt`, `cache/` contain patient-level data: never share them.

Defaults reproduce the parameter counts of Table 3 (122,246 CareVue / 88,869 MetaVision):
W=40, d_model=16, n_head=4, d_ff=32, L=3, capsules 8x8 -> 4x16, R=3.
