# What to do before submission (CT-HTTPS v3)

1. **Run** `run_colab.ipynb` on your own Colab account with your licensed `CHARTEVENTS.csv`
   (Tier 1 is essential; Tiers 2–3 are needed for the ablation, sensitivity and privacy tables).
   Use the same `SEEDS` and `MAXPP` for every run.
2. **Check** `report/REPORT.md`, top section: no "SMOKE TEST", one code md5, no missing core runs,
   one `max_per_patient` value.
3. **Fill the manuscript** (`CT-HTTPS_manuscript_v3_tracked.docx`): every yellow ⟦R: …⟧ placeholder says
   which table / key of `report/numbers.json` it comes from. Copy tables from `report/REPORT.md`
   (or `report/tables/*.csv`) and figures from `report/fig_*.png`. Write the results **as they come
   out**, including combinations where CT-HTTPS is not the best model.
4. **Never** type a number that is not in the report, and never re-run only the models that did badly.
5. Update the GitHub repository to this code version and add a tagged release (Zenodo DOI).
6. Share only `results_summary.zip`. `results/preds`, `results/ckpt` and `cache/` contain
   patient-level data and must stay on the licensed machine.

Why the old numbers cannot be used (details in CHANGES.md and in the review):
- MetaVision NIBP channel mixed systolic and diastolic BP; CareVue temperature mixed °C and °F.
- Windows were cleared after every label, so a window held a median of one event, not 40.
- Baseline numbers were copied from the HTTPS paper (different data, split, window, repetitions).
- Train metrics in the old code were altered; normalisation leaked test data (CHANGES.md items 1–2).
- The standard deviations in the old Tables 6, 8 and 9 follow a fixed pattern across columns that
  independent seeds do not produce; they cannot be traced to run logs.
