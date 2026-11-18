# What was fixed (old -> new)

1. **Train metrics were altered in `evaluate_train`** (`MSE - 32`, `RMSE - 4`, `R2` with `3*ss_tot`). Removed;
   train/val/test are now computed by one function. Any figure made with the old code (e.g. Fig. 3 gap) must be regenerated.
2. **Normalisation leakage**: mean/std were computed on all data (train+test) and included zero padding. Now per-channel,
   train-only, observed entries only; labels standardised for training, metrics reported in original units.
3. **Split**: sequential split on concatenated samples -> seeded PATIENT-level train/val/test split.
4. **No validation / stopping rule**: now early stopping on validation MSE (patience 10, max 60 epochs, best weights restored);
   test is logged for curves but never used for selection. Added ReduceLROnPlateau and grad-clip 1.0.
5. **Architecture != paper**: old code put the Transformer on the client and the final regression on the server.
   Now: client front (AEs) -> server (Transformer+Capsule) -> client back (regression). Verified: split gradients == end-to-end gradients.
6. **Defaults != paper**: d_model 64 / d_ff 128 -> d_model 16 / d_ff 32 (reproduces Table 3 exactly); notebook used W=3 -> W=40.
7. **Window bug**: a full window dropped the next label; now sliding (oldest event dropped). Events sorted by charttime.
8. AE loss normalised by valid positions; asymmetric LR for warm-started AEs (Eq. 21) implemented (`ae_lr_factor`, default 0.1 - check vs. paper).
9. `transmitter_simulation.py` was broken (wrong kwarg) and unused; replaced by `SplitChannel`. Old HTTP transmitter removed.
10. Data extraction is vectorised per patient and cached (.npz).

# v3 (this release) - what was fixed and added

## Data extraction (`new_dataset.py`) - all results must be regenerated
11. **MetaVision NIBP**: v2 mapped 220179 (NIBP *systolic*) and 220180 (NIBP *diastolic*) into one
    "NIBP mean" channel. Now 220181 (NIBP mean), as stated in the manuscript. (Verified in D_ITEMS.)
12. **CareVue temperature**: 676 (deg C) and 678 (deg F) shared one channel without conversion. 678 is
    now converted to deg C. (Verified in D_ITEMS.)
13. **Charting errors**: pre-specified plausibility ranges (SpO2 50-100 %, MAP/NIBP 20-200 mmHg,
    RR 1-70 /min, HR 20-250 /min, T 25-45 C) and the MIMIC `error` flag. The MIMIC-III demo contains
    e.g. NIBP 11,647 mmHg, ABP -8 mmHg and many zeros. Sensitivity run without the filter: family S.
14. **Window content**: v2 (like HTTPS) cleared the buffer after every label; on the MIMIC-III demo the
    median window held 1 event (~96 % padding), so W=40 was never used. Default is now a sliding
    history (last W events before each label); the old behaviour is `history="reset"` (family S).
15. **Concurrent-measurement leakage**: events with the same charttime as the label are excluded
    (`strict_past=True`); the gap between last input and label (horizon) is recorded and reported.
16. **Cache key** now contains the full extraction configuration (v2 would silently reuse an old
    cache after a mapping change). The 33 GB CSV is streamed once and cached in compact form.
17. Optional `max_per_patient` cap (fixed random subset of label times per patient) for run time.

## Experiments
18. **Baselines on the same data** (`baselines.py`, family E): mean, linear, MLP, GRU, LSTM, centralised
    Transformer, same split / normalisation / early stopping; lr chosen on validation from {1e-3, 1e-2}.
    The manuscript's baseline numbers were copied from the HTTPS paper (different extraction, split,
    W, repetitions) and are now shown for reference only.
19. **Statistics**: per-sample test predictions are saved; `make_report.py` reports mean +- SD over
    seeds and a patient-level paired bootstrap 95 % CI of the relative MSE difference.
20. **Privacy** (`privacy_attack.py`, families A/full and P): input reconstruction from the latents
    and label recovery from the returned gradient (exact for a linear client head), plus a
    privacy-utility curve with client-side latent noise.
21. **Traffic accounting** fixed (v2 mixed directions and omitted the server->client gradient).
22. `dataset_audit.py` rewritten (v2 imported a non-existent function and crashed).
23. `make_report.py` replaces `ablation_report.py`: every table, figure and quoted number is
    generated from the run JSONs, with provenance (code md5, GPU, seeds, completeness check).
