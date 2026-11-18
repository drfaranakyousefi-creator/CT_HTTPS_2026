"""
Data extraction for CT-HTTPS (MIMIC-III CHARTEVENTS).

v3 changes (see CHANGES.md, section "v3"):
  * MetaVision non-invasive BP now uses the MEAN itemid 220181 (v2 mixed systolic 220179 and
    diastolic 220180 into one channel, contradicting the manuscript);
  * CareVue temperature 678 (Fahrenheit) is converted to Celsius before it shares a channel
    with 676 (Celsius);
  * pre-specified physiological plausibility ranges remove charting errors (switchable, so a
    sensitivity analysis without the filter can be reported);
  * strict-past windows: an input event recorded at the same charttime as the label (or later)
    is never part of that label's window (no concurrent-measurement leakage);
  * the time gap between the last input event and the label is stored (prediction horizon);
  * sliding history (default): the buffer is no longer cleared after every label. In v2 (and in
    HTTPS) a window only contained the events since the previous label - on the MIMIC-III demo the
    median window held ONE event (~96% padding), so W=40 was never used. history="reset"
    reproduces the old behaviour for comparison;
  * the cache key contains the extraction configuration, so a changed mapping/filter can never
    silently reuse an old cache.
Unchanged from v2: patient-level seeded split, train-only per-channel normalisation on observed
entries, labels standardised for training and metrics reported in original units.
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch

EXTRACTION_VERSION = "v3"

# itemid groups -------------------------------------------------------------
ITEM_IDS = {
    "metavision": [220045, 220210, 220181, 220052, 220277],
    "carevue":    [211, 618, 52, 456, 676, 678, 646],
}
SPO2 = {646, 220277}            # SpO2 / O2 saturation pulseoxymetry
ABP = {52, 220052}              # arterial BP mean
RR = {618, 220210}              # respiratory rate
HR = {211, 220045}              # heart rate
NIBP = {456, 220181}            # non-invasive BP mean
TEMP_C = {676}                  # temperature, Celsius
TEMP_F = {678}                  # temperature, Fahrenheit -> converted to Celsius
TEMP = TEMP_C | TEMP_F

N_CHANNELS = {"metavision": 4, "carevue": 5}
TARGETS = ("spO2", "BP", "RR")

# Pre-specified plausibility ranges (inclusive), applied to every itemid of the group BEFORE any
# experiment. Values outside are treated as charting errors and dropped. Report them in Methods.
PHYS_RANGES = {
    "SPO2": (50.0, 100.0),      # %
    "ABP": (20.0, 200.0),       # mmHg, mean arterial pressure
    "NIBP": (20.0, 200.0),      # mmHg, mean
    "RR": (1.0, 70.0),          # breaths/min
    "HR": (20.0, 250.0),        # beats/min
    "TEMP": (25.0, 45.0),       # deg C (after F->C conversion)
}
_GROUPS = {"SPO2": SPO2, "ABP": ABP, "NIBP": NIBP, "RR": RR, "HR": HR, "TEMP": TEMP}


def _role(item_id, target):
    """Return ('label', None) | ('channel', idx) | (None, None).
    Channel order: 0 = ABP (or SpO2 when target is BP), 1 = RR (or SpO2 when target is RR),
    2 = HR, 3 = NIBP mean, 4 = Temperature (CareVue only)."""
    if item_id in SPO2:
        if target == "spO2":
            return "label", None
        return "channel", (0 if target == "BP" else 1)
    if item_id in ABP:
        return ("label", None) if target == "BP" else ("channel", 0)
    if item_id in RR:
        return ("label", None) if target == "RR" else ("channel", 1)
    if item_id in HR:
        return "channel", 2
    if item_id in NIBP:
        return "channel", 3
    if item_id in TEMP:
        return "channel", 4
    return None, None


def _extract_subject(item_ids, values, times, W, N, target, strict_past=True, history="sliding"):
    """One patient -> lists of (window (W,N), label, mask (W,), gap_hours).

    Events are processed in time order. Every input event updates a carry-forward state vector
    (last observed value per channel) and appends a copy of it to the history buffer (the most
    recent W events are kept). Every label event emits one sample.
      history="sliding" (default, v3): the buffer is NOT cleared after a label, so each window holds
                         the last W input events before the label (overlapping windows; W is the
                         real context length).
      history="reset"  : HTTPS-style - the buffer and the carry-forward state are cleared after
                         every label, so a window only holds the events since the previous label.
    With strict_past=True only events with charttime < label charttime enter the window."""
    X, Y, M, G = [], [], [], []
    buf, tbuf = [], []
    e = np.zeros(N, dtype=np.float32)
    have_t = times is not None
    for k, (it, v) in enumerate(zip(item_ids, values)):
        kind, ch = _role(int(it), target)
        if kind is None:
            continue
        t = times[k] if have_t else None
        if kind == "label":
            if have_t and strict_past:
                keep = [i for i, tt in enumerate(tbuf) if tt < t]
                rows = [buf[i] for i in keep][-W:]; rt = [tbuf[i] for i in keep][-W:]
            else:
                rows, rt = buf[-W:], tbuf[-W:]
            if len(rows) > 0:
                w = np.zeros((W, N), dtype=np.float32)
                m = np.zeros(W, dtype=np.float32)
                w[:len(rows)] = np.stack(rows)
                m[:len(rows)] = 1.0
                gap = float((t - rt[-1]) / np.timedelta64(1, "h")) if have_t else np.nan
                X.append(w); Y.append(v); M.append(m); G.append(gap)
            if history == "reset":
                buf, tbuf = [], []
                e = np.zeros(N, dtype=np.float32)
        else:
            if ch >= N:
                continue
            e[ch] = v
            buf.append(e.copy()); tbuf.append(t)
            if len(buf) > 4 * W:                  # bounded memory; same-time events may be skipped
                del buf[:-2 * W], tbuf[:-2 * W]
    return X, Y, M, G


def extraction_config(filter_ranges=True, strict_past=True, history="sliding", max_per_patient=None):
    return {"version": EXTRACTION_VERSION, "item_ids": ITEM_IDS, "history": history,
            "max_per_patient": max_per_patient,
            "nibp": sorted(NIBP), "temp_f_to_c": sorted(TEMP_F),
            "ranges": PHYS_RANGES if filter_ranges else None, "strict_past": strict_past}


def _cache_path(csv_path, dataset_name, W, target, cache_dir, cfg):
    st = os.stat(csv_path)
    key = (f"{os.path.abspath(csv_path)}|{st.st_size}|{int(st.st_mtime)}|{dataset_name}|{W}|{target}|"
           + json.dumps(cfg, sort_keys=True))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{dataset_name}_{target}_W{W}_{EXTRACTION_VERSION}_{h}.npz")


def read_chartevents(csv_path, keep_items=None, cache_dir="./cache", verbose=True):
    """Stream CHARTEVENTS.csv (~33 GB) in chunks and keep a compact frame of the needed itemids:
    subject_id, itemid, _v (numeric value), _t (charttime), with MIMIC error==1 rows removed.
    The compact frame is cached once per CSV (pickle), so later runs never re-read the CSV."""
    keep = sorted(set(keep_items) if keep_items is not None
                  else set(ITEM_IDS["carevue"]) | set(ITEM_IDS["metavision"]))
    st = os.stat(csv_path)
    key = f"{os.path.abspath(csv_path)}|{st.st_size}|{int(st.st_mtime)}|{keep}|{EXTRACTION_VERSION}"
    os.makedirs(cache_dir, exist_ok=True)
    cp = os.path.join(cache_dir, f"events_{hashlib.md5(key.encode()).hexdigest()[:12]}.pkl")
    if os.path.isfile(cp):
        return pd.read_pickle(cp)
    head = pd.read_csv(csv_path, nrows=0)
    cols = {c.lower(): c for c in head.columns}
    usecols = [cols[c] for c in ("subject_id", "itemid", "value", "valuenum", "charttime", "error") if c in cols]
    parts, n_err, n_rows = [], 0, 0
    for ch in pd.read_csv(csv_path, usecols=usecols, low_memory=False, chunksize=5_000_000):
        ch = ch.rename(columns={c: c.lower() for c in ch.columns})
        n_rows += len(ch)
        ch = ch[ch["itemid"].isin(keep)]
        if "error" in ch.columns:
            err = pd.to_numeric(ch["error"], errors="coerce").fillna(0) == 1
            n_err += int(err.sum()); ch = ch[~err]
        v = pd.to_numeric(ch["valuenum"], errors="coerce") if "valuenum" in ch.columns else None
        if "value" in ch.columns:
            vv = pd.to_numeric(ch["value"], errors="coerce")
            v = vv if v is None else v.fillna(vv)
        out = pd.DataFrame({"subject_id": ch["subject_id"].astype(np.int64),
                            "itemid": ch["itemid"].astype(np.int32),
                            "_v": v.astype(np.float32)})
        if "charttime" in ch.columns:
            out["_t"] = pd.to_datetime(ch["charttime"], errors="coerce")
        parts.append(out)
        if verbose:
            print(f"[read] {n_rows/1e6:.0f}M rows scanned, {sum(len(p) for p in parts)/1e6:.1f}M kept", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.attrs["events_error_flag_removed"] = n_err
    df.to_pickle(cp)
    return df


def clean_events(df, dataset_name, filter_ranges=True):
    """Numeric value, F->C, plausibility filter. Accepts a raw CHARTEVENTS frame or the compact
    frame of read_chartevents. Returns (df, report)."""
    n_err = df.attrs.get("events_error_flag_removed")
    df = df.rename(columns={c: c.lower() if c not in ("_v", "_t") else c for c in df.columns})
    df = df[df["itemid"].isin(ITEM_IDS[dataset_name])].copy()
    rep = {"events_raw": int(len(df))}
    if "_v" not in df.columns:
        vcol = "valuenum" if "valuenum" in df.columns else "value"
        df["_v"] = pd.to_numeric(df[vcol], errors="coerce")
        if vcol == "valuenum" and "value" in df.columns:
            df["_v"] = df["_v"].fillna(pd.to_numeric(df["value"], errors="coerce"))
        if "error" in df.columns:
            err = pd.to_numeric(df["error"], errors="coerce").fillna(0) == 1
            n_err = int(err.sum()); df = df[~err]
    if "_t" not in df.columns and "charttime" in df.columns:
        df["_t"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df[np.isfinite(df["_v"])]
    rep["events_numeric"] = int(len(df))
    if n_err is not None:
        rep["events_error_flag_removed_all_cohorts"] = int(n_err)
    f = df["itemid"].isin(TEMP_F)
    df.loc[f, "_v"] = (df.loc[f, "_v"] - 32.0) * 5.0 / 9.0
    rep["temp_f_converted"] = int(f.sum())
    if filter_ranges:
        ok = np.ones(len(df), dtype=bool)
        removed = {}
        for g, (lo, hi) in PHYS_RANGES.items():
            sel = df["itemid"].isin(_GROUPS[g]).to_numpy()
            bad = sel & ~((df["_v"] >= lo) & (df["_v"] <= hi)).to_numpy()
            removed[g] = int(bad.sum()); ok &= ~bad
        df = df[ok]
        rep["removed_out_of_range"] = removed
    rep["events_clean"] = int(len(df))
    return df, rep


def load_samples(csv_path, dataset_name, W, target, cache_dir="./cache", df=None, verbose=True,
                 filter_ranges=True, strict_past=True, history="sliding", max_per_patient=None,
                 return_info=False):
    """Returns raw (un-normalised) arrays x (S,W,N), y (S,), mask (S,W), subject (S,)
    [and, with return_info=True, a dict with gap_hours (S,) and the cleaning report]."""
    cfg = extraction_config(filter_ranges, strict_past, history, max_per_patient)
    path = _cache_path(csv_path, dataset_name, W, target, cache_dir, cfg) if csv_path else None
    if path and os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        out = (z["x"], z["y"], z["mask"], z["subject"])
        info = {"gap_hours": z["gap"], "clean_report": json.loads(str(z["report"])), "cache": path}
        return out + (info,) if return_info else out

    if df is None:
        df = read_chartevents(csv_path, cache_dir=cache_dir, verbose=verbose)
    df, rep = clean_events(df, dataset_name, filter_ranges)
    N = N_CHANNELS[dataset_name]
    has_t = "_t" in df.columns
    if has_t:
        df = df[df["_t"].notna()]
        df = df.sort_values(["subject_id", "_t"], kind="mergesort")
    else:
        df = df.sort_values(["subject_id"], kind="mergesort")

    Xs, Ys, Ms, Ss, Gs = [], [], [], [], []
    for sid, g in df.groupby("subject_id", sort=False):
        t = g["_t"].to_numpy() if has_t else None
        X, Y, M, G = _extract_subject(g["itemid"].to_numpy(), g["_v"].to_numpy(np.float32), t, W, N,
                                      target, strict_past, history)
        if Y and max_per_patient and len(Y) > max_per_patient:
            # fixed (seed-independent) random subset of label times per patient, kept in time order
            keep = np.sort(np.random.RandomState(int(sid) % (2**31)).choice(len(Y), max_per_patient, replace=False))
            X = [X[i] for i in keep]; Y = [Y[i] for i in keep]; M = [M[i] for i in keep]; G = [G[i] for i in keep]
        if Y:
            Xs.append(np.stack(X)); Ys.append(np.asarray(Y, np.float32)); Ms.append(np.stack(M))
            Ss.append(np.full(len(Y), sid, dtype=np.int64)); Gs.append(np.asarray(G, np.float32))
    if not Ys:
        raise ValueError(f"no samples for dataset={dataset_name} target={target}")
    x = np.concatenate(Xs); y = np.concatenate(Ys); m = np.concatenate(Ms)
    s = np.concatenate(Ss); gap = np.concatenate(Gs)
    rep["samples"] = int(len(y)); rep["patients"] = int(len(np.unique(s)))
    if path:
        np.savez_compressed(path, x=x, y=y, mask=m, subject=s, gap=gap, report=json.dumps(rep))
    if verbose:
        print(f"[data] {dataset_name}/{target} W={W}: {len(y)} samples, {len(np.unique(s))} patients")
    info = {"gap_hours": gap, "clean_report": rep, "cache": path}
    return (x, y, m, s, info) if return_info else (x, y, m, s)


class Normalizer:
    """Per-channel z-score on observed entries (value != 0 inside the valid mask) + label z-score.
    After the plausibility filter no vital sign can be 0, so 0 unambiguously means 'not observed'."""
    def __init__(self, x, y, mask):
        obs = (x != 0) & (mask[:, :, None] > 0)
        N = x.shape[2]
        self.mean = np.zeros(N, np.float32); self.std = np.ones(N, np.float32)
        for c in range(N):
            v = x[:, :, c][obs[:, :, c]]
            if v.size > 1:
                self.mean[c] = v.mean(); self.std[c] = v.std() + 1e-4
        self.y_mean = float(y.mean()); self.y_std = float(y.std() + 1e-4)

    def x(self, x, mask):
        obs = (x != 0) & (mask[:, :, None] > 0)
        out = (x - self.mean) / self.std
        return np.where(obs, out, 0.0).astype(np.float32)

    def y(self, y):
        return ((y - self.y_mean) / self.y_std).astype(np.float32)


class Split:
    """Holds tensors for one split; iterable in mini-batches (all data lives on `device`)."""
    def __init__(self, x, y_norm, y_raw, mask, device, batch_size, shuffle, subject=None):
        self.x = torch.as_tensor(x, device=device)
        self.y = torch.as_tensor(y_norm, device=device)
        self.y_raw = torch.as_tensor(y_raw, device=device)
        self.mask = torch.as_tensor(mask, device=device)
        self.subject = subject
        self.bs, self.shuffle, self.device = batch_size, shuffle, device

    def __len__(self):
        return self.x.shape[0]

    def __iter__(self):
        n = len(self)
        idx = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, self.bs):
            j = idx[i:i + self.bs]
            yield self.x[j], self.y[j], self.mask[j], self.y_raw[j]


def patient_split(subjects, seed, test_size=0.2, val_size=0.1):
    rng = np.random.RandomState(seed)
    subj = np.unique(subjects); rng.shuffle(subj)
    n_te = max(1, int(round(test_size * len(subj))))
    n_va = max(1, int(round(val_size * (len(subj) - n_te))))
    return subj[n_te + n_va:], subj[n_te:n_te + n_va], subj[:n_te]


class data_preparing:
    def __init__(self, csv_path, dataset_name, W, target, batch_size, seed=0,
                 test_size=0.2, val_size=0.1, device="cpu", cache_dir="./cache", df=None,
                 normalize=True, filter_ranges=True, strict_past=True, history="sliding",
                 max_per_patient=None):
        x, y, m, s, info = load_samples(csv_path, dataset_name, W, target, cache_dir, df=df,
                                        filter_ranges=filter_ranges, strict_past=strict_past,
                                        history=history, max_per_patient=max_per_patient,
                                        return_info=True)
        tr_s, va_s, te_s = patient_split(s, seed, test_size, val_size)
        idx = {k: np.where(np.isin(s, v))[0] for k, v in (("train", tr_s), ("val", va_s), ("test", te_s))}
        assert not (set(tr_s) & set(te_s)) and not (set(tr_s) & set(va_s)) and not (set(va_s) & set(te_s))

        self.norm = Normalizer(x[idx["train"]], y[idx["train"]], m[idx["train"]]) if normalize else None
        def prep(k, shuffle):
            i = idx[k]
            xx = self.norm.x(x[i], m[i]) if normalize else x[i]
            yy = self.norm.y(y[i]) if normalize else y[i]
            return Split(xx, yy, y[i], m[i], device, batch_size, shuffle, subject=s[i])
        self.train = prep("train", True)
        self.train_eval = prep("train", False)
        self.val = prep("val", False)
        self.test = prep("test", False)
        self.y_mean = self.norm.y_mean if normalize else 0.0
        self.y_std = self.norm.y_std if normalize else 1.0
        gap = info["gap_hours"]
        te_ids = np.sort(te_s).astype(np.int64)
        self.stats = {
            "dataset": dataset_name, "target": target, "W": W, "seed": seed,
            "extraction": extraction_config(filter_ranges, strict_past, history, max_per_patient),
            "clean_report": info["clean_report"],
            "patients_total": int(len(np.unique(s))), "patients_train": int(len(tr_s)),
            "patients_val": int(len(va_s)), "patients_test": int(len(te_s)),
            "samples_train": int(len(idx["train"])), "samples_val": int(len(idx["val"])),
            "samples_test": int(len(idx["test"])), "samples_total": int(len(y)),
            "valid_timestep_pct": float(m.mean() * 100),
            "events_per_window_median": float(np.median(m.sum(1))),
            "gap_hours_median": float(np.nanmedian(gap)) if np.isfinite(gap).any() else None,
            "gap_hours_iqr": [float(np.nanpercentile(gap, 25)), float(np.nanpercentile(gap, 75))]
            if np.isfinite(gap).any() else None,
            "label_train_mean": float(y[idx["train"]].mean()), "label_train_std": float(y[idx["train"]].std()),
            "test_subjects_md5": hashlib.md5(te_ids.tobytes()).hexdigest(),
        }
