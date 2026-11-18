"""Convergence plots (lines only, optional log-scale y axis; no IPython dependency)."""
import os
import matplotlib
import matplotlib.pyplot as plt

_META = {"MSE": "MSE", "RMSE": "RMSE", "MAE": "MAE", "R2": "R$^2$", "MAPE": "MAPE (%)"}


def plot_history(history, save_dir=".", log_y=True, show=False):
    os.makedirs(save_dir, exist_ok=True)
    n = len(history["train_MSE"])
    ep = list(range(1, n + 1))
    saved = []
    for k, lab in _META.items():
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for split, c, ls in (("train", "#2563EB", "-"), ("val", "#16A34A", "-"), ("test", "#DC2626", "--")):
            ax.plot(ep, history[f"{split}_{k}"], color=c, linestyle=ls, linewidth=2, label=split)
        if log_y and k != "R2":
            ax.set_yscale("log")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3); ax.legend(frameon=False)
        fig.tight_layout()
        p = os.path.join(save_dir, f"metric_{k}.png")
        fig.savefig(p, dpi=200)
        (plt.show if show else plt.close)()
        saved.append(p)
    return saved
