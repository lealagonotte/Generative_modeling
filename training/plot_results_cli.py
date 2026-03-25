#!/usr/bin/env python3
"""
Publication-quality plots for Ambient Diffusion experiments.

Usage:
  # Inpainting (expects columns: method, p, delta, Avg wd, Std wd, ...)
  python plot_results_cli.py inpainting \
      --spirals spirals_inpainting.csv \
      --moons   two_moons_inpainting.csv \
      --outdir  figures/

  # Compressed sensing (expects columns: method, m, m_prime, Avg wd, Std wd, ...)
  python plot_results_cli.py cs \
      --spirals spirals_cs.csv \
      --moons   two_moons_cs.csv \
      --outdir  figures/

  # Both at once (all four CSV files)
  python plot_results_cli.py all \
      --inp-spirals spirals_inpainting.csv \
      --inp-moons   two_moons_inpainting.csv \
      --cs-spirals  spirals_cs.csv \
      --cs-moons    two_moons_cs.csv \
      --outdir      figures/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

INP_PALETTE = {
    r"Naive ($\delta=0$)":     "#2c7bb6",
    r"Ambient ($\delta=0.1$)": "#d7191c",
    r"Ambient ($\delta=0.3$)": "#fdae61",
}
INP_MARKERS = {
    r"Naive ($\delta=0$)":     "o",
    r"Ambient ($\delta=0.1$)": "s",
    r"Ambient ($\delta=0.3$)": "D",
}
CS_COLORS = ["#2c7bb6", "#d7191c"]

METRICS_WD_SWD_CD = [
    ("Avg wd",  "Std wd",  "Wasserstein Distance"),
    ("Avg swd", "Std swd", "Sliced Wasserstein Distance"),
    ("Avg cd",  "Std cd",  "Chamfer Distance"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  INPAINTING
# ═══════════════════════════════════════════════════════════════════════════════

def reshape_inpainting(raw: pd.DataFrame, avg: str, std: str) -> pd.DataFrame:
    """Reshape to tidy format.  Only 'naive' rows feed the δ=0 line."""
    rows = []
    for _, r in raw.iterrows():
        method = r["method"]
        delta  = float(r["delta"])
        if method == "naive":
            label = r"Naive ($\delta=0$)"
        elif method == "ambient" and delta == 0.1:
            label = r"Ambient ($\delta=0.1$)"
        elif method == "ambient" and delta == 0.3:
            label = r"Ambient ($\delta=0.3$)"
        else:
            continue
        rows.append(dict(p=float(r["p"]), Setting=label,
                         mean=float(r[avg]), std=float(r[std])))
    return pd.DataFrame(rows)


def plot_inpainting(csv_path: str, dataset: str, out_path: Path):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Inpainting Results - {dataset}",
                 fontsize=17, fontweight="bold", y=1.03)

    settings_order = list(INP_PALETTE.keys())

    for ax, (avg_c, std_c, title) in zip(axes, METRICS_WD_SWD_CD):
        pdf = reshape_inpainting(df, avg_c, std_c)
        for setting in settings_order:
            sub = pdf[pdf["Setting"] == setting].sort_values("p")
            if sub.empty:
                continue
            c, m = INP_PALETTE[setting], INP_MARKERS[setting]
            ax.plot(sub["p"], sub["mean"],
                    marker=m, color=c, linewidth=2.2, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=setting, zorder=4)
            ax.fill_between(sub["p"],
                            sub["mean"] - sub["std"],
                            sub["mean"] + sub["std"],
                            alpha=0.13, color=c, zorder=2)
        ax.set_xlabel("Corruption probability  $p$", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPRESSED SENSING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cs(csv_spirals: str, csv_moons: str, out_path: Path):
    df_s = pd.read_csv(csv_spirals)
    df_m = pd.read_csv(csv_moons)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle("Compressed Sensing Results  ($m=2$)",
                 fontsize=17, fontweight="bold", y=1.03)

    setting_labels = [r"Naive ($m'=0$)", r"Ambient ($m'=1$)"]
    datasets = ["Spirals", "Two Moons"]

    for ax, (avg_c, std_c, title) in zip(axes, METRICS_WD_SWD_CD):
        rows = []
        for df, ds in [(df_s, "Spirals"), (df_m, "Two Moons")]:
            for _, r in df.iterrows():
                mp = int(float(r["m_prime"]))
                lab = setting_labels[0] if mp == 0 else setting_labels[1]
                rows.append(dict(Dataset=ds, Setting=lab,
                                 mean=float(r[avg_c]), std=float(r[std_c])))
        pdf = pd.DataFrame(rows)
        x = np.arange(len(datasets))
        width = 0.34
        for i, setting in enumerate(setting_labels):
            sub = pdf[pdf["Setting"] == setting]
            means = [sub[sub["Dataset"] == d]["mean"].values[0] for d in datasets]
            stds  = [sub[sub["Dataset"] == d]["std"].values[0]  for d in datasets]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   color=CS_COLORS[i], edgecolor="white", linewidth=0.9,
                   error_kw=dict(capsize=5, capthick=1.5, elinewidth=1.5,
                                 color="#333333"),
                   label=setting, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication plots for Ambient Diffusion experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- inpainting sub-command ---
    p_inp = sub.add_parser("inpainting",
        help="Plot inpainting results (line plots with ±1σ bands).")
    p_inp.add_argument("--spirals", required=True,
        help="CSV with inpainting results for Spirals "
             "(cols: method, p, delta, Avg wd, Std wd, …)")
    p_inp.add_argument("--moons", required=True,
        help="CSV with inpainting results for Two Moons")
    p_inp.add_argument("--outdir", default=".", help="Output directory")

    # --- cs sub-command ---
    p_cs = sub.add_parser("cs",
        help="Plot compressed-sensing results (grouped bar chart).")
    p_cs.add_argument("--spirals", required=True,
        help="CSV with CS results for Spirals "
             "(cols: method, m, m_prime, Avg wd, Std wd, …)")
    p_cs.add_argument("--moons", required=True,
        help="CSV with CS results for Two Moons")
    p_cs.add_argument("--outdir", default=".", help="Output directory")

    # --- all sub-command ---
    p_all = sub.add_parser("all",
        help="Plot everything (inpainting + CS).")
    p_all.add_argument("--inp-spirals", required=True)
    p_all.add_argument("--inp-moons",   required=True)
    p_all.add_argument("--cs-spirals",  required=True)
    p_all.add_argument("--cs-moons",    required=True)
    p_all.add_argument("--outdir", default=".", help="Output directory")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.command == "inpainting":
        plot_inpainting(args.spirals, "Spirals",
                        outdir / "inpainting_spirals.png")
        plot_inpainting(args.moons,  "Two Moons",
                        outdir / "inpainting_two_moons.png")

    elif args.command == "cs":
        plot_cs(args.spirals, args.moons,
                outdir / "compressed_sensing.png")

    elif args.command == "all":
        plot_inpainting(args.inp_spirals, "Spirals",
                        outdir / "inpainting_spirals.png")
        plot_inpainting(args.inp_moons,  "Two Moons",
                        outdir / "inpainting_two_moons.png")
        plot_cs(args.cs_spirals, args.cs_moons,
                outdir / "compressed_sensing.png")

    print("\nDone ✓")


"""
python plot_results_cli.py inpainting --spirals spirals.csv --moons moons.csv --outdir figures/
python plot_results_cli.py cs --spirals spirals_cs.csv --moons moons_cs.csv --outdir figures/
python plot_results_cli.py all --inp-spirals ... --inp-moons ... --cs-spirals ... --cs-moons ... --outdir figures/
"""
if __name__ == "__main__":
    main()