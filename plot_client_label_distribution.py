import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


# NPG-inspired palette (muted, publication-friendly)
NATURE_COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

# Stable label colors (fallback to NATURE_COLORS for unseen labels)
LABEL_COLOR_OVERRIDES = {
    "glioma": "#E64B35",
    "meningioma": "#4DBBD5",
    "pituitary": "#00A087",
    "brain_metastases": "#3C5488",
    "no_tumor": "#B0B0B0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked bar chart of label distribution per client."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to split_manifest.csv. If omitted, script auto-detects common locations.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Use all samples or only one split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (.png). Default: same folder as manifest.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Output PDF path. Default: same folder as manifest.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Keep empty for a cleaner publication style.",
    )
    return parser.parse_args()


def resolve_manifest(manifest: Path | None) -> Path:
    if manifest is not None:
        return manifest
    candidates = [
        Path("data_brisc_style/split_manifest.csv"),
        Path("input/split_manifest.csv"),
        Path("split_manifest.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Manifest not found. Tried: "
        + ", ".join(str(x) for x in candidates)
        + ". Please pass --manifest."
    )


def set_nature_style() -> None:
    # Reset first to avoid inheriting environment/global styles.
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#2B2B2B",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def pretty_label(label: str) -> str:
    return label.replace("_", " ")


def main() -> None:
    args = parse_args()
    manifest = resolve_manifest(args.manifest)
    output_png = args.output or manifest.parent / "client_label_distribution.png"
    output_pdf = args.output_pdf or manifest.parent / "client_label_distribution.pdf"

    df = pd.read_csv(manifest)
    required = {"dataset", "split", "label", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    # Ensure one sample counts once
    df = df.drop_duplicates(subset=["dataset", "split", "label", "sample_id"])

    table = (
        df.groupby(["dataset", "label"])["sample_id"]
        .count()
        .rename("count")
        .reset_index()
        .pivot(index="dataset", columns="label", values="count")
        .fillna(0)
        .astype(int)
    )

    # Fixed client order for readability and consistency.
    preferred_client_order = ["BraTS", "Brisc2025", "Figshare", "Shanghai", "Yale"]
    present_clients = list(table.index)
    ordered_clients = [c for c in preferred_client_order if c in present_clients] + [
        c for c in present_clients if c not in preferred_client_order
    ]
    table = table.loc[ordered_clients]

    # Label order by global frequency
    label_order = table.sum(axis=0).sort_values(ascending=False).index.tolist()
    table = table[label_order]

    set_nature_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)

    bottom = None
    x = list(range(len(table.index)))
    color_iter = iter(NATURE_COLORS)
    label_to_color = {}
    for label in table.columns:
        if label in LABEL_COLOR_OVERRIDES:
            label_to_color[label] = LABEL_COLOR_OVERRIDES[label]
        else:
            label_to_color[label] = next(color_iter, "#7F7F7F")

    for i, label in enumerate(table.columns):
        y = table[label].values
        color = label_to_color[label]
        ax.bar(
            x,
            y,
            bottom=bottom,
            label=pretty_label(label),
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.55,
            width=0.62,
        )
        bottom = y if bottom is None else bottom + y

    ax.set_xlabel("")
    ax.set_ylabel("Sample Count")
    if args.title.strip():
        ax.set_title(args.title.strip(), pad=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(table.index, rotation=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.18, color="#6E7781")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ymax = table.sum(axis=1).max()
    ax.set_ylim(0, ymax * 1.08)

    # Put legend above the axes to avoid overlapping title/data.
    legend = fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=min(5, len(table.columns)),
        fontsize=10.0,
        title=None,
        handlelength=1.5,
        columnspacing=1.2,
    )
    legend._legend_box.align = "left"

    fig.subplots_adjust(top=0.86, left=0.10, right=0.99, bottom=0.13)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=400)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved PNG: {output_png}")
    print(f"Saved PDF: {output_pdf}")


if __name__ == "__main__":
    main()
