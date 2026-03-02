from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run t-SNE or UMAP on an embedding-matrix CSV and export 2D coordinates."
    )
    parser.add_argument("--input", required=True, help="Input embedding CSV path.")
    parser.add_argument("--output", required=True, help="Output coordinates CSV path.")
    parser.add_argument(
        "--json-output",
        help="Optional output JSON path with [{id,label,color,x,y}, ...].",
    )
    parser.add_argument(
        "--method",
        default="tsne",
        choices=["tsne", "umap"],
        help="Dimension reduction method (default: tsne).",
    )
    parser.add_argument(
        "--id-col",
        default="id",
        help="Column used as unique row ID (default: id).",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Column used as display label (default: label).",
    )
    parser.add_argument(
        "--color-col",
        help="Optional column used for scatter color categories.",
    )
    parser.add_argument(
        "--feature-start-col",
        type=int,
        default=-1,
        help="0-based index where embedding feature columns begin; use -1 to auto-detect 'feat_' columns (default).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply StandardScaler before DR.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--plot",
        help="Optional output path for a static scatter plot (PDF/PNG).",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with label text in the static plot.",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=0,
        help="Maximum number of annotations to draw (0 means annotate all selected points).",
    )
    parser.add_argument("--point-size", type=float, default=15.0, help="Scatter point size (default: 15).")
    parser.add_argument("--label-fontsize", type=float, default=4.0, help="Annotation font size (default: 4).")
    parser.add_argument("--fig-width", type=float, default=14.0, help="Plot width in inches (default: 14).")
    parser.add_argument("--fig-height", type=float, default=10.0, help="Plot height in inches (default: 10).")
    parser.add_argument("--dpi", type=int, default=200, help="Plot DPI (default: 200).")

    # t-SNE params
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--tsne-learning-rate", default="auto", help="t-SNE learning rate.")
    parser.add_argument(
        "--tsne-metric",
        default="euclidean",
        help="Distance metric for t-SNE (default: euclidean).",
    )

    # UMAP params
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument(
        "--umap-metric",
        default="euclidean",
        help="Distance metric for UMAP (default: euclidean).",
    )

    return parser.parse_args()


def build_color_map(series: pd.Series) -> tuple[list[str], dict[str, str]]:
    def canonical(value: str) -> str:
        return value.strip().lower()

    special_map = {
        "female": "#d62728",
        "male": "#1f77b4",
        "mixed": "#2ca02c",
        "neutral": "#2ca02c",
        "nuetral": "#2ca02c",
        "na": "#2ca02c",
        "n/a": "#2ca02c",
        "mixed/neutral/na": "#2ca02c",
        "mixed/nuetral/na": "#2ca02c",
    }

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    raw_values = series.fillna("unknown").astype(str).tolist()
    categories = list(dict.fromkeys(raw_values))
    category_to_color = {}
    next_palette_idx = 0
    for cat in categories:
        key = canonical(cat)
        if key in special_map:
            category_to_color[cat] = special_map[key]
        else:
            category_to_color[cat] = palette[next_palette_idx % len(palette)]
            next_palette_idx += 1
    return [category_to_color[value] for value in raw_values], category_to_color


def run_tsne(features, args: argparse.Namespace):
    model = TSNE(
        n_components=2,
        random_state=args.random_state,
        perplexity=args.perplexity,
        learning_rate=args.tsne_learning_rate,
        metric=args.tsne_metric,
        init="pca",
    )
    return model.fit_transform(features)


def run_umap(features, args: argparse.Namespace):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP support requires 'umap-learn'. Install with: pip install -e '.[dr]'"
        ) from exc

    model = umap.UMAP(
        n_components=2,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.umap_metric,
    )
    return model.fit_transform(features)


def maybe_plot(
    coords,
    labels: pd.Series,
    colors: list[str],
    category_to_color: dict[str, str] | None,
    args: argparse.Namespace,
) -> None:
    if not args.plot:
        return

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=args.point_size)
    plt.gca().set_aspect("equal", "datalim")

    if args.annotate:
        label_values = labels.astype(str).tolist()
        indices = list(range(len(label_values)))
        if args.max_annotations > 0:
            indices = indices[: args.max_annotations]
        for idx in indices:
            plt.annotate(label_values[idx], (coords[:, 0][idx], coords[:, 1][idx]), fontsize=args.label_fontsize)

    plt.title(f"{args.method.upper()} projection")
    if category_to_color:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=cat)
            for cat, color in category_to_color.items()
        ]
        plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    plt.savefig(plot_path, format=plot_path.suffix.lstrip(".") or "pdf")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    dataframe = pd.read_csv(input_path)

    missing = [col for col in [args.id_col, args.label_col] if col not in dataframe.columns]
    if missing:
        raise ValueError(f"Required columns missing from input CSV: {missing}")

    feature_start_col = args.feature_start_col
    if feature_start_col < 0:
        feature_columns = [col for col in dataframe.columns if str(col).startswith("feat_")]
        if feature_columns:
            features = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        else:
            feature_start_col = 2
            features = dataframe.iloc[:, feature_start_col:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        features = dataframe.iloc[:, feature_start_col:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if features.shape[1] == 0:
        raise ValueError("No feature columns found. Check --feature-start-col and input format.")

    matrix = features.to_numpy(dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    non_constant_cols = np.std(matrix, axis=0) > 0
    if non_constant_cols.any():
        matrix = matrix[:, non_constant_cols]

    if args.standardize:
        matrix = StandardScaler().fit_transform(matrix)

    if args.method == "tsne":
        coords = run_tsne(matrix, args)
    else:
        coords = run_umap(matrix, args)

    if args.color_col and args.color_col in dataframe.columns:
        color_values = dataframe[args.color_col].astype(str)
        colors, category_to_color = build_color_map(color_values)
    else:
        colors = ["#1f77b4"] * len(dataframe)
        category_to_color = None

    result_df = pd.DataFrame(
        {
            "id": dataframe[args.id_col],
            "label": dataframe[args.label_col],
            "color": colors,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(result_df.to_dict(orient="records"), fp)

    maybe_plot(coords, result_df["label"], colors, category_to_color, args)


if __name__ == "__main__":
    main()
