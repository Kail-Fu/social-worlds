from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
        default=2,
        help="0-based index where embedding feature columns begin (default: 2).",
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


def build_color_map(series: pd.Series) -> list[str]:
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
    categories = list(dict.fromkeys(series.fillna("unknown").astype(str).tolist()))
    category_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    return [category_to_color[value] for value in series.fillna("unknown").astype(str).tolist()]


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


def maybe_plot(coords, labels: pd.Series, colors: list[str], args: argparse.Namespace) -> None:
    if not args.plot:
        return

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10), dpi=200)
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=15)
    plt.gca().set_aspect("equal", "datalim")

    if args.annotate:
        for idx, text in enumerate(labels.astype(str).tolist()):
            plt.annotate(text, (coords[:, 0][idx], coords[:, 1][idx]), fontsize=4)

    plt.title(f"{args.method.upper()} projection")
    plt.savefig(plot_path, format=plot_path.suffix.lstrip(".") or "pdf")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    dataframe = pd.read_csv(input_path)

    missing = [col for col in [args.id_col, args.label_col] if col not in dataframe.columns]
    if missing:
        raise ValueError(f"Required columns missing from input CSV: {missing}")

    features = dataframe.iloc[:, args.feature_start_col :].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if features.shape[1] == 0:
        raise ValueError("No feature columns found. Check --feature-start-col and input format.")

    matrix = features.to_numpy()
    if args.standardize:
        matrix = StandardScaler().fit_transform(matrix)

    if args.method == "tsne":
        coords = run_tsne(matrix, args)
    else:
        coords = run_umap(matrix, args)

    if args.color_col and args.color_col in dataframe.columns:
        color_values = dataframe[args.color_col].astype(str)
        colors = build_color_map(color_values)
    else:
        colors = ["#1f77b4"] * len(dataframe)

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

    maybe_plot(coords, result_df["label"], colors, args)


if __name__ == "__main__":
    main()
