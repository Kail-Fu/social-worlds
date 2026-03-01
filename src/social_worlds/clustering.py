from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a hierarchical clustering dendrogram from an annotated matrix CSV."
    )
    parser.add_argument(
        "--input",
        default="english/english_4454.csv",
        help="Input annotated matrix CSV (default: english/english_4454.csv).",
    )
    parser.add_argument(
        "--output",
        default="english/hierarchical_clustering.pdf",
        help="Output PDF path (default: english/hierarchical_clustering.pdf).",
    )
    parser.add_argument(
        "--method",
        default="average",
        choices=["single", "complete", "average", "weighted", "centroid", "median", "ward"],
        help="Linkage method (default: average).",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="Distance metric passed to scipy.cluster.hierarchy.linkage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    similarity_matrix = []
    labels = []

    with input_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue
            labels.append(row[0])
            similarity_matrix.append([float(value) for value in row[4:]])

    linkage_matrix = hierarchy.linkage(similarity_matrix, method=args.method, metric=args.metric)

    plt.figure(figsize=(10, max(8, len(labels) / 14)))
    plt.rcParams.update({"font.size": 7})
    hierarchy.dendrogram(linkage_matrix, labels=labels, leaf_rotation=0, orientation="right")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.subplots_adjust(left=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
