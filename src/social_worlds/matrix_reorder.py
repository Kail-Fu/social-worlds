from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from fastcluster import linkage
from scipy.spatial.distance import squareform

sys.setrecursionlimit(30000)


LANGUAGE_CONFIG = {
    "english": {
        "tagged": "english_similarity_matrix_pm_mpnet.csv",
        "numeric": "english_similarity_matrix_pm_mpnet_notext.csv",
        "output": "english_reordered_matrix_off_caption_vectors_pm_mpnet.csv",
    },
    "french": {
        "tagged": "french_similarity_matrix_pm_mpnet.csv",
        "numeric": "french_similarity_matrix_pm_mpnet_notext.csv",
        "output": "french_reordered_matrix_off_caption_vectors_pm_mpnet.csv",
    },
    "viet": {
        "tagged": "viet_similarity_matrix_viet_sbert.csv",
        "numeric": "viet_similarity_matrix_viet_sbert_notext.csv",
        "output": "viet_reordered_matrix_off_caption_vectors_viet_sbert.csv",
    },
}


def seriation(tree, count: int, current_idx: int) -> list[int]:
    if current_idx < count:
        return [current_idx]
    left = int(tree[current_idx - count, 0])
    right = int(tree[current_idx - count, 1])
    return seriation(tree, count, left) + seriation(tree, count, right)


def compute_serial_matrix(dist_mat: np.ndarray, idx_list: list[int], method: str = "average"):
    count = len(dist_mat)
    flat_dist_mat = squareform(np.array(dist_mat))
    tree = linkage(flat_dist_mat, method=method, preserve_input=True)
    order = seriation(tree, count, count + count - 2)

    seriated_dist = np.zeros((count, count))
    row_idx, col_idx = np.triu_indices(count, k=1)
    seriated_dist[row_idx, col_idx] = dist_mat[[order[i] for i in row_idx], [order[j] for j in col_idx]]
    seriated_dist[col_idx, row_idx] = seriated_dist[row_idx, col_idx]

    reordered_indices = [idx_list[order[i]] for i in range(count)]
    return seriated_dist, reordered_indices


def load_row_labels(path: Path) -> list[int]:
    rows = []
    with path.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            rows.append(row)
    return list(range(len(rows)))


def reorder_language(base_dir: Path, language: str, method: str) -> Path:
    config = LANGUAGE_CONFIG[language]
    language_dir = base_dir / language

    tagged_path = language_dir / config["tagged"]
    numeric_path = language_dir / config["numeric"]
    output_path = language_dir / config["output"]

    idx_list = load_row_labels(tagged_path)
    matrix = np.genfromtxt(numeric_path, delimiter=",")

    size = len(matrix)
    for i in range(size):
        for j in range(size):
            matrix[i][j] = 1 - abs(round(matrix[i][j], 3))

    ordered_matrix, reordered_indices = compute_serial_matrix(matrix, idx_list, method=method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for i, row in enumerate(ordered_matrix):
            writer.writerow([reordered_indices[i], *row])

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder language similarity matrices by hierarchical clustering seriation."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing english/, french/, and viet/ folders.",
    )
    parser.add_argument(
        "--method",
        default="average",
        choices=["ward", "single", "average", "complete"],
        help="Linkage method (default: average).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["english", "french", "viet"],
        choices=["english", "french", "viet"],
        help="Subset of languages to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)

    for language in args.languages:
        output_path = reorder_language(base_dir, language, method=args.method)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
