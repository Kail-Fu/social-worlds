from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate radial tree JSON from an annotated matrix CSV."
    )
    parser.add_argument(
        "--input",
        default="english/english_4454.csv",
        help="Input annotated matrix CSV (default: english/english_4454.csv).",
    )
    parser.add_argument(
        "--keyword",
        required=True,
        help="Root caption to place at the center of the radial tree.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Maximum similarity value threshold for child candidate selection (default: 0.5).",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=4,
        help="Maximum children per node (default: 4).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth (default: 4).",
    )
    parser.add_argument(
        "--output",
        default="radial.json",
        help="Output JSON path (default: radial.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    captions = []
    adjacency = {}

    with input_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue

            captions.append(row[0])
            similarities = [float(value) for value in row[4:]]
            ordered_indices = sorted(range(len(similarities)), key=lambda idx: similarities[idx])

            threshold_cutoff = 0
            while threshold_cutoff < len(ordered_indices) and similarities[ordered_indices[threshold_cutoff]] < args.threshold:
                threshold_cutoff += 1

            adjacency[row_index - 1] = ordered_indices[:threshold_cutoff]

    if args.keyword not in captions:
        raise ValueError(f"Keyword not found in captions: {args.keyword}")

    start = captions.index(args.keyword)
    seen = {args.keyword}

    def to_tree(node_idx: int, level: int):
        if level == args.max_depth:
            return {"name": captions[node_idx], "value": 0}

        children = []
        for neighbor_idx in adjacency[node_idx]:
            caption = captions[neighbor_idx]
            if caption in seen:
                continue
            seen.add(caption)
            children.append(to_tree(neighbor_idx, level + 1))
            if len(children) == args.max_children:
                break

        return {"name": captions[node_idx], "children": children}

    tree = to_tree(start, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(tree, outfile)


if __name__ == "__main__":
    main()
