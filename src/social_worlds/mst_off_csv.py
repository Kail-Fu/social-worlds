from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pygraphviz as pg
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and render a minimum spanning tree from an annotated similarity matrix."
    )
    parser.add_argument(
        "--input",
        default="english/english_4454.csv",
        help="Input annotated matrix CSV (default: english/english_4454.csv).",
    )
    parser.add_argument(
        "--images-dir",
        default="web_low_res",
        help="Directory containing image files referenced by 'Link to Image'.",
    )
    parser.add_argument(
        "--output",
        default="english/english_mst.pdf",
        help="Output PDF path (default: english/english_mst.pdf).",
    )
    parser.add_argument(
        "--graphviz-prog",
        default="neato",
        help="Graphviz program used by pygraphviz draw() (default: neato).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    cos_all: list[list[float]] = []
    unique_text: list[tuple[str, str]] = []
    links: list[str] = []
    empty_list: list[int] = []

    with input_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row_idx, row in enumerate(reader):
            if row_idx == 0:
                continue
            if row[0]:
                cos_all.append([float(value) for value in row[4:]])
                links.append(row[3])
                label = f"{row_idx}_{row[0]}\\n{row[1]}\\n{row[2]}"
                unique_text.append((label, "black"))
            else:
                empty_list.append(row_idx - 1)

    empty_list.reverse()
    for idx in empty_list:
        for matrix_row in cos_all:
            matrix_row.pop(idx)

    dif_all = np.zeros((len(cos_all), len(cos_all[0])))
    for i in range(len(cos_all)):
        for j in range(len(cos_all[0])):
            dif_all[i][j] = 1 if i == j else cos_all[i][j]

    graph_matrix = csr_matrix(dif_all)
    mst_sparse = minimum_spanning_tree(graph_matrix)
    edge_row_indices, edge_col_indices = mst_sparse.nonzero()
    nonzero_value_list = mst_sparse[edge_row_indices, edge_col_indices]

    graph = pg.AGraph()

    for i in range(len(cos_all)):
        image_name = links[i].split("/")[-1]
        graph_loc = images_dir / image_name

        if not graph_loc.exists():
            raise FileNotFoundError(f"Image file not found: {graph_loc}")

        with Image.open(graph_loc) as image:
            ideal_height = image.size[1] / 67.2 + 2

        graph.add_node(
            unique_text[i][0],
            image=str(graph_loc),
            label=unique_text[i][0],
            labelloc="b",
            imagepos="tc",
            fontsize="15",
            height=ideal_height,
            fontcolor=unique_text[i][1],
        )

    for i in range(len(edge_row_indices)):
        start = unique_text[edge_row_indices[i]][0]
        end = unique_text[edge_col_indices[i]][0]
        distance = nonzero_value_list[0, i] + 1.3
        distance = (distance**2) * 7.5
        graph.add_edge(start, end, len=abs(distance))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.draw(str(output_path), format="pdf", prog=args.graphviz_prog)


if __name__ == "__main__":
    main()
