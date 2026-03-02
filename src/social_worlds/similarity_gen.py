from __future__ import annotations

import argparse
import csv
import difflib
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


LANGUAGE_CONFIG = {
    "english": {
        "text_column": "English Translation",
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "prefix": "english_similarity_matrix_pm_mpnet",
        "embedding_matrix": "english_mpnet_embedding_matrix.csv",
    },
    "french": {
        "text_column": "French Text",
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "prefix": "french_similarity_matrix_pm_mpnet",
        "embedding_matrix": "french_mpnet_embedding_matrix.csv",
    },
    "viet": {
        "text_column": "Char Text",
        "model": "keepitreal/vietnamese-sbert",
        "prefix": "viet_similarity_matrix_viet_sbert",
        "embedding_matrix": "viet_sbert_embedding_matrix.csv",
    },
}


def clean_text(series: pd.Series) -> list[str]:
    return [value if isinstance(value, str) else "" for value in series.tolist()]


def write_matrix(path: Path, matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)


def write_matrix_with_labels(path: Path, row_labels: list[str], matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Tags", *row_labels])
        for idx, row in enumerate(matrix):
            writer.writerow([row_labels[idx], *row])


def write_embedding_matrix(path: Path, labels: list[str], embeddings, metadata_columns: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = len(embeddings[0]) if len(embeddings) else 0
    feature_headers = [f"feat_{i:04d}" for i in range(dims)]
    metadata_headers = list(metadata_columns.keys())

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label", *metadata_headers, *feature_headers])
        for idx, embedding in enumerate(embeddings):
            metadata_values = [metadata_columns[col][idx] for col in metadata_headers]
            writer.writerow([idx, labels[idx], *metadata_values, *embedding])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate language-specific caption similarity matrices from an Excel sheet."
    )
    parser.add_argument(
        "--sheet",
        required=True,
        help="Path to the compiled Excel sheet.",
    )
    parser.add_argument(
        "--sheet-tab",
        default="Sheet1",
        help="Worksheet name containing caption columns (default: Sheet1).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Base output directory where english/french/viet folders are written.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip writing per-language embedding matrix CSV files for downstream DR workflows.",
    )
    parser.add_argument(
        "--embedding-metadata-cols",
        nargs="*",
        default=[],
        help="Optional sheet columns to include in embedding CSVs (for DR coloring/grouping).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workbook_path = Path(args.sheet)
    output_dir = Path(args.output_dir)

    dataframe = pd.ExcelFile(workbook_path).parse(args.sheet_tab)
    shared_labels = clean_text(dataframe["Char Text"])
    metadata_columns = {}
    normalized_to_actual = {str(col).strip().lower(): col for col in dataframe.columns}
    for column in args.embedding_metadata_cols:
        normalized = column.strip().lower()
        actual_col = normalized_to_actual.get(normalized)
        if actual_col is None:
            close = difflib.get_close_matches(column, list(dataframe.columns), n=5, cutoff=0.5)
            raise ValueError(
                f"Metadata column not found in sheet: {column}. "
                f"Closest matches: {close if close else 'none'}. "
                f"Available columns: {list(dataframe.columns)}"
            )
        metadata_columns[str(actual_col)] = dataframe[actual_col].fillna("").tolist()

    model_cache: dict[str, SentenceTransformer] = {}

    for language, config in LANGUAGE_CONFIG.items():
        text_values = clean_text(dataframe[config["text_column"]])
        model_name = config["model"]

        if model_name not in model_cache:
            model_cache[model_name] = SentenceTransformer(model_name)

        embeddings = model_cache[model_name].encode(text_values)
        similarity = cosine_similarity(embeddings)

        folder = output_dir / language
        prefix = config["prefix"]
        write_matrix(folder / f"{prefix}_notext.csv", similarity)
        write_matrix_with_labels(folder / f"{prefix}.csv", shared_labels, similarity)
        if not args.no_embeddings:
            write_embedding_matrix(
                folder / config["embedding_matrix"],
                text_values,
                embeddings,
                metadata_columns,
            )


if __name__ == "__main__":
    main()
