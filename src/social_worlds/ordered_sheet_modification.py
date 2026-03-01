from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LANGUAGE_CONFIG = {
    "english": {
        "reordered": "english_reordered_matrix_off_caption_vectors_pm_mpnet.csv",
        "output": "english_4454.csv",
        "primary_col": "English Translation",
        "secondary_cols": ["French Text", "Char Text", "Link to Image"],
    },
    "french": {
        "reordered": "french_reordered_matrix_off_caption_vectors_pm_mpnet.csv",
        "output": "french_4454.csv",
        "primary_col": "French Text",
        "secondary_cols": ["English Translation", "Char Text", "Link to Image"],
    },
    "viet": {
        "reordered": "viet_reordered_matrix_off_caption_vectors_viet_sbert.csv",
        "output": "viet_4454.csv",
        "primary_col": "Char Text",
        "secondary_cols": ["English Translation", "French Text", "Link to Image"],
    },
}


def to_clean_list(series: pd.Series) -> list[str]:
    return [value if isinstance(value, str) else "" for value in series.tolist()]


def write_language_sheet(base_dir: Path, language: str, reference_df: pd.DataFrame) -> Path:
    config = LANGUAGE_CONFIG[language]
    language_dir = base_dir / language

    reordered_path = language_dir / config["reordered"]
    output_path = language_dir / config["output"]

    matrix_df = pd.read_csv(reordered_path, header=None)
    idx_list = matrix_df.iloc[:, 0].astype(int).tolist()

    primary_values = [config["primary_col"]]
    secondary_values = [[column] for column in config["secondary_cols"]]

    for idx in idx_list:
        primary_values.append(reference_df.iloc[idx][config["primary_col"]])
        for sec_idx, column in enumerate(config["secondary_cols"]):
            secondary_values[sec_idx].append(reference_df.iloc[idx][column])

    primary_values = to_clean_list(pd.Series(primary_values))
    secondary_values = [to_clean_list(pd.Series(values)) for values in secondary_values]

    matrix_df.loc[-1] = primary_values
    matrix_df.index = matrix_df.index + 1
    matrix_df.sort_index(inplace=True)
    matrix_df = matrix_df.drop(matrix_df.columns[[0]], axis=1)
    matrix_df.insert(loc=0, column=0, value=primary_values)
    for insert_pos, values in enumerate(secondary_values, start=1):
        matrix_df.insert(loc=insert_pos, column=-insert_pos, value=values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(output_path, encoding="utf-8-sig", index=False, header=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach multilingual captions and image links to reordered similarity matrices."
    )
    parser.add_argument("--sheet", required=True, help="Path to compiled Excel sheet.")
    parser.add_argument("--sheet-tab", default="Sheet1", help="Worksheet name (default: Sheet1).")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing english/, french/, and viet/ folders.",
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

    reference_df = pd.ExcelFile(Path(args.sheet)).parse(args.sheet_tab)
    base_dir = Path(args.base_dir)

    for language in args.languages:
        output_path = write_language_sheet(base_dir, language, reference_df)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
