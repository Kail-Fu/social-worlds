from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sw-dr coordinate output to PixPlot-friendly metadata, manifest, and layout files."
    )
    parser.add_argument("--dr-input", required=True, help="Input CSV from sw-dr (must include x and y columns).")
    parser.add_argument("--metadata-output", required=True, help="Output metadata CSV path.")
    parser.add_argument(
        "--layout-output",
        help="Optional output layout JSON path (array of [x, y] pairs in row order).",
    )
    parser.add_argument(
        "--manifest-output",
        help="Optional output image manifest path (newline-delimited image paths).",
    )
    parser.add_argument(
        "--metadata-input",
        help="Optional extra metadata CSV to join before export.",
    )
    parser.add_argument("--dr-key", default="id", help="Join key column in dr-input (default: id).")
    parser.add_argument(
        "--metadata-key",
        default="id",
        help="Join key column in metadata-input (default: id).",
    )
    parser.add_argument(
        "--filename-col",
        default="filename",
        help="Column containing image filename (default: filename).",
    )
    parser.add_argument(
        "--link-col",
        default="link_to_image",
        help="Column containing image URL/path used to derive filename if filename-col is missing.",
    )
    parser.add_argument(
        "--image-dir",
        help="Optional directory to prepend when writing manifest lines.",
    )
    parser.add_argument(
        "--category-col",
        default="color",
        help="Column exported as PixPlot category (default: color).",
    )
    parser.add_argument(
        "--description-col",
        default="label",
        help="Column exported as PixPlot description (default: label).",
    )
    return parser.parse_args()


def _basename_from_link(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text.split("?")[0].rstrip("/").split("/")[-1]


def _load_and_merge(args: argparse.Namespace) -> pd.DataFrame:
    dr_df = pd.read_csv(args.dr_input)
    if args.metadata_input:
        meta_df = pd.read_csv(args.metadata_input)
        if args.dr_key not in dr_df.columns:
            raise ValueError(f"dr key column not found: {args.dr_key}")
        if args.metadata_key not in meta_df.columns:
            raise ValueError(f"metadata key column not found: {args.metadata_key}")

        merged = dr_df.merge(
            meta_df,
            left_on=args.dr_key,
            right_on=args.metadata_key,
            how="left",
            suffixes=("", "_meta"),
        )
        return merged
    return dr_df


def _resolve_filename_series(df: pd.DataFrame, args: argparse.Namespace) -> Optional[pd.Series]:
    if args.filename_col in df.columns:
        series = df[args.filename_col].fillna("").astype(str)
        if (series != "").any():
            return series

    if args.link_col in df.columns:
        return df[args.link_col].fillna("").astype(str).map(_basename_from_link)

    if args.image_dir and args.dr_key in df.columns:
        image_dir = Path(args.image_dir)
        if image_dir.exists():
            resolved = []
            for raw_id in df[args.dr_key].tolist():
                id_text = str(raw_id)
                candidates = sorted(image_dir.glob(f"{id_text}_*"))
                if not candidates:
                    candidates = sorted(image_dir.glob(f"{id_text}.*"))
                if candidates:
                    resolved.append(candidates[0].name)
                else:
                    resolved.append("")
            series = pd.Series(resolved, index=df.index).astype(str)
            if (series != "").any():
                return series

    return None


def _write_manifest(filenames: pd.Series, args: argparse.Namespace) -> None:
    if not args.manifest_output:
        return

    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fp:
        for name in filenames.tolist():
            if args.image_dir:
                fp.write(str(Path(args.image_dir) / name) + "\n")
            else:
                fp.write(str(name) + "\n")


def _write_layout(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if not args.layout_output:
        return

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Input must include x and y columns for layout export")

    layout_path = Path(args.layout_output)
    layout_path.parent.mkdir(parents=True, exist_ok=True)

    coords = [[float(x), float(y)] for x, y in zip(df["x"], df["y"])]
    with layout_path.open("w", encoding="utf-8") as fp:
        json.dump(coords, fp)


def main() -> None:
    args = parse_args()
    merged = _load_and_merge(args)

    if "label" not in merged.columns:
        raise ValueError("Input must include a label column")
    if "x" not in merged.columns or "y" not in merged.columns:
        raise ValueError("Input must include x and y columns")

    filename_series = _resolve_filename_series(merged, args)
    if filename_series is None:
        raise ValueError(
            "Could not resolve filenames. Provide --filename-col or --link-col via --metadata-input."
        )

    category_series = (
        merged[args.category_col].astype(str)
        if args.category_col in merged.columns
        else pd.Series(["unknown"] * len(merged))
    )
    description_series = (
        merged[args.description_col].astype(str)
        if args.description_col in merged.columns
        else merged["label"].astype(str)
    )

    out_df = pd.DataFrame(
        {
            "filename": filename_series.astype(str),
            "label": merged["label"].astype(str),
            "category": category_series,
            "description": description_series,
            "x": merged["x"],
            "y": merged["y"],
        }
    )

    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(metadata_path, index=False)

    _write_manifest(out_df["filename"], args)
    _write_layout(out_df, args)


if __name__ == "__main__":
    main()
