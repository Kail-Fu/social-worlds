# Social Worlds
*Visualization code and starter data for exploring connections in captioned image corpora.*

<p align="left">
  <a href="https://doi.org/10.1109/MCG.2026.3660122">
    <img src="https://img.shields.io/badge/Paper-DOI-red?style=plastic&logo=adobeacrobatreader&logoColor=red" alt="Paper DOI">
  </a>
  <a href="https://osf.io/zpr8c/overview">
    <img src="https://img.shields.io/badge/Dataset-OSF-blue?style=plastic&logo=Open%20Access&logoColor=blue" alt="Dataset OSF">
  </a>
  <a href="https://youtu.be/CKkZ1sL8y68">
    <img src="https://img.shields.io/badge/Video-YouTube-FF0000?style=plastic&logo=youtube&logoColor=white" alt="YouTube video">
  </a>
</p>

This repository open-sources tools used to generate visualizations in *Visual Exploration of a Historical Vietnamese Corpus of Captioned Drawings: A Case Study*.

<p align="center">
  <img src="./figures/teaser.jpg" width="75%" alt="Social Worlds teaser">
</p>

## Package Layout
- Source package: `src/social_worlds/`
- Build config: `pyproject.toml`
- CLI pipeline commands: `sw-similarity`, `sw-reorder`, `sw-enrich`, `sw-mst`, `sw-cluster`, `sw-radial`

## Requirements
- Python 3.9+
- Graphviz system package (required by `pygraphviz` for MST rendering)

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Data Inputs
1. Download the compiled sheet (Excel):
   [Final Compiled Captions Sheet](https://docs.google.com/spreadsheets/d/16yMkE7Mq18QrLiFVpHl8XWvVdxIKrZy8/edit?usp=sharing&ouid=107527649313550538089&rtpof=true&sd=true)
2. Download images ZIP:
   [Image Archive](https://drive.google.com/file/d/10gt402_PHWq2Q_trCCoWJQYXRksBGYBg/view?usp=sharing)
3. Unzip images into `web_low_res/` (or pass a custom path to `sw-mst --images-dir`).

## Quickstart Pipeline
Set your sheet path once:

```bash
SHEET_PATH="/absolute/path/to/Final_Compiled_Captions.xlsx"
```

1. Generate similarity matrices:
```bash
sw-similarity --sheet "$SHEET_PATH" --sheet-tab Sheet1 --output-dir .
```

2. Reorder matrices:
```bash
sw-reorder --base-dir . --method average
```

3. Attach multilingual metadata and image links:
```bash
sw-enrich --sheet "$SHEET_PATH" --sheet-tab Sheet1 --base-dir .
```

4. Render MST (English default):
```bash
sw-mst --input english/english_4454.csv --images-dir web_low_res --output english/english_mst.pdf
```

5. Generate hierarchical clustering dendrogram:
```bash
sw-cluster --input english/english_4454.csv --output english/hierarchical_clustering.pdf
```

6. Generate radial JSON:
```bash
sw-radial \
  --input english/english_4454.csv \
  --keyword "A praying monk (earthenware toy)." \
  --output radial.json
```

Upload `radial.json` to [this Observable notebook](https://observablehq.com/d/c7cbeabbeffbc1c2) to view the radial tree.

## Make Targets
After installation, the same pipeline is available via:

```bash
make similarity SHEET="$SHEET_PATH"
make reorder
make enrich SHEET="$SHEET_PATH"
make mst
make cluster
make radial
```

Or run everything:

```bash
make pipeline SHEET="$SHEET_PATH"
```

## Video
- IEEE CG&A talk: [Visual Exploration of a Historical Vietnamese Corpus of Captioned Drawings: A Case Study](https://youtu.be/CKkZ1sL8y68)
- Channel: IEEE Computer Society

## Citation
If you use this code, please cite the paper:

```bibtex
@article{fu2026visual,
  title={Visual Exploration of a Historical Vietnamese Corpus of Captioned Drawings: A Case Study},
  author={Fu, Kailiang and Gurth, Tyler and Laidlaw, David H. and Nguyen, Cindy Anh},
  journal={IEEE Computer Graphics and Applications},
  year={2026},
  doi={10.1109/MCG.2026.3660122}
}
```

## License
MIT. See [LICENSE](LICENSE).
