# Social Worlds
*Visualization code and starter data for exploring connections and rebuilding "social worlds" in captioned image corpora.*

<p align="left">
  <a href="https://doi.org/10.1109/MCG.2026.3660122">
    <img src="https://img.shields.io/badge/Paper-DOI-red?style=plastic&logo=adobeacrobatreader&logoColor=red" alt="Paper DOI">
  </a>
  <a href="https://osf.io/zpr8c/overview">
    <img src="https://img.shields.io/badge/Dataset-OSF-blue?style=plastic&logo=Open%20Access&logoColor=blue" alt="Dataset OSF">
  </a>
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-lightgrey?style=plastic" alt="Code coming soon">
</p>

This repository open-sources the code used to generate the visualizations in *Visual Exploration of a Historical Vietnamese Corpus of Captioned Drawings: A Case Study*.

It is designed to (1) reproduce the paper figures and (2) serve as a template for building similar “social worlds” visualizations on your own captioned image datasets (multilingual text + images + metadata).

<p align="center">
  <img src="./figures/teaser.jpg" width="75%" alt="Social Worlds teaser">
</p>

---

## What’s included (and what’s coming)
**Included (or planned):**
- Embedding pipelines for captions and images (text and vision feature vectors)
- Similarity structures and visualizations:
  - Distance matrix + hierarchical reordering
  - Minimum spanning tree
  - 2D projections (t-SNE / UMAP / PixPlot-style layout)

**Coming soon:**
- One-command setup and environment files
- Scripts to reproduce each figure from the paper
- A small starter dataset download and loaders
- Documentation for uploading and visualizing your own data

---

## Quickstart
Coming soon.

---

## Data
A detailed schema and examples are coming soon.

---

## Reproducing the paper figures
Coming soon. Target outputs will mirror the paper’s visualization set (distance matrix, MST, t-SNE/UMAP, PixPlot projection).

---

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
````

---

## License

MIT. See [LICENSE](LICENSE).
