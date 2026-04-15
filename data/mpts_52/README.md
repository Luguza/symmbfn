# MPTS-52

MPTS-52 (Materials Project Time Split, max 52 atoms per unit cell) contains inorganic crystal structures sourced from the Materials Project and split by discovery/publication date. The materials have up to 52 atoms in the unit cell and span a wide range of compositions and structure types, making it the most comparable benchmark to MP-20 among the time-split datasets.

## What is in the dataset?

MPTS-52 is derived from the Materials Project database. Unlike random train/test splits, the dataset is partitioned **temporally**: structures are ordered by their earliest publication year, so the test set contains materials discovered after those in the training set. This time-based split is intended to simulate realistic generative model evaluation, where a model trained on known structures is assessed on its ability to generate structures resembling future discoveries.

The dataset provides standard `train.csv`, `val.csv`, and `test.csv` splits.

## Stability of curated materials

All structures are taken from the Materials Project and have been relaxed via DFT.

## Citation

Please consider citing the following paper:

```
@article{Baird2024,
  doi     = {10.21105/joss.05618},
  year    = {2024},
  volume  = {9},
  number  = {97},
  pages   = {5618},
  author  = {Sterling G. Baird and Hasan M. Sayeed and Joseph Montoya and Taylor D. Sparks},
  title   = {matbench-genmetrics: A Python library for benchmarking crystal structure generative models using time-based splits of Materials Project structures},
  journal = {Journal of Open Source Software}
}
```
