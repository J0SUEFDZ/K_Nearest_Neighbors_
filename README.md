# K Nearest Neighbors — Image classification

## What it does

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify small images. It is trained on a labeled dataset; given a new image, it finds the K closest training examples and returns the most frequent class among them (e.g. “dog”, “ship”). Three distance metrics are implemented: Manhattan, Chebyshev, and Levenshtein.

## Dataset: CIFAR-10

The code uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:

- **60,000** 32×32 color images in **10 classes** (6,000 images per class).
- **50,000** training images (in 5 batches of 10,000) and **10,000** test images.
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The dataset is split into pickled batch files: `data_batch_1` … `data_batch_5`, `test_batch`, and `batches.meta` (label names). This layout matches the official [Python version](https://www.cs.toronto.edu/~kriz/cifar.html) of CIFAR-10.

## Context

- **Course / subject:** Artificial Intelligence
- **When:** *Feb 2018*

## How to run it

1. Create a virtual environment and install dependencies (once):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the CIFAR-10 dataset and place it in the project:

   - Go to [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html) and download the **CIFAR-10 Python version** (~163 MB).
   - Unpack the archive. Put the folder that contains `data_batch_1`, `test_batch`, and `batches.meta` in this project:
     - Either name it `cifar-10-batches-py` at the same level as `knn.py`, or
     - Keep the default structure `cifar-10-python/cifar-10-batches-py`.

3. Run the script:

   ```bash
   python3 knn.py
   ```

## Technologies used

Python, NumPy. Optional: Pillow (PIL) for the image-display function.

**Versions:** Python **3.8+**; NumPy **1.20+** (or 2.x). Recommended: Python 3.10+ and NumPy 1.24+ or 2.x. NumPy 2.4+ may show a deprecation when loading CIFAR-10 pickles; the script suppresses it.

## Report: `main.tex`

`main.tex` is the LaTeX source for the **project report** (in Spanish). It describes the KNN methodology, the three distance functions (Manhattan, Chebyshev, Levenshtein) with equations, the experiments run on CIFAR-10, and the conclusions. The PDF is built from this file together with `biblio.bib`.

**View the report online (read-only, Spanish):**  
[Reporte en Overleaf — K Nearest Neighbors Analysis](https://j0suefdz.github.io/LaTex-Templates/k_nearest_neighbours.pdf)

## Source and reference

- **Dataset and description:** [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html) (Alex Krizhevsky, Vinod Nair, Geoffrey Hinton).
- **Citation:** *Learning Multiple Layers of Features from Tiny Images*, Alex Krizhevsky, 2009. If you use CIFAR-10, please cite the [tech report](https://www.cs.toronto.edu/~kriz/cifar.html) as indicated on the dataset page.

---

*Archived coursework; code has been lightly adapted for cross-platform paths and data lookup.*
