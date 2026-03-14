# Handwritten Digit Recognition (MNIST) — From Scratch

A from-scratch implementation of classical ML algorithms for MNIST digit classification, achieving ~99% accuracy with cubic polynomial kernel features and softmax regression.

## What It Does

Classifies handwritten digits (0–9) from the MNIST dataset using algorithms implemented entirely in NumPy — no high-level ML frameworks for the core math.

### Implemented Algorithms

- **Linear Regression** — closed-form solution with L2 regularization
- **SVM** — one-vs-rest and multiclass classification via `LinearSVC`
- **Softmax Regression** — manually derived gradient descent with temperature scaling
- **PCA** — dimensionality reduction with reconstruction
- **Kernel Features** — polynomial and Gaussian RBF kernels, plus cubic feature mapping

### Approach

1. Reduce dimensionality with PCA (10 components)
2. Apply cubic polynomial kernel feature mapping
3. Train softmax regression on the expanded feature space
4. Evaluate on the MNIST test set → ~99% accuracy

## Project Structure

```
├── utils.py                  # Data loading, plotting, pickle I/O
└── part1/
    ├── main.py               # Entry point — runs the full pipeline
    ├── linear_regression.py  # Closed-form linear regression
    ├── svm.py                # SVM classifiers (sklearn LinearSVC)
    ├── softmax.py            # Softmax regression + gradient descent
    ├── features.py           # PCA, cubic features, plotting
    ├── kernel.py             # Polynomial & RBF kernel functions
    ├── test.py               # Unit tests for all modules
    └── cubic_features_checker.py  # Verification for cubic feature mapping
```

## Dataset

MNIST — 60,000 training + 10,000 test images of 28×28 grayscale handwritten digits.

The dataset is **not included** in this repo. Download `mnist.pkl.gz` and place it in a `Datasets/` directory at the project root.

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Language |
| 🔢 NumPy | Linear algebra, core math |
| 📊 Matplotlib | Visualization |
| 📐 SciPy | Sparse matrix utilities |
| 🤖 scikit-learn | SVM classifier (`LinearSVC`) |

## Getting Started

```bash
pip install numpy matplotlib scipy scikit-learn

# Run the full pipeline
cd part1
python main.py

# Run unit tests
python test.py
```

## ⚠️ Known Issues

- The MNIST dataset must be downloaded separately and placed at `../Datasets/mnist.pkl.gz` relative to `utils.py`
- `main.py` calls `plt.show()` which blocks execution — close each plot window to continue
- Some functions in `main.py` are commented out (linear regression, SVM runners) as they serve as exercises
