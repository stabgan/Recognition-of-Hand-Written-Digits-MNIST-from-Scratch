# ✍️ Handwritten Digit Recognition (MNIST) — From Scratch

A from-scratch implementation of classical machine learning algorithms for recognizing handwritten digits on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). No high-level ML frameworks are used for the core algorithms — SVM, Softmax Regression, PCA, and kernel methods are all implemented manually using NumPy.

## 📖 Description

This project tackles the MNIST digit classification problem (0–9) by building fundamental ML components from the ground up:

- **Closed-form Linear Regression** with L2 regularization
- **Support Vector Machine (SVM)** — one-vs-rest and multiclass via `sklearn.svm.LinearSVC`
- **Softmax (Multinomial Logistic) Regression** with hand-derived gradient descent
- **Principal Component Analysis (PCA)** for dimensionality reduction
- **Kernel methods** — Polynomial and Gaussian RBF kernels implemented from scratch
- **Cubic feature mapping** to project data into higher-dimensional space

Achieves ~99% accuracy using cubic polynomial kernel features on 10-dimensional PCA representations with Softmax regression.

## 🔬 Methodology

1. **Data loading** — MNIST is loaded via pickle and split into train/validation/test sets.
2. **Linear Regression** — Closed-form solution `θ = (XᵀX + λI)⁻¹XᵀY` as a baseline.
3. **SVM** — Binary (one-vs-rest) and multiclass classification using a linear kernel.
4. **Softmax Regression** — Full forward pass (probability computation via temperature-scaled softmax), cost function with regularization, and batch gradient descent — all implemented from scratch.
5. **PCA** — Eigenvalue decomposition of the covariance matrix to extract principal components; data is projected onto the top-k eigenvectors.
6. **Kernel Features** — Polynomial and RBF kernels computed element-wise; cubic feature mapping expands 10-dim PCA vectors into a rich feature space for improved classification.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🐍 Language | Python 3 |
| 🔢 Numerics | NumPy |
| 📊 Plotting | Matplotlib |
| 🤖 SVM | scikit-learn (`LinearSVC`) |
| 📐 Linear Algebra | NumPy (`linalg`) |

## 📦 Dependencies

```
numpy
matplotlib
scipy
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib scipy scikit-learn
```

## 🚀 How to Run

1. **Get the MNIST dataset** — download `mnist.pkl.gz` and place it in a `Datasets/` folder one level above this repo (i.e. `../Datasets/mnist.pkl.gz`).

2. **Run the unit tests:**
   ```bash
   cd part1
   python test.py
   ```

3. **Run the main pipeline** (trains models, plots PCA, evaluates accuracy):
   ```bash
   cd part1
   python main.py
   ```

4. **Verify cubic feature mapping:**
   ```bash
   cd part1
   python cubic_features_checker.py
   ```

## 📁 Project Structure

```
.
├── utils.py                        # Data loading, plotting, pickle I/O
└── part1/
    ├── main.py                     # Main training & evaluation pipeline
    ├── linear_regression.py        # Closed-form linear regression
    ├── svm.py                      # SVM (one-vs-rest & multiclass)
    ├── softmax.py                  # Softmax regression + gradient descent
    ├── features.py                 # PCA, cubic features, centering
    ├── kernel.py                   # Polynomial & RBF kernels
    ├── cubic_features_checker.py   # Verification for cubic feature mapping
    ├── test.py                     # Unit tests for all modules
    └── theta.pkl.gz                # Pre-trained softmax parameters
```

## ⚠️ Known Issues

- The MNIST dataset (`mnist.pkl.gz`) is not included in the repo — you must download it separately and place it at `../Datasets/mnist.pkl.gz`.
- `main.py` calls `plt.show()` which blocks execution in non-interactive environments. Use a matplotlib backend like `Agg` if running headless.
- Kernel functions in `kernel.py` use explicit Python loops (O(n·m)) rather than vectorized operations, so they can be slow on large inputs.
- Part 2 (Neural Network approach) is not yet implemented.
