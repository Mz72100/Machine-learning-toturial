# Regularisation Paths and Model Stability in Logistic Regression

## Overview

This tutorial explores how **regularisation hyperparameters** influence:

- The **magnitude and sparsity** of logistic regression coefficients
- The **stability** of learned models across different train–test splits
- The **generalisation performance** on unseen data

The focus is on **L1** and **L2** penalties applied to a **high-dimensional synthetic dataset** with 100 features. Regularisation paths are examined by varying the inverse-regularisation parameter `C` over several orders of magnitude.

## Key Concepts

### Regularisation as Stability Control

Regularisation acts as a **stability control knob**:

- **No regularisation**: Logistic regression can produce large and unstable coefficients, especially in high dimensions with noisy or correlated features
- **Strong regularisation**: The solution becomes more stable, but the model may be oversimplified and underfit the data

### Two Common Penalties

1. **L2 Regularisation (Ridge)**
   - Shrinks all coefficients smoothly toward zero
   - Discourages extreme values but typically keeps most coefficients non-zero

2. **L1 Regularisation (Lasso)**
   - Pushes some coefficients **exactly** to zero
   - Performs implicit feature selection and can produce very sparse models

## Dataset

The project uses a **synthetic high-dimensional dataset** with:
- **2,000 samples**
- **100 total features**
- **10 informative features** (truly predictive)
- **10 redundant features** (linear combinations of informative features)
- Remaining features as noise
- Binary classification task

This setup mimics a typical high-dimensional ML scenario where only a fraction of features are truly useful.

## Methodology

### 1. Data Preparation
- 80% training, 20% testing split
- StandardScaler for feature normalization (important for regularised models)

### 2. Regularisation Paths
- Scan `C` values from `10^-3` to `10^3` (logarithmic grid)
- Fit logistic regression models for each `C` value
- Record coefficients and test accuracy
- Visualize how coefficients evolve as regularisation strength changes

### 3. Stability Analysis
- Repeat train–test splitting 20 times with different random seeds
- Measure coefficient variability and accuracy distributions
- Compare stability between L1 and L2 regularisation

## Key Functions

### `compute_regularisation_path`
Computes regularisation paths by:
- Scanning a list of `C` values
- Fitting logistic regression models for each
- Recording coefficients and test accuracy

### `stability_experiment`
Assesses model stability by:
- Repeating train–test splits multiple times
- Fitting models with fixed penalty type and `C` value
- Measuring coefficient and accuracy variability across runs

## Results and Observations

### L2 Regularisation
- Coefficients shrink **smoothly** as `C` decreases
- Almost all features remain non-zero, but magnitudes are controlled
- Test accuracy is relatively stable for intermediate `C` values
- Coefficients tend to be less variable across random train–test splits

### L1 Regularisation
- Coefficients are often **exactly zero** for small `C`, producing sparse models
- As `C` increases, features "switch on" suddenly
- Accuracy can be comparable to L2, but may be more sensitive to split variations
- Coefficient variability is often higher for active features

## Mathematical Background

### Logistic Regression Model
For binary classification:
$$p(y = 1 \mid x) = \sigma(w^\top x + b)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

### L2 Regularisation (Ridge)
$$\mathcal{J}_{\text{L2}}(w,b) = \mathcal{L}(w,b) + \frac{\lambda}{2}\|w\|_2^2$$

### L1 Regularisation (Lasso)
$$\mathcal{J}_{\text{L1}}(w,b) = \mathcal{L}(w,b) + \lambda\|w\|_1$$

### Parameter Relationship
In scikit-learn, we use:
$$C = \frac{1}{\lambda}$$

- **Small C → Strong regularisation** (large $\lambda$)
- **Large C → Weak regularisation** (small $\lambda$)

## Visualizations

The notebook includes several visualizations:

1. **Coefficient paths** showing how individual feature coefficients change with `C`
2. **Test accuracy vs C** plots for both L1 and L2
3. **L1 vs L2 comparison** for selected features
4. **Accuracy distributions** across multiple runs
5. **Coefficient variability** distributions

## Usage

1. Install dependencies (see `requirements.txt`)
2. Open and run `notebook14.ipynb`
3. The notebook is self-contained and can be executed cell by cell

## Technical Details

- **Solver**: `saga` (used for both L1 and L2 in high dimensions)
- **Max iterations**: 5000
- **Random state**: 42 (for reproducibility)
- **Number of stability runs**: 20

## Author

Muhammad Zubair

## Course Information

**Course**: Machine Learning Neural Networks  
**Semester**: 2, Jan-2025  
**Institution**: University of [Institution Name]  
**Assignment**: Individual Assignment (40%)


