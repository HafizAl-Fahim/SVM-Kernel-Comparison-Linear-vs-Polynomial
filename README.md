# SVM Kernel Comparison: Linear vs Polynomial

This project compares two Support Vector Machine (SVM) models trained on the **Aust Credit Approval Dataset**:

- **Linear Kernel:** `SVC(C=20, gamma='auto', kernel='linear')`
- **Polynomial Kernel:** `SVC(C=20, gamma='auto', kernel='poly', degree=3)`

## Project Overview

Support Vector Machines are powerful classifiers, but kernel choice greatly impacts performance.  
In this notebook, we:
1. Preprocess data using `StandardScaler` and `PCA`.
2. Train both Linear and Polynomial SVM models.
3. Evaluate accuracy, precision, recall, and confusion matrices.
4. Visualize results side by side for comparison.

## Results

- **Linear Kernel:** Higher accuracy and better generalization  
- **Polynomial Kernel:** More complex, slower, and prone to overfitting on this dataset  

**Key takeaway:** Start simple with a **linear kernel**. Use polynomial or non-linear kernels only when the data truly requires it.

## Code Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Linear SVM Pipeline
pipeline_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('svc', SVC(C=20, gamma='auto', kernel='linear'))
])

# Polynomial SVM Pipeline
pipeline_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('svc', SVC(C=20, gamma='auto', kernel='poly', degree=3))
])
