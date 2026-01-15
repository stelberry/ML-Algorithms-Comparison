## Overview

This project implements and compares classic supervised learning algorithms — 1-Nearest Neighbour (1-NN), k-Nearest Neighbour (k-NN), and Decision Trees (CART) — from scratch using Python and NumPy.

Rather than relying on high-level machine learning libraries, the focus of this project is to develop a deep understanding of:

- algorithmic logic,

- computational trade-offs,

- data preprocessing decisions,

- and practical limitations on real-world datasets.

The algorithms are evaluated on benchmark datasets, including an imbalanced real-world dataset from the UCI Machine Learning Repository.

## Algorithms Implemented
**1-Nearest Neighbour (1-NN)**

- Computes Euclidean distance between a test sample and all training samples

- Uses NumPy vectorisation (np.linalg.norm) to avoid Python loops

- Predicts the label of the single closest training instance

**k-Nearest Neighbour (k-NN)**

- Extends 1-NN to consider the k closest neighbours

- Majority voting for classification

- Distance-based tie-breaking to ensure deterministic predictions

- Manual grid search used to select optimal k

- Highlights the computational cost of lazy learning at inference time

**Decision Tree (CART)**

- Binary decision tree using impurity-based splits

- Measures:

  - Gini impurity

  - Shannon entropy

- Recursive tree construction with configurable stopping criteria:

  - maximum depth

  - minimum samples per node

- Efficient prediction via tree traversal

- Designed to balance interpretability and generalisation


## Dataset

The primary real-world dataset used is:

**Default of Credit Card Clients Dataset**
Source: UCI Machine Learning Repository

- 30,000 samples

- Highly imbalanced classes (≈22% default, ≈78% non-default)

- Mixed feature scales and categorical values

- This dataset highlights practical challenges such as:

- feature scaling for distance-based methods,

- overfitting in decision trees.

## Engineering Decisions

- Vectorisation over loops for performance-critical code

Distance-based tie-breaking instead of random or label-priority rules

Depth-limited trees to reduce overfitting

Stratified sampling to preserve class distribution

Manual implementation of tuning and evaluation logic for transparency

## Tools & Libraries

Python

NumPy — numerical computation and vectorisation

scikit-learn - data splitting, scaling, resampling

Matplotlib — result visualisation

Seaborn — statistical plotting

## Results

The experiments demonstrate clear trade-offs:

- k-NN achieves strong accuracy but suffers from high inference cost

- Decision Trees train more slowly but provide fast predictions and interpretability

- Preprocessing decisions (scaling, stratification) significantly impact performance

## Future Work

**Extended Evaluation Metrics**
Extend evaluation beyond accuracy by incorporating precision, recall, confusion matrices, and ROC analysis. This is essential for the imbalanced credit card default dataset, where accuracy alone can be misleading.

**Cross-Validation Framework**
Implement a full k-fold cross-validation pipeline to ensure fair and reproducible performance estimates, with preprocessing performed independently within each fold to prevent data leakage.

**Advanced Classification Algorithms**
Implement more advanced algorithms such as kernel-based k-NN and Support Vector Machines (SVMs) to enable fair comparison with the baseline models developed in this project.

**Graphical User Interface (GUI)**
Develop a simple graphical user interface to allow users to load datasets, select algorithms and hyperparameters, and visualise predictions and performance metrics without modifying code.

**Systematic Multi-Dataset Evaluation**
Evaluate all models on additional benchmark datasets to assess robustness and generalisation across different data distributions.

