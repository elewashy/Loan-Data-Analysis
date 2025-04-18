# Logistic Regression Project

This project implements and compares different variants of Logistic Regression algorithms for binary classification on a loan default prediction dataset.

## Project Overview

The project includes:
- Custom implementation of Logistic Regression from scratch
- Three different optimization methods:
  - Basic Logistic Regression (Loss only)
  - L2 Regularization (Ridge)
  - Stochastic Gradient Descent (SGD)
- Comparison with scikit-learn implementations
- Comprehensive data preprocessing and visualization
- Model evaluation using various metrics

## Dataset

The project uses the `loan_data.csv` dataset which contains features like:
- Person demographics (age, gender, education)
- Financial information (income, employment experience)
- Loan details (amount, interest rate, intent)
- Credit history (credit score, defaults history)

## Data Preprocessing Steps

1. Outlier removal (age > 100)
2. Missing value imputation
   - Numeric columns: Mean imputation
   - Categorical columns: Mode imputation
3. Categorical encoding using factorization
4. Feature normalization (standardization)

## Model Implementations

### Custom Implementation
- Implemented sigmoid activation function
- Binary cross-entropy loss function
- Optional L2 regularization
- Support for both batch and stochastic gradient descent
- Custom evaluation metrics calculation

### Scikit-learn Implementation
- LogisticRegression with no regularization
- Ridge Regression (L2 regularization)
- SGDClassifier

## Model Evaluation

Models are evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Loss curves

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook `Logistic_Regression_Project.ipynb` to:
1. Load and preprocess the data
2. Train different models
3. Compare model performances
4. Visualize results

## Results Visualization

- Distribution plots of numeric features
- Box plots for outlier detection
- Correlation heatmap
- Loss comparison curves
- Confusion matrices
- Accuracy comparison bar charts

## Key Findings

- All models achieve comparable performance
- L2 regularization helps prevent overfitting
- SGD provides faster training with similar accuracy
- Custom implementation matches scikit-learn performance
