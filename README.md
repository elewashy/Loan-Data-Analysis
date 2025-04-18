# Logistic Regression Project

This project implements and compares different variants of Logistic Regression algorithms for binary classification on a loan default prediction dataset.

## Project Steps

### 1. Data Visualization
- First, we analyze the data through visualization
- Check all columns and identify the target variable
- Use barplots to understand categorical variables
- Create correlation matrix and heatmap
  - This helps identify which features affect our target
  - Shows relationships between different columns
- After visualization, we know which columns are important for prediction

### 2. Data Preprocessing
- Check for data quality issues:
  - Handle null values using median or mean
  - Deal with outliers if found
  - Convert categorical columns using encoding (one-hot or label)
- Data cleaning steps:
  1. Fill missing values
  2. Remove outliers
  3. Convert categorical to numerical values
  4. Normalize/scale numerical features

### 3. Model Implementation (From Scratch)
- Implement logistic regression algorithm
- Create simple loss function
- Build the model step by step:
  1. Basic implementation with loss function
  2. Add gradient descent
  3. Implement prediction functionality

## Tools Used
- Python
- Pandas for data handling
- Matplotlib/Seaborn for visualization
- NumPy for numerical operations

## Key Features
- Simple data exploration
- Basic data preprocessing
- Basic model implementation
- Easy to understand visualization

## How to Run
1. Load the data
2. Run visualization cells
3. Run preprocessing steps
4. Train the model
5. Check results

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
