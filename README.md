# Heart Disease Prediction: Decision Tree vs MLP

This project compares two machine learning models — a pruned Decision Tree and a Multi-Layer Perceptron (MLP) — to predict heart disease based on patient features.

## Dataset

We use the `heart.csv` dataset, which contains medical information about patients such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Maximum heart rate achieved
- ST depression, and more.

**Target:**  
- `0` → No heart disease  
- `1` → Heart disease present

## Models Compared

### 1. Decision Tree Classifier
- Pruned using `ccp_alpha=0.01203502` (cost-complexity pruning).
- Tuned to avoid overfitting and increase generalization.

### 2. MLPClassifier (Simple Perceptron)
- Fully-connected neural network with default parameters.
- Trained on the same data for comparison.

## Results

| Model           | Accuracy (Test) |
|----------------|------------------|
| Decision Tree  | 0.84             |
| MLPClassifier  | 0.87             |

### Conclusions

- The **MLPClassifier** performed slightly better on the test set.
- However, the Decision Tree provides more interpretability and can be pruned for simplicity.
- Depending on the use case (e.g. accuracy vs explainability), both models can be valid choices.
