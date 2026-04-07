# Titanic Survival Prediction

Predicting passenger survival on the Titanic using multiple classification models.

## Dataset

The [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data) contains 891 passenger records with 12 features including class, sex, age, fare, and embarkation port.

**To run the notebook**, download `train.csv` from Kaggle, rename it to `titanic-training-data.csv`, and place it in this directory.

## Models Compared

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline with feature scaling |
| Decision Tree | Single tree with depth limit |
| Random Forest | Ensemble of 100 decision trees |
| Bagging Classifier | Bootstrap aggregating with 20 estimators |
| AdaBoost | Adaptive boosting with 100 estimators |
| Gradient Boosting | Sequential boosting with 100 estimators |
| K-Nearest Neighbors | Distance-based classification (k=7) |

## Preprocessing

- **Missing values:** Age filled by class-median, Embarked filled with mode, Cabin deck letter extracted
- **Feature engineering:** FamilySize, IsAlone derived features
- **Encoding:** One-hot encoding for Sex, Embarked, Deck
- **Scaling:** StandardScaler applied for Logistic Regression and KNN
- **Split:** 70/30 stratified train/test split with fixed random seed

## Key Results

- Ensemble methods (Gradient Boosting, Random Forest) achieve the best test accuracy (~82-84%)
- Top predictive features: Sex, Fare, Age, Passenger Class
- All results validated with 5-fold cross-validation and ROC-AUC scoring

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Titanic_Survival_Prediction.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib
