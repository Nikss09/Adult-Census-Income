Machine Learning Assignment 2

Adult Income Classification

Problem Statement: The objective of this project is to build and compare multiple classification models to predict whether a person's annual income is above or below $50k based on demographic and employement realted features.

Dataset Description:

The dataset used is the Adult Income Dataset(https://www.kaggle.com/datasets/uciml/adult-census-income/data).

Source: UCI Machine Learning Repository
Number of instances after cleaning: ~30,000+
Number of features after encoding: 96

Target variable: income
0 → <=50K
1 → >50K

Features include age, education level, occupation, workclass, etc

Models used
        Model           Accuracy  AUC        Precision Recall    F1        MCC
0  Logistic Regression  0.842035  0.896977   0.718975  0.598667  0.653328  0.556026
1        Decision Tree  0.805735  0.744672   0.606355  0.623333  0.614727  0.484967
2                  KNN  0.814023  0.830714   0.644275  0.562667  0.600712  0.482086
3          Naive Bayes  0.402122  0.648931   0.288411  0.957333  0.443278  0.200031
4        Random Forest  0.843030  0.897054   0.715175  0.612667  0.659964  0.561542
5              XGBoost  0.864081  0.923257   0.770701  0.645333  0.702467  0.619354

Observations on Model Performance:

1. Logistic Regression: Performs well as a strong baseline model. Shows good AUC and balanced performance. Works well on scaled tabular data.
2. Decision Tree: Slightly lower performance. Likely overfitting due to single tree structure. Less stable than ensemble methods.
3. kNN: Moderate performance. Sensitive to high dimensionality (96 features after encoding). Scaling improved results.
4. Naive Bayes: Moderate performance. Sensitive to high dimensionality (96 features after encoding). Scaling improved results.
5. Random Forest: Improved stability and performance over single Decision Tree. Handles feature interactions better.
6. XGBoost: Best performing model across all metrics. Highest AUC and MCC. Boosting improves overall predictive power and generalization.