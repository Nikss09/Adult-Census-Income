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

<img width="1022" height="310" alt="image" src="https://github.com/user-attachments/assets/ad7666dc-d18d-4395-8f62-2501354d6547" />


Observations on Model Performance:

1. Logistic Regression: Performs well as a strong baseline model. Shows good AUC and balanced performance. Works well on scaled tabular data.
2. Decision Tree: Slightly lower performance. Likely overfitting due to single tree structure. Less stable than ensemble methods.
3. kNN: Moderate performance. Sensitive to high dimensionality (96 features after encoding). Scaling improved results.
4. Naive Bayes: Moderate performance. Sensitive to high dimensionality (96 features after encoding). Scaling improved results.
5. Random Forest: Improved stability and performance over single Decision Tree. Handles feature interactions better.

6. XGBoost: Best performing model across all metrics. Highest AUC and MCC. Boosting improves overall predictive power and generalization.
