import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Adult Income Classification")

# Load shared objects
scaler = joblib.load("model/scaler.pkl")
training_columns = joblib.load("model/columns.pkl")

model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

model_name = st.selectbox("Select Model", list(model_map.keys()))
model = joblib.load(f"model/{model_map[model_name]}")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    df["income"] = df["income"].str.replace(".", "", regex=False)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    df.drop(columns=["fnlwgt", "education"], inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"]

    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=training_columns, fill_value=0)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("AUC:", roc_auc_score(y, y_prob))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
