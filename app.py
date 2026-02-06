import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

st.title("🍷 Red Wine Quality Prediction")

# ================= CSV Upload =================
uploaded_file = st.file_uploader("Upload CSV file (Test Data)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(df.head())

    # Binary target
    df["quality"] = (df["quality"] >= 7).astype(int)

    X = df.drop("quality", axis=1)
    y = df["quality"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ================= Model Selection =================
    model_name = st.selectbox(
        "Select ML Model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    model = models[model_name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # ================= Metrics =================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", round(acc,3))
    st.write("AUC:", round(auc,3))
    st.write("Precision:", round(prec,3))
    st.write("Recall:", round(rec,3))
    st.write("F1 Score:", round(f1,3))
    st.write("MCC:", round(mcc,3))

    # ================= Confusion Matrix =================
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # ================= Classification Report =================
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload CSV file to continue.")

