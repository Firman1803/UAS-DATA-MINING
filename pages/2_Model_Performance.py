import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("📈 Model Performance")

# Load data dan model
df = pd.read_csv("data/dataset.csv", header=None)
df = df[0].str.split(";", expand=True)
df.columns = ["Asal", "Biaya", "Minat", "Akses", "Kualitas", "Memilih"]

X = df[["Asal", "Biaya", "Minat", "Akses", "Kualitas"]].astype(int)
y = df["Memilih"]

model = joblib.load("model/model.pkl")

# Prediksi
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

# Akurasi dan laporan klasifikasi
st.subheader("🔹 Classification Report")
report = classification_report(y, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("🔹 Confusion Matrix")
cm = confusion_matrix(y, y_pred, labels=["Ya", "Tidak"])
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ya", "Tidak"])
disp.plot(ax=ax)
st.pyplot(fig)

# ROC AUC Score
if y_proba is not None:
    try:
        y_binary = y.map({"Ya": 1, "Tidak": 0})
        auc_score = roc_auc_score(y_binary, y_proba)
        st.subheader("🔹 ROC AUC Score")
        st.metric(label="AUC", value=round(auc_score, 3))
    except:
        st.warning("Tidak bisa menghitung ROC AUC Score.")
