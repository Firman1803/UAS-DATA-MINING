import streamlit as st
import joblib

st.title("📈 Model Performance")

st.markdown("""
Model klasifikasi digunakan untuk memprediksi apakah calon mahasiswa akan memilih kampus berdasarkan fitur tertentu.
""")

# Visualisasi dummy
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/ROC_curve.svg/512px-ROC_curve.svg.png", caption="Contoh ROC Curve")
