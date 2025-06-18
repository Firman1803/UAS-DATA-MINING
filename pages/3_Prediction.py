import streamlit as st
import numpy as np
import joblib

st.title("🔮 Prediksi Pemilihan Kampus")

model = joblib.load("model/model.pkl")

asal = st.selectbox("Asal Wilayah", [1, 2, 3])
biaya = st.number_input("Kemampuan Biaya (Rp)", value=3000000, step=500000)
minat = st.selectbox("Minat Jurusan", [1, 2, 3])
akses = st.selectbox("Kemudahan Akses", [1, 2, 3])
kualitas = st.selectbox("Persepsi Kualitas", [1, 2, 3])

if st.button("Prediksi"):
    X = np.array([[asal, biaya, minat, akses, kualitas]])
    hasil = model.predict(X)[0]
    st.success(f"Prediksi: {'Memilih' if hasil == 'Ya' else 'Tidak Memilih'}")
