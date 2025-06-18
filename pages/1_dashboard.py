import streamlit as st
import pandas as pd

st.title("📊 Dashboard")
data = pd.read_csv("data/dataset.csv", header=None)
data = data[0].str.split(";", expand=True)
data.columns = ["Asal", "Biaya", "Minat", "Akses", "Kualitas", "Memilih"]

st.subheader("Preview Data")
st.dataframe(data.head())

st.subheader("Distribusi Keputusan")
st.bar_chart(data["Memilih"].value_counts())
