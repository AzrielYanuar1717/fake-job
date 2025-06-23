import streamlit as st
import joblib
import numpy as np

# Load model & encoder
tfidf = joblib.load('models/tfidf.pkl')
ohe = joblib.load('models/ohe.pkl')
rf = joblib.load('models/rf.pkl')

st.title("Prediksi Lowongan Kerja Palsu/Asli")

# Input user
desc = st.text_area("Deskripsi Lowongan", "")
emp_type = st.selectbox("Employment Type", ohe.categories_[0])
exp = st.selectbox("Experience", ohe.categories_[1])
edu = st.selectbox("Education", ohe.categories_[2])
industry = st.selectbox("Industry", ohe.categories_[3])
function = st.selectbox("Function", ohe.categories_[4])
telecommuting = st.selectbox("Telecommuting", [0,1])
has_logo = st.selectbox("Ada Logo?", [0,1])
has_questions = st.selectbox("Ada Questions?", [0,1])

if st.button("Prediksi"):
    # Proses sama seperti di preprocessing
    desc_vec = tfidf.transform([desc]).toarray()
    cat_vec = ohe.transform([[emp_type, exp, edu, industry, function]])
    num_vec = np.array([[telecommuting, has_logo, has_questions]])
    X_input = np.hstack([desc_vec, cat_vec, num_vec])
    pred = rf.predict(X_input)
    st.success("Lowongan Palsu!" if pred[0]==1 else "Lowongan Asli!")

