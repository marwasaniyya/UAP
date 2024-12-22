import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path  # Untuk mengelola path file

# Fungsi untuk prediksi skor
def predict_score(model, tokenizer, comment, max_len=100):
    sequences = tokenizer.texts_to_sequences([comment])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)
    predicted_score = np.argmax(prediction) + 1  # Tambahkan 1 karena skor mulai dari 1
    return predicted_score

# Halaman Utama
def main_page():
    st.title("Selamat Datang Pada Ujian Praktikum Machine Learning")
    st.write("Aplikasi ini dibuat untuk memprediksi skor komentar berdasarkan teks yang diberikan.")
    
    if st.button("Masuk ke Halaman Prediksi"):
        st.session_state['page'] = 'prediksi'

# Halaman Prediksi
def prediction_page():
    st.title("Prediksi Skor Komentar")
    st.write("Aplikasi ini memprediksi skor komentar pada skala 1 hingga 5 berdasarkan teks yang diberikan.")
    
    # Path untuk model dan tokenizer
    tokenizer_path = Path(__file__).parent / "model/tokenizer.joblib"  # Path tokenizer
    model_path = Path(__file__).parent / "model/model_lstm.h5"  # Path model

    # Cek keberadaan file tokenizer dan model
    if not tokenizer_path.exists():
        st.error(f"File tokenizer tidak ditemukan: {tokenizer_path}")
        return
    if not model_path.exists():
        st.error(f"File model tidak ditemukan: {model_path}")
        return

    # Muat tokenizer dan model
    tokenizer = joblib.load(tokenizer_path)  # Muat tokenizer
    model = load_model(model_path)  # Muat model
    
    # Input komentar
    user_input = st.text_area("Masukkan komentar di sini:")
    
    # Tombol prediksi
    if st.button("Prediksi Skor"):
        if user_input.strip() == "":
            st.warning("Harap masukkan komentar terlebih dahulu!")
        else:
            # Prediksi dan tampilkan hasil
            predicted_score = predict_score(model, tokenizer, user_input)
            st.success(f"Skor prediksi untuk komentar ini adalah: {predicted_score}")
    
    # Tombol kembali ke halaman utama
    if st.button("Kembali ke Halaman Utama"):
        st.session_state['page'] = 'utama'

# Navigasi antar halaman
if 'page' not in st.session_state:
    st.session_state['page'] = 'utama'

if st.session_state['page'] == 'utama':
    main_page()
elif st.session_state['page'] == 'prediksi':
    prediction_page()
