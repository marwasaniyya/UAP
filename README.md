# ✨ Analisis Sentimen Ulasan Pengguna Aplikasi Gojek: Perspektif Pengalaman Pengguna di Indonesia ✨

## 📑 Table of Contents
1. [📋 Deskripsi Proyek](#deskripsi-proyek)
2. [⚙️ Langkah Instalasi](#langkah-instalasi)
3. [🗂️ Overview Dataset](#overview-dataset)
4. [🧠 Deskripsi Model](#deskripsi-model)
   - [📈 LSTM](#lstm)
   - [📊 BERT](#bert)
5. [📊 Hasil dan Analisis](#hasil-dan-analisis)
6. [📊 Kesimpulan perbandingan model LSTM dan BERT](#Kesimpulan-perbandingan-model-LSTM-dan-BERT)
7. [🔗 Link Live Demo](#link-live-demo)
8. [👨‍💻 Author](#author)

---

## 📋 Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model analisis sentimen pada ulasan pengguna aplikasi Gojek, dengan memanfaatkan rating bintang 1 hingga 5 sebagai indikator sentimen. Pendekatan ini dirancang untuk memahami perspektif pengalaman pengguna di Indonesia dan memberikan wawasan berharga guna meningkatkan kualitas layanan aplikasi.  

Analisis ini mengelompokkan ulasan berdasarkan rating bintang, di mana bintang 1 dan 2 mencerminkan sentimen negatif, bintang 3 dianggap netral, dan bintang 4 serta 5 mencerminkan sentimen positif. Dengan model ini, ulasan pengguna dapat diolah menjadi informasi strategis untuk mendukung pengambilan keputusan dan perbaikan layanan.

---

## ⚙️ Langkah Instalasi
Ikuti langkah-langkah berikut untuk menginstal dependencies dan menjalankan aplikasi:

1. **Clone Repository:**
   ```bash
   git init
   git add .
   git commit -m "Inisialisasi proyek"
   git remote add origin https://github.com/marwasaniyya/UAP.git
   git branch -M main
   git push -u origin main

   commit
   git status
   git add (sesuai file yang ditambahkan)
   git commit -m "coba"
   git push origin main
  
   ```

2. **Buat Virtual Environment:**
   ```bash
   python -m venv env
   env\Scripts\activate   # Untuk Windows
   ```

3. **Instal Dependencies:**
   ```bash
   pip install pdm
   pdm init
   pdm add streamlit
   pdm add tensorflow
   pdm add joblib
   pdm add scikit-learn
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Web:**
   ```bash
   streamlit run app.py
   ```

---
## 🗂️ Overview Dataset
Dataset diambil dari [Master Data File Full Columns](https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia). Dataset ini mencakup:
- **Periode Pengumpulan Data:** 05 November 2021 hingga 02 Februari 2024
- **Jumlah Variabel:** 5 kolom
- **Jumlah Observasi:** 25.011 baris
  Data dibagi menjadi dua bagian: 80% untuk Training Set dan 20% untuk Testing Set. Dimana pada setiap Set, terdapat 5 Label Class yaitu:
  - Label "0" menunjukkan bahwa berita memiliki rating 1, yang merupakan rating terjelek.
  - Label "1" menunjukkan bahwa berita memiliki rating 2, yang sedikit lebih baik dibandingkan rating 1.
  - Label "2" menunjukkan bahwa berita memiliki rating 3, yang merupakan rating sedang atau netral.
  - Label "3" menunjukkan bahwa berita memiliki rating 4, yang cukup bagus.
  - Label "4" menunjukkan bahwa berita memiliki rating 5, yang merupakan rating terbaik.
    
## 🧠 Deskripsi Model

### 📈 LSTM
Model LSTM (Long Short-Term Memory) digunakan untuk menangkap konteks temporal dalam teks. Model ini dirancang untuk memahami hubungan antar kata dalam urutan teks sehingga dapat meningkatkan akurasi klasifikasi sentimen.

#### 🧠 Struktur LSTM 
Model LSTM memiliki arsitektur khusus yang dirancang untuk menangani masalah vanishing gradient dalam jaringan saraf berulang (RNN). Struktur utamanya terdiri dari tiga gerbang utama:  Forget Gate: Memutuskan informasi mana yang akan dilupakan dari memori. Input Gate: Memutuskan informasi baru mana yang akan ditambahkan ke memori. Output Gate: Memutuskan bagian dari memori yang akan dikeluarkan sebagai output.

Berikut adalah ilustrasi struktur LSTM: 

![image](https://github.com/user-attachments/assets/c0801fac-bf4c-417a-ba27-28d797fabfae)

#### 🔧Prepocessing 

Pada tahap preprocessing  mencakup data cleaning untuk menghilangkan noise atau data yang tidak relevan, tokenisasi untuk memecah teks menjadi kata atau token, dan stopword removal untuk menghapus kata-kata yang tidak penting seperti "dan", "atau", dan "adalah". dilanjut dengan melakukan splitting dataset menjadi 2 (Training, dan Testing) sesuai dengan penjelasan pada Dataset.

Hasil dari model lstm yang telah di bangun

| Layer Type               | Output Shape          | Number of Parameters |
|--------------------------|-----------------------|----------------------|
| Embedding                | (None, 100, 256)      | 1,280,000            |
| LSTM                     | (None, 256)           | 525,312              |
| Dense                    | (None, 5)             | 1,285                |



#### 📊 Hasil Evaluasi Model LSTM
   1. **Training Accuracy**: Signifikan hampir mencapai 100%.
   2. **Validation Accuracy**: Stabil di sekitar 85%, menunjukkan kemampuan model untuk menggeneralisasi dengan baik.
   3. **Testing Accuracy**: Sedikit lebih rendah (82%), yang dapat menunjukkan tantangan dalam generalisasi pada data yang lebih bervariasi atau berbeda dari data pelatihan.

#### 📝 Classification Report LSTM

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.69      | 0.74   | 0.71     | 1445    |
| 1     | 0.15      | 0.13   | 0.14     | 248     |
| 2     | 0.10      | 0.09   | 0.09     | 241     |
| 3     | 0.09      | 0.05   | 0.06     | 283     |
| 4     | 0.87      | 0.90   | 0.88     | 2785    |

Accuracy: 73%

#### 📉 Grafik Akurasi LSTM

![image](https://github.com/user-attachments/assets/3532e560-be20-4637-b592-70193d31c729)

Training accuracy meningkat secara signifikan dan mencapai hampir 100% setelah beberapa epoch. Testing accuracy stabil di sekitar 83%, menunjukkan adanya sedikit overfitting. 

#### 📉 Grafik Loss LSTM

![image](https://github.com/user-attachments/assets/d513718a-6fd7-4602-93f9-fd5440756dc7)

Training loss terus menurun secara konsisten. Testing loss meningkat setelah beberapa epoch, menandakan adanya overfitting. 

#### 🧩 Confusion Matrix 📊
Hasil dari Confusion Matrix

![image](https://github.com/user-attachments/assets/559b97a4-ca57-4b20-9808-1ab7881c06c5)

----

### 📊 BERT
Model BERT (Bidirectional Encoder Representations from Transformers) digunakan untuk memanfaatkan representasi teks berbasis transformer yang lebih kaya. Model ini sangat efektif dalam memahami konteks dua arah dalam teks.

#### 🧠 Struktur LSTM 
Arsitektur BERT terdiri dari tiga bagian utama: input layer yang menggabungkan token, segment, dan position embeddings; encoder berbasis Transformer dengan self-attention untuk pemrosesan bidirectional; dan output layer yang menghasilkan representasi vektor untuk tugas lanjutan seperti klasifikasi atau NER.

Berikut adalah ilustrasi struktur BERT: 

![image](https://github.com/user-attachments/assets/583004e0-9546-47c4-bb65-26435c08936d)


#### 🔧Prepocessing 

Pada tahap preprocessing  mencakup data cleaning untuk menghilangkan noise atau data yang tidak relevan, tokenisasi untuk memecah teks menjadi kata atau token, dan stopword removal untuk menghapus kata-kata yang tidak penting seperti "dan", "atau", dan "adalah". dilanjut dengan melakukan splitting dataset menjadi 2 (Training, dan Testing) sesuai dengan penjelasan pada Dataset.

#### 📊 Hasil Evaluasi Model BERT
1. **Training Accuracy**: Meningkat secara signifikan mecapai 95%.
2. **Validation Accuracy**: Stabil di sekitar 80%-87%, mencerminkan kemampuan model untuk menggeneralisasi data yang tidak terlihat.
3. **Testing Accuracy**: Sekitar 85%, menandakan bahwa model cukup baik dalam melakukan prediksi pada data baru setelah pelatihan.

#### 📝 Classification Report BERT
| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.70      | 0.71   | 0.71     | 890     |
| 1     | 0.67      | 0.73   | 0.70     | 1013    |
| 2     | 0.93      | 0.90   | 0.91     | 4894    |

Accuracy : 85%

#### 📉 Grafik BERT

1. **Accuracy:**
   
![image](https://github.com/user-attachments/assets/5ce50b4a-094e-4f37-8260-8fbc237ea297)

Training accuracy meningkat hingga hampir 95%, sementara validation accuracy stabil di sekitar 85%-87%, menunjukkan model BERT dapat menggeneralisasi dengan baik.

3. **Loss:**
   
![image](https://github.com/user-attachments/assets/c3bb8c68-adb4-4643-a7b7-b88acb4df143)

Training loss terus menurun, sementara validation loss sedikit meningkat, menunjukkan tanda overfitting ringan.


#### 🧩 Confusion Matrix 📊
Hasil dari Confusion Matrix

![image](https://github.com/user-attachments/assets/babcb283-66d9-4840-8b17-c7813e391b38)


## 📊 Hasil dan Analisis

### 📈 Perbandingan Performa Model
| Model | Training Accuracy | Validation Accuracy | Testing Accuracy |
|-------|-------------------|---------------------|------------------|
| LSTM  | 90%               | 85%                 | 82%              |
| BERT  |90%                | 80%-87%             | 85%              |

---

### 📊 Kesimpulan perbandingan model LSTM dan BERT

1. **Akurasi** :
   - *LSTM* :
     * Akurasi training mencapai 90% dan akurasi test mencapai 82%.
     * Performa bagus dalam menangkap konteks temporal, namun terdapat bukti overfitting karena terdapat kesenjangan antara akurasi training dan test.
   - *BERT*:
     * Akurasi training meningkat menjadi 905% dan akurasi test mencapai 85%.
     * Memahami konteks teks dua arah lebih baik daripada LSTM, sehingga menghasilkan generalisasi yang lebih kuat.

2. **Loss**:
   - *LSTM*
      * Loss training terus menurun, tetapi test pengujian meningkat setelah beberapa epoch. Hal ini menunjukkan bahwa model cenderung overfit. 
   - *BERT*
     * Loss training terus menurun, dan peningkatan kerugian test relatif kecil, menunjukkan sedikit overfitting namun masih dapat dikelola. 
 
 3. **Kemampuan generalisasi**:
    - *LSTM*
       * Dapat memahami urutan teks dengan baik, namun cenderung overfit.
    - *BERT*
       * Model dengan  arsitektur transformator dua arah ini lebih baik dalam memahami hubungan yang kompleks dan menunjukkan kemampuan generalisasi yang lebih baik.

4. **Fleksibilitas penyesuaian**:
   - *LSTM*
      * Cocok untuk data dengan urutan kronologis yang jelas, namun terbatas dalam memahami hubungan nonlinier yang kompleks.
   - *BERT*
      * Lebih baik dalam menangani konteks kompleks dan hubungan non-linier, memungkinkan dapat melakukan tugas analisis sentimen terperinci dengan lebih efektif.

### **Kesimpulan akhir**:
Akurasi dan generalisasi yang lebih tinggi untuk analisis sentimen, **BERT** adalah pilihan yang lebih baik daripada **LSTM**. Namun, ketika sumber daya komputasi terbatas, **LSTM** tetap menjadi pilihan yang efisien dan cukup handal.

---

## 🔗 Link Live Demo
Aplikasi web telah di-deploy dan dapat diakses melalui tautan berikut:  
[Live Demo Aplikasi](https://uaplatihan.streamlit.app/)


---

## 👨‍💻 Author  
👨‍💻 [Nadiya Dewi Al Khlifi](https://github.com/Nadiyaal)
