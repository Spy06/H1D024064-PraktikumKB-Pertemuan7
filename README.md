## 📁 Struktur Folder
```
pertemuan7/
├── jst7.py                # Source code utama
├── iris.data              # Dataset Iris (lokal)
├── README.md              # Dokumentasi
├── training_history.png   # Grafik loss & accuracy
└── confusion_matrix.png   # Confusion matrix
```

## 🗃️ Dataset
Menggunakan Iris Dataset lokal (iris.data) dengan 150 sampel (80% training, 20% testing).

Input (4 Fitur): Panjang/lebar sepal & panjang/lebar petal (cm).

Output (3 Kelas): Iris-setosa, Iris-versicolor, Iris-virginica.

## 🏗️ Arsitektur & Konfigurasi Model
Ringkasan Layer
Input: 4 Fitur

Dense 1: 1000 neuron, aktivasi ReLU

Dense 2: 500 neuron, aktivasi ReLU

Dense 3: 300 neuron, aktivasi ReLU

Output: 3 neuron, aktivasi Softmax

Total Parameter: 656,703

Setelan Training
Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Epochs: 50 | Batch Size: 32

## 📊 Hasil Evaluasi
```
Loss: 0.0689

Accuracy: 96.67% (29 dari 30 data uji benar)
```
## 🚀 Cara Menjalankan

1. Install Dependencies

Bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

2. Jalankan Program
Pastikan file iris.data berada di folder yang sama dengan jst2.py, lalu eksekusi:
```
Bash
python jst2.py
```
3. Input Interaktif
Masukkan data baru saat diminta di terminal untuk langsung melihat hasil prediksi spesiesnya.