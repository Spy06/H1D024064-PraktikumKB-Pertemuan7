import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)

dataset = pd.read_csv('iris.data', header=None, sep=',')

print("Dataset shape:", dataset.shape)
print("5 data pertama:")
print(dataset.head())

X = dataset.iloc[:, :-1].values   
y = dataset.iloc[:, -1].values    

print("\nFitur (X) shape:", X.shape)
print("Label (y) contoh:", y[:5])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)   

print("\nLabel setelah encoding:", y[:10])
print("Kelas:", label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nJumlah data latih:", X_train.shape[0])
print("Jumlah data uji:", X_test.shape[0])

model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500,  activation='relu'),
    Dense(300,  activation='relu'),
    Dense(3,    activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLoss: {loss:.4f}, Accuracy: {accuracy:.4f}")

pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Training History - Loss & Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Grafik disimpan sebagai 'training_history.png'")

predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("\nPrediksi:", predicted_classes)
print("Label Asli:", y_test)

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix disimpan sebagai 'confusion_matrix.png'")

def predict_new_data():
    print("\n--- Prediksi Spesies Bunga Iris Baru ---")
    sepal_length = float(input("Masukkan sepal length (cm): "))
    sepal_width  = float(input("Masukkan sepal width (cm): "))
    petal_length = float(input("Masukkan petal length (cm): "))
    petal_width  = float(input("Masukkan petal width (cm): "))

    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)

    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"\nProbabilitas tiap kelas: {prediction[0]}")
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()