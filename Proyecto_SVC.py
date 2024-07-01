import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_folder(folder, label):
    images = []
    labels = []
    print(f"Cargando imágenes de {folder}...")
    for filename in os.listdir(folder):
        print(f"Procesando archivo: {filename}")
        if filename.endswith(('.JPG', '.jpeg', '.png', '.bmp')):  # Asegurarse de que solo se carguen archivos de imagen
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('L')  # Convertir a escala de grises
                img = img.resize((512, 512))  # Redimensionar la imagen a 512x512
                img = np.array(img)  # Convertir la imagen a un arreglo numpy
                img = img.flatten()  # Aplanar la imagen
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error al procesar la imagen {img_path}: {e}")
    print(f"Se cargaron {len(images)} imágenes de {folder}.")
    return images, labels

# Cargar imágenes
folder_cancer = r'C:\Users\Eric\Desktop\ISIC MALIGNAS'
folder_no_cancer = r'C:\Users\Eric\Desktop\ISIC BENIGNAS'
images_cancer, labels_cancer = load_images_from_folder(folder_cancer, 1)
images_no_cancer, labels_no_cancer = load_images_from_folder(folder_no_cancer, 0)

# Combinar datasets
X = images_cancer + images_no_cancer
y = labels_cancer + labels_no_cancer
X = np.array(X)
y = np.array(y)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    img = img.resize((512, 512))  # Redimensionar la imagen a 512x512
    img = np.array(img)  # Convertir la imagen a un arreglo numpy
    img = img.flatten()  # Aplanar la imagen
    img = scaler.transform([img])  # Asegúrate de usar el mismo scaler que usaste para los datos de entrenamiento
    prediction = model.predict(img)
    probability = model.predict_proba(img)
    return prediction, probability

# Usar la función para predecir una nueva imagen
image_path = r"C:\Users\Eric\Desktop\ISIC_4027194.JPG"
prediction, probability = predict_image(image_path)
print(f'Prediction: {"Cancer" if prediction[0] == 1 else "No Cancer"}')
print(f'Probability: {probability}')

# Crear la matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns

def predict_images_from_folder(folder, label):
    predictions = []
    true_labels = []
    for filename in os.listdir(folder):
        if filename.endswith(('.JPG', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder, filename)
            prediction, _ = predict_image(img_path)
            predictions.append(prediction[0])
            true_labels.append(label)
    return predictions, true_labels

predictions_cancer, true_labels_cancer = predict_images_from_folder(folder_cancer, 1)
predictions_no_cancer, true_labels_no_cancer = predict_images_from_folder(folder_no_cancer, 0)

all_predictions = predictions_cancer + predictions_no_cancer
all_true_labels = true_labels_cancer + true_labels_no_cancer

cm = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
