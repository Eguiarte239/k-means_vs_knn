import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Configuración para imprimir todos los resultados en una sola línea
pd.set_option('display.max_columns', None)

# Carga el conjunto de datos desde un archivo xlsx
data = pd.read_excel("diabetes_prediction_dataset.xlsx")

# Codifica las variables categóricas en el conjunto de datos
label_encoder = LabelEncoder()
data["gender"] = label_encoder.fit_transform(data["gender"])
data["smoking_history"] = label_encoder.fit_transform(data["smoking_history"])
data["diabetes"] = label_encoder.fit_transform(data["diabetes"])  # Codificamos la variable objetivo

# Selecciona las características y la variable objetivo
X = data[["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]]
y = data["diabetes"]

# Divide el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplica K-means para agrupar los datos en k clústeres (ajusta k según tus necesidades)
num_clusters = 2  # ajusta el número de clústeres según tus necesidades
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_train)

# Obtiene las etiquetas de clúster para los datos de prueba
kmeans_labels = kmeans.predict(X_test)

# Aplica K-nearest neighbors (KNN) para predecir la diabetes
k_neighbors = 3  # ajusta el número de vecinos según tus necesidades
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Evalúa el rendimiento de K-means y KNN
print("Resultados de K-means:")
print(confusion_matrix(y_test, kmeans_labels))
print(classification_report(y_test, kmeans_labels))

print("\nResultados de K-nearest neighbors (KNN):")
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))