import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar el dataset
file_path = r"C:\Users\s2dan\OneDrive\Documentos\WorkSpace\Proyect_AI\ObesityDataSet_raw_and_data_sinthetic.csv"
data = pd.read_csv(file_path)

# Codificar las variables categóricas
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Dividir el dataset en características (X) y la etiqueta (y)
X = data.drop('NObeyesdad', axis=1)  # Características
y = data['NObeyesdad']  # Etiqueta

# Estandarizar los datos (PCA es sensible a la escala de las características)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inicializar el clasificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Definir las cantidades de componentes principales para las ejecuciones
components_list = [12, 10, 11, 9, 5, 3]

# Almacenar los resultados
accuracies_pca = {}

# Aplicar PCA y entrenar el modelo para cada número de componentes
for n_components in components_list:
    # Reducir la dimensionalidad con PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Dividir los datos en conjunto de entrenamiento y prueba (80%/20%)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Entrenar el clasificador
    rf_classifier.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = rf_classifier.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_pca[n_components] = accuracy

# Mostrar los resultados de precisión para cada configuración de PCA
for n_components, accuracy in accuracies_pca.items():
    print(f"Precisión con {n_components} componentes principales: {accuracy * 100:.2f}%")

# ---------------------------------------------
# Explicación de PCA (Análisis de Componentes Principales):
# ---------------------------------------------
# PCA es una técnica de reducción de dimensionalidad que se utiliza para reducir el número de 
# características en un conjunto de datos mientras se conserva la mayor parte de la varianza posible.
# Esto se logra proyectando los datos originales en un nuevo conjunto de coordenadas llamado "componentes
# principales", que son combinaciones lineales de las características originales.

# Desde el punto de vista del álgebra lineal:
# 1. PCA busca los "vectores propios" (eigenvectors) y sus correspondientes "valores propios" (eigenvalues) 
#    de la matriz de covarianza de los datos. Los vectores propios son las direcciones de máxima varianza 
#    en los datos, y los valores propios representan la magnitud de esa varianza.
# 2. Los componentes principales son simplemente los primeros vectores propios ordenados por el valor propio 
#    (de mayor a menor).
# 3. La cantidad de componentes seleccionados depende de cuánta varianza queremos retener del conjunto de datos 
#    original. Cada componente principal captura una fracción de la varianza total de los datos.

# En este caso, estamos utilizando PCA para reducir las características originales de 12 a 3 (o cualquier otra 
# cantidad especificada), con el objetivo de observar cómo el modelo de clasificación se comporta con un menor número 
# de características.
