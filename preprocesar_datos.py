import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib 

# --- 1. Cargar Datos ---
NOMBRE_ARCHIVO_IN = "solana_historico_YFINANCE.csv"
try:
    df = pd.read_csv(NOMBRE_ARCHIVO_IN)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {NOMBRE_ARCHIVO_IN}")
    exit()

print(f"Datos cargados desde {NOMBRE_ARCHIVO_IN} ({len(df)} registros)")

# --- 2.1 Limpieza y Conversión ---
# Definimos las columnas que deben ser numéricas
features = ['open', 'high', 'low', 'close']

# <-- INICIO DE LÍNEAS NUEVAS -->
# Forzar todas las columnas de 'features' a ser numéricas.
# 'errors='coerce'' convertirá cualquier string (como 'SOL-USD') en NaN (Nulo).
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# <-- FIN DE LÍNEAS NUEVAS -->

# Inspeccionar si hay valores nulos (ahora puede incluir los que creamos)
print(f"\nValores nulos antes de limpiar:\n{df.isnull().sum()}")

# Rellenar con el método 'forward fill' (usar el valor del día anterior)
df = df.ffill() # <-- LÍNEA MODIFICADA (elimina el FutureWarning)

# Si aún quedan nulos al inicio, rellenar con 'backward fill'
df = df.bfill() # <-- LÍNEA MODIFICADA (elimina el FutureWarning)

print("\nValores nulos después de limpiar (deberían ser 0):")
print(df.isnull().sum())

# --- 2.2 Escalado (Normalización) ---
# Ahora 'data_a_escalar' está garantizado que solo contiene números
data_a_escalar = df[features] 

scaler = MinMaxScaler(feature_range=(0, 1))
data_escalada = scaler.fit_transform(data_a_escalar)

# Guardar el scaler para el Paso 4 (Desescalado)
joblib.dump(scaler, 'scaler_multivariado.joblib')
print("\nDatos escalados. Scaler guardado en 'scaler_multivariado.joblib'")

# --- 2.3 Creación de Secuencias Temporales (Ventanas) ---
TIME_STEP = 60
N_FEATURES = 4
# ------------------------------------

X = [] 
y = [] 

for i in range(TIME_STEP, len(data_escalada)):
    X.append(data_escalada[i-TIME_STEP:i, :])
    y.append(data_escalada[i, 3]) # Columna 3 es 'close'

X, y = np.array(X), np.array(y)

# --- 3. División de Datos (Train/Test Split) ---
split_ratio = 0.8
train_size = int(len(X) * split_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("-" * 50)
print("Formato de datos (shape) listo para el modelo LSTM/GRU:")
print(f"  X_train (Muestras, TimeSteps, Features): {X_train.shape}")
print(f"  y_train (Muestras, Target):           {y_train.shape}")
print(f"  X_test (Muestras, TimeSteps, Features):  {X_test.shape}")
print(f"  y_test (Muestras, Target):           {y_test.shape}")
print("-" * 50)

# --- 4. Guardar los datos procesados ---
np.savez('datos_procesados.npz', 
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test)

print("Datos de entrenamiento y prueba guardados en 'datos_procesados.npz'")