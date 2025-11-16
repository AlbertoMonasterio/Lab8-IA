import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import matplotlib.pyplot as plt
import os

# Suprimir logs de TensorFlow (opcional, para una salida más limpia)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Cargar Datos Procesados ---
# Cargamos los datos que generamos en el Paso 2
try:
    data = np.load('datos_procesados.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
except FileNotFoundError:
    print("Error: No se encontró el archivo 'datos_procesados.npz'.")
    print("Asegúrate de ejecutar 'preprocesar_datos.py' primero.")
    exit()

print("Datos cargados. Formas (Shapes):")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# --- 2. Definición de la Arquitectura (Paso 3.2) ---
# Necesitamos la 'forma de entrada' (time_step, features)
# X_train.shape[1] = 60 (el TIME_STEP)
# X_train.shape[2] = 4 (las N_FEATURES: open, high, low, close)
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential()

# Capa Inicial: LSTM (como pide tu guía)
# Usaremos 50 unidades. 'input_shape' solo se pone en la primera capa.
# 'return_sequences=True' es necesario si vas a apilar otra capa LSTM.
model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))

# Capa Adicional: Dropout (para prevenir sobreajuste)
model.add(Dropout(0.2))

# Capa Adicional: Otra capa LSTM
# No es necesario 'input_shape' aquí.
# 'return_sequences=False' (es el valor por defecto) porque es la última capa recurrente.
model.add(LSTM(units=50))

# Capa Adicional: Dropout
model.add(Dropout(0.2))

# Capa de Salida: Dense (como pide tu guía)
# Una sola unidad porque queremos predecir 1 solo valor (el precio de cierre)
model.add(Dense(units=1))

print("\n--- Resumen del Modelo ---")
model.summary()

# --- 3. Compilación y Entrenamiento (Paso 3.3) ---

# Compilación
# Optimizador: 'adam' (recomendado)
# Función de Pérdida (loss): 'mean_squared_error' (mse) (recomendado para regresión)
model.compile(optimizer='adam', loss='mean_squared_error')

print("\n--- Iniciando Entrenamiento ---")

# Entrenamiento
# Usaremos 50 épocas y un batch_size de 32
# 'validation_data' nos permite ver cómo rinde el modelo en los datos de prueba
# al final de cada época.
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1 # Muestra la barra de progreso
)

print("\n--- Entrenamiento Finalizado ---")

# --- 4. Guardar el Modelo Entrenado ---
# Guardamos el modelo para usarlo en el Paso 4 (Predicciones)
model.save('modelo_lstm.keras')
print("Modelo guardado exitosamente como 'modelo_lstm.keras'")

# --- 5. Visualizar Pérdida (Loss) ---
# Es una buena práctica ver si el modelo aprendió bien
print("Generando gráfico de historial de pérdida...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(history.history['val_loss'], label='Pérdida (Validación)')
plt.title('Historial de Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('loss_history.png')
print("Gráfico guardado como 'loss_history.png'")
