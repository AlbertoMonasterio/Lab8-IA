import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Suprimir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- DEFINIR EL ASSET ID ---
ASSET_ID = "solana" # <--- ¡LÍNEA AÑADIDA PARA CORREGIR EL ERROR!
# ---------------------------

# --- 1. Cargar Archivos ---
try:
    model = load_model('modelo_lstm.keras')
    scaler = joblib.load('scaler_multivariado.joblib')
    data = np.load('datos_procesados.npz')
    X_test = data['X_test']
    y_test = data['y_test'] # Este 'y_test' está escalado
except Exception as e:
    print(f"Error cargando archivos: {e}")
    print("Asegúrate de tener 'modelo_lstm.keras', 'scaler_multivariado.joblib', y 'datos_procesados.npz' en la carpeta.")
    exit()

print("Archivos cargados (modelo, scaler, datos de prueba).")

# --- 2. Predicción en el Conjunto de Prueba ---
y_pred_scaled = model.predict(X_test)

# --- 3. Desescalado (Paso 4.2) ---
dummy_shape = (len(y_test), 4)

y_test_dummy = np.zeros(dummy_shape)
y_test_dummy[:, 3] = y_test.reshape(-1) 
y_test_real = scaler.inverse_transform(y_test_dummy)[:, 3] 

y_pred_dummy = np.zeros(dummy_shape)
y_pred_dummy[:, 3] = y_pred_scaled.reshape(-1) 
y_pred_real = scaler.inverse_transform(y_pred_dummy)[:, 3]

print("\nPredicciones y valores reales han sido desescalados a USD.")

# --- 4. Evaluación (Paso 4.3) ---
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)

print("\n" + "="*50)
print("--- Evaluación del Modelo (en USD) ---")
print(f"  RMSE (Raíz del Error Cuadrático Medio): ${rmse:.2f}")
print(f"  MAE  (Error Absoluto Medio):          ${mae:.2f}")
print(f"\n(Esto significa que, en promedio, las predicciones del modelo")
print(f" sobre el conjunto de prueba se equivocan en ${mae:.2f})")
print("="*50)

# --- 5. Predicción a 10 Días (Autoregresiva) (Paso 4.4) ---
df_raw = pd.read_csv('solana_historico_YFINANCE.csv')
features = ['open', 'high', 'low', 'close']

for col in features:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
df_raw = df_raw.ffill().bfill()

data_total_scaled = scaler.transform(df_raw[features])

TIME_STEP = 60
N_FEATURES = 4
last_sequence_scaled = data_total_scaled[-TIME_STEP:]

future_predictions_scaled = []
current_sequence = last_sequence_scaled.reshape(1, TIME_STEP, N_FEATURES)

print("\n--- Iniciando Predicción Autoregresiva a 10 Días ---")

for i in range(10): # Repetir 10 veces
    pred_scaled = model.predict(current_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled[0])
    
    # Asumimos que o,h,l serán iguales al 'close' predicho
    new_row = np.array([pred_scaled[0]] * N_FEATURES) 
    
    new_sequence_base = current_sequence[0][1:]
    new_sequence = np.append(new_sequence_base, new_row.reshape(1, N_FEATURES), axis=0)
    current_sequence = new_sequence.reshape(1, TIME_STEP, N_FEATURES)

print("Predicción a 10 días completada.")

pred_10_days_dummy = np.zeros((len(future_predictions_scaled), 4))
pred_10_days_dummy[:, 3] = future_predictions_scaled
pred_10_days_real = scaler.inverse_transform(pred_10_days_dummy)[:, 3]

print("\n" + "-"*50)
print(f"--- Predicciones de {ASSET_ID.capitalize()} a 10 Días (en USD) ---")
for i, price in enumerate(pred_10_days_real):
    print(f"  Día {i+1}: ${price:.2f}")
print("-"*50)

# --- 6. Visualización (Paso 4.5) ---
print("\nGenerando gráficos...")

# Gráfico 1: Comparación en Conjunto de Prueba
plt.figure(figsize=(14, 7))
plt.plot(y_test_real, color='blue', label='Precio Real (Test)', marker='.', markersize=5)
plt.plot(y_pred_real, color='red', label='Precio Predicho (Test)', linestyle='--')
plt.title('Comparación: Precio Real vs. Predicho (Conjunto de Prueba)')
plt.xlabel('Días (en el conjunto de prueba)')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.grid(True)
plt.savefig('test_vs_predicted.png')
print("Gráfico 'test_vs_predicted.png' guardado.")

# Gráfico 2: Predicción a 10 Días Futuros
last_30_days_real = y_test_real[-30:]
dias_historial = np.arange(len(last_30_days_real))
dias_futuros = np.arange(len(last_30_days_real), len(last_30_days_real) + 10)

plt.figure(figsize=(14, 7))
plt.plot(dias_historial, last_30_days_real, color='blue', label='Últimos 30 Días (Historial Real)')
plt.plot(dias_futuros, pred_10_days_real, color='red', label='Predicción a 10 Días', linestyle='--', marker='o')
# Esta línea ahora funcionará
plt.title(f'Predicción de {ASSET_ID.capitalize()} a 10 Días Futuros (Autoregresiva)') 
plt.xlabel('Días')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.grid(True)
plt.savefig('forecast_10_dias.png')
print("Gráfico 'forecast_10_dias.png' guardado.")
print("\n--- ¡PROYECTO COMPLETADO! ---")