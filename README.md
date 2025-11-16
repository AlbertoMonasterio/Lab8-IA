🤖 Proyecto de Predicción de Criptomonedas (LSTM)
Este proyecto utiliza un modelo de Red Neuronal Recurrente (LSTM) para predecir el precio futuro de la criptomoneda Solana (SOL) basándose en datos históricos.
📋 Requisitos
Para ejecutar este proyecto, necesitarás tener Python 3.x instalado. Puedes instalar todas las librerías necesarias ejecutando el siguiente comando en tu terminal:
pip install pandas yfinance numpy scikit-learn tensorflow matplotlib

Librerías utilizadas:
 * yfinance: Para descargar los datos históricos de precios (OHLC).
 * pandas: Para la manipulación y limpieza de datos.
 * numpy: Para operaciones numéricas y la creación de secuencias.
 * scikit-learn: Para el MinMaxScaler (escalado de datos) y las métricas de evaluación (MAE, RMSE).
 * tensorflow: Para construir y entrenar el modelo LSTM.
 * matplotlib: Para visualizar los resultados (gráficos).
🚀 Cómo Ejecutar el Proyecto
El proyecto está dividido en 4 scripts que deben ejecutarse en orden:
1. Obtener Datos
Descarga el historial de 1 año de Solana (OHLC) desde Yahoo Finance.
python obtener_historico_YFINANCE.py

 * Salida: solana_historico_YFINANCE.csv
2. Preprocesar Datos
Limpia, escala (normaliza) y transforma los datos en "ventanas deslizantes" para el modelo.
python preprocesar_datos.py

 * Salida: datos_procesados.npz y scaler_multivariado.joblib
3. Entrenar el Modelo
Construye la arquitectura LSTM, entrena el modelo con los datos procesados y guarda el modelo final.
python entrenar_modelo.py

 * Salida: modelo_lstm.keras y loss_history.png
4. Predecir y Evaluar
Carga el modelo entrenado para:
 * Evaluar su rendimiento (MAE/RMSE) en el conjunto de prueba.
 * Realizar una predicción autoregresiva de los próximos 10 días.
<!-- end list -->
python predecir_evaluar.py

 * Salida: Métricas en la terminal y los gráficos test_vs_predicted.png y forecast_10_dias.png.
