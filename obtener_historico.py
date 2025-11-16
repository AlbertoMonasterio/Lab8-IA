import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Configuración ---
# El "Ticker" de Solana en Yahoo Finance es 'SOL-USD'
ASSET_TICKER = 'SOL-USD'
NOMBRE_ARCHIVO_CSV = "solana_historico_YFINANCE.csv"
# ---------------------

print(f"Descargando datos para {ASSET_TICKER} desde Yahoo Finance...")

try:
    # 1. Definir el rango de 1 año
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 2. Descargar datos (start, end, intervalo diario)
    df = yf.download(ASSET_TICKER, 
                     start=start_date.strftime('%Y-%m-%d'), 
                     end=end_date.strftime('%Y-%m-%d'), 
                     interval="1d")
    
    if not df.empty:
        # 3. Limpieza de datos
        df = df.reset_index() # Mover la fecha de índice a columna
        
        # 4. Renombrar columnas a minúsculas (como pide tu proyecto)
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        })
        
        # 5. Seleccionar solo las columnas que necesitas
        # (Yahoo Finance da 'Adj Close' y 'Volume' que no usaremos)
        df_final = df[['date', 'open', 'high', 'low', 'close']]
        
        # 6. Guardar a CSV
        df_final.to_csv(NOMBRE_ARCHIVO_CSV, index=False)
        
        print("-" * 50)
        print(f"¡Éxito! Datos guardados en {NOMBRE_ARCHIVO_CSV}")
        print(f"Total de registros obtenidos: {len(df_final)}")
        print("\nVista previa de los datos:")
        print(df_final.head())
        print("-" * 50)
    
    else:
        print("No se pudieron descargar datos. Verifica el ticker 'SOL-USD'.")

except Exception as e:
    print(f"Ocurrió un error: {e}")