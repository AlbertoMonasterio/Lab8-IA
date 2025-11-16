import requests
import json

# Clave de API y endpoint que proporcionaste
API_KEY = "01617ff674242d25f37a1e7da4c4d8865a21a6e3475291b9189ce7a6c2378297"
URL = "https://rest.coincap.io/v3/assets"

# Configuramos los headers para incluir la clave de API
# El formato estándar es "Bearer <token>"
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

print("Conectando a la API de CoinCap (v3) para obtener la lista de activos...")

try:
    # Realizamos la petición GET
    response = requests.get(URL, headers=headers)

    # Verificar si la petición fue exitosa (código 200)
    if response.status_code == 200:
        data = response.json()
        
        # La API v3 devuelve los datos bajo la clave "data"
        if "data" in data:
            print("-" * 50)
            print(f"{'Nombre':<25} | {'ID del Activo (para la API)':<20}")
            print("-" * 50)
            
            # Iteramos sobre la lista de activos e imprimimos el nombre y el ID
            for asset in data["data"]:
                print(f"{asset.get('name', 'N/A'):<25} | {asset.get('id', 'N/A'):<20}")
        
        else:
            print("Error: La respuesta JSON no contiene la clave 'data'.")
            print("Respuesta recibida:", json.dumps(data, indent=2))

    else:
        # Si hay un error (ej. clave inválida, error 401 o 403)
        print(f"Error al conectar con la API. Código de estado: {response.status_code}")
        print("Respuesta:", response.text)

except requests.exceptions.RequestException as e:
    print(f"Error de conexión: {e}")