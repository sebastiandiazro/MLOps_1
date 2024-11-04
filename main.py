# Importar librerias
from fastapi import FastAPI
import pandas as pd
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Crear la aplicación FastAPI
app = FastAPI()

#Crear titulo
app.title = "MLOps plataforma de streaming"

# Especificar las rutas absolutas a los archivos Parquet usando raw strings para evitar problemas con las barras invertidas
movies_df_path = 'Datasets/movies_df.parquet' 

# Verificar que los archivos existen
if not os.path.exists(movies_df_path):
    raise FileNotFoundError(f"Archivo no encontrado: {movies_df_path}")

# Leer los archivos Parquet y cargar DataFrames
df_movies  = pd.read_parquet(movies_df_path)

# Ruta raíz que devuelve un mensaje de bienvenida
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de películas"}

# Endpoint 1: Se ingresa un mes en idioma Español, devuelve la cantidad de películas que fueron estrenadas en el mes consultado
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):

   # Diccionario para mapear los meses en español a sus números correspondientes
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    # Convertir el mes ingresado a minúsculas
    mes = mes.lower()
    
    # Verificar si el mes ingresado es válido
    if mes not in meses:
        return "Mes ingresado no es válido. Por favor, ingrese un mes en español."
    
    # Obtener el número del mes correspondiente
    mes_numero = meses[mes]
    
    # Filtrar el DataFrame por el mes de estreno
    cantidad = df_movies[df_movies['release_date'].dt.month == mes_numero].shape[0]
    
    return f"{cantidad} películas fueron estrenadas en el mes de {mes.capitalize()}"

# Endpoint 2: Se ingresa un día en idioma Español, devuelve la cantidad de películas que fueron estrenadas en día consultado
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):

    # Diccionario para mapear los días en español a sus números correspondientes
    dias_semana = {
        'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 'viernes': 4, 'sabado': 5, 'domingo': 6
    }
    
    # Convertir el día ingresado a minúsculas
    dia = dia.lower()
    
    # Verificar si el día ingresado es válido
    if dia not in dias_semana:
        return "Día ingresado no es válido. Por favor, ingrese un día en español."
    
    # Obtener el número del día correspondiente
    dia_numero = dias_semana[dia]
    
    # Filtrar el DataFrame por el día de estreno
    cantidad = df_movies[df_movies['release_date'].dt.dayofweek == dia_numero].shape[0]
    
    return f"{cantidad} películas fueron estrenadas el día {dia.capitalize()}"

# Endpoint 3: Se ingresa el título de una filmación, devuelve el título, el año de estreno y el score
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):

   # Filtrar el DataFrame para encontrar la fila con el título especificado
    filmacion = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    
    # Verificar si se encontró la filmación
    if filmacion.empty:
        return f"No se encontró ninguna filmación con el título '{titulo}'"
    
    # Extraer la información de la filmación
    titulo = filmacion['title'].values[0]
    año_estreno = filmacion['release_year'].values[0]
    score = filmacion['popularity'].values[0]
    
    return f"La película '{titulo}' fue estrenada en el año {año_estreno} con un score de {score}"

# Endpoint 4: Se ingresa el título de una filmación, devuelve el título, la cantidad de votos y el valor promedio de las votaciones (No devuelve ningun valor si cuenta con menos de 2000 valoraciones) 
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):

    # Filtrar el DataFrame para encontrar la fila con el título especificado
    filmacion = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    
    # Verificar si se encontró la filmación
    if filmacion.empty:
        return f"No se encontró ninguna filmación con el título '{titulo}'"
    
    # Extraer la cantidad de votos
    cantidad_votos = filmacion['vote_count'].values[0]
    
    # Verificar si la cantidad de votos es al menos 2000
    if cantidad_votos < 2000:
        return f"La película '{titulo}' no cumple con la condición de tener al menos 2000 valoraciones."
    
    # Extraer el valor promedio de votaciones
    promedio_votos = filmacion['vote_average'].values[0]
    titulo = filmacion['title'].values[0]
    año_estreno = filmacion['release_year'].values[0]
    
    return f"La película '{titulo}' fue estrenada en el año {año_estreno}. La misma cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votos}"



# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
