# Importar librerias
from fastapi import FastAPI, HTTPException
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
@app.get('/score_titulo/{titulo}')
async def score_titulo(titulo: str):

    #Convierto el a minuscula para que la funcion distinga entre minusculas y mayusculas(case-insensitive)
    titulo = titulo.lower()

    #Busco la pelicula
    pelicula = df_movies[df_movies['title'].str.lower() == titulo]

    #Si no esta la pelicula, devuelve un error de tipo 404
    if pelicula.empty:
        raise HTTPException(status_code=404, detail=f"No se encontro la pelicula: {titulo}")
    
    if len(pelicula) > 1:
        pelicula = pelicula.iloc[0]
    else:
        pelicula = pelicula.iloc[0]

    #Extraigo el titulo, el año de lanzamiento y el puntaje
    titulo_original = pelicula['title']
    año_estreno = pelicula['release_year']
    score = pelicula['vote_average']

    return f"La pelicula {titulo_original} fue estrenada en el año {año_estreno} con un score/popularidad de {score}"

# Endpoint 4: Se ingresa el título de una filmación, devuelve el título, la cantidad de votos y el valor promedio de las votaciones (No devuelve ningun valor si cuenta con menos de 2000 valoraciones) 
@app.get('/votos_titulo/{titulo}')
async def votos_titulo(titulo: str):

    #Covierto el titulo en minuscula
    titulo = titulo.lower()

    #Busco la pelicula
    pelicula = df_movies[df_movies['title'].str.lower() == titulo]

    #Devuelvo un error si no encuentra la pelicula
    if pelicula.empty:
        raise HTTPException(status_code=404, detail=f"No se encontro la pelicula: {titulo}")

    if len(pelicula) > 1:
        pelicula = pelicula.iloc[0]
    else:
        pelicula = pelicula.iloc[0]
        
    #Extraigo el titulo,la cantidad de votos y valor promedio de las votaciones
    titulo_original = pelicula['title']
    año_estreno = pelicula['release_year']
    votos_totales = pelicula['vote_count']
    promedio_votos = pelicula['vote_average']

    #Verifico si la pelicula tiene al menos 2000 valoraciones
    if votos_totales < 2000:
        return f"La pelicula {titulo_original} no cumple con la condicion de contar al menos 2000 valoraciones. Cuenta con {votos_totales} valoraciones"

    return f"La pelicula {titulo_original} fue estrenada en el año {año_estreno}. La misma cuenta con un total de {votos_totales} valoraciones, con un promedio de {promedio_votos}"

# Endpoint 5: Se ingresa el nombre de un actor, devuelve el éxito del mismo medido a través del retorno, la cantidad de películas en las que ha participado y el promedio de retorno
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str) -> str:
    # Convertir el nombre del actor a minúsculas para la búsqueda
    nombre_actor = nombre_actor.lower()

    # Filtrar por actor en la lista y verificar que no es el director
    peliculas_actor = df_movies.loc[
        df_movies['actors'].apply(lambda actores: nombre_actor in [actor.lower() for actor in actores]) & 
        (df_movies['director'].str.lower() != nombre_actor)
    ]

    # Manejo de error si no hay películas con el actor solicitado
    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail=f"No se encontró el actor: {nombre_actor} o aparece solo como director")

    # Calcular cantidad de películas, retorno total y promedio de retorno
    cantidad_peliculas = len(peliculas_actor)
    retorno_total = peliculas_actor['return'].fillna(0).sum()
    promedio_retorno = retorno_total / cantidad_peliculas

    # Formatear y retornar la respuesta
    respuesta = (f"El actor {nombre_actor.title()} ha participado en {cantidad_peliculas} filmaciones, "
                 f"obteniendo un retorno total de {retorno_total:.2f} y un promedio de {promedio_retorno:.2f} por filmación.")
    return respuesta


# Endpoint 6: Se ingresa el nombre de un director, devuelve el éxito del mismo medido a través del retorno. Además, devuelve el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str) -> dict:
    # Convertir a minúscula para búsqueda
    nombre_director = nombre_director.lower()

    # Filtrar películas por director
    peliculas_director = df_movies[df_movies['director'].str.lower() == nombre_director]

    # Comprobación si no hay resultados
    if peliculas_director.empty:
        raise HTTPException(status_code=404, detail=f"No se encontró el director: {nombre_director}.")
    
    # Calcular el retorno total
    retorno_total = peliculas_director['return'].fillna(0).sum()

    # Preparar la información de las películas
    peliculas_info = []
    for _, pelicula in peliculas_director.iterrows():
        # Validación de datos individuales
        titulo = pelicula.get('title', 'Título desconocido')
        fecha = pelicula['release_date'] if pd.notnull(pelicula['release_date']) else "Fecha desconocida"
        retorno = pelicula.get('return', 0)
        costo = pelicula.get('budget', 0)
        ganancia = pelicula.get('revenue', 0) - costo if pd.notnull(pelicula.get('revenue')) else 0

        peliculas_info.append({
            "titulo": titulo,
            "fecha_lanzamiento": fecha.strftime('%Y-%m-%d') if isinstance(fecha, pd.Timestamp) else fecha,
            "retorno": round(retorno, 2),
            "costo": round(costo, 2),
            "ganancia": round(ganancia, 2)
        })


    respuesta = {
        "director": nombre_director.title(), 
        "retorno_total": round(retorno_total, 2), 
        "peliculas": peliculas_info
    }

    return respuesta


# Sistema de recomendacion: Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.

# Cargar el DataFrame de recomendaciones
df_recomendacion = pd.read_parquet('Datasets/df_recomendacion.parquet')

# Asegurarse de que 'features' sea una cadena unificada si es una lista
df_recomendacion['features'] = df_recomendacion['features'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Vectorización del texto usando TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_recomendacion['features'])

# Crear un índice para los títulos
indices = pd.Series(df_recomendacion.index, index=df_recomendacion['title']).drop_duplicates()

@app.get('/recomendacion_pelicula/{titulo}')
def recomendacion(titulo: str) -> list:

    # Convertir a minúscula para búsqueda
    titulo = titulo.lower()
    
    # Verificar si el título existe en el índice
    if titulo not in indices:
        raise ValueError("Película no encontrada")

    # Obtener el índice de la película en el DataFrame
    idx = indices[titulo]

    # Calcular la similitud del coseno entre la película seleccionada y todas las demás
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Obtener las películas más similares, ordenadas por similitud
    sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)

    # Seleccionar las 5 películas más similares, excluyendo la misma película
    recommended_indices = [i for i, score in sim_scores[1:6]]
    recommended_movies = df_recomendacion['title'].iloc[recommended_indices].tolist()

    return recommended_movies

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
