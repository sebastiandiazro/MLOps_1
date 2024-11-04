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

# Endpoint 5: Se ingresa el nombre de un actor, devuelve el éxito del mismo medido a través del retorno, la cantidad de películas en las que ha participado y el promedio de retorno
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):

    #Convierto a minuscula
    nombre_actor = nombre_actor.lower()

    #filtro las peliculas donde aparece el actor y no es director
    peliculas_actor = df_movies[
        (df_movies['actors'].apply(lambda x: nombre_actor in [actor.lower() for actor in x])) & 
        (df_movies['director'].str.lower() != nombre_actor)
    ]

    #Si no hay películas del actor, devuelvo un error
    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail=f"No se encontró el actor: {nombre_actor} o aparece como director")

    #Calculos requeridos
    cantidad_peliculas = len(peliculas_actor)
    retorno = peliculas_actor['return'].sum()
    promedio_retorno = retorno / cantidad_peliculas

    return f"El actor {nombre_actor.title()} ha participado de {cantidad_peliculas} filmaciones, el mismo ha conseguido un retorno de {retorno:.2f} con un promedio de {promedio_retorno:.2f} por filmacion"

# Endpoint 6: Se ingresa el nombre de un director, devuelve el éxito del mismo medido a través del retorno. Además, devuelve el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str) -> dict:

    #Convierto a minuscula
    nombre_director = nombre_director.lower()

    #Filtro las peliculas por director
    peliculas_director = df_movies[df_movies['director'].str.lower() == nombre_director]

    #Si no encuentra las peliculas de acuerdo al director, devuelve error
    if peliculas_director.empty:
        raise HTTPException(status_code=404, detail=f"No se encontro el director: {nombre_director}.")
    
    #Calculo el exito total del director a traves del retorno
    retorno = peliculas_director['return'].sum()

    #Preparo los detalles que necesito de las peliculas
    peliculas_info = []
    for _, pelicula in peliculas_director.iterrows():
        peliculas_info.append({
            "titulo": pelicula['title'],
            "fecha_lanzamiento": pelicula['release_date'].strftime('%Y-%m-%d'),
            "retorno": round(pelicula['return'], 2),
            "costo": round(pelicula['budget'], 2),
            "ganancia": round(pelicula['revenue'] - pelicula['budget'], 2)
        })
    

    respuesta = {
        "director": nombre_director.title(), 
        "retorno": round(retorno, 2), 
        "peliculas": peliculas_info
    }

    return respuesta

# Sistema de recomendacion: Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.

df_recomendacion = pd.read_parquet('Datasets/df_recomendacion.parquet')

# Asegúrate de que cada entrada en 'features' sea un string
df_recomendacion['features'] = df_recomendacion['features'].astype(str)

# Pasa la columna como lista de strings
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_recomendacion['features'].tolist())

# Vectorización del texto usando TF-IDF
tfidf_matrix = vectorizer.fit_transform(df_recomendacion['features'])

#Obtengo el índice del título ingresado.
indices = pd.Series(df_recomendacion.index, index=df_recomendacion['title']).drop_duplicates()

@app.get('/recomendacion_pelicula/{titulo}')
async def recomendacion_pelicula(titulo: str):

    #Verificar si el titulo se encuentra en los datos
    if titulo not in indices:
        #Si no lo encuentra devuelvo un error
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    #Obtener el indice del título ingresado.
    idx = indices[titulo]
    
    #Calcular la similitud del coseno
    cosine_sim = cosine_similarity(tfidf_matriz[idx:idx+1], tfidf_matriz).flatten()

    #Guardar los scores de similitud en una lista de tuplas, donde el primer elemento es el índice y el segundo es el score.
    sim_scores = list(enumerate(cosine_sim))
    #Ordenar la lista de mayor a menor.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


    #Obtener las 6 películas más similares (incluyendo la misma pelicula)
    sim_scores = sim_scores[:6]

    #Excluir la primera película si es la misma que se ingresó
    if sim_scores[0][0] == idx:
        sim_scores = sim_scores[1:6]
    else:
        sim_scores = sim_scores[:5]

    #Obtener los títulos de las películas recomendadas y los convierto en lista.
    movies_indices = [i[0] for i in sim_scores]
    recommended_movies = df_recomendacion['title'].iloc[movies_indices].tolist()
    

    #Devolver las películas recomendadas.
    return recommended_movies


# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
