import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar el modelo KNN guardado con joblib
knn_model = joblib.load('knn_model.bin')
scaler = joblib.load('scaler.bin')

# Función para realizar la predicción
def predict(edad, colesterol):
    # Crear el dataframe con los datos ingresados
    input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})
    
    # Normalizar los datos usando el scaler cargado
    input_data_normalized = scaler.transform(input_data)
    
    # Realizar la predicción
    prediction = knn_model.predict(input_data_normalized)
    
    return prediction[0]

# Título de la aplicación
st.title("Modelo predicción problemas cardiacos con IA")

# Subtítulo con el nombre del autor
st.subheader("Realizado por Alfredo Diaz")

# Introducción a la aplicación
st.write("""
    Esta aplicación utiliza un modelo de inteligencia artificial basado en el algoritmo K-Nearest Neighbors (KNN) para predecir
    si una persona tiene riesgos de problemas cardíacos. Solo se requieren dos datos: la edad y el colesterol en sangre. 
    Esta herramienta puede ser útil para evaluar el riesgo y tomar decisiones preventivas.
""")

# Mostrar la imagen
st.image("https://images.emojiterra.com/google/noto-emoji/unicode-15/color/512px/1fac0.png", width=100)

# Inputs del usuario
edad = st.slider('Edad', min_value=20, max_value=80, value=30, step=1)
colesterol = st.slider('Colesterol (mg/dl)', min_value=100, max_value=600, value=200, step=1)

# Predicción
if st.button('Predecir'):
    # Realizar la predicción
    resultado = predict(edad, colesterol)
    
    # Mostrar el resultado con el fondo adecuado
    if resultado == 0:
        st.markdown(
            f"<div style='background-color:blue; color:white; padding:20px; border-radius:5px;'>"
            f"<h3>¡No tendrá problemas cardíacos! 😊</h3>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background-color:red; color:white; padding:20px; border-radius:5px;'>"
            f"<h3>¡Tiene riesgos de problemas cardíacos! ⚠️</h3>"
            f"</div>", unsafe_allow_html=True)
    
    # Dibujar una línea de separación
    st.markdown("---")
    
# Símbolo de copyright
st.markdown("<footer style='text-align:center;'>© Unab2025</footer>", unsafe_allow_html=True)
