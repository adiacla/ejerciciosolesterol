#pip install
#

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Título de la aplicación
st.title("Predicción de problemas cardiacos")

# Subtítulo
st.subheader("Realizado por Alfredo Diaz")

# Instrucciones de manejo y objetivo
st.write("""
Este modelo permite predecir el riesgo de problemas cardiacos basándose en dos variables: **Edad** y **Colesterol**.
El modelo fue entrenado utilizando el algoritmo KNN (K-Nearest Neighbors) y preprocesado con un escalador MinMaxScaler.
Proporcione los valores de las variables y haga clic en **Predecir** para conocer el resultado.
""")

# Mostrar la imagen proporcionada
st.image("https://images.ecestaticos.com/aTyeFebpQ-BqHJ7FIQjnjzcN2og=/334x4:1953x1213/1200x900/filters:fill(white):format(jpg)/f.elconfidencial.com%2Foriginal%2Fb9e%2Fd37%2F516%2Fb9ed3751689578efdbb19ed1b8b401e9.jpg")

# Definir los sliders para los datos de entrada
edad = st.slider("Edad", 20, 80, 50)
colesterol = st.slider("Colesterol", 100, 600, 200)

# Cargar el scaler y el modelo entrenado
scaler = joblib.load('scaler.bin')
knn_model = joblib.load('knn_model.bin')

# Crear el DataFrame para la predicción
data = pd.DataFrame({
    'edad': [edad],
    'colesterol': [colesterol]
})

# Escalar los datos utilizando el scaler cargado
data_scaled = scaler.transform(data)

# Realizar la predicción con el modelo KNN
prediccion = knn_model.predict(data_scaled)

# Mostrar el resultado
if st.button("Predecir"):
    if prediccion[0] == 0:
        st.markdown(f'<div style="background-color: green; padding: 10px; color: white; font-size: 20px;">No tiene riesgo de problemas cardiacos. Predicción: {prediccion[0]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: red; padding: 10px; color: white; font-size: 20px;">Tiene riesgo de problemas cardiacos. Predicción: {prediccion[0]}</div>', unsafe_allow_html=True)

# Traza una línea y coloca el símbolo de copyright
st.markdown("---")
st.write("© UNAB 2025")
