# Archivo base para el despliegue del Agente en Streamlit
# Librerías
import streamlit as st 
from sklearn.linear_model import LinearRegression 
import numpy as np 

# En streamlit vamos a agregar un título para la página web
st.title("Configuracion inicial") 
# Agregaremos un textbox en nuestra página web
st.write("Primera prueba de uso de streamlit y ambiente de MA2026") 

# El slider de streamlit me permite ingresar por un slider el parámetro invresión
gasto=st.slider("Seleccione nivel de gasto en publicicdad", 10,200,50) 

# Variables de nuestro modelo
variable_x = np.array([[10], [20], [30], [40],[50]]) 
variable_y = np.array([15,25,35,45,55]) 
modelo_lr = LinearRegression()

# Entrenamiento d enuestro modelo LR
modelo_lr.fit(variable_x,variable_y) 

# En Streamlit, tenemos un botón que dice Predecir y al dar Click activará las líneas del código If
if st.button("Predecir"): 
    resultado = modelo_lr.predict([[gasto]]) 
    # Streamlit muestra un mensaje de éxito en verde bonito usando gasto
    st.success(f"Las ventas proyectadas para una inversion de ${gasto} son: ${resultado[0]}") 