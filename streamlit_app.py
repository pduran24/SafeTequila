import streamlit as st

from oraculo.crear_datos import CrearDatos
from oraculo.modelo import ModeloOraculo


st.image ("https://imgproxy.attic.sh/insecure/f:png/plain/https://attic.sh/d7soo271ta5vqrjszhp9tfn30y89")
st.title(" SAFE TEQUILA ")
st.markdown("Consultar el estado de Tequila y su probabilidad de estar perdida")


@st.cache_data
def entrenar_oraculo():
    generador = CrearDatos(seed=42, num_ejemplos=1000)
    datos = generador.generar()
    modelo = ModeloOraculo()
    resultados = modelo.entrenar(datos)
    return modelo, resultados

modelo, resultados = entrenar_oraculo()


with st.expander("Ver precisión del Oráculo"):
    st.write(f"**Precisión**: {resultados['precision']:.2f}")
    st.write("**Matriz de confusión**:")
    st.dataframe(resultados['confusion_matrix'])

st.subheader("Introduce el destino de Tequila")
temperatura = st.slider("Temperatura: ", min_value=-20, max_value=50, value=10, step=1)
hora = st.slider("Hora del día: ", min_value=0, max_value=23, value=12, step=1)
niebla = st.selectbox("¿Hay niebla?", options=["No", "Sí"])

niebla_binaria = 1 if niebla == "Sí" else 0

if st.button("Consultar al Oráculo"):
    pred = modelo.predecir(temperatura, niebla_binaria, hora)
    if pred == 1:
        st.error("⚠️ Tequila se encuentra perdida!!")
        st.image("images/Tequila.jpeg")
    else:
        st.success("El camino es seguro, Tequila puede avanzar sin miedo.")
        st.image("images/Tequila.jpeg")