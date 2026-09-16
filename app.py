import streamlit as st
import pandas as pd
import joblib


# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="AutoValue AI",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# TÍTULO
# ==========================================

st.title("🚗 AutoValue AI")
st.subheader("Predicción inteligente del precio de coches")


# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_coches.pkl")


modelo = cargar_modelo()


# ==========================================
# OBTENER OPCIONES DEL MODELO
# ==========================================

categorical_features = [
    "manufacturer",
    "model",
    "engine",
    "transmission",
    "drivetrain",
    "fuel_type",
    "accidents_or_damage",
    "one_owner",
    "personal_use_only"
]


encoder = (
    modelo
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
)


opciones = dict(
    zip(
        categorical_features,
        encoder.categories_
    )
)


# ==========================================
# DATOS DEL COCHE
# ==========================================

st.markdown("### 🚘 Características del vehículo")

col1, col2, col3 = st.columns(3)


with col1:

    manufacturer = st.selectbox(
        "Fabricante",
        opciones["manufacturer"]
    )

    model = st.selectbox(
        "Modelo",
        opciones["model"]
    )

    year = st.number_input(
        "Año",
        min_value=1980,
        max_value=2026,
        value=2020,
        step=1
    )

    mileage = st.number_input(
        "Kilometraje (km)",
        min_value=0,
        max_value=1000000,
        value=60000,
        step=1000
    )


with col2:

    engine = st.selectbox(
        "Motor",
        opciones["engine"]
    )

    transmission = st.selectbox(
        "Transmisión",
        opciones["transmission"]
    )

    drivetrain = st.selectbox(
        "Tracción",
        opciones["drivetrain"]
    )

    fuel_type = st.selectbox(
        "Combustible",
        opciones["fuel_type"]
    )


with col3:

    mpg = st.number_input(
        "MPG",
        min_value=1.0,
        max_value=150.0,
        value=30.0,
        step=0.5
    )

    accidents_or_damage = st.selectbox(
        "Accidentes o daños",
        opciones["accidents_or_damage"]
    )

    one_owner = st.selectbox(
        "Un solo propietario",
        opciones["one_owner"]
    )

    personal_use_only = st.selectbox(
        "Solo uso personal",
        opciones["personal_use_only"]
    )


# ==========================================
# PREDICCIÓN
# ==========================================

st.markdown("---")

if st.button("💰 PREDECIR PRECIO", use_container_width=True):

    datos = pd.DataFrame([{
        "manufacturer": manufacturer,
        "model": model,
        "year": year,
        "mileage": mileage,
        "engine": engine,
        "transmission": transmission,
        "drivetrain": drivetrain,
        "fuel_type": fuel_type,
        "mpg": mpg,
        "accidents_or_damage": accidents_or_damage,
        "one_owner": one_owner,
        "personal_use_only": personal_use_only
    }])

    prediccion = modelo.predict(datos)[0]

    st.success(
        f"💰 PRECIO ESTIMADO: {prediccion:,.2f} €"
    )

    st.info(
        "La estimación ha sido generada mediante el modelo "
        "de Machine Learning entrenado con 100.000 vehículos."
    )


# ==========================================
# INFORMACIÓN DEL MODELO
# ==========================================

st.markdown("---")

st.markdown("### 📊 Información del modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Vehículos de entrenamiento",
        "100.000"
    )

with col2:
    st.metric(
        "R²",
        "86,56%"
    )

with col3:
    st.metric(
        "Error medio",
        "4.116 €"
    )


st.caption(
    "AutoValue AI — Predicción de precios de vehículos mediante Machine Learning"
)