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
# CARGAR MODELO Y OPCIONES
# ==========================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_coches.pkl")


@st.cache_data
def cargar_opciones():
    return pd.read_csv("data/opciones.csv")


modelo = cargar_modelo()
opciones = cargar_opciones()


# ==========================================
# TÍTULO
# ==========================================

st.title("🚗 AutoValue AI")
st.subheader("Predicción inteligente del precio de coches")


# ==========================================
# OPCIONES DEL MODELO
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

categorias_modelo = dict(
    zip(
        categorical_features,
        encoder.categories_
    )
)


# ==========================================
# SELECCIÓN DEL VEHÍCULO
# ==========================================

st.markdown("### 🚘 Características del vehículo")

col1, col2, col3 = st.columns(3)


# ==========================================
# FABRICANTE
# ==========================================

fabricantes = sorted(
    opciones["manufacturer"]
    .dropna()
    .unique()
)

with col1:

    manufacturer = st.selectbox(
        "Fabricante",
        fabricantes
    )


# ==========================================
# MODELO
# ==========================================

modelos = sorted(
    opciones[
        opciones["manufacturer"] == manufacturer
    ]["model"]
    .dropna()
    .unique()
)

with col1:

    model = st.selectbox(
        "Modelo",
        modelos
    )


# ==========================================
# FILTRAR FABRICANTE + MODELO
# ==========================================

config = opciones[
    (opciones["manufacturer"] == manufacturer) &
    (opciones["model"] == model)
].copy()


# ==========================================
# MOTOR
# ==========================================

motores = sorted(
    config["engine"]
    .dropna()
    .unique()
)

with col2:

    engine = st.selectbox(
        "Motor",
        motores
    )


# ==========================================
# FILTRAR MOTOR
# ==========================================

config_motor = config[
    config["engine"] == engine
]


# ==========================================
# TRANSMISIÓN
# ==========================================

transmisiones = sorted(
    config_motor["transmission"]
    .dropna()
    .unique()
)

with col2:

    transmission = st.selectbox(
        "Transmisión",
        transmisiones
    )


# ==========================================
# FILTRAR TRANSMISIÓN
# ==========================================

config_trans = config_motor[
    config_motor["transmission"] == transmission
]


# ==========================================
# TRACCIÓN
# ==========================================

tracciones = sorted(
    config_trans["drivetrain"]
    .dropna()
    .unique()
)

with col2:

    drivetrain = st.selectbox(
        "Tracción",
        tracciones
    )


# ==========================================
# FILTRAR TRACCIÓN
# ==========================================

config_drive = config_trans[
    config_trans["drivetrain"] == drivetrain
]


# ==========================================
# COMBUSTIBLE
# ==========================================

combustibles = sorted(
    config_drive["fuel_type"]
    .dropna()
    .unique()
)

with col3:

    fuel_type = st.selectbox(
        "Combustible",
        combustibles
    )


# ==========================================
# AÑO
# ==========================================

with col1:

    year = st.number_input(
        "Año",
        min_value=1980,
        max_value=2026,
        value=2020,
        step=1
    )


# ==========================================
# KILOMETRAJE
# ==========================================

with col1:

    mileage = st.number_input(
        "Kilometraje (km)",
        min_value=0,
        max_value=1000000,
        value=60000,
        step=1000
    )


# ==========================================
# MPG
# ==========================================

with col2:

    mpg = st.number_input(
        "MPG",
        min_value=1.0,
        max_value=150.0,
        value=30.0,
        step=0.5
    )


# ==========================================
# ACCIDENTES / DAÑOS
# ==========================================

accidentes = sorted(
    [str(x) for x in categorias_modelo["accidents_or_damage"]]
)

with col3:

    accidentes_or_damage = st.selectbox(
        "Accidentes o daños",
        accidentes
    )


# ==========================================
# UN SOLO PROPIETARIO
# ==========================================

propietarios = sorted(
    [str(x) for x in categorias_modelo["one_owner"]]
)

with col3:

    one_owner = st.selectbox(
        "Un solo propietario",
        propietarios
    )


# ==========================================
# USO PERSONAL
# ==========================================

uso_personal = sorted(
    [str(x) for x in categorias_modelo["personal_use_only"]]
)

with col3:

    personal_use_only = st.selectbox(
        "Solo uso personal",
        uso_personal
    )


# ==========================================
# PREDICCIÓN
# ==========================================

st.markdown("---")

if st.button(
    "💰 PREDECIR PRECIO",
    use_container_width=True
):

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
        "accidents_or_damage": accidentes_or_damage,
        "one_owner": one_owner,
        "personal_use_only": personal_use_only
    }])

    prediccion = modelo.predict(datos)[0]

    st.success(
        f"💰 PRECIO ESTIMADO: {prediccion:,.2f} €"
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