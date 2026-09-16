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
# CARGAR MODELO Y DATOS
# ==========================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_coches.pkl")


@st.cache_data
def cargar_datos():
    return pd.read_csv("data/cars.csv")


model = cargar_modelo()
df = cargar_datos()

# ==========================================
# CABECERA
# ==========================================

st.title("🚗 AutoValue AI")
st.subheader("Predicción inteligente del precio de coches")

st.write(
    "Introduce las características del vehículo y "
    "obtén una estimación de su precio."
)

st.divider()

# ==========================================
# SELECCIÓN DEL COCHE
# ==========================================

col1, col2 = st.columns(2)

with col1:

    marcas = sorted(
        df["manufacturer"]
        .dropna()
        .unique()
    )

    marca = st.selectbox(
        "🏭 Marca",
        marcas
    )

with col2:

    modelos = sorted(
        df[
            df["manufacturer"] == marca
        ]["model"]
        .dropna()
        .unique()
    )

    modelo = st.selectbox(
        "🚘 Modelo",
        modelos
    )

# ==========================================
# FILTRAR DATOS DEL COCHE
# ==========================================

coches_filtrados = df[
    (df["manufacturer"] == marca) &
    (df["model"] == modelo)
]

# ==========================================
# CARACTERÍSTICAS
# ==========================================

st.divider()

st.subheader("🔧 Características del vehículo")

col1, col2, col3 = st.columns(3)

with col1:

    motores = sorted(
        coches_filtrados["engine"]
        .dropna()
        .unique()
    )

    motor = st.selectbox(
        "⚙️ Motor",
        motores
    )

with col2:

    transmisiones = sorted(
        coches_filtrados["transmission"]
        .dropna()
        .unique()
    )

    transmision = st.selectbox(
        "⚙️ Transmisión",
        transmisiones
    )

with col3:

    combustibles = sorted(
        coches_filtrados["fuel_type"]
        .dropna()
        .unique()
    )

    combustible = st.selectbox(
        "⛽ Combustible",
        combustibles
    )

# ==========================================
# DATOS NUMÉRICOS
# ==========================================

col1, col2 = st.columns(2)

with col1:

    año = st.number_input(
        "📅 Año",
        min_value=1950,
        max_value=2026,
        value=2020,
        step=1
    )

with col2:

    kilometraje = st.number_input(
        "🛣️ Kilometraje",
        min_value=0,
        max_value=1000000,
        value=60000,
        step=1000
    )

# ==========================================
# DATOS ADICIONALES
# ==========================================

st.subheader("📋 Información adicional")

col1, col2 = st.columns(2)

with col1:

    drivetrains = sorted(
        coches_filtrados["drivetrain"]
        .dropna()
        .unique()
    )

    if len(drivetrains) > 0:
        drivetrain = st.selectbox(
            "🚙 Tracción",
            drivetrains
        )
    else:
        drivetrain = None

with col2:

    mpg_values = coches_filtrados["mpg"].dropna()

    mpg = None

    if len(mpg_values) > 0:
        mpg = st.number_input(
            "⛽ MPG",
            min_value=0.0,
            max_value=200.0,
            value=30.0,
            step=0.1
        )

# ==========================================
# HISTORIAL
# ==========================================

col1, col2 = st.columns(2)

with col1:

    accidentes = st.selectbox(
        "💥 ¿Tiene accidentes o daños?",
        ["No", "Yes"]
    )

with col2:

    propietario = st.selectbox(
        "👤 ¿Un solo propietario?",
        ["No", "Yes"]
    )

uso_personal = st.selectbox(
    "🚗 ¿Uso exclusivamente personal?",
    ["No", "Yes"]
)

# ==========================================
# PREDICCIÓN
# ==========================================

st.divider()

if st.button(
    "💰 CALCULAR PRECIO",
    use_container_width=True
):

    # --------------------------------------
    # Crear DataFrame con TODAS las columnas
    # que necesita el modelo
    # --------------------------------------

    coche = pd.DataFrame([{
        "manufacturer": marca,
        "model": modelo,
        "year": año,
        "mileage": kilometraje,
        "engine": motor,
        "transmission": transmision,
        "drivetrain": drivetrain,
        "fuel_type": combustible,
        "mpg": mpg,
        "accidents_or_damage": accidentes,
        "one_owner": propietario,
        "personal_use_only": uso_personal
    }])

    # --------------------------------------
    # Predicción
    # --------------------------------------

    try:

        precio = model.predict(coche)[0]

        precio = max(0, precio)

        st.success("✅ Predicción realizada correctamente")

        st.metric(
            "💰 PRECIO ESTIMADO",
            f"{precio:,.2f} €"
        )

        st.divider()

        st.subheader("📋 Datos utilizados")

        resultado = pd.DataFrame({
            "Característica": [
                "Marca",
                "Modelo",
                "Año",
                "Kilometraje",
                "Motor",
                "Transmisión",
                "Tracción",
                "Combustible"
            ],
            "Valor": [
                marca,
                modelo,
                año,
                f"{kilometraje:,} km",
                motor,
                transmision,
                drivetrain,
                combustible
            ]
        })

        st.dataframe(
            resultado,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"❌ Error al realizar la predicción: {e}"
        )

# ==========================================
# INFORMACIÓN DEL MODELO
# ==========================================

st.divider()

st.subheader("🤖 Información del modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🚗 Coches de entrenamiento",
        "100.000"
    )

with col2:
    st.metric(
        "📊 R²",
        "86,56 %"
    )

with col3:
    st.metric(
        "💶 Error medio",
        "4.116,13 €"
    )

st.caption(
    "AutoValue AI — Modelo de predicción basado en aprendizaje automático."
)