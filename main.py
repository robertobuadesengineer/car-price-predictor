import pandas as pd
import joblib


# ============================================================
# CARGAR MODELO Y DATASET
# ============================================================

model = joblib.load("modelo_coches.pkl")
df = pd.read_csv("data/cars.csv")


print("\n====================================")
print("        🚗 AUTOVALUE AI")
print("====================================")
print("Predictor de precio de coches\n")


# ============================================================
# MARCAS
# ============================================================

marcas = sorted(
    df["manufacturer"]
    .dropna()
    .unique()
)

print("Marcas disponibles:\n")

for i, marca_opcion in enumerate(marcas, 1):
    print(f"{i}. {marca_opcion}")


marca = input(
    "\nIntroduce la marca: "
).strip()


# ============================================================
# MODELOS
# ============================================================

modelos = sorted(
    df[
        df["manufacturer"] == marca
    ]["model"]
    .dropna()
    .unique()
)

print("\nModelos disponibles:\n")

for i, modelo_opcion in enumerate(modelos, 1):
    print(f"{i}. {modelo_opcion}")


modelo = input(
    "\nIntroduce el modelo: "
).strip()


# ============================================================
# FILTRAR COCHES DE ESA MARCA Y MODELO
# ============================================================

coches_filtrados = df[
    (df["manufacturer"] == marca) &
    (df["model"] == modelo)
]


# ============================================================
# COMPROBAR QUE EXISTE EL COCHE
# ============================================================

if coches_filtrados.empty:

    print("\n❌ No se encontraron datos para ese coche.")

    raise SystemExit


# ============================================================
# MOTORES
# ============================================================

motores = sorted(
    coches_filtrados["engine"]
    .dropna()
    .unique()
)

print("\n====================================")
print("Motores disponibles")
print("====================================\n")

for i, motor_opcion in enumerate(motores, 1):
    print(f"{i}. {motor_opcion}")


motor = input(
    "\nEscribe el motor exactamente como aparece: "
).strip()


# ============================================================
# TRANSMISIONES
# ============================================================

transmisiones = sorted(
    coches_filtrados["transmission"]
    .dropna()
    .unique()
)

print("\n====================================")
print("Transmisiones disponibles")
print("====================================\n")

for i, transmision_opcion in enumerate(transmisiones, 1):
    print(f"{i}. {transmision_opcion}")


transmision = input(
    "\nEscribe la transmisión exactamente como aparece: "
).strip()


# ============================================================
# COMBUSTIBLES
# ============================================================

combustibles = sorted(
    coches_filtrados["fuel_type"]
    .dropna()
    .unique()
)

print("\n====================================")
print("Combustibles disponibles")
print("====================================\n")

for i, combustible_opcion in enumerate(combustibles, 1):
    print(f"{i}. {combustible_opcion}")


combustible = input(
    "\nEscribe el combustible exactamente como aparece: "
).strip()


# ============================================================
# AÑO
# ============================================================

while True:

    try:

        año = int(
            input("\nAño: ")
        )

        break

    except ValueError:

        print("❌ Introduce un año válido.")


# ============================================================
# KILOMETRAJE
# ============================================================

while True:

    try:

        kilometros = int(
            input("Kilometraje: ")
        )

        break

    except ValueError:

        print("❌ Introduce un kilometraje válido.")


# ============================================================
# CREAR VEHÍCULO
# ============================================================

coche = pd.DataFrame([{

    "manufacturer": marca,

    "model": modelo,

    "year": año,

    "mileage": kilometros,

    "engine": motor,

    "transmission": transmision,

    "drivetrain": None,

    "fuel_type": combustible,

    "mpg": None,

    "accidents_or_damage": None,

    "one_owner": None,

    "personal_use_only": None

}])


# ============================================================
# PREDICCIÓN
# ============================================================

print("\n====================================")
print("       🤖 CALCULANDO...")
print("====================================")


precio = model.predict(coche)[0]


# Evitar valores negativos

precio = max(
    0,
    precio
)


# ============================================================
# RESULTADO
# ============================================================

print("\n====================================")
print("          💰 VALORACIÓN")
print("====================================")

print(f"\nMarca:         {marca}")
print(f"Modelo:        {modelo}")
print(f"Año:           {año}")
print(f"Kilometraje:   {kilometros:,} km")
print(f"Motor:         {motor}")
print(f"Transmisión:   {transmision}")
print(f"Combustible:   {combustible}")

print("\n------------------------------------")

print(
    f"💰 PRECIO ESTIMADO: {precio:,.2f} €"
)

print("------------------------------------")


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

print("\n====================================")
print("        📊 MODELO UTILIZADO")
print("====================================")

print("Coches de entrenamiento: 100.000")
print("R²: 86.56%")
print("Error medio: 4.116,13 €")

print("\n====================================")
print("        🚗 AUTOVALUE AI")
print("====================================\n")