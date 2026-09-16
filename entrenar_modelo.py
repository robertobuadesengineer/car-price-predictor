import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


print("====================================")
print("AUTOVALUE AI - ENTRENAMIENTO RÁPIDO")
print("====================================")


# ============================================================
# 1. CARGAR DATOS
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv("data/cars.csv")

print(f"Dataset original: {len(df):,} coches")


# ============================================================
# 2. LIMPIAR PRECIOS
# ============================================================

df = df[
    (df["price"] >= 1000) &
    (df["price"] <= 200000)
].copy()

print(f"Coches después de limpiar precios: {len(df):,}")


# ============================================================
# 3. LIMPIAR MPG
# ============================================================

def convertir_mpg(valor):

    if pd.isna(valor):
        return None

    numeros = re.findall(
        r"\d+(?:\.\d+)?",
        str(valor)
    )

    if not numeros:
        return None

    numeros = [float(x) for x in numeros]

    if len(numeros) >= 2:
        return (numeros[0] + numeros[1]) / 2

    return numeros[0]


df["mpg"] = df["mpg"].apply(convertir_mpg)


# ============================================================
# 4. USAR SOLO 100.000 COCHES
# ============================================================

df = df.sample(
    n=min(100000, len(df)),
    random_state=42
)

print(f"Coches utilizados: {len(df):,}")


# ============================================================
# 5. VARIABLES
# ============================================================

features = [
    "manufacturer",
    "model",
    "year",
    "mileage",
    "engine",
    "transmission",
    "drivetrain",
    "fuel_type",
    "mpg",
    "accidents_or_damage",
    "one_owner",
    "personal_use_only"
]

X = df[features]

y = df["price"]


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

numeric_features = [
    "year",
    "mileage",
    "mpg"
]


# ============================================================
# 6. PREPROCESAMIENTO
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10
        )
    )
])


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


preprocessor = ColumnTransformer([
    (
        "cat",
        categorical_pipeline,
        categorical_features
    ),
    (
        "num",
        numeric_pipeline,
        numeric_features
    )
])


# ============================================================
# 7. RANDOM FOREST MÁS LIGERO
# ============================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=40,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# 8. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 9. ENTRENAR
# ============================================================

print("\n====================================")
print("ENTRENANDO...")
print("====================================\n")

model.fit(
    X_train,
    y_train
)


# ============================================================
# 10. EVALUAR
# ============================================================

print("\nCalculando resultados...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n====================================")
print("RESULTADOS")
print("====================================")

print(f"Error medio: {mae:,.2f} €")
print(f"R²: {r2:.4f}")
print(f"R² porcentaje: {r2 * 100:.2f}%")


# ============================================================
# 11. GUARDAR
# ============================================================

joblib.dump(
    model,
    "modelo_coches.pkl"
)

print("\n====================================")
print("MODELO GUARDADO")
print("====================================")

print("modelo_coches.pkl")