#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def cargar_y_preprocesar(ruta):
    """Carga y preprocesa uno de los datasets."""

    dataframe = pd.read_csv(
        ruta,
        compression="zip",
    )

    # Crear la edad del vehículo.
    dataframe["Age"] = 2021 - dataframe["Year"]

    # Eliminar las columnas indicadas.
    dataframe = dataframe.drop(
        columns=[
            "Year",
            "Car_Name",
        ]
    )

    return dataframe


def calcular_metricas(modelo, x_data, y_data, dataset):
    """Calcula las métricas solicitadas."""

    predicciones = modelo.predict(x_data)

    return {
        "type": "metrics",
        "dataset": dataset,
        "r2": float(r2_score(y_data, predicciones)),
        "mse": float(mean_squared_error(y_data, predicciones)),
        "mad": float(
            median_absolute_error(
                y_data,
                predicciones,
            )
        ),
    }


def main():
    """Entrena el modelo y genera los archivos solicitados."""

    # -------------------------------------------------------------------------
    # Cargar y preprocesar los datasets
    # -------------------------------------------------------------------------
    train_data = cargar_y_preprocesar("files/input/train_data.csv.zip")

    test_data = cargar_y_preprocesar("files/input/test_data.csv.zip")

    # -------------------------------------------------------------------------
    # Separar variables predictoras y variable objetivo
    # -------------------------------------------------------------------------
    x_train = train_data.drop(columns=["Present_Price"])
    y_train = train_data["Present_Price"]

    x_test = test_data.drop(columns=["Present_Price"])
    y_test = test_data["Present_Price"]

    # -------------------------------------------------------------------------
    # Definir variables categóricas y numéricas
    # -------------------------------------------------------------------------
    variables_categoricas = [
        "Fuel_Type",
        "Selling_type",
        "Transmission",
    ]

    variables_numericas = [
        "Selling_Price",
        "Driven_kms",
        "Owner",
        "Age",
    ]

    # -------------------------------------------------------------------------
    # Preprocesamiento
    # -------------------------------------------------------------------------
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                variables_categoricas,
            ),
            (
                "numerical",
                MinMaxScaler(),
                variables_numericas,
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocesador,
            ),
            (
                "feature_selection",
                SelectKBest(score_func=f_regression),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    # Después del OneHotEncoder se generan 11 variables.
    parametros = {
        "feature_selection__k": range(1, 12),
        "regressor__fit_intercept": [
            True,
            False,
        ],
    }

    # -------------------------------------------------------------------------
    # Optimización mediante validación cruzada
    # -------------------------------------------------------------------------
    modelo = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )

    modelo.fit(
        x_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Crear carpetas de salida
    # -------------------------------------------------------------------------
    os.makedirs(
        "files/models",
        exist_ok=True,
    )

    os.makedirs(
        "files/output",
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Guardar el modelo comprimido
    # -------------------------------------------------------------------------
    with gzip.open(
        "files/models/model.pkl.gz",
        "wb",
    ) as file:
        pickle.dump(
            modelo,
            file,
        )

    # -------------------------------------------------------------------------
    # Calcular métricas
    # -------------------------------------------------------------------------
    metricas = [
        calcular_metricas(
            modelo=modelo,
            x_data=x_train,
            y_data=y_train,
            dataset="train",
        ),
        calcular_metricas(
            modelo=modelo,
            x_data=x_test,
            y_data=y_test,
            dataset="test",
        ),
    ]

    # -------------------------------------------------------------------------
    # Guardar una métrica por línea en formato JSON
    # -------------------------------------------------------------------------
    with open(
        "files/output/metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        for metrica in metricas:
            file.write(json.dumps(metrica) + "\n")


if __name__ == "__main__":
    main()