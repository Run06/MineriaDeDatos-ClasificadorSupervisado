import pandas as pd
import numpy as np
from pathlib import Path

for ruta in Path("resultados").glob("*/metricas_por_clase.csv"):

    df = pd.read_csv(ruta)

    datos = df[
        (df["particion"] == "Test") &
        (df["k_elegido_dev"] == True)
    ]

    print(f"\nModelo: {ruta.parent.name}")

    for estrategia, grupo in datos.groupby("estrategia"):

        pesos = grupo["soporte"]

        print(f"\nEstrategia: {estrategia}")
        print(f"k: {grupo['k'].iloc[0]}")

        for metrica in ["precision", "recall", "f1"]:
            valor = np.average(grupo[metrica], weights=pesos)
            print(f"{metrica}_weighted: {valor:.4f}")