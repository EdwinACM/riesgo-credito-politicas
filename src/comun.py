"""Carga, limpieza y variables compartidas por todos los pasos del proyecto."""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos" / "credito_tarjetas.csv"
SALIDA = RAIZ / "resultados"
SEMILLA = 42

# Del mes más reciente al más antiguo. En la base original PAY_0 es septiembre
# y PAY_2 a PAY_6 son agosto a abril (no existe PAY_1).
MESES = ["sep", "ago", "jul", "jun", "may", "abr"]
MORAS = [f"mora_{m}" for m in MESES]

# Variables de comportamiento de la cuenta que usa el modelo.
# Sexo, estado civil, educación y edad quedan fuera a propósito: no deben
# decidir una política de crédito.
# Se compararon cinco conjuntos solo con validación cruzada del entrenamiento.
# Este tiene casi la misma capacidad de ordenar riesgo que el conjunto completo
# (AUC 0.750 contra 0.751) y la mitad de variables.
VARIABLES_PD = ["limite_log", "mora_sep", "meses_con_atraso", "uso_sep", "razon_pago"]


def cargar() -> pd.DataFrame:
    """Lee la base, renombra columnas y corrige categorías sin significado."""
    df = pd.read_csv(DATOS)
    nombres = {
        "default": "incumple", "LIMIT_BAL": "limite", "AGE": "edad",
        "SEX": "sexo", "EDUCATION": "educacion", "MARRIAGE": "estado_civil",
    }
    for original, mes in zip(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], MESES):
        nombres[original] = f"mora_{mes}"
    for i, mes in enumerate(MESES, start=1):
        nombres[f"BILL_AMT{i}"] = f"saldo_{mes}"
        nombres[f"PAY_AMT{i}"] = f"pago_{mes}"
    df = df.rename(columns=nombres)
    # La documentación solo define 1 a 4 en educación y 1 a 3 en estado civil.
    df["educacion"] = df["educacion"].replace({0: 4, 5: 4, 6: 4})
    df["estado_civil"] = df["estado_civil"].replace({0: 3})
    return df


def variables(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables de comportamiento que se usan en el análisis."""
    d = df.copy()
    saldos = [f"saldo_{m}" for m in MESES]
    d["mora_max"] = d[MORAS].max(axis=1)
    d["meses_con_atraso"] = (d[MORAS] >= 1).sum(axis=1)
    d["uso_sep"] = (d["saldo_sep"] / d["limite"]).clip(-0.5, 2)
    d["uso_prom"] = (d[saldos].mean(axis=1) / d["limite"]).clip(-0.5, 2)
    # Lo pagado en un mes liquida el saldo del mes anterior.
    facturado = sum(d[f"saldo_{m}"].clip(lower=0) for m in MESES[1:])
    pagado = sum(d[f"pago_{m}"] for m in MESES[:-1])
    d["razon_pago"] = (pagado / facturado.where(facturado > 0)).clip(0, 2).fillna(1.0)
    d["tendencia_saldo"] = ((d["saldo_sep"] - d["saldo_abr"]) / d["limite"]).clip(-1, 1)
    d["limite_log"] = np.log(d["limite"])
    d["exposicion"] = d["saldo_sep"].clip(lower=0)
    return d


def preparada() -> pd.DataFrame:
    return variables(cargar())
