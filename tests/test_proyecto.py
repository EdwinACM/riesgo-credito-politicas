"""Pruebas básicas de integridad. Se corren con: python -m pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from comun import MORAS, VARIABLES_PD, cargar, preparada
from paso3_modelo_pd import puntaje


def test_base_completa_y_sin_nulos():
    df = cargar()
    assert len(df) == 30000
    assert df["ID"].is_unique
    assert df.isna().sum().sum() == 0
    assert set(df["incumple"].unique()) == {0, 1}


def test_categorias_corregidas():
    df = cargar()
    assert df["educacion"].isin([1, 2, 3, 4]).all()
    assert df["estado_civil"].isin([1, 2, 3]).all()


def test_variables_sin_nulos_y_sin_variables_sensibles():
    d = preparada()
    assert d[VARIABLES_PD].isna().sum().sum() == 0
    for prohibida in ["sexo", "estado_civil", "educacion", "edad"]:
        assert prohibida not in VARIABLES_PD


def test_meses_con_atraso_consistente():
    d = preparada()
    esperado = (d[MORAS] >= 1).sum(axis=1)
    assert (d["meses_con_atraso"] == esperado).all()


def test_puntaje_baja_cuando_sube_la_pd():
    pds = np.array([0.02, 0.10, 0.30, 0.60])
    assert (np.diff(puntaje(pds)) < 0).all()


def test_puntaje_en_la_base_de_la_escala():
    # 1 malo por cada 50 buenos equivale a 600 puntos
    assert abs(float(puntaje(np.array([1 / 51]))[0]) - 600) < 0.5
