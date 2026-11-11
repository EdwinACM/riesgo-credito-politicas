"""Paso 3. Modelo de probabilidad de incumplimiento (PD) con regresión logística,
validación, calibración y puntaje tipo scorecard."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comun import SALIDA, SEMILLA, VARIABLES_PD, preparada

# Escala del puntaje: 600 puntos con relación de 50 buenos por cada malo y
# 20 puntos por cada vez que esa relación se duplica.
PUNTOS_BASE, RELACION_BASE, PDO = 600, 50, 20


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def puntaje(pd_: np.ndarray) -> np.ndarray:
    pd_ = np.clip(pd_, 1e-6, 1 - 1e-6)
    factor = PDO / np.log(2)
    desplazamiento = PUNTOS_BASE - factor * np.log(RELACION_BASE)
    return desplazamiento + factor * np.log((1 - pd_) / pd_)


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    d = preparada()
    X, y = d[VARIABLES_PD], d["incumple"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEMILLA)

    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # PD fuera de muestra en entrenamiento: sirve para fijar cortes sin ver la prueba.
    pd_ent = cross_val_predict(modelo, X_ent, y_ent, cv=5, method="predict_proba")[:, 1]
    modelo.fit(X_ent, y_ent)
    pd_pru = modelo.predict_proba(X_pru)[:, 1]

    auc = roc_auc_score(y_pru, pd_pru)
    base_auc = roc_auc_score(y_pru, d.loc[X_pru.index, "mora_sep"])
    metricas = pd.DataFrame([
        {"medida": "AUC en prueba", "valor": auc},
        {"medida": "Gini en prueba", "valor": 2 * auc - 1},
        {"medida": "KS en prueba", "valor": ks(y_pru, pd_pru)},
        {"medida": "Brier en prueba", "valor": brier_score_loss(y_pru, pd_pru)},
        {"medida": "AUC solo con mora de septiembre", "valor": base_auc},
        {"medida": "AUC en validación cruzada (entrenamiento)", "valor": roc_auc_score(y_ent, pd_ent)},
        {"medida": "Incumplimiento observado en prueba", "valor": float(y_pru.mean())},
        {"medida": "PD promedio predicha en prueba", "valor": float(pd_pru.mean())},
    ])
    metricas.to_csv(SALIDA / "modelo_metricas.csv", index=False)

    coef = modelo.named_steps["logisticregression"].coef_[0]
    tabla = pd.DataFrame({
        "variable": VARIABLES_PD,
        "coeficiente_estandarizado": coef,
        "razon_de_momios_por_desv_est": np.exp(coef),
    }).sort_values("coeficiente_estandarizado", ascending=False)
    tabla.to_csv(SALIDA / "modelo_coeficientes.csv", index=False)

    # Calibración por deciles de PD en prueba
    deciles = pd.DataFrame({"pd": pd_pru, "y": y_pru.values})
    deciles["decil"] = pd.qcut(deciles["pd"], 10, labels=False, duplicates="drop") + 1
    cal = deciles.groupby("decil").agg(clientes=("y", "size"), pd_predicha=("pd", "mean"),
                                       incumplimiento_observado=("y", "mean")).reset_index()
    cal.to_csv(SALIDA / "modelo_calibracion.csv", index=False)

    # Predicciones para el paso 4
    ent = d.loc[X_ent.index, ["ID", "incumple", "mora_sep", "uso_sep", "exposicion"]].assign(pd=pd_ent)
    pru = d.loc[X_pru.index, ["ID", "incumple", "mora_sep", "uso_sep", "exposicion"]].assign(pd=pd_pru)
    pru["puntaje"] = puntaje(pd_pru)
    ent.to_csv(SALIDA / "predicciones_entrenamiento.csv", index=False)
    pru.to_csv(SALIDA / "predicciones_prueba.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    fpr, tpr, _ = roc_curve(y_pru, pd_pru)
    ejes[0].plot(fpr, tpr, color="#1F3864", label=f"Modelo (AUC {auc:.3f})")
    ejes[0].plot([0, 1], [0, 1], "--", color="gray")
    ejes[0].set_xlabel("Falsos positivos")
    ejes[0].set_ylabel("Verdaderos positivos")
    ejes[0].set_title("Curva ROC en la muestra de prueba")
    ejes[0].legend()
    ejes[1].plot(cal["pd_predicha"] * 100, cal["incumplimiento_observado"] * 100, marker="o", color="#1F3864")
    tope = max(cal["pd_predicha"].max(), cal["incumplimiento_observado"].max()) * 100
    ejes[1].plot([0, tope], [0, tope], "--", color="gray")
    ejes[1].set_xlabel("PD predicha (%)")
    ejes[1].set_ylabel("Incumplimiento observado (%)")
    ejes[1].set_title("Calibración por deciles")
    fig.tight_layout()
    fig.savefig(SALIDA / "03_modelo_pd.png", dpi=150)

    print(metricas.round(4).to_string(index=False))
    print(tabla.round(3).to_string(index=False))
    print(cal.round(3).to_string(index=False))
    print("Puntaje en prueba, percentiles 5/50/95:", np.percentile(pru["puntaje"], [5, 50, 95]).round(0))


if __name__ == "__main__":
    main()
