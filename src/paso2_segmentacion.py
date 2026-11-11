"""Paso 2. Segmentación de clientes por comportamiento con K-means."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from comun import SALIDA, SEMILLA, preparada

COLUMNAS = ["uso_prom", "razon_pago", "mora_max", "meses_con_atraso", "limite_log", "tendencia_saldo"]


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    d = preparada()
    X = StandardScaler().fit_transform(d[COLUMNAS])
    rng = np.random.default_rng(SEMILLA)
    muestra = rng.choice(len(d), 8000, replace=False)

    candidatos = []
    for k in range(3, 7):
        etiquetas = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(X)
        candidatos.append((k, silhouette_score(X[muestra], etiquetas[muestra])))
    pd.DataFrame(candidatos, columns=["k", "silhouette"]).to_csv(SALIDA / "segmentacion_k.csv", index=False)
    k_elegido = max(candidatos, key=lambda c: c[1])[0]

    d["cluster"] = KMeans(n_clusters=k_elegido, n_init=10, random_state=SEMILLA).fit_predict(X)
    perfil = d.groupby("cluster").agg(
        clientes=("incumple", "size"),
        tasa_incumplimiento=("incumple", "mean"),
        exposicion=("exposicion", "sum"),
        uso_prom=("uso_prom", "mean"),
        razon_pago=("razon_pago", "mean"),
        mora_max=("mora_max", "mean"),
        meses_con_atraso=("meses_con_atraso", "mean"),
        limite_mediano=("limite", "median"),
        edad_media=("edad", "mean"),
    )
    perfil["pct_clientes"] = perfil["clientes"] / perfil["clientes"].sum()
    perfil["pct_exposicion"] = perfil["exposicion"] / perfil["exposicion"].sum()
    # Se numeran de menor a mayor incumplimiento para leerlos como niveles de riesgo.
    perfil = perfil.sort_values("tasa_incumplimiento")
    perfil.insert(0, "segmento", [f"S{i}" for i in range(1, len(perfil) + 1)])
    perfil.to_csv(SALIDA / "segmentacion_perfil.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot([c[0] for c in candidatos], [c[1] for c in candidatos], marker="o", color="#1F3864")
    ejes[0].set_xlabel("Número de segmentos")
    ejes[0].set_ylabel("Índice de silueta")
    ejes[0].set_title("Elección del número de segmentos")
    ejes[1].bar(perfil["segmento"], perfil["tasa_incumplimiento"] * 100, color="#1F3864")
    ejes[1].set_ylabel("Incumplimiento el mes siguiente (%)")
    ejes[1].set_title("Riesgo observado por segmento")
    fig.tight_layout()
    fig.savefig(SALIDA / "02_segmentacion.png", dpi=150)

    print(f"Segmentos elegidos: {k_elegido}")
    print(pd.DataFrame(candidatos, columns=["k", "silhouette"]).round(3).to_string(index=False))
    print(perfil.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
