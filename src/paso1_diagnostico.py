"""Paso 1. Diagnóstico de la cartera: incumplimiento por segmento, concentración
y matrices de transición de mora (roll rates)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import MESES, SALIDA, preparada


def tabla_segmento(d: pd.DataFrame, columna: str, nombre: str) -> pd.DataFrame:
    g = d.groupby(columna, observed=True).agg(
        clientes=("incumple", "size"),
        incumplen=("incumple", "sum"),
        exposicion=("exposicion", "sum"),
    )
    g["tasa_incumplimiento"] = g["incumplen"] / g["clientes"]
    g["pct_clientes"] = g["clientes"] / g["clientes"].sum()
    g["pct_exposicion"] = g["exposicion"] / g["exposicion"].sum()
    g["pct_incumplimientos"] = g["incumplen"] / g["incumplen"].sum()
    g.insert(0, "variable", nombre)
    return g.reset_index().rename(columns={columna: "segmento"})


def estado_mora(serie: pd.Series) -> pd.Series:
    """0 al corriente, 1 de uno a dos meses de atraso, 2 tres o más.

    De abril a agosto la base casi no registra "1 mes" (pasa de 0 a 2), pero en
    septiembre sí. Por eso se agrupan uno y dos meses: así los estados significan
    lo mismo en todos los meses."""
    return pd.cut(serie, [-10, 0, 2, 10], labels=[0, 1, 2]).astype(int)


def roll_rates(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cronologico = MESES[::-1]  # abril a septiembre
    pares = []
    for antes, despues in zip(cronologico[:-1], cronologico[1:]):
        pares.append(pd.DataFrame({
            "desde": estado_mora(d[f"mora_{antes}"]),
            "hacia": estado_mora(d[f"mora_{despues}"]),
        }))
    t = pd.concat(pares)
    etiquetas = {0: "al corriente", 1: "1 a 2 meses", 2: "3 o más"}
    conteo = pd.crosstab(t["desde"], t["hacia"]).rename(index=etiquetas, columns=etiquetas)
    tasas = pd.crosstab(t["desde"], t["hacia"], normalize="index").rename(index=etiquetas, columns=etiquetas)
    return tasas, conteo


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    d = preparada()

    d["banda_limite"] = pd.qcut(d["limite"], 5, duplicates="drop")
    d["banda_limite"] = d["banda_limite"].apply(lambda x: f"{int(x.left):,} a {int(x.right):,}")
    d["banda_uso"] = pd.cut(d["uso_sep"], [-1, 0.2, 0.5, 0.8, 1.0, 3],
                            labels=["hasta 20%", "20 a 50%", "50 a 80%", "80 a 100%", "más de 100%"])
    d["banda_edad"] = pd.cut(d["edad"], [20, 25, 30, 40, 50, 80], right=False,
                             labels=["21 a 24", "25 a 29", "30 a 39", "40 a 49", "50 o más"])
    d["situacion_sep"] = d["mora_sep"].map(
        lambda v: "sin consumo" if v == -2 else "pago completo" if v == -1
        else "revolvente al corriente" if v == 0 else f"{v} mes(es) de atraso" if v < 3 else "3 o más meses")

    tablas = [
        tabla_segmento(d, "banda_limite", "límite de crédito"),
        tabla_segmento(d, "banda_uso", "uso de la línea en septiembre"),
        tabla_segmento(d, "situacion_sep", "situación de pago en septiembre"),
        tabla_segmento(d, "banda_edad", "edad (solo descriptivo)"),
        tabla_segmento(d, "educacion", "educación (solo descriptivo)"),
    ]
    seg = pd.concat(tablas, ignore_index=True)
    seg.to_csv(SALIDA / "diagnostico_segmentos.csv", index=False)

    rr, rr_n = roll_rates(d)
    rr.to_csv(SALIDA / "roll_rates.csv")
    rr_n.to_csv(SALIDA / "roll_rates_conteo.csv")

    # Gráficas
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    s = seg[seg["variable"] == "situación de pago en septiembre"].sort_values("tasa_incumplimiento")
    ejes[0].barh(s["segmento"], s["tasa_incumplimiento"] * 100, color="#1F3864")
    ejes[0].set_xlabel("Incumplimiento el mes siguiente (%)")
    ejes[0].set_title("Incumplimiento según situación de pago")
    im = ejes[1].imshow(rr.values * 100, cmap="Blues", vmin=0, vmax=100)
    ejes[1].set_xticks(range(3), rr.columns, rotation=20)
    ejes[1].set_yticks(range(3), rr.index)
    ejes[1].set_xlabel("Estado al mes siguiente")
    ejes[1].set_ylabel("Estado de partida")
    ejes[1].set_title("Matriz de transición de mora (%)")
    for i in range(3):
        for j in range(3):
            ejes[1].text(j, i, f"{rr.values[i, j] * 100:.0f}", ha="center", va="center",
                         color="white" if rr.values[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ejes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(SALIDA / "01_diagnostico.png", dpi=150)

    print(f"Clientes: {len(d):,}  Incumplimiento: {d['incumple'].mean():.1%}")
    print(seg[seg["variable"] == "situación de pago en septiembre"][
        ["segmento", "clientes", "tasa_incumplimiento", "pct_exposicion"]].round(3).to_string(index=False))
    print((rr * 100).round(1))
    print(rr_n)


if __name__ == "__main__":
    main()
