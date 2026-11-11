"""Paso 4. Del modelo a la política: matriz de decisión, simulación
champion/challenger y precio basado en riesgo.

La base no trae resultados de acciones reales, así que el valor de cada acción
se calcula con los supuestos de abajo. Son ilustrativos y se pueden cambiar."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import SALIDA, SEMILLA

# Supuestos explícitos (montos en la misma moneda de la base, dólares de Taiwán)
SUPUESTOS = {
    "LGD": 0.70,                      # parte del saldo que no se recupera si incumple
    "costo_intervencion": 150.0,      # por cliente: contacto, revisión de línea
    "efectividad": 0.30,              # parte de la pérdida que se evita al intervenir
    "fondeo_mensual": 0.007,          # costo de fondeo, cerca de 8.4% anual
    "costo_operativo_mensual": 0.005,
    "remuestreos_bootstrap": 2000,
}
BANDAS = [0, 0.08, 0.20, 0.40, 1.0001]
NOMBRES_BANDA = ["menos de 8%", "8 a 20%", "20 a 40%", "40% o más"]


def valor(df: pd.DataFrame, interviene: pd.Series, efectividad=None, costo=None) -> pd.Series:
    """Valor neto de cada cliente bajo una política: pérdida evitada menos costo."""
    efectividad = SUPUESTOS["efectividad"] if efectividad is None else efectividad
    costo = SUPUESTOS["costo_intervencion"] if costo is None else costo
    perdida = df["incumple"] * df["exposicion"] * SUPUESTOS["LGD"]
    return interviene * (efectividad * perdida - costo)


def medidas(df: pd.DataFrame, interviene: pd.Series) -> dict:
    inc = df["incumple"].sum()
    capturados = (df["incumple"] * interviene).sum()
    return {
        "clientes": len(df),
        "tasa_intervencion": interviene.mean(),
        "incumplimientos_capturados": capturados / inc,
        "precision": capturados / max(interviene.sum(), 1),
        "valor_neto_por_cliente": valor(df, interviene).mean(),
    }


def accion(pd_: float, uso: float) -> str:
    if pd_ < 0.08:
        a = "sin acción"
    elif pd_ < 0.20:
        a = "alerta preventiva" if uso >= 0.5 else "monitoreo"
    elif pd_ < 0.40:
        a = "cobranza preventiva y sin aumentos de línea" if uso >= 0.5 else "cobranza preventiva"
    else:
        a = "revisión de línea y contacto prioritario"
    return a


RESPONSABLE = {
    "sin acción": "ninguno",
    "monitoreo": "analista de portafolio",
    "alerta preventiva": "mensaje automático",
    "cobranza preventiva": "ejecutivo de cobranza",
    "cobranza preventiva y sin aumentos de línea": "ejecutivo de cobranza",
    "revisión de línea y contacto prioritario": "comité de riesgos",
}


def main() -> None:
    ent = pd.read_csv(SALIDA / "predicciones_entrenamiento.csv")
    pru = pd.read_csv(SALIDA / "predicciones_prueba.csv")
    rng = np.random.default_rng(SEMILLA)

    # 1. Corte del challenger, fijado solo con entrenamiento (PD fuera de muestra)
    candidatos = np.quantile(ent["pd"], np.linspace(0.50, 0.98, 49))
    valores = [valor(ent, ent["pd"] >= c).sum() for c in candidatos]
    corte = float(candidatos[int(np.argmax(valores))])
    champion_ent = valor(ent, ent["mora_sep"] >= 1).sum()
    pd.DataFrame({"corte_pd": candidatos, "valor_neto_total": valores}).to_csv(
        SALIDA / "challenger_busqueda_corte.csv", index=False)

    # 2. Experimento: la muestra de prueba se reparte al azar en dos grupos
    orden = rng.permutation(len(pru))
    grupo_a, grupo_b = pru.iloc[orden[: len(pru) // 2]], pru.iloc[orden[len(pru) // 2:]]
    pol_champion = lambda df: (df["mora_sep"] >= 1)
    pol_challenger = lambda df: (df["pd"] >= corte)
    fila_a = {"política": "champion: mora de septiembre de 1 mes o más", "grupo": "A",
              **medidas(grupo_a, pol_champion(grupo_a))}
    fila_b = {"política": f"challenger: PD de {corte:.1%} o más", "grupo": "B",
              **medidas(grupo_b, pol_challenger(grupo_b))}
    va, vb = valor(grupo_a, pol_champion(grupo_a)).values, valor(grupo_b, pol_challenger(grupo_b)).values
    diffs = np.array([
        rng.choice(vb, len(vb)).mean() - rng.choice(va, len(va)).mean()
        for _ in range(SUPUESTOS["remuestreos_bootstrap"])])
    inferior, superior = np.percentile(diffs, [2.5, 97.5])
    experimento = pd.DataFrame([fila_a, fila_b])
    experimento.to_csv(SALIDA / "champion_challenger.csv", index=False)

    # 3. Comparación pareada con toda la muestra de prueba (misma gente, ambas políticas)
    pareada = pd.DataFrame([
        {"política": "champion", **medidas(pru, pol_champion(pru))},
        {"política": "challenger", **medidas(pru, pol_challenger(pru))},
    ])
    pareada.to_csv(SALIDA / "champion_challenger_pareada.csv", index=False)
    # Sensibilidad: ¿la ventaja del challenger depende de los supuestos?
    filas = []
    for efec in [0.10, 0.20, 0.30, 0.45]:
        for costo in [50, 150, 300, 600]:
            vc = valor(pru, pol_champion(pru), efec, costo).mean()
            vh = valor(pru, pol_challenger(pru), efec, costo).mean()
            filas.append({"efectividad": efec, "costo_intervencion": costo,
                          "valor_champion": vc, "valor_challenger": vh,
                          "challenger_mejor": bool(vh > vc)})
    sens = pd.DataFrame(filas)
    sens.to_csv(SALIDA / "champion_challenger_sensibilidad.csv", index=False)
    resumen = {
        "corte_challenger": corte,
        "valor_champion_entrenamiento": float(champion_ent),
        "valor_challenger_entrenamiento": float(max(valores)),
        "diferencia_valor_por_cliente": float(vb.mean() - va.mean()),
        "intervalo_95": [float(inferior), float(superior)],
        "supuestos": SUPUESTOS,
    }
    (SALIDA / "champion_challenger_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Matriz de decisión sobre la muestra de prueba
    p = pru.copy()
    p["banda_pd"] = pd.cut(p["pd"], BANDAS, labels=NOMBRES_BANDA, right=False)
    p["uso_linea"] = np.where(p["uso_sep"] >= 0.5, "50% o más", "menos de 50%")
    p["acción"] = [accion(a, b) for a, b in zip(p["pd"], p["uso_sep"])]
    matriz = p.groupby(["banda_pd", "uso_linea", "acción"], observed=True).agg(
        clientes=("incumple", "size"), incumplimiento_observado=("incumple", "mean"),
        exposicion=("exposicion", "sum")).reset_index()
    matriz["pct_clientes"] = matriz["clientes"] / matriz["clientes"].sum()
    matriz["responsable"] = matriz["acción"].map(RESPONSABLE)
    matriz.to_csv(SALIDA / "matriz_decision.csv", index=False)

    # Excepción propuesta y puesta a prueba: quien pagó completo el último mes
    # no debería entrar a cobranza. Solo se adoptaría si los datos la respaldan.
    media = p[p["banda_pd"] == "20 a 40%"].copy()
    media["pagó completo"] = media["mora_sep"] <= -1
    excepcion = media.groupby("pagó completo").agg(
        clientes=("incumple", "size"), incumplimiento_observado=("incumple", "mean")).reset_index()
    excepcion.to_csv(SALIDA / "excepcion_pago_completo.csv", index=False)

    # 5. Precio basado en riesgo
    p["perdida_esperada"] = p["pd"] * SUPUESTOS["LGD"]
    p["tasa_minima"] = (SUPUESTOS["fondeo_mensual"] + p["perdida_esperada"]
                        + SUPUESTOS["costo_operativo_mensual"])
    plana = np.average(p["tasa_minima"], weights=p["exposicion"].clip(lower=1))
    precio = p.groupby("banda_pd", observed=True).agg(
        clientes=("pd", "size"), pd_media=("pd", "mean"),
        perdida_esperada_mensual=("perdida_esperada", "mean"),
        tasa_minima_mensual=("tasa_minima", "mean")).reset_index()
    precio["tasa_plana_mensual"] = plana
    precio["subsidio_cruzado_mensual"] = plana - precio["tasa_minima_mensual"]
    precio.to_csv(SALIDA / "precio_riesgo.csv", index=False)

    # Gráfica
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    etiquetas = ["champion", "challenger"]
    valores_b = [va.mean(), vb.mean()]
    ejes[0].bar(etiquetas, valores_b, color=["#7F7F7F", "#1F3864"])
    ejes[0].errorbar(1, vb.mean(), yerr=[[vb.mean() - (va.mean() + inferior)], [(va.mean() + superior) - vb.mean()]],
                     color="black", capsize=6)
    ejes[0].set_ylabel("Valor neto por cliente")
    ejes[0].set_title("Experimento A/B con supuestos explícitos")
    ejes[0].text(0.5, -0.17, "La barra negra es el intervalo de 95% de la diferencia, sumado al valor del champion",
                 transform=ejes[0].transAxes, ha="center", fontsize=8)
    ejes[1].bar(precio["banda_pd"].astype(str), precio["subsidio_cruzado_mensual"] * 100, color="#1F3864")
    ejes[1].axhline(0, color="black", linewidth=0.8)
    ejes[1].set_ylabel("Subsidio cruzado (puntos porcentuales al mes)")
    ejes[1].set_title("Tasa plana contra tasa por riesgo")
    ejes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(SALIDA / "04_politica.png", dpi=150)

    print(f"Corte del challenger (fijado en entrenamiento): PD {corte:.3f}")
    print(f"Valor entrenamiento. champion {champion_ent:,.0f}  challenger {max(valores):,.0f}")
    print(experimento.round(3).to_string(index=False))
    print(f"Diferencia B menos A por cliente: {vb.mean() - va.mean():,.1f}  IC95 [{inferior:,.1f}; {superior:,.1f}]")
    print(pareada.round(3).to_string(index=False))
    print(f"Sensibilidad: el challenger gana en {int(sens['challenger_mejor'].sum())} de {len(sens)} combinaciones")
    print(sens[~sens["challenger_mejor"]].round(1).to_string(index=False))
    print(matriz.round(3).to_string(index=False))
    print(excepcion.round(3).to_string(index=False))
    print(precio.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
