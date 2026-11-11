"""Paso 0. Descarga la base pública y la convierte a CSV.

Fuente: UCI Machine Learning Repository, "Default of Credit Card Clients"
(Yeh y Lien, 2009). Son clientes de tarjeta de crédito de Taiwán, abril a
septiembre de 2005. No contiene datos de ningún banco mexicano."""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def main() -> None:
    carpeta = RAIZ / "datos"
    carpeta.mkdir(exist_ok=True)
    destino = carpeta / "uci_default.zip"
    if not destino.exists():
        urllib.request.urlretrieve(URL, destino)
    with zipfile.ZipFile(destino) as z:
        z.extractall(carpeta)
    df = pd.read_excel(carpeta / "default of credit card clients.xls", header=1, engine="xlrd")
    df = df.rename(columns={"default payment next month": "default"})
    df.to_csv(carpeta / "credito_tarjetas.csv", index=False)
    print(f"Base lista: {df.shape[0]:,} clientes y {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
