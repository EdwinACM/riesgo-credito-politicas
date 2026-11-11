# Políticas de riesgo de crédito con datos públicos

Proyecto personal que recorre el ciclo de una política de riesgo sobre una cartera de tarjetas de crédito: diagnóstico, segmentación, modelo de probabilidad de incumplimiento, conversión del modelo a reglas y una simulación champion/challenger con supuestos explícitos. Todo corre con un solo comando y da los mismos resultados cada vez.

Es un ejercicio académico con datos públicos. No usa información de ningún banco ni de sus clientes.

## Los datos

Base "Default of Credit Card Clients" del repositorio UCI (Yeh y Lien, 2009): 30,000 clientes de tarjeta de crédito de Taiwán, de abril a septiembre de 2005, con límite de crédito, saldos, pagos y estado de pago de cada mes. La variable a explicar es si el cliente incumple el mes siguiente. En esta base lo hace el 22.1%, una cifra alta que refleja la crisis de tarjetas de esos años.

## Qué encontré

**Diagnóstico.** El estado de pago del último mes ordena el riesgo mejor que cualquier otra variable. Los clientes revolventes al corriente incumplen el 12.8% y concentran el 75% de la exposición. Con un mes de atraso el incumplimiento sube a 33.9%, y con dos meses a 69.1%. Las matrices de transición muestran que un cliente con uno o dos meses de atraso se queda ahí el 68% de las veces, regresa al corriente el 25% y empeora el 6%.

**Segmentación.** K-means con seis variables de comportamiento separa tres segmentos. El primero paga casi todo lo que debe (14.8% de incumplimiento). El segundo es revolvente con uso alto y concentra el 72% de la exposición (14.9%). El tercero acumula atrasos, es el 20% de los clientes y tiene 50.6% de incumplimiento.

**Modelo de PD.** Regresión logística con cinco variables de la cuenta. En la muestra de prueba tiene AUC de 0.733, Gini de 0.467 y KS de 0.389; la validación cruzada del entrenamiento da AUC de 0.750. La regla simple de mirar solo la mora de septiembre llega a 0.683. La PD media predicha es 21.9% contra 22.1% observado. En el decil de menor riesgo el modelo predice 8% y se observa 13%, así que ahí subestima.

**Política.** La matriz de decisión cruza cuatro bandas de PD con el uso de la línea y asigna una acción y un responsable. Por ejemplo, la banda de 40% o más, que es el 13% de la muestra de prueba y tiene 60% de incumplimiento, pasa a revisión de línea por el comité de riesgos.

**Champion/challenger.** El champion interviene sobre quien tiene un mes o más de atraso. El challenger interviene sobre quien supera un corte de PD de 14.6%, fijado solo con la muestra de entrenamiento y antes de ver la de prueba. La muestra de prueba se reparte al azar en dos grupos de 4,500. El champion interviene al 22.8% y captura la mitad de los incumplimientos. El challenger interviene al 49.2% y captura el 74.8%, con menor precisión (32.4% contra 50.4%). Con los supuestos del proyecto, el valor neto por cliente es 1,217 para el champion y 1,493 para el challenger, una diferencia de 275 con intervalo de 95% de 15 a 543. En el análisis de sensibilidad el challenger gana en 15 de 16 combinaciones de efectividad y costo; solo pierde cuando la acción casi no funciona y es muy cara.

**Excepción descartada.** Probé que quien pagó completo el último mes no entrara a cobranza. En la banda de 20 a 40% esos clientes incumplen el 26.9% contra 29.9% del resto, sobre solo 104 casos. La diferencia es pequeña y no justifica una regla, así que no la adopté.

**Precio por riesgo.** Una tasa plana subsidia a los clientes de mayor riesgo con lo que pagan los de menor riesgo. En la ilustración, las dos bandas bajas pagan entre 5 y 9 puntos porcentuales al mes de más y la banda de 40% o más queda 29 puntos por debajo de su costo. Las magnitudes son exageradas porque el incumplimiento mensual de esta base es muy alto; sirve para mostrar la mecánica, no como tasas reales.

## Límites que conviene tener presentes

- La base es de tarjetas en Taiwán en 2005. No hay fecha de originación, así que no se pueden hacer cosechas (vintages). Usé matrices de transición de mora, que responden una pregunta cercana.
- De abril a agosto la base casi no registra "1 mes de atraso" y salta de 0 a 2. Por eso agrupé uno y dos meses en las matrices.
- Son clientes vigentes, no solicitantes. No hay rechazados, así que no se puede medir el sesgo de selección de una política de aprobación.
- La simulación usa supuestos propios de pérdida, costo y efectividad de la intervención porque la base no trae resultados de acciones reales. Están al inicio de `src/paso4_politica.py` y se pueden cambiar.
- Los coeficientes individuales del modelo no deben leerse de forma aislada: la razón de pago tiene signo positivo una vez controlada la mora, aunque sola se asocia con menor riesgo.
- Excluí sexo, estado civil, educación y edad del modelo. No evalué impacto desigual entre grupos.

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/paso0_preparar_datos.py
python src/ejecutar_todo.py
python -m pytest
```

Los pasos, en orden: `paso1_diagnostico.py`, `paso2_segmentacion.py`, `paso3_modelo_pd.py` y `paso4_politica.py`. Los resultados quedan en `resultados/` como tablas CSV y cuatro gráficas.

## Documentación de decisiones

`docs/bitacora.md` registra cada decisión del análisis, qué se midió y por qué se eligió así. La semilla aleatoria es 42 en todos los pasos.

## Referencia

Yeh, I. C. y Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2).

Licencia MIT.
