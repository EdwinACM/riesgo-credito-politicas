# Bitácora de decisiones

Fecha de la corrida: 10 de noviembre de 2026. Semilla 42 en todos los pasos.

## Datos

1. Elegí la base de UCI "Default of Credit Card Clients" porque es gratuita, no exige cuenta y trae seis meses de comportamiento. No trae fecha de originación.
2. En educación recodifiqué los valores 0, 5 y 6 como "otros" (4), y en estado civil el valor 0 como "otros" (3), porque la documentación de la base solo define 1 a 4 y 1 a 3.
3. La columna PAY_0 corresponde a septiembre y las columnas PAY_2 a PAY_6 a agosto hasta abril. Las renombré por mes para evitar confusiones.

## Diagnóstico

4. Las matrices de transición agrupan "1 a 2 meses" porque de abril a agosto casi no existen registros de 1 mes (entre 0 y 28 clientes por mes, contra 3,688 en septiembre). Con cuatro estados, la matriz mezclaba definiciones distintas.

## Modelo

5. Comparé cinco conjuntos de variables solo con validación cruzada de la muestra de entrenamiento, sin mirar la de prueba. Elegí el de cinco variables: AUC de 0.750 contra 0.751 del conjunto de diez, con coeficientes más fáciles de explicar.
6. Sexo, estado civil, educación y edad no entran al modelo. La edad se usa solo para describir segmentos.
7. La muestra se divide 70% y 30% con estratificación por incumplimiento. La PD de entrenamiento se obtiene con validación cruzada de cinco pliegues para que fijar el corte no use predicciones optimistas.

## Política y champion/challenger

8. El corte del challenger se eligió maximizando el valor neto sobre la muestra de entrenamiento (PD fuera de muestra) y se aplicó sin cambios a la de prueba.
9. Supuestos: pérdida en caso de incumplimiento de 70% del saldo, costo de intervención de 150 por cliente y efectividad de 30% de la pérdida. Son ilustrativos. Probé 16 combinaciones de efectividad (10%, 20%, 30%, 45%) y costo (50, 150, 300, 600).
10. La excepción "quien pagó completo no entra a cobranza" se probó y se descartó: 26.9% de incumplimiento contra 29.9% del resto, con 104 casos.
11. Retiré la tasa anualizada del precio por riesgo porque el incumplimiento mensual de la base (22%) produce cifras sin sentido económico. Conservo la comparación relativa entre bandas.

## Cambios durante el análisis

12. La primera versión del modelo tenía diez variables. La simplifiqué a cinco por la decisión 5.
13. La primera versión de las matrices de transición tenía cuatro estados. La corregí por la decisión 4.
14. La primera versión de la política incluía la excepción de la decisión 10. La quité de las reglas por falta de respaldo.
