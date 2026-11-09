# TFM — Clasificación de enfermedad cardíaca (Cleveland)

Código reproducible para el **Proyecto 1 del Trabajo Fin de Máster de Inteligencia Artificial aplicada a la Sanidad**.

Se analizan 303 registros con 13 variables predictoras y la variable original `num`. El objetivo binario se define como ausencia de enfermedad (`num = 0`) frente a presencia de enfermedad (`num = 1, 2, 3 o 4`). Se comparan una **regresión logística** y un **Random Forest**.

## Contenido

- `TFM_Proyecto1_analisis.py`: exploración de los datos, entrenamiento, validación y generación de gráficos.
- `heart.csv`: copia del archivo proporcionado para este TFM.
- `requirements.txt`: dependencias y versiones utilizadas.
- `Figuras_TFM/`, `TFM_Resultados_EDA.json` y `TFM_Borrador_EDA_y_Modelos.md`: archivos generados al ejecutar el script (no es necesario crearlos a mano).

## Reproducir los resultados

Se recomienda Python 3.11 o superior (la ejecución verificada se realizó con Python 3.13.5).

```bash
git clone https://github.com/JoRex-bot/TFM-Heart-Disease.git
cd TFM-Heart-Disease
python -m pip install -r requirements.txt
python TFM_Proyecto1_analisis.py
```

El programa guarda los gráficos en `Figuras_TFM/`, las métricas en `TFM_Resultados_EDA.json` y el resumen en `TFM_Borrador_EDA_y_Modelos.md`.

## Procedimiento metodológico

- Reparto **estratificado 80/20**: 242 registros para entrenamiento y 61 para prueba, con `random_state=42`.
- Validación cruzada estratificada de **5 particiones** exclusivamente en los registros de entrenamiento.
- Escalado de variables numéricas para regresión logística y codificación one-hot de variables categóricas, todo dentro de un `Pipeline` para evitar fugas de información.
- Regresión logística con `max_iter=3000`.
- Random Forest con `n_estimators=300`, `min_samples_leaf=2` y `random_state=42`.

## Resultados reproducidos

| Métrica en el conjunto de prueba | Regresión logística | Random Forest |
|---|---:|---:|
| Exactitud | 88,52 % | 86,89 % |
| Sensibilidad | 92,86 % | 92,86 % |
| Especificidad | 84,85 % | 81,82 % |
| AUC-ROC | 0,9665 | 0,9502 |

La regresión logística produjo 28 verdaderos negativos, 5 falsos positivos, 2 falsos negativos y 26 verdaderos positivos.

## Fuente y limitaciones

El archivo se proporcionó para el TFM y es compatible con las 14 columnas del conjunto **Heart Disease (Cleveland)** de UCI: https://archive.ics.uci.edu/dataset/45/heart+disease . Se desconoce si el CSV suministrado pasó por un procesamiento previo con respecto al original de UCI; por ello se conserva una copia exacta de sus valores en este repositorio (se normalizan únicamente los saltos de línea).

Los 303 registros no representan una muestra poblacional. Este ejercicio **no es un dispositivo médico**, no dispone de validación clínica externa y no debe utilizarse para tomar decisiones asistenciales individuales.

### Referencia de los datos

Janosi A, Steinbrunn W, Pfisterer M, Detrano R. *Heart Disease*. UCI Machine Learning Repository. 1989. https://doi.org/10.24432/C52P4X

## Verificación adicional para el tribunal

Tras ejecutar el análisis, ejecute:

```bash
python verificar_resultados.py
```

Este comprobador independiente comprueba que existen los siete gráficos, los dos archivos de resultados y los indicadores publicados (303 casos, partición 242/61, matriz de confusión, exactitud, sensibilidad, especificidad y AUC). Si algún resultado no coincide, falla con un error y muestra la comprobación que falló.

### Ejecución automática en GitHub

En [Actions](https://github.com/JoRex-bot/TFM-Heart-Disease/actions), seleccione **Verificar reproducibilidad del TFM**. El flujo también se ejecuta al actualizar `main`. Si está habilitado en la configuración del repositorio, puede seleccionarse **Run workflow** para repetir la evaluación sobre Ubuntu con Python 3.11. El flujo instala dependencias, ejecuta el modelo, verifica las métricas y publica un artefacto `resultados-tfm` con las figuras, el JSON y el informe para su inspección. La disponibilidad de GitHub Actions depende de la configuración del repositorio y de la plataforma.

**No es una aplicación web:** el proyecto es un script reproducible de investigación académica. Para evaluarlo, descargue el repositorio y ejecute los dos comandos anteriores. No solicita credenciales, no utiliza datos clínicos identificables y no toma decisiones asistenciales.
