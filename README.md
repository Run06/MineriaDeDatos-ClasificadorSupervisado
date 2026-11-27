# Clasificación supervisada de textos de Reddit

Proyecto de clasificación de usuarios a partir de sus publicaciones mediante representaciones vectoriales, re-etiquetado en cascada y algoritmos de aprendizaje supervisado. Incluye la evaluación de modelos, la medición del coste computacional y la generación automática de gráficas.

## 1. Instalación

Instalar las dependencias desde la terminal del entorno Python del proyecto:

```bash
python -m pip install numpy pandas "scikit-learn>=1.7" gensim nltk torch sentence-transformers psutil matplotlib
```

Se requiere scikit-learn 1.7 o posterior para utilizar `sample_weight` en el entrenamiento del MLP. Se recomienda trabajar en un entorno virtual con una versión de Python compatible con todas las dependencias (por ejemplo, Python 3.12). Sentence Transformers descarga el modelo preentrenado si no está disponible localmente.

## 2. Estructura del proyecto

```text
proyecto/
├── main.py
├── preproceso.py
├── re_etiquetado.py
├── eval.py
├── medicion.py
├── graficas.py
├── 500_Reddit_users_posts_labels.csv
├── semilla.csv
├── weighted.py     # Opcional: cálculo de métricas ponderadas (sección 9)
├── README.md
├── cache/          # Generada automáticamente
└── resultados/     # Generada automáticamente
```

| Archivo | Descripción |
| --- | --- |
| `main.py` | Carga de datos y ejecución del pipeline. |
| `preproceso.py` | División de publicaciones, preprocesamiento y vectorización. |
| `re_etiquetado.py` | Pseudo-etiquetado en cascada mediante regresión logística. |
| `eval.py` | Entrenamiento, predicción y evaluación a nivel de usuario. |
| `medicion.py` | Registro de tiempos de ejecución y memoria RAM. |
| `graficas.py` | Generación de gráficas individuales y comparativas. |
| `weighted.py` (opcional) | Recuperación de métricas Weighted Avg desde los CSV existentes; no requiere reentrenar. |

## 3. Datos de entrada

- `500_Reddit_users_posts_labels.csv`: dataset de usuarios, publicaciones y etiquetas. Cada fila representa un usuario.
- `semilla.csv`: conjunto de mensajes etiquetados manualmente utilizado para iniciar el re-etiquetado.

Ambos archivos deben contener las columnas de texto y etiqueta, denominadas `Post` y `Label` por defecto. Sus nombres pueden configurarse mediante los argumentos `-t` y `-l`.

El dataset contiene 500 usuarios y 9.127 mensajes individuales, distribuidos en cinco clases: `Attempt`, `Behavior`, `Ideation`, `Indicator` y `Supportive`. Las publicaciones se separan en mensajes individuales conservando el identificador de usuario y la etiqueta original.

La partición estratificada se realiza antes de separar los mensajes (`random_state=42`):

| Partición | Usuarios | Mensajes |
| --- | ---: | ---: |
| Train (70 %) | 350 | 6.298 |
| Dev (15 %) | 75 | 1.564 |
| Test (15 %) | 75 | 1.265 |
| Total | 500 | 9.127 |

Las 200 semillas manuales se incorporan adicionalmente al entrenamiento; no están incluidas en los recuentos de la tabla. Debe comprobarse que las semillas no procedan de usuarios de Dev o Test para evitar fugas de información.

## 4. Ejecución

### Ejecución básica

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -s semilla.csv
```

La configuración predeterminada utiliza Sentence Transformers y SVM lineal. El baseline experimental del póster (Sentence Transformers + SVC RBF) se selecciona explícitamente con `-m svm_rbf`. No debe confundirse con el control opcional `--baseline-original`.

### Argumentos disponibles

| Argumento | Descripción | Valor predeterminado |
| --- | --- | --- |
| `-f`, `--file` | Ruta del dataset CSV. | Obligatorio |
| `-s`, `--seed` | Ruta del CSV de semillas. | `semilla.csv` |
| `-t`, `--target` | Columna de texto. | `Post` |
| `-l`, `--label` | Columna de etiquetas. | `Label` |
| `-v`, `--vectorizer` | Método de vectorización. | `embeddings` |
| `-m`, `--model` | Clasificador supervisado. | `svm_linear` |
| `--baseline-original` | Añade el entrenamiento con semillas y etiquetas originales. | Desactivado |
| `--limpiar-cache` | Elimina la carpeta `cache/` antes de ejecutar. | Desactivado |
| `--resultados` | Directorio de exportación de resultados. | `resultados` |
| `-h`, `--help` | Muestra la ayuda del programa. | — |

### Métodos de vectorización

| Valor | Representación |
| --- | --- |
| `fasttext` | FastText Skip-gram, vectores de 100 dimensiones. |
| `embeddings` | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`), vectores de 384 dimensiones. |

- **FastText:** entrenamiento desde cero sobre Train y las semillas manuales, mediante Gensim. Skip-gram, 15 épocas, ventana de contexto de 5 palabras, frecuencia mínima de 2 y subpalabras de 3 a 6 caracteres. Se aplica una limpieza ligera sin stemming ni eliminación de stopwords. Los vectores de las palabras se promedian y se normalizan mediante L2.
- **Sentence Transformers:** modelo preentrenado, sin ajuste fino. Los mensajes se dividen en bloques de hasta 80 palabras, con un máximo de 50 bloques por mensaje (muestreo uniforme si se supera el límite). Se promedian los embeddings normalizados de los bloques y se normaliza el vector resultante mediante L2. Se utiliza CUDA si está disponible; en caso contrario, CPU.

En ambos casos se obtiene un vector por mensaje. Dev y Test no se utilizan para entrenar FastText.

### Clasificadores disponibles

| Valor        | Clasificador                           |
|--------------|----------------------------------------|
| `svm_linear` | Support Vector Machine lineal.         |
| `svm_rbf`    | Support Vector Machine con kernel RBF. |
| `rf`         | Random Forest.                         |
| `mlp`         | Multilayer Perceptron (perceptrón multicapa). |

Configuración de los clasificadores:

| Modelo | Parámetros principales |
| --- | --- |
| SVM lineal | `C=1.0`, `class_weight="balanced"`, `max_iter=10000` |
| SVC RBF | `C=2.0`, `gamma="scale"`, `class_weight="balanced"` |
| Random Forest | 500 árboles, `max_features="sqrt"`, profundidad máxima 300, `min_samples_split=5`, `min_samples_leaf=3`, `class_weight="balanced"` |
| MLP | Una capa oculta de 256 neuronas, ReLU, `alpha=1e-4`, `max_iter=300`, `early_stopping=True`, `n_iter_no_change=10`; pesos de muestra balanceados |

Todos los clasificadores utilizan `random_state=42`. El early stopping del MLP emplea su propia validación interna de entrenamiento, distinta de Dev.

### Ejemplos

Modelo propuesto: FastText + MLP (medición en frío):

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -s semilla.csv -m mlp -v fasttext --limpiar-cache
```

Baseline experimental: Sentence Transformers + SVC RBF (medición en frío):

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -s semilla.csv -m svm_rbf -v embeddings --limpiar-cache
```

FastText y Random Forest con control de etiquetas originales y caché limpia:

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -v fasttext -m rf --baseline-original --limpiar-cache
```

## 5. Proceso de entrenamiento y evaluación

1. Carga y partición estratificada del dataset a nivel de usuario: Train (70 %), Dev (15 %) y Test (15 %).
2. Separación de las publicaciones individuales y vectorización de los textos.
3. Re-etiquetado en cascada con umbrales de confianza de 0.95, 0.90, 0.85, 0.80, 0.75 y 0.70.
4. Entrenamiento del clasificador seleccionado para cada estrategia.
5. Evaluación a nivel de usuario mediante Max-Pooling de las puntuaciones por clase, con ventanas de 1, 3, 5, 10, 15 y todos los mensajes.
6. Selección de la ventana con mejor F1-Macro en Dev, de forma independiente para cada configuración y estrategia. En caso de empate, se selecciona la primera ventana (la menor).
7. Evaluación de la ventana seleccionada sobre Test y exportación de métricas y gráficas.

### Re-etiquetado en cascada

Se utiliza regresión logística como modelo maestro (`C=1.0`, `class_weight="balanced"`, `max_iter=500`, `solver="lbfgs"`). A partir de las semillas, se incorporan progresivamente mensajes de Train cuya confianza supera o iguala el umbral correspondiente, con un máximo de 10 iteraciones por nivel.

Se exige una longitud mínima de 8 palabras para los candidatos a pseudo-etiquetado y se aplica balanceo relativo (factor 2) entre las clases presentes en los candidatos. Los mensajes seleccionados se retiran del conjunto pendiente y el modelo maestro se vuelve a entrenar. Si la primera iteración del primer nivel cubre todas las clases de las semillas, se sustituyen temporalmente las semillas en el entrenamiento del maestro; se recuperan siempre para el entrenamiento del clasificador final.

Los mensajes que no se pseudo-etiquetan conservan su etiqueta original cuando se utiliza fallback.

### Estrategias de entrenamiento

| Estrategia | Datos utilizados |
| --- | --- |
| Solo pseudo | Semillas manuales y mensajes pseudo-etiquetados. |
| Con fallback | Semillas, mensajes pseudo-etiquetados y mensajes restantes con sus etiquetas originales. |
| Etiquetas originales | Semillas y todos los mensajes de Train con sus etiquetas originales. Se activa mediante `--baseline-original`. |

### Agregación de mensajes y selección de k

El clasificador produce puntuaciones por clase para cada mensaje (`decision_function` en SVM o `predict_proba` en Random Forest y MLP). Para cada usuario se seleccionan sus primeros `k` mensajes disponibles, según su orden en el dataset. Mediante Max-Pooling se obtiene la máxima puntuación de cada clase entre esos mensajes y se asigna al usuario la clase con la mayor puntuación agregada.

`k` es el número máximo de mensajes utilizados por usuario durante la evaluación; no modifica el entrenamiento. Si el usuario dispone de menos de `k` mensajes, se utilizan todos los disponibles. Se exploran las ventanas `1`, `3`, `5`, `10`, `15` y `Todos`.

La ventana óptima se selecciona únicamente mediante F1-Macro en Dev. Los resultados de Test para las demás ventanas son exploratorios y no deben utilizarse para escoger `k`.

**Importante:** en la versión actual de `eval.py`, el `classification_report` mostrado en terminal corresponde a `k=Todos`. Las gráficas de resultados finales utilizan, en cambio, el `k` seleccionado en Dev. Para construir las tablas del póster se deben utilizar los CSV filtrando por `particion = Test` y `k_elegido_dev = True` (y por estrategia cuando corresponda).

## 6. Ejecución de todos los experimentos

El siguiente script de PowerShell ejecuta las ocho combinaciones de vectorización y clasificador, incluyendo las tres estrategias de entrenamiento y eliminando la caché antes de cada ejecución:

```powershell
foreach ($v in @('fasttext', 'embeddings')) {
    foreach ($m in @('svm_linear', 'svm_rbf', 'rf', 'mlp')) {
        python main.py -f 500_Reddit_users_posts_labels.csv -s semilla.csv -v $v -m $m --baseline-original --limpiar-cache

        if ($LASTEXITCODE -ne 0) {
            throw "Error en $v / $m"
        }
    }
}
```

Para ejecutar únicamente las dos configuraciones principales del póster, sin el control de etiquetas originales, utilizar los dos comandos de la sección 4. Se entrenan las estrategias `Solo pseudo` y `Con fallback` en cada ejecución.

## 7. Generación de gráficas comparativas

Comparar las ejecuciones realizadas con `--limpiar-cache`:

```bash
python graficas.py --modo-cache frio
```

Comparar las ejecuciones realizadas sin `--limpiar-cache`:

```bash
python graficas.py --modo-cache normal
```

Comparar experimentos almacenados en un directorio personalizado:

```bash
python graficas.py --resultados resultados_experimento --modo-cache frio
```

El comparador utiliza la última ejecución de cada combinación de vectorizador, clasificador y uso del control de etiquetas originales. Lee los CSV existentes sin volver a entrenar los modelos. Los experimentos de una comparación deben utilizar el mismo dataset, semillas y columnas. Se verifica la firma de los datos antes de generar la comparación.

Para aislar los experimentos del póster de otras ejecuciones, puede emplearse el mismo directorio personalizado (`--resultados resultados_experimento`) tanto en `main.py` como en `graficas.py`.

La comparación de tiempo utiliza la duración total del pipeline (todas las estrategias ejecutadas), sin incluir la exportación de gráficas y CSV. La RAM corresponde al pico de memoria residente (RSS) del proceso, muestreado cada 50 ms; no representa la VRAM ni la memoria total del sistema. `--limpiar-cache` elimina la caché propia del proyecto, no la caché de descarga de modelos preentrenados. Comparar experimentos en las mismas condiciones de hardware y caché.

## 8. Resultados generados

Cada ejecución genera un directorio independiente con el formato:

```text
resultados/<fecha>_<hora>_<microsegundos>_<vectorizador>_<modelo>/
```

Todas las gráficas se exportan en PDF vectorial y PNG a 300 DPI.

### Gráficas individuales

| Archivo (sin extensión) | Contenido |
| --- | --- |
| `f1_ventana` | F1-Macro en Dev y Test según el máximo de mensajes por usuario. |
| `metricas_test` | Precision-Macro, Recall-Macro, F1-Macro y Accuracy en Test para la ventana seleccionada en Dev. |
| `metricas_clase_*` | Precision, Recall y F1 por clase y estrategia. |
| `confusion_*` | Matrices de confusión con frecuencias y porcentajes por clase real. |
| `cascada` | Nuevas pseudo-etiquetas por umbral y cobertura acumulada. |
| `distribucion_usuario` | Distribución de clases a nivel de usuario. |
| `distribucion_mensaje` | Distribución de clases a nivel de mensaje. |
| `longitudes` | Diagramas de caja de la longitud de los mensajes. |
| `tiempo_fases` | Tiempo de ejecución por fase, en segundos. |
| `ram_fases` | Pico de memoria residente del proceso (RSS), en MiB. |

### Archivos de resultados

| Archivo | Contenido |
| --- | --- |
| `metricas.csv` | Precision-Macro, Recall-Macro, F1-Macro, Accuracy y Kappa por partición, estrategia y ventana. Incluye `k_elegido_dev`. |
| `metricas_por_clase.csv` | Precision, Recall, F1 y soporte de cada clase por partición, estrategia y ventana. Incluye `k_elegido_dev`. |
| `confusiones.csv` | Frecuencias de las matrices de confusión. |
| `recursos.csv` | Tiempo y memoria RAM por fase. |
| `dataset_resumen.csv` | Número de usuarios, mensajes, estadísticas de longitud y vocabulario bruto. |
| `distribucion.csv` | Distribución de clases por partición y unidad de análisis. |
| `cascada_iteraciones.csv` | Historial del re-etiquetado por iteración. |
| `cascada_umbrales.csv` | Resultados acumulados por umbral de confianza. |
| `config.json` | Configuración y resumen de la ejecución. |

### Gráficas comparativas

Se almacenan en `resultados/comparacion_frio/` o `resultados/comparacion_normal/`, según el modo seleccionado.

| Archivo (sin extensión) | Contenido |
| --- | --- |
| `comparacion_f1_*` | Comparación del F1-Macro entre configuraciones y estrategias. |
| `comparacion_segundos_*` | Comparación del tiempo total del pipeline. |
| `comparacion_ram_pico_mib_*` | Comparación del pico de memoria RAM. |

El sufijo `con_control` o `sin_control` identifica si se utilizó la opción `--baseline-original`.

También se generan `metricas_comparadas.csv` y `recursos_comparados.csv`, que reúnen los resultados de las configuraciones comparadas.


## 9. Obtención de métricas ponderadas (Weighted Avg)

Los archivos `metricas.csv` y `metricas_comparadas.csv` almacenan directamente los promedios macro, pero no los ponderados. Estos últimos se pueden recuperar a partir de `metricas_por_clase.csv` sin volver a entrenar los modelos.

- **Macro Avg:** media de las métricas de las clases, asignando el mismo peso a cada clase.
- **Weighted Avg:** media ponderada por el soporte (número de usuarios reales de cada clase).
- **Accuracy:** proporción de usuarios correctamente clasificados. En clasificación multiclase de etiqueta única coincide con el Recall-Weighted.

Para cada métrica, se multiplica el valor de cada clase por su soporte, se suman los productos y se divide entre la suma de los soportes.

Crear opcionalmente un archivo `weighted.py` en la raíz del proyecto con el siguiente contenido. Utiliza las ejecuciones incluidas en `metricas_comparadas.csv` para mantener la correspondencia con las gráficas y selecciona únicamente Test con el `k` elegido en Dev.

```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--resultados", default="resultados")
parser.add_argument("--modo-cache", choices=["frio", "normal"], default="frio")
args = parser.parse_args()

raiz = Path(args.resultados)
comparacion = raiz / f"comparacion_{args.modo_cache}"
metricas = pd.read_csv(comparacion / "metricas_comparadas.csv", dtype={"k": str})

filas = []
for ejecucion in metricas["ejecucion"].unique():
    df = pd.read_csv(raiz / ejecucion / "metricas_por_clase.csv", dtype={"k": str})
    datos = df[(df["particion"] == "Test") & df["k_elegido_dev"]]
    for (estrategia, k), grupo in datos.groupby(["estrategia", "k"], sort=False):
        pesos = grupo["soporte"]
        fila = {"ejecucion": ejecucion, "estrategia": estrategia, "k": k}
        for metrica in ["precision", "recall", "f1"]:
            fila[f"{metrica}_weighted"] = np.average(grupo[metrica], weights=pesos)
        filas.append(fila)

resultado = pd.DataFrame(filas)
print(resultado.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
resultado.to_csv(comparacion / "metricas_weighted.csv", index=False)
```

Después de generar las gráficas comparativas, ejecutar:

```bash
python weighted.py --modo-cache frio
```

El script muestra las métricas ponderadas y crea `resultados/comparacion_frio/metricas_weighted.csv`. Para mediciones normales, utilizar `--modo-cache normal`.

No se debe confundir este cálculo posterior con una modificación de `eval.py`: el pipeline original sigue exportando las métricas macro y por clase. La gráfica de F1-Macro y una tabla de métricas Weighted Avg son compatibles siempre que se identifique el tipo de promedio y se utilice la misma configuración experimental.

## 10. Resultados de referencia del póster

Experimentos del 27/11/2026, con caché del proyecto limpia, 200 semillas y sin el control opcional de etiquetas originales. En las cuatro configuraciones de evaluación se seleccionó `k=3` mediante Dev. Los resultados siguientes corresponden a los 75 usuarios de Test.

| Vectorización + clasificador | Estrategia | F1-Macro | Precision-Weighted | Recall-Weighted | F1-Weighted | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ST + SVC RBF | Solo pseudo | 0.2718 | 0.2996 | 0.3467 | 0.2971 | 0.3467 |
| ST + SVC RBF | Con fallback | 0.2649 | 0.3239 | 0.3467 | 0.3131 | 0.3467 |
| FastText + MLP | Solo pseudo | 0.1109 | 0.2065 | 0.2400 | 0.1262 | 0.2400 |
| FastText + MLP | Con fallback | 0.2148 | 0.2723 | 0.2000 | 0.1933 | 0.2000 |

| Configuración | Pseudo-etiquetas | Mensajes con fallback | Tiempo total (s) | Pico RSS muestreado (MiB) |
| --- | ---: | ---: | ---: | ---: |
| ST + SVC RBF | 761 | 5.537 | 172.1 | 1667 |
| FastText + MLP | 1 | 6.297 | 25.1 | 1175 |

Estos valores son resultados de las ejecuciones de referencia, no garantías de rendimiento para futuras ejecuciones. FastText + MLP reduce el coste computacional, pero obtiene un menor F1-Macro. El fallback mejora el F1-Weighted de ambos modelos y el F1-Macro de FastText + MLP.

Como referencia teórica, un clasificador aleatorio uniforme de cinco clases tiene una Accuracy esperada de 0.20. No se ha medido aquí su F1-Macro ni su F1-Weighted; tampoco se ha comprobado la significación estadística de las diferencias. Esta referencia no constituye un límite inferior del rendimiento ni un clasificador implementado en el pipeline.
