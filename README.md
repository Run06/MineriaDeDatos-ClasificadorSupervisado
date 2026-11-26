# Clasificación supervisada de textos de Reddit

Proyecto de clasificación de usuarios a partir de sus publicaciones mediante representaciones vectoriales, re-etiquetado en cascada y algoritmos de aprendizaje supervisado. Incluye la evaluación de modelos, la medición del coste computacional y la generación automática de gráficas.

## 1. Instalación

Instalar las dependencias desde la terminal del entorno Python del proyecto:

```bash
python -m pip install numpy pandas scikit-learn gensim nltk torch sentence-transformers psutil matplotlib
```

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

## 3. Datos de entrada

- `500_Reddit_users_posts_labels.csv`: dataset de usuarios, publicaciones y etiquetas. Cada fila representa un usuario.
- `semilla.csv`: conjunto de mensajes etiquetados manualmente utilizado para iniciar el re-etiquetado.

Ambos archivos deben contener las columnas de texto y etiqueta, denominadas `Post` y `Label` por defecto. Sus nombres pueden configurarse mediante los argumentos `-t` y `-l`.

## 4. Ejecución

### Ejecución básica

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -s semilla.csv
```

La configuración predeterminada utiliza Sentence Transformers y SVM lineal.

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
| `embeddings` | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`). |

### Clasificadores disponibles

| Valor | Clasificador |
| --- | --- |
| `svm_linear` | Support Vector Machine lineal. |
| `svm_rbf` | Support Vector Machine con kernel RBF. |
| `rf` | Random Forest. |

### Ejemplos

FastText y SVM lineal:

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -v fasttext -m svm_linear
```

Sentence Transformers y SVM RBF:

```bash
python main.py -f 500_Reddit_users_posts_labels.csv -v embeddings -m svm_rbf
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
6. Selección de la ventana con mejor F1-Macro en Dev y exportación de métricas y gráficas.

### Estrategias de entrenamiento

| Estrategia | Datos utilizados |
| --- | --- |
| Solo pseudo | Semillas manuales y mensajes pseudo-etiquetados. |
| Con fallback | Semillas, mensajes pseudo-etiquetados y mensajes restantes con sus etiquetas originales. |
| Etiquetas originales | Semillas y todos los mensajes de Train con sus etiquetas originales. Se activa mediante `--baseline-original`. |

## 6. Ejecución de todos los experimentos

El siguiente script de PowerShell ejecuta las seis combinaciones de vectorización y clasificador, incluyendo las tres estrategias de entrenamiento y eliminando la caché antes de cada ejecución:

```powershell
foreach ($v in @('fasttext', 'embeddings')) {
    foreach ($m in @('svm_linear', 'svm_rbf', 'rf')) {
        python main.py -f 500_Reddit_users_posts_labels.csv -s semilla.csv -v $v -m $m --baseline-original --limpiar-cache

        if ($LASTEXITCODE -ne 0) {
            throw "Error en $v / $m"
        }
    }
}
```

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

El comparador utiliza la última ejecución de cada combinación de vectorizador, clasificador y uso del control de etiquetas originales. Lee los CSV existentes sin volver a entrenar los modelos. Los experimentos de una comparación deben utilizar el mismo dataset, semillas y columnas.

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
| `metricas.csv` | Métricas globales por partición, estrategia y ventana. |
| `metricas_por_clase.csv` | Métricas individuales por clase. |
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
