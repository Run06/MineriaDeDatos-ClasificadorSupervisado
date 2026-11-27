# MineriaDeDatos-ClasificadorSupervisado
Proyecto grupal para la asignatura de Minería de Datos del grado en Ingeniería Informática de Gestión y Sistemas de Información en la EHU.

## Branches
- baseline: contiene el baseline sobre el que experimentaremos los tres integrantes del grupo.
- marcos: experimentación de Marcos _**(branch actual)**_.
- ibai: experimentación de Ibai.
- keneth: experimentación de Keneth.

## Arquitectura y Archivos del Proyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.py` | Punto de entrada del flujo base. Particiona los datos por usuario (`70% / 15% / 15%`), vectoriza con embeddings, ejecuta el re-etiquetado en cascada y evalúa un clasificador `SVC` comparando los conjuntos `clean` (solo re-etiquetados + semillas) y `full` (con *fallback*) para distintas ventanas $k$. |
| `experimentos_vader_xgb.py` | Pipeline experimental principal (rama `marcos`). Ejecuta múltiples repeticiones (*runs*), compara 5 configuraciones de modelos (`SVM` y `XGBoost` con Embeddings, VADER y su concatenación), selecciona la ventana óptima $k$ en *Dev*, exporta métricas y gráficas a `poster_figuras/`, y lanza las pruebas cualitativas y estadísticas de robustez (RQ1) y falsos positivos graves con intervalos *bootstrap* (RQ2). |
| `preproceso.py` | Módulo de limpieza, segmentación y representación vectorial. Incluye el desglose de listas de posts por usuario (`dividir_mensajes`), limpieza/stemming y TF-IDF clásico, troceado uniforme de textos largos (`trocear`), extracción cacheada de embeddings (`SentenceTransformer`) y cálculo de polaridad VADER (`[neg, neu, pos, compound]`). |
| `re_etiquetado.py` | Implementa `cascading_self_training`: bucle iterativo de pseudo-etiquetado por umbrales descendentes (`0.95` a `0.70`) usando regresión logística balanceada, con filtro de longitud mínima de palabras, balanceo relativo entre clases y almacenamiento en caché `.npz`. |
| `eval.py` | Contiene la agregación por usuario mediante *max-pooling* sobre los primeros $k$ mensajes (`predecir_por_usuario`) y el entrenamiento/evaluación comparativa por ventanas del modelo base (`entrenar_y_evaluar`). |
| `config.json` | Archivo centralizado con todos los hiperparámetros, rutas por defecto, umbrales de cascada y parámetros de evaluación manteniendo los valores exactos de los experimentos. |

## Branch: Marcos:

Python recomendado: 3.12.

Instalar requirements.txt.
```
pip install -r requirements.txt
```
Datos - Reddit C-SSRS Suicide Dataset: https://zenodo.org/records/2667859

## Guía de Ejecución

### 1. Ejecución del Pipeline Base (`main.py`)

Ejecuta el flujo estándar de particionado por usuario, re-etiquetado en cascada y evaluación con `SVC` en distintas ventanas $k$:

```bash
python main.py --help
```
```bash
python main.py -f ./500_Reddit_users_posts_labels.csv -t Post -l Label -s semilla.csv
```

| Argumento | Por defecto | Descripción |
| :--- | :--- | :--- |
| `-f`, `--file` | *(Obligatorio)* | Ruta al archivo CSV de usuarios y listas de publicaciones. |
| `-t`, `--target` | `Post` | Nombre de la columna que contiene los textos. |
| `-l`, `--label` | `Label` | Nombre de la columna con la etiqueta de gravedad. |
| `-s`, `--seed` | `semilla.csv` | Ruta al CSV con instancias semilla anotadas manualmente. |

---

### 2. Ejecución de Experimentos Completos (`experimentos_vader_xgb.py`)

Permite consultar todas las opciones disponibles mediante:

```bash
python experimentos_vader_xgb.py --help
```
```bash
python experimentos_vader_xgb.py -f .\500_Reddit_users_posts_labels.csv -t Post -l Label -s semilla.csv --conjunto-train full --runs 3
```

| Argumento | Por defecto | Descripción |
| :--- | :--- | :--- |
| `-f`, `--file` | *(Obligatorio)* | CSV de usuarios y mensajes de entrada. |
| `-t`, `--target` | `Post` | Columna de texto en el dataset principal. |
| `-l`, `--label` | `Label` | Columna de etiquetas de clase. |
| `-s`, `--seed` | `semilla.csv` | CSV con las instancias semilla de anclaje. |
| `--seed-target` | `None` | Columna de texto en `semilla.csv` (usa `Post` si existe, si no `--target`). |
| `--conjunto-train` | `full` | Conjunto de entrenamiento: `sin_pseudo`, `clean` o `full`. |
| `--orden-gravedad` | `Supportive Indicator Ideation Behavior Attempt` | Jerarquía ordenada de menor a mayor gravedad clínica. |
| `--runs` | `3` | Número de repeticiones con semillas consecutivas (`42 + i*10`). |

---

## Resumen de Modelos y Métricas Evaluadas

| Configuración | Tipo de Modelo | Representación de Entrada |
| :--- | :--- | :--- |
| `Baseline (Embeddings + SVM)` | `SVC` (kernel RBF) | Embeddings densos ($D$) |
| `Embeddings + VADER (SVM)` | `SVC` (kernel RBF) | Embeddings ($D$) + VADER ($4$) |
| `Embeddings (XGBoost)` | `XGBClassifier` | Embeddings densos ($D$) |
| `VADER solo (XGBoost)` | `XGBClassifier` | Características VADER ($4$) |
| `Embeddings + VADER (XGBoost)` | `XGBClassifier` | Embeddings ($D$) + VADER ($4$) |

* **Métricas por usuario:** `Macro-F1`, `Weighted-F1`, `Cohen's Kappa`, `F1 Attempt` (clase minoritaria), tasa de falsos positivos graves (`FP_grave_tasa`) y exceso medio de gravedad (`Gap_gravedad`).
* **Salidas generadas:** Los resultados por ejecución, resúmenes estadísticos y gráficos comparativos se guardan automáticamente en la carpeta `poster_figuras/`.
* **Cache:** Se almacenan los modelos y el reetiquetado en `/cache`.

---

## Configuración de Hiperparámetros (`config.json`)

El archivo `config.json` centraliza los parámetros principales del autoentrenamiento y de los modelos finales para facilitar su ajuste sin modificar el código fuente:

| Sección | Clave | Valor | Descripción |
| :--- | :--- | :--- | :--- |
| `re_etiquetado` | `umbrales` | `[0.95, 0.9, 0.85, 0.8, 0.75, 0.7]` | Niveles descendentes de probabilidad mínima exigida para aceptar pseudo-etiquetas en cada etapa de la cascada. |
| `re_etiquetado` | `max_iter_interna` | `10` | Número máximo de iteraciones internas de reentrenamiento dentro de un mismo umbral antes de pasar al siguiente. |
| `re_etiquetado` | `factor_balanceo` | `2` | Multiplicador respecto a la clase menos frecuente en cada ronda para limitar el exceso de pseudo-etiquetas de clases mayoritarias. |
| `re_etiquetado` | `min_palabras` | `8` | Longitud mínima (en palabras) que debe tener un mensaje para poder ser seleccionado como pseudo-etiqueta. |
| `re_etiquetado` | `random_state` | `42` | Semilla aleatoria para el muestreo balanceado y el modelo maestro. |
| `re_etiquetado.modelo_maestro` | `class_weight` | `"balanced"` | Ajuste automático de pesos inversamente proporcional a la frecuencia de cada clase en la regresión logística. |
| `re_etiquetado.modelo_maestro` | `max_iter` | `500` | Número máximo de iteraciones del optimizador de `LogisticRegression`. |
| `re_etiquetado.modelo_maestro` | `C` | `1.0` | Parámetro inverso de regularización L2 del modelo maestro. |
| `re_etiquetado.modelo_maestro` | `solver` | `"lbfgs"` | Algoritmo de optimización utilizado para ajustar la regresión logística multiclase. |
| `modelos.svm` | `kernel` | `"rbf"` | Función kernel radial utilizada por el clasificador `SVC`. |
| `modelos.svm` | `C` | `2.0` | Parámetro de penalización del error (regularización) en `SVC`. |
| `modelos.svm` | `gamma` | `"scale"` | Coeficiente del kernel RBF calculado automáticamente como $1 / (n\_features \cdot \text{Var}(X))$. |
| `modelos.xgb` | `n_estimators` | `250` | Número de árboles de decisión (*boosting rounds*) en `XGBClassifier`. |
| `modelos.xgb` | `max_depth` | `5` | Profundidad máxima de cada árbol para controlar la complejidad y evitar sobreajuste. |
| `modelos.xgb` | `learning_rate` | `0.05` | Tasa de aprendizaje (*shrinkage*) que pondera la contribución de cada nuevo árbol. |
| `modelos.xgb` | `subsample` | `0.8` | Proporción de instancias muestreadas aleatoriamente para entrenar cada árbol. |
| `modelos.xgb` | `colsample_bytree` | `0.8` | Proporción de características (columnas) submuestreadas al construir cada árbol. |
