# MineriaDeDatos-ClasificadorSupervisado
Proyecto grupal para la asignatura de Minería de Datos del grado en Ingeniería Informática de Gestión y Sistemas de Información en la EHU.

## Branches
- baseline: contiene el baseline sobre el que experimentaremos los tres integrantes del grupo.
- marcos: experimentación de Marcos.
- ibai: experimentación de Ibai.
- keneth: experimentación de Keneth.

# Ibai
## Clasificación Jerárquica de Riesgo Suicida (C-SSRS) con MentalRoBERTa-LoRA y BiLSTM-MIL

Sistema de aprendizaje profundo jerárquico para estimar el nivel de riesgo suicida a nivel de usuario sobre historiales de publicaciones en Reddit según la escala clínica **C-SSRS** de 5 niveles ordenados:

`Supportive (0)` → `Indicator (1)` → `Ideation (2)` → `Behavior (3)` → `Attempt (4)`

El proyecto combina adaptación eficiente de parámetros (**LoRA** en las 12 capas de `mental/mental-roberta-base`, solo el **0,41 %** de los pesos entrenables), modelado secuencial bidireccional (**BiLSTM** residual), agregación **Multiple Instance Learning (MIL)** invariante a la longitud del historial para evitar la dilución de mensajes críticos de `Attempt`, aprendizaje **Multi-Task** con mensajes semilla limpios (`semilla.csv`) y pérdida **Balanced Softmax + Soft Ordinal Targets (SORD)** para abordar simultáneamente el desbalanceo severo y la naturaleza ordinal de la tarea.

---

## Estructura del repositorio

```text
.
├── main.py                  # Orquestación: split estratificado, CV, OOF, Ensemble y registro
├── modelo_jerarquico.py     # Arquitectura MentalRoBERTa-LoRA + Token Attn + BiLSTM-MIL y entrenamiento
├── preproceso.py            # Gestión de dispositivo (CUDA/CPU) y expansión cronológica de mensajes
├── utilidades.py            # Carga de JSON/CSV, filtrado sin fuga, métricas, tiempos y RAM/VRAM
├── parametros.json          # Configuración centralizada de datos, arquitectura, entrenamiento y salidas
├── semilla.csv              # 200 mensajes individuales anotados manualmente para la rama Multi-Task
├── graficas.py              # Generación de figuras vectoriales (PDF) y 300 DPI (PNG) para el póster
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Documentación del proyecto
└── output/                  # Generada automáticamente al ejecutar
    ├── resultados_modelos.csv   # Registro incremental de folds, OOF y Ensemble (métricas + recursos)
    ├── mejor_modelo.pt          # Checkpoint ligero (~2 MB) del mejor modelo por Test Macro F1
    └── graficas/                # Figuras generadas por graficas.py (PDF y PNG a 300 DPI)
```


## Instalación

Requiere **Python 3.10+** (recomendado entorno con GPU CUDA para el entrenamiento; en CPU la inferencia y generación de gráficas funcionan sin restricciones).

```bash
# 1. Crear y activar entorno virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

Dependencias mínimas (`requirements.txt`):

```text
torch>=2.1.0
transformers>=4.40.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

> **Acceso a `mental/mental-roberta-base`:** Este modelo es un repositorio con acceso solicitado (*gated*) en Hugging Face. Si es la primera vez que se descarga en el equipo, autentíquese previamente con `huggingface-cli login` (o `hf auth login`) tras aceptar sus condiciones de uso.

---

## Configuración (`parametros.json`)

Todos los hiperparámetros y rutas del experimento se centralizan en `parametros.json`, organizado en cinco bloques:

| Bloque | Parámetros clave | Valor por defecto | Descripción |
| :--- | :--- | :--- | :--- |
| **`datos`** | `file`, `seed_file` | CSV usuarios / `"semilla.csv"` | Rutas al dataset principal y a los 200 mensajes semilla. |
| | `target_col`, `label_col` | `"Post"`, `"Label"` | Nombres de las columnas de texto y etiqueta clínica. |
| | `test_size`, `split_seed` | `0.15`, `42` | Partición congelada de Test (idéntica al baseline). |
| **`modelo`** | `model_name` | `"mental/mental-roberta-base"` | Encoder preentrenado base. |
| | `lora_layers`, `lora_r`, `lora_alpha`, `lora_include_out` | `12`, `8`, `16.0`, `true` | Adaptación LoRA en atención (`Q`, `V` y proyección de salida). |
| | `hidden_dim`, `dropout`, `multi_sample_dropout_n` | `32`, `0.4`, `5` | Dimensión de la BiLSTM y regularización en la cabeza clasificadora. |
| | `grad_checkpointing` | `false` | Reduce el pico de VRAM a costa de mayor tiempo de entrenamiento. |
| **`entrenamiento`** | `max_length`, `max_msgs_train`, `msg_dropout` | `384`, `32`, `0.10` | Longitud máxima por mensaje, tope por usuario y dropout de mensajes. |
| | `batch_size`, `accum_steps` | `2`, `8` | Tamaño de lote efectivo de 16 usuarios por actualización. |
| | `lr_lora`, `lr_head` | `2e-4`, `1e-3` | Tasas de aprendizaje diferenciadas para LoRA y cabeza MIL. |
| | `tau`, `sord_sigma` | `0.75`, `0.38` | Escalar de *Balanced Softmax* y suavizado ordinal (SORD). |
| | `seed_alpha`, `use_top2_soup` | `0.20`, `true` | Peso Multi-Task de semillas y promediado de los 2 mejores checkpoints. |
| **`validacion`** | `n_folds`, `n_repeats`, `seed` | `5`, `1`, `42` | Validación cruzada estratificada sobre `Train+Dev`. |
| | `ventanas_k` | `[1, 3, 5, 10, 15, null]` | Límites de mensajes evaluados para detección temprana (RQ2). |
| **`salidas`** | `metrics_csv` | `"output/resultados_modelos.csv"` | Registro incremental de métricas y consumo de recursos. |
| | `best_model_path` | `"output/mejor_modelo.pt"` | Checkpoint ligero del mejor modelo según `Test Macro F1`. |

---

## Ejecución

### 1. Entrenamiento y evaluación completa (CV + OOF + Ensemble)

Edite `"file"` en `parametros.json` con la ruta de su CSV de 500 usuarios (o pásela mediante `-f`):

```bash
# Ejecución estándar usando parametros.json
python main.py -p parametros.json

# Sobreescribir dataset o esquema de repeticiones desde consola (ej. 2x5-Fold = 10 modelos)
python main.py -p parametros.json -f ruta/al/dataset.csv --n_repeats 2 --n_folds 5
```

Durante la ejecución:
1. Se separa el **Test congelado** (`15%`, 75 usuarios, `random_state=42`) sin tocarlo en ningún momento del entrenamiento ni en el filtrado de `semilla.csv`.
2. Se entrena validación cruzada estratificada sobre los **425 usuarios restantes** (`Train=340`, `Val=85` por fold), filtrando además cualquier coincidencia de `semilla.csv` con `Val` o `Test`.
3. Al finalizar cada fold se anexa su fila a `output/resultados_modelos.csv` y se actualiza `output/mejor_modelo.pt` si mejora el mejor `Test Macro F1` registrado.
4. Al cierre se calculan y guardan la validación **Out-Of-Fold (425 usuarios)**, las métricas **Media ± DE** por ventana temprana $k$, el **Ensemble en Test** y el resumen computacional (tiempo, RAM y VRAM).

### 2. Generación de figuras para el póster (PDF vectorial y PNG a 300 DPI)

```bash
# Genera las figuras en output/graficas/ con los datos consolidados del estudio
python graficas.py

# Opcional: actualizar la curva RQ2 leyendo el último Ensemble registrado en el CSV
python graficas.py --csv output/resultados_modelos.csv
```

### 3. Carga del mejor checkpoint guardado (`output/mejor_modelo.pt`)

El checkpoint almacena únicamente los pesos entrenables (LoRA + cabeza MIL) junto a su configuración y métricas:

```python
from modelo_jerarquico import cargar_checkpoint_mil

model, clases, metricas, hiperparametros = cargar_checkpoint_mil(
    "output/mejor_modelo.pt", device="cuda"
)
print(metricas)
```

---

## Esquema de `output/resultados_modelos.csv`

El archivo registra filas de forma incremental con columnas separadas para distinguir sin ambigüedad la validación interna de cada fold (`dev_fold_*`) de la evaluación principal de esa fila (`eval_*`):

| Bloque | Columnas | Descripción |
| :--- | :--- | :--- |
| **Identificación** | `timestamp`, `run_id`, `tipo`, `conjunto_eval`, `n_usuarios_eval`, `rep`, `fold`, `epoca_sel`, `ventana` | `tipo` distingue `modelo_fold`, `oof` y `ensemble`. `conjunto_eval` indica explícitamente si las columnas `eval_*` miden `test` (`75` usuarios) o `dev_oof` (`425` usuarios). |
| **Validación interna** | `dev_fold_auc`, `dev_fold_macro_f1`, `dev_fold_graded_f1` | Rendimiento sobre el fold de validación (`85` usuarios) usado para *early stopping* y selección de época. Quedan vacías en `oof` y `ensemble`. |
| **Evaluación principal** | `eval_macro_f1`, `eval_weighted_f1`, `eval_graded_f1`, `eval_graded_p`, `eval_graded_r`, `eval_auc`, `eval_qwk`, `eval_kappa`, `eval_mae_ordinal`, `eval_acc_pm1` | Métricas de clasificación y ordinales calculadas sobre `conjunto_eval`. `eval_acc_pm1` indica la proporción de predicciones exactas o a ±1 nivel clínico. |
| **F1 por clase** | `eval_f1_Supportive`, `eval_f1_Indicator`, `eval_f1_Ideation`, `eval_f1_Behavior`, `eval_f1_Attempt` | F1-score individual de cada clase clínica en `conjunto_eval` (incluyendo la clase minoritaria `Attempt`). |
| **Recursos** | `dispositivo`, `tiempo_entrenamiento_seg`, `tiempo_entrenamiento_min`, `ram_actual_mb`, `ram_pico_mb`, `vram_pico_asignada_mb`, `vram_pico_reservada_mb` | Tiempo de entrenamiento y consumo de memoria RAM/VRAM por fold y acumulado/pico global del *run*. |
| **Hiperparámetros** | `model_name`, `lora_layers`, `lora_r`, `max_length`, `tau`, `sord_sigma`, `seed_alpha`, `lr_lora`, `lr_head` | Configuración clave para trazabilidad entre experimentos. |

---
