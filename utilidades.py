import csv
from datetime import datetime
import gc
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    import resource
except ImportError:
    resource = None

from modelo_jerarquico import calcular_metricas_graded


def cargar_parametros(ruta):
    """
    Carga la configuración de hiperparámetros y rutas desde un archivo JSON.
    Precondición:
        `ruta` debe apuntar a un archivo JSON existente, legible y con sintaxis válida
        en codificación UTF-8.
    Entradas:
        ruta (str): Ruta al archivo JSON de parámetros.
    Salidas:
        dict: Estructura de configuración deserializada desde el JSON.
    Postcondición:
        Devuelve el diccionario cargado. Si el archivo no existe, imprime un mensaje de
        error y finaliza la ejecución del proceso con código de salida 1.
    """
    try:
        with open(ruta, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"Error: El archivo de parámetros '{ruta}' no existe.")
        sys.exit(1)


def cargar_csv(ruta):
    """
    Carga un archivo CSV en un DataFrame de pandas con control de error de ruta.
    Precondición:
        `ruta` debe corresponder a un archivo CSV existente y legible por `pandas.read_csv`.
    Entradas:
        ruta (str): Ruta al archivo CSV a cargar.
    Salidas:
        pd.DataFrame: Tabla cargada desde el archivo CSV.
    Postcondición:
        Devuelve el DataFrame leído. Si el archivo no existe, imprime un mensaje de error
        y finaliza la ejecución del proceso con código de salida 1.
    """
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)


def filtrar_semillas_sin_fuga(seed_texts, seed_labels, textos_prohibidos_norm):
    """
    Filtra los mensajes semilla eliminando cualquier coincidencia exacta (normalizada)
    presente en los conjuntos de validación o test para garantizar cero fuga de datos.
    Precondición:
        - `seed_texts` y `seed_labels` deben ser secuencias de igual longitud.
        - `textos_prohibidos_norm` debe contener los textos prohibidos previamente
          normalizados mediante `.strip().lower()`.
    Entradas:
        seed_texts (Sequence[str]): Lista de textos de los mensajes semilla.
        seed_labels (Sequence[int]): Etiquetas numéricas correspondientes a cada semilla.
        textos_prohibidos_norm (set[str]): Conjunto de textos normalizados de Val/Test.
    Salidas:
        tuple[list[str], np.ndarray]:
            - `t_limpios`: lista de mensajes semilla permitidos.
            - `y_limpios`: array NumPy `int64` con las etiquetas alineadas.
    Postcondición:
        Ningún elemento de `t_limpios` coincide tras aplicar `.strip().lower()` con
        ningún elemento de `textos_prohibidos_norm`, manteniendo el orden relativo original.
    """
    t_limpios, y_limpios = [], []
    for texto, etiqueta in zip(seed_texts, seed_labels):
        if str(texto).strip().lower() not in textos_prohibidos_norm:
            t_limpios.append(texto)
            y_limpios.append(etiqueta)
    return t_limpios, np.array(y_limpios, dtype=np.int64)


def formatear_duracion(segundos):
    """
    Convierte una duración expresada en segundos a una cadena legible (horas, minutos y segundos).
    Precondición:
        `segundos` debe ser un valor numérico real no negativo (`segundos >= 0`).
    Entradas:
        segundos (float | int): Duración en segundos.
    Salidas:
        str: Cadena formateada como `"Xh YYm ZZs"`, `"Ym ZZs"` o `"X.Xs"`.
    Postcondición:
        Devuelve una representación textual compacta y determinista de la duración recibida.
    """
    seg_totales = int(round(segundos))
    horas, resto = divmod(seg_totales, 3600)
    minutos, segs = divmod(resto, 60)
    if horas > 0:
        return f"{horas}h {minutos:02d}m {segs:02d}s"
    if minutos > 0:
        return f"{minutos}m {segs:02d}s"
    return f"{segundos:.1f}s"


def reiniciar_estadisticas_memoria(device):
    """
    Ejecuta recolección de basura en Python y reinicia los contadores de pico de VRAM en CUDA.
    Precondición:
        `device` debe ser una instancia válida de `torch.device` (con GPU inicializada si es `"cuda"`).
    Entradas:
        device (torch.device): Dispositivo de cómputo activo.
    Salidas:
        None.
    Postcondición:
        Libera objetos no referenciados en CPU; si `device.type == "cuda"`, vacía la caché
        del asignador de PyTorch y pone a cero los picos de memoria registrados para `device`.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _leer_ram_actual_mb():
    """
    Consulta la memoria física residente (RSS) actual del proceso en megabytes (MB).
    Precondición:
        Se ejecuta dentro de un proceso Python activo (en Linux usa `/proc/self/status`
        y, en otros sistemas, recurre a `psutil` si está instalado).
    Entradas:
        Ninguna.
    Salidas:
        float: Consumo actual de RAM del proceso en MB, o `float("nan")` si no puede consultarse.
    Postcondición:
        Devuelve un flotante no negativo o `NaN` sin alterar el estado del proceso.
    """
    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fh:
                for linea in fh:
                    if linea.startswith("VmRSS:"):
                        return float(linea.split()[1]) / 1024.0
        except OSError:
            pass
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
    except ImportError:
        return float("nan")


def _leer_ram_pico_mb():
    """
    Consulta el pico histórico de memoria RAM (RSS) alcanzado por el proceso en megabytes (MB).
    Precondición:
        Ninguna; si el módulo estándar `resource` no está disponible (ej. Windows),
        delega en `_leer_ram_actual_mb()`.
    Entradas:
        Ninguna.
    Salidas:
        float: Pico máximo de RAM del proceso en MB (o `float("nan")` si no está disponible).
    Postcondición:
        Ajusta la unidad devuelta por `ru_maxrss` según la plataforma (bytes en macOS,
        kilobytes en Linux) devolviendo siempre el valor en MB.
    """
    if resource is not None:
        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # En macOS ru_maxrss viene en bytes; en Linux en kilobytes
        return max_rss / (1024.0 ** 2) if sys.platform == "darwin" else max_rss / 1024.0
    return _leer_ram_actual_mb()


def obtener_consumo_recursos(device):
    """
    Recopila el nombre del hardware y el consumo actual/pico de RAM y VRAM del proceso.
    Precondición:
        `device` debe ser una instancia válida de `torch.device`.
    Entradas:
        device (torch.device): Dispositivo sobre el que se mide el consumo (`cuda` o `cpu`).
    Salidas:
        dict[str, object]: Diccionario con las claves `"dispositivo"`, `"ram_actual_mb"`,
        `"ram_pico_mb"`, `"vram_pico_asignada_mb"` y `"vram_pico_reservada_mb"`.
    Postcondición:
        Si `device.type == "cuda"`, sincroniza la GPU antes de leer los picos de VRAM en MB;
        en CPU devuelve `0.0` en los campos de VRAM.
    """
    ram_actual = _leer_ram_actual_mb()
    ram_pico = _leer_ram_pico_mb()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        vram_asig = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        vram_res = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
        nombre_disp = torch.cuda.get_device_name(device)
    else:
        vram_asig = 0.0
        vram_res = 0.0
        nombre_disp = "CPU"

    return {
        "dispositivo": nombre_disp,
        "ram_actual_mb": ram_actual,
        "ram_pico_mb": ram_pico,
        "vram_pico_asignada_mb": vram_asig,
        "vram_pico_reservada_mb": vram_res,
    }


def anexar_fila_csv(ruta_csv, fila):
    """
    Anexa una fila de métricas a un archivo CSV, escribiendo la cabecera solo si es nuevo o está vacío.
    Precondición:
        - `fila` debe ser un diccionario no vacío cuyas claves coincidan con el esquema
          del CSV existente (si el archivo ya contiene datos).
        - Debe existir permiso de escritura en la ubicación de `ruta_csv`.
    Entradas:
        ruta_csv (str): Ruta al archivo CSV de registro de métricas.
        fila (dict[str, object]): Diccionario con los pares columna-valor a registrar.
    Salidas:
        None.
    Postcondición:
        Crea el directorio padre si no existía, añade exactamente una fila al final de
        `ruta_csv` en codificación UTF-8 y cierra el descriptor de archivo.
    """
    directorio = os.path.dirname(ruta_csv)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    escribir_cabecera = not os.path.exists(ruta_csv) or os.path.getsize(ruta_csv) == 0
    with open(ruta_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fila.keys()))
        if escribir_cabecera:
            writer.writeheader()
        writer.writerow(fila)


def obtener_mejor_f1_en_disco(ruta_ckpt):
    """
    Lee el valor de `test_macro_f1` registrado en un modelo entrenado previamente guardado en disco.
    Precondición:
        `ruta_ckpt` debe ser una ruta válida como cadena (puede no existir todavía en disco).
    Entradas:
        ruta_ckpt (str): Ruta al archivo `.pt` del modelo entrenado guardado previamente.
    Salidas:
        float: Valor de `test_macro_f1` almacenado en el archivo, o `-1.0` si el archivo
        no existe, está corrupto o carece de dicha métrica.
    Postcondición:
        Carga únicamente en CPU sin consumir VRAM ni modificar el archivo en disco.
    """
    if not os.path.exists(ruta_ckpt):
        return -1.0
    try:
        try:
            ckpt = torch.load(ruta_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ruta_ckpt, map_location="cpu")
        return float(ckpt.get("metricas", {}).get("test_macro_f1", -1.0))
    except Exception:
        return -1.0


def _fmt(valor, decimales=4):
    """
    Formatea un valor numérico con un número fijo de decimales o devuelve cadena vacía si falta.
    Precondición:
        `valor` debe ser `None` o un número real (`int` o `float`), y `decimales >= 0`.
    Entradas:
        valor (float | int | None): Valor numérico a formatear.
        decimales (int): Número de cifras decimales deseadas.
    Salidas:
        str: Número formateado como cadena o `""` si `valor` es `None` o `NaN`.
    Postcondición:
        Devuelve siempre una cadena segura para su escritura directa en CSV.
    """
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    return f"{valor:.{decimales}f}"


def construir_fila_metricas(
    run_id,
    tipo,
    conjunto_eval,
    rep,
    fold,
    epoca_sel,
    ventana,
    clases,
    y_true,
    y_pred,
    y_prob,
    cfg_mod,
    cfg_ent,
    dev_fold_auc=None,
    dev_fold_f1=None,
    dev_fold_gf1=None,
    tiempo_seg=None,
    recursos=None,
):
    """
    Calcula las métricas de clasificación, ordinales y por clase y construye una fila uniforme
    para el CSV diferenciando validación interna (`dev_fold_*`) de evaluación principal (`eval_*`).
    Precondición:
        - `y_true` e `y_pred` deben ser arrays 1D no vacíos de igual longitud `N` con
          etiquetas enteras en `[0, len(clases) - 1]`.
        - `y_prob` debe tener forma `(N, len(clases))` con probabilidades válidas por fila.
        - `cfg_mod` debe contener `model_name`, `lora_layers` y `lora_r`; `cfg_ent` debe
          contener `max_length`, `tau`, `sord_sigma`, `seed_alpha`, `lr_lora` y `lr_head`.
    Entradas:
        run_id (str): Identificador único de la ejecución.
        tipo (str): Tipo de registro (`"modelo_fold"`, `"oof"` o `"ensemble"`).
        conjunto_eval (str): Partición evaluada (`"test"` o `"dev_oof"`).
        rep (int): Número de repetición o total de repeticiones.
        fold (int | str): Índice de fold o `"ALL"`.
        epoca_sel (str): Época seleccionada o par Top-2 Soup (`"-"` en agregados).
        ventana (str): Límite de mensajes evaluado (`"1"`, `"10"`, `"Todos"`, etc.).
        clases (list[str]): Nombres de clases en orden clínico.
        y_true (np.ndarray): Etiquetas reales en orden clínico.
        y_pred (np.ndarray): Etiquetas predichas en orden clínico.
        y_prob (np.ndarray): Probabilidades predichas por clase.
        cfg_mod (dict): Bloque `modelo` de `parametros.json`.
        cfg_ent (dict): Bloque `entrenamiento` de `parametros.json`.
        dev_fold_auc (float | None): AUC de validación del fold interno (si aplica).
        dev_fold_f1 (float | None): Macro F1 de validación del fold interno (si aplica).
        dev_fold_gf1 (float | None): Graded F1 de validación del fold interno (si aplica).
        tiempo_seg (float | None): Tiempo de entrenamiento asociado en segundos.
        recursos (dict | None): Diccionario devuelto por `obtener_consumo_recursos`.
    Salidas:
        dict[str, object]: Fila completa lista para ser escrita por `anexar_fila_csv`.
    Postcondición:
        Devuelve un diccionario con orden de claves constante que incluye identificación,
        métricas `dev_fold_*`, métricas `eval_*`, F1 por clase, recursos e hiperparámetros.
    """
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    graded = calcular_metricas_graded(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    kap = cohen_kappa_score(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    acc_pm1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    f1_clases = f1_score(
        y_true, y_pred, labels=list(range(len(clases))), average=None, zero_division=0
    )
    recursos = recursos or {}

    fila = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "tipo": tipo,
        "conjunto_eval": conjunto_eval,
        "n_usuarios_eval": len(y_true),
        "rep": rep,
        "fold": fold,
        "epoca_sel": epoca_sel,
        "ventana": ventana,
        # Referencia del fold de validación (Dev interno)
        "dev_fold_auc": _fmt(dev_fold_auc),
        "dev_fold_macro_f1": _fmt(dev_fold_f1),
        "dev_fold_graded_f1": _fmt(dev_fold_gf1),
        # Métricas principales evaluadas sobre `conjunto_eval`
        "eval_macro_f1": _fmt(mf1),
        "eval_weighted_f1": _fmt(wf1),
        "eval_graded_f1": _fmt(graded["graded_f1"]),
        "eval_graded_p": _fmt(graded["graded_p"]),
        "eval_graded_r": _fmt(graded["graded_r"]),
        "eval_auc": _fmt(auc),
        "eval_qwk": _fmt(qwk),
        "eval_kappa": _fmt(kap),
        "eval_mae_ordinal": _fmt(mae),
        "eval_acc_pm1": _fmt(acc_pm1),
    }
    for c_name, c_f1 in zip(clases, f1_clases):
        fila[f"eval_f1_{c_name}"] = _fmt(c_f1)

    fila.update(
        {
            # Coste computacional y memoria
            "dispositivo": recursos.get("dispositivo", ""),
            "tiempo_entrenamiento_seg": _fmt(tiempo_seg, 2),
            "tiempo_entrenamiento_min": _fmt(
                None if tiempo_seg is None else tiempo_seg / 60.0, 2
            ),
            "ram_actual_mb": _fmt(recursos.get("ram_actual_mb"), 1),
            "ram_pico_mb": _fmt(recursos.get("ram_pico_mb"), 1),
            "vram_pico_asignada_mb": _fmt(recursos.get("vram_pico_asignada_mb"), 1),
            "vram_pico_reservada_mb": _fmt(recursos.get("vram_pico_reservada_mb"), 1),
            # Hiperparámetros clave
            "model_name": cfg_mod["model_name"],
            "lora_layers": cfg_mod["lora_layers"],
            "lora_r": cfg_mod["lora_r"],
            "max_length": cfg_ent["max_length"],
            "tau": cfg_ent["tau"],
            "sord_sigma": cfg_ent["sord_sigma"],
            "seed_alpha": cfg_ent["seed_alpha"],
            "lr_lora": cfg_ent["lr_lora"],
            "lr_head": cfg_ent["lr_head"],
        }
    )
    return fila


def imprimir_tabla_individuales(ventanas_k, resultados_val, resultados_test, total_modelos):
    """
    Imprime por consola la tabla resumen Media ± Desviación Estándar de los modelos
    individuales evaluados en cada ventana temporal `k`.

    Precondición:
        - Todas las claves de `ventanas_k` deben existir en `resultados_val` y `resultados_test`.
        - Cada subdiccionario debe contener listas numéricas no vacías para las métricas
          requeridas (`f1`, `gf1`, `auc`, `kappa`, `msgs`).
    Entradas:
        ventanas_k (Sequence[int | None]): Secuencia de ventanas evaluadas.
        resultados_val (dict): Métricas acumuladas en validación por ventana `k`.
        resultados_test (dict): Métricas acumuladas en test por ventana `k`.
        total_modelos (int): Número total de modelos promediados.
    Salidas:
        None.
    Postcondición:
        Muestra la tabla formateada por salida estándar sin modificar los diccionarios de entrada.
    """
    print("\n" + "=" * 118)
    print(f"RESULTADOS INDIVIDUALES (Media ± StDev sobre {total_modelos} modelos)")
    print("=" * 118)
    print(
        f"{'Ventana':>8} | {'Msgs':>5} | {'Dev Macro F1':>15} | {'Test Macro F1':>15} | "
        f"{'Test Graded F1':>15} | {'Test AUC':>15} | {'Test Kappa':>15}"
    )
    print("-" * 118)

    for k in ventanas_k:
        etiqueta_k = "Todos" if k is None else str(k)
        msgs = np.mean(resultados_test[k]["msgs"])
        v_m, v_s = np.mean(resultados_val[k]["f1"]), np.std(resultados_val[k]["f1"])
        t_m, t_s = np.mean(resultados_test[k]["f1"]), np.std(resultados_test[k]["f1"])
        g_m, g_s = np.mean(resultados_test[k]["gf1"]), np.std(resultados_test[k]["gf1"])
        a_m, a_s = np.mean(resultados_test[k]["auc"]), np.std(resultados_test[k]["auc"])
        k_m, k_s = np.mean(resultados_test[k]["kappa"]), np.std(resultados_test[k]["kappa"])

        print(
            f"{etiqueta_k:>8} | {msgs:5.1f} | {v_m:.4f} ± {v_s:.4f} | {t_m:.4f} ± {t_s:.4f} | "
            f"{g_m:.4f} ± {g_s:.4f} | {a_m:.4f} ± {a_s:.4f} | {k_m:.4f} ± {k_s:.4f}"
        )


def imprimir_resumen_recursos(
    tiempos_folds, tiempo_total_run, recursos_globales, device
):
    """
    Imprime un resumen estructurado del coste temporal y consumo de RAM/VRAM de la ejecución.
    Precondición:
        - `tiempos_folds` debe ser una secuencia de tiempos numéricos no negativos en segundos.
        - `recursos_globales` debe contener `"dispositivo"`, `"ram_actual_mb"`, `"ram_pico_mb"`,
          `"vram_pico_asignada_mb"` y `"vram_pico_reservada_mb"`.
        - `device` debe ser una instancia válida de `torch.device`.
    Entradas:
        tiempos_folds (Sequence[float]): Duración de entrenamiento de cada fold en segundos.
        tiempo_total_run (float): Tiempo total de pared de la ejecución completa en segundos.
        recursos_globales (dict[str, object]): Consumo global de memoria y dispositivo.
        device (torch.device): Dispositivo empleado durante la ejecución.
    Salidas:
        None.
    Postcondición:
        Imprime por salida estándar los tiempos medio/acumulado/total y los picos de RAM
        y VRAM (o `"N/A"` si la ejecución fue en CPU).
    """
    tiempo_ent_total = float(np.sum(tiempos_folds))
    tiempo_ent_medio = float(np.mean(tiempos_folds)) if tiempos_folds else 0.0
    tiempo_ent_std = float(np.std(tiempos_folds)) if tiempos_folds else 0.0

    print("\n" + "=" * 95)
    print("RESUMEN COMPUTACIONAL Y MEMORIA (para el póster)")
    print("=" * 95)
    print(f"Dispositivo                 : {recursos_globales['dispositivo']} ({device})")
    print(
        f"Tiempo entrenamiento/modelo : {formatear_duracion(tiempo_ent_medio)} "
        f"(± {tiempo_ent_std:.1f}s) | Acumulado ({len(tiempos_folds)} modelos): "
        f"{formatear_duracion(tiempo_ent_total)}"
    )
    print(f"Tiempo total ejecución      : {formatear_duracion(tiempo_total_run)}")
    print(
        f"RAM proceso (actual / pico) : {recursos_globales['ram_actual_mb']:.1f} MB / "
        f"{recursos_globales['ram_pico_mb']:.1f} MB ({recursos_globales['ram_pico_mb'] / 1024.0:.2f} GB)"
    )
    if device.type == "cuda":
        asig = recursos_globales["vram_pico_asignada_mb"]
        res = recursos_globales["vram_pico_reservada_mb"]
        print(
            f"VRAM CUDA pico (asig / res) : {asig:.1f} MB ({asig / 1024.0:.2f} GB) / "
            f"{res:.1f} MB ({res / 1024.0:.2f} GB)"
        )
    else:
        print("VRAM CUDA pico              : N/A (ejecución en CPU)")
    print("=" * 95)


def imprimir_diagnostico_test(y_true, y_pred, y_prob, clases, titulo="Test, k=Todos"):
    """
    Imprime un diagnóstico clínico detallado sobre un conjunto de evaluación: AUC OvR
    por clase, matriz de confusión ordinal, métricas Graded/QWK/MAE y radiografía de `Attempt`.
    Precondición:
        - `y_true` e `y_pred` deben ser arrays NumPy 1D de igual longitud `N` con índices
          en `[0, len(clases) - 1]`.
        - `y_prob` debe ser un array NumPy 2D de forma `(N, len(clases))`.
    Entradas:
        y_true (np.ndarray): Etiquetas reales en orden clínico.
        y_pred (np.ndarray): Etiquetas predichas en orden clínico.
        y_prob (np.ndarray): Matriz de probabilidades predichas por clase.
        clases (list[str]): Lista de nombres de clases en orden clínico creciente.
        titulo (str): Etiqueta descriptiva mostrada en las cabeceras del diagnóstico.
    Salidas:
        None.
    Postcondición:
        Imprime por salida estándar el desglose de rendimiento clínico sin modificar los
        arrays de entrada; si `"Attempt"` figura en `clases`, muestra además las
        probabilidades individuales de todos los usuarios reales de dicha clase.
    """
    aucs_clase = []
    for i, c in enumerate(clases):
        try:
            auc_c = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            aucs_clase.append(f"{c}:{auc_c:.3f}")
        except ValueError:
            aucs_clase.append(f"{c}:nan")
    print(f"\nAUC OvR por clase ({titulo}): " + " | ".join(aucs_clase))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(clases))))
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real:{c[:4]}" for c in clases],
        columns=[f"Pred:{c[:4]}" for c in clases],
    )
    print(f"\nMatriz de confusión ({titulo}):")
    print(df_cm.to_string())

    graded = calcular_metricas_graded(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    distancias = np.abs(y_true - y_pred)
    print(
        f"\nGraded Precision: {graded['graded_p']:.4f} | "
        f"Graded Recall: {graded['graded_r']:.4f} | "
        f"Graded F1: {graded['graded_f1']:.4f}"
    )
    print(
        f"QWK (Kappa cuadrático): {qwk:.4f} | "
        f"MAE ordinal (0-4): {distancias.mean():.3f} | "
        f"Exacto o ±1 nivel: {100 * np.mean(distancias <= 1):.1f}%"
    )

    if "Attempt" in clases:
        idx_att = clases.index("Attempt")
        indices_att = np.where(y_true == idx_att)[0]
        cabecera = "  ".join(f"{c[:4]:>6}" for c in clases)
        print(f"\nRadiografía de los {len(indices_att)} usuarios reales de Attempt ({titulo}):")
        print(f"  {'Usr':>3} | {'Pred':<10} | {cabecera}")
        print("  " + "-" * (18 + 8 * len(clases)))
        for pos in indices_att:
            pred_nombre = clases[y_pred[pos]]
            probs_str = "  ".join(f"{y_prob[pos, i]:6.3f}" for i in range(len(clases)))
            print(f"  {pos:3d} | {pred_nombre:<10} | {probs_str}")