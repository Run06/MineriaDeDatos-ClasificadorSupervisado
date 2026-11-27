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
    """Carga la configuración de hiperparámetros desde un archivo JSON."""
    try:
        with open(ruta, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"Error: El archivo de parámetros '{ruta}' no existe.")
        sys.exit(1)


def cargar_csv(ruta):
    """Carga un CSV con manejo limpio de error si no existe."""
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)


def filtrar_semillas_sin_fuga(seed_texts, seed_labels, textos_prohibidos_norm):
    """Elimina cualquier semilla cuyo texto aparezca en Val o Test."""
    t_limpios, y_limpios = [], []
    for texto, etiqueta in zip(seed_texts, seed_labels):
        if str(texto).strip().lower() not in textos_prohibidos_norm:
            t_limpios.append(texto)
            y_limpios.append(etiqueta)
    return t_limpios, np.array(y_limpios, dtype=np.int64)


def formatear_duracion(segundos):
    """Convierte segundos a formato legible (ej. 2m 15s o 1h 04m 10s)."""
    seg_totales = int(round(segundos))
    horas, resto = divmod(seg_totales, 3600)
    minutos, segs = divmod(resto, 60)
    if horas > 0:
        return f"{horas}h {minutos:02d}m {segs:02d}s"
    if minutos > 0:
        return f"{minutos}m {segs:02d}s"
    return f"{segundos:.1f}s"


def reiniciar_estadisticas_memoria(device):
    """Limpia caché y reinicia los contadores de pico de VRAM antes de cada fold."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _leer_ram_actual_mb():
    """Lee la memoria RSS actual del proceso en MB (Linux /proc o psutil si existe)."""
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
    """Lee el pico histórico de RAM (RSS) del proceso en MB."""
    if resource is not None:
        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # En macOS ru_maxrss viene en bytes; en Linux en kilobytes
        return max_rss / (1024.0 ** 2) if sys.platform == "darwin" else max_rss / 1024.0
    return _leer_ram_actual_mb()


def obtener_consumo_recursos(device):
    """Devuelve consumo actual/pico de RAM y pico de VRAM (si el dispositivo es CUDA)."""
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
    """Añade una fila al CSV de métricas creando cabecera solo si el archivo es nuevo."""
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
    """Lee el mejor Test Macro F1 previamente guardado en disco (o -1.0 si no existe)."""
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
    Construye una fila sin ambigüedad entre validación (dev_fold_*) y evaluación principal (eval_*):
      - conjunto_eval: 'test' (75 usuarios congelados) o 'dev_oof' (425 usuarios Out-Of-Fold).
      - dev_fold_*: métricas del fold de validación (85 usuarios) usado para early stopping.
      - eval_*: métricas calculadas exactamente sobre `conjunto_eval`.
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
    """Imprime la tabla Media ± StDev de todos los modelos individuales por ventana k."""
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
    """Imprime un bloque limpio con tiempos y picos de RAM/VRAM listo para el póster."""
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
    """Imprime AUC por clase, métricas ordinales/Graded, matriz clínica y radiografía de Attempt."""
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