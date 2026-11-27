"""
Genera las figuras definitivas en formato vectorial PDF y PNG a 300 DPI:
  1. output/graficas/rq2_deteccion_temprana.(pdf|png)
  2. output/graficas/rq1_matriz_propuesto.(pdf|png)
  3. output/graficas/rq1_matrices_comparativa.(pdf|png)

Uso:
  python graficas.py
  python graficas.py --csv output/resultados_modelos.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns


# DATOS EXPERIMENTALES (Test congelado común: n = 75 usuarios)

VENTANAS = ["1", "3", "5", "10", "15", "Todos"]
MSGS_MEDIOS = [1.0, 2.7, 4.1, 7.0, 9.1, 16.9]
CLASES = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]

LBL_PROPUESTO = "Propuesto: BiLSTM-MIL"
LBL_BASELINE = "Baseline: SVC"

# Modelo propuesto: Ensemble 2x5-Fold en Test congelado
PROPUESTO_RQ2 = {
    "macro_f1":  np.array([0.2504, 0.3432, 0.3970, 0.4357, 0.4392, 0.4453]),
    "graded_f1": np.array([0.3504, 0.4252, 0.4839, 0.5354, 0.5397, 0.5556]),
    "auc":       np.array([0.6269, 0.7068, 0.7416, 0.7704, 0.7719, 0.7654]),
}

# Media ± DE de los 10 modelos individuales (estabilidad entre folds)
PROPUESTO_STD_RQ2 = {
    "macro_f1_mean":  np.array([0.2352, 0.3469, 0.3913, 0.4526, 0.4499, 0.4329]),
    "macro_f1_std":   np.array([0.0264, 0.0463, 0.0372, 0.0323, 0.0257, 0.0347]),
    "graded_f1_mean": np.array([0.3447, 0.4269, 0.4685, 0.5426, 0.5428, 0.5276]),
    "graded_f1_std":  np.array([0.0295, 0.0408, 0.0395, 0.0207, 0.0237, 0.0395]),
    "auc_mean":       np.array([0.6222, 0.6916, 0.7277, 0.7566, 0.7554, 0.7542]),
    "auc_std":        np.array([0.0194, 0.0210, 0.0188, 0.0136, 0.0104, 0.0114]),
}

# Baseline: SVC completo (re-etiquetados + fallback, 6498 mensajes)
BASE_COMPLETO_RQ2 = {
    "macro_f1":  np.array([0.3115, 0.2649, 0.2880, 0.2835, 0.2493, 0.2394]),
    "graded_f1": np.array([0.3906, 0.3796, 0.4118, 0.4088, 0.3636, 0.3538]),
    "auc":       np.array([0.6265, 0.5979, 0.5883, 0.6212, 0.5877, 0.5701]),
}

# Matrices de confusión en orden clínico (Test congelado, historial completo, n=75)
CM_PROPUESTO = np.array(
    [
        [6, 6,  2, 1, 1],  # Supportive (16)
        [1, 6,  5, 3, 0],  # Indicator  (15)
        [3, 8, 14, 1, 0],  # Ideation   (26)
        [0, 0,  2, 7, 2],  # Behavior   (11)
        [0, 1,  1, 3, 2],  # Attempt    (7)
    ],
    dtype=int,
)

CM_BASE_COMPLETO = np.array(
    [
        [9, 2,  4, 1, 0],  # Supportive (16)
        [3, 2,  9, 1, 0],  # Indicator  (15)
        [9, 5, 10, 1, 1],  # Ideation   (26)
        [1, 1,  4, 2, 3],  # Behavior   (11)
        [2, 2,  2, 1, 0],  # Attempt    (7)
    ],
    dtype=int,
)

RESUMEN_PROPUESTO = {"macro_f1": 0.445, "graded_f1": 0.556, "qwk": 0.521, "acc_pm1": 84.0}
RESUMEN_BASE_COMPLETO = {"macro_f1": 0.239, "graded_f1": 0.354, "qwk": 0.240, "acc_pm1": 68.0}


PALETA = {
    "propuesto": "#0072B2",
    "baseline": "#D55E00",
    "texto": "#1F2937",
}


def configurar_estilo():
    """
        Configura los estilos globales de Seaborn y Matplotlib para figuras.
        Precondición:
            Las librerías `matplotlib.pyplot` y `seaborn` deben estar importadas correctamente.
        Entradas:
            Ninguna.
        Salidas:
            None.
        Postcondición:
            Modifica el estado global `plt.rcParams` y el tema activo de Seaborn con tipografía
            legible, resolución de 300 DPI y fuentes TrueType (`fonttype=42`) para exportación PDF.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.2,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 8.8,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def cargar_propuesto_desde_csv(ruta_csv, run_id=None):
    """
    Carga las métricas por ventana del modelo Ensemble desde un CSV de resultados.
    Precondición:
        - Debe existir el archivo `ruta_csv` en formato CSV legible por pandas.
        - El CSV debe contener al menos una fila con `tipo == "ensemble"` y las columnas
          `run_id`, `ventana` y las métricas de `macro_f1`, `graded_f1` y `auc`
          (con o sin prefijo `eval_`).
        - Para el `run_id` seleccionado deben existir todas las ventanas definidas en `VENTANAS`.
    Entradas:
        ruta_csv (str): Ruta al archivo CSV de resultados (`output/resultados_modelos.csv`).
        run_id (str | None): Identificador de ejecución a cargar; si es `None`, selecciona el último.
    Salidas:
        dict[str, np.ndarray]: Diccionario con claves `"macro_f1"`, `"graded_f1"` y `"auc"`,
        cada una asociada a un array NumPy de longitud `len(VENTANAS)` en el orden de `VENTANAS`.
    Postcondición:
        Devuelve el diccionario de arrays numéricos e imprime el `run_id` cargado. Si faltan
        filas del ensemble o alguna ventana, lanza `ValueError`.
    """
    df = pd.read_csv(ruta_csv)
    df_ens = df[df["tipo"] == "ensemble"].copy()
    if df_ens.empty:
        raise ValueError(f"No se encontraron filas con tipo='ensemble' en '{ruta_csv}'.")

    target_run = run_id if run_id is not None else df_ens["run_id"].iloc[-1]
    df_run = df_ens[df_ens["run_id"] == target_run].copy()
    df_run["ventana"] = df_run["ventana"].astype(str)

    col_mf1 = "eval_macro_f1" if "eval_macro_f1" in df_run.columns else "macro_f1"
    col_gf1 = "eval_graded_f1" if "eval_graded_f1" in df_run.columns else "graded_f1"
    col_auc = "eval_auc" if "eval_auc" in df_run.columns else "auc"

    out = {"macro_f1": [], "graded_f1": [], "auc": []}
    for v in VENTANAS:
        fila = df_run[df_run["ventana"] == v]
        if fila.empty:
            raise ValueError(f"Falta la ventana '{v}' para run_id={target_run} en '{ruta_csv}'.")
        out["macro_f1"].append(float(fila[col_mf1].iloc[0]))
        out["graded_f1"].append(float(fila[col_gf1].iloc[0]))
        out["auc"].append(float(fila[col_auc].iloc[0]))

    print(f"Métricas del Ensemble cargadas desde '{ruta_csv}' (run_id={target_run}).")
    return {k: np.array(val) for k, val in out.items()}


def guardar_figura(fig, ruta_base):
    """
        Guarda una figura de Matplotlib en formato vectorial PDF y en PNG a 300 DPI.
        Precondición:
            - `fig` debe ser una instancia válida y abierta de `matplotlib.figure.Figure`.
            - `ruta_base` debe incluir una ruta con directorio válido (sin extensión) y permisos
              de escritura en disco.
        Entradas:
            fig (matplotlib.figure.Figure): Figura a exportar.
            ruta_base (str): Ruta destino sin extensión (ej. `"output/graficas/rq2_deteccion_temprana"`).
        Salidas:
            None.
        Postcondición:
            Crea el directorio padre si no existía, genera los archivos `{ruta_base}.pdf` y
            `{ruta_base}.png` en disco e imprime la confirmación por consola.
    """
    os.makedirs(os.path.dirname(ruta_base), exist_ok=True)
    fig.savefig(f"{ruta_base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{ruta_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Guardado: {ruta_base}.pdf (vectorial LaTeX) y {ruta_base}.png (300 DPI)")


# GRÁFICA 1: CURVA DE DETECCIÓN TEMPRANA RQ2

def generar_grafica_rq2(prop_data, prop_std, base_comp, out_dir):
    """
        Genera y guarda la gráfica apilada (2×1) de detección temprana (Macro F1 y AUC Macro)
        comparando el modelo propuesto frente al baseline.
        Precondición:
            - `prop_data` y `base_comp` deben contener las claves `"macro_f1"` y `"auc"` con
              arrays de longitud igual a `len(VENTANAS)`.
            - `prop_std` debe contener `"macro_f1_mean"`, `"macro_f1_std"`, `"auc_mean"` y
              `"auc_std"` con arrays de longitud igual a `len(VENTANAS)`.
            - `VENTANAS` debe incluir el valor `"10"` y coincidir en longitud con `MSGS_MEDIOS`.
        Entradas:
            prop_data (dict[str, np.ndarray]): Métricas del Ensemble propuesto por ventana.
            prop_std (dict[str, np.ndarray]): Media y desviación estándar de los folds individuales.
            base_comp (dict[str, np.ndarray]): Métricas del baseline SVC por ventana.
            out_dir (str): Carpeta destino donde se guardarán las figuras.
        Salidas:
            None.
        Postcondición:
            Exporta `rq2_deteccion_temprana.pdf` y `rq2_deteccion_temprana.png` en `out_dir`
            y cierra la figura para liberar memoria.
    """
    x = np.arange(len(VENTANAS))
    etiquetas_x = [
        f"1 msg\n({MSGS_MEDIOS[0]:.1f})" if v == "1"
        else (f"{v} msgs\n({m:.1f})" if v != "Todos" else f"Todos\n({m:.1f})")
        for v, m in zip(VENTANAS, MSGS_MEDIOS)
    ]
    idx_k10 = VENTANAS.index("10")

    fig, (ax_f1, ax_auc) = plt.subplots(
        2, 1, figsize=(6.9, 6.0), sharex=True, gridspec_kw={"hspace": 0.16}
    )

    paneles = [
        (ax_f1, "macro_f1", "Macro F1-score\n(escala 0.0 – 1.0)"),
        (ax_auc, "auc", "AUC Macro OvR\n(escala 0.0 – 1.0)"),
    ]

    for ax, metrica, ylabel in paneles:
        media_ind = prop_std[f"{metrica}_mean"]
        std_ind = prop_std[f"{metrica}_std"]

        ax.fill_between(
            x,
            np.clip(media_ind - std_ind, 0.0, 1.0),
            np.clip(media_ind + std_ind, 0.0, 1.0),
            color=PALETA["propuesto"],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            prop_data[metrica],
            label=LBL_PROPUESTO,
            color=PALETA["propuesto"],
            marker="o",
            markersize=6.5,
            linewidth=2.3,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        ax.plot(
            x,
            base_comp[metrica],
            label=LBL_BASELINE,
            color=PALETA["baseline"],
            linestyle="--",
            marker="s",
            markersize=5.6,
            linewidth=1.9,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )

        ax.axvline(idx_k10, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)

    # Anotaciones Panel Macro F1
    ax_f1.annotate(
        "Estabilización temprana (10 msgs)\n"
        f"Macro F1 = {prop_data['macro_f1'][idx_k10]:.3f} (98% del valor final)",
        xy=(idx_k10, prop_data["macro_f1"][idx_k10]),
        xytext=(0.15, 0.72),
        fontsize=8.4,
        fontweight="bold",
        color=PALETA["texto"],
        bbox=dict(boxstyle="round,pad=0.33", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.9),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.1, connectionstyle="arc3,rad=-0.12"),
    )
    ax_f1.text(x[-1] + 0.12, prop_data["macro_f1"][-1], f"{prop_data['macro_f1'][-1]:.3f}",
               color=PALETA["propuesto"], fontsize=8.6, fontweight="bold", va="center", ha="left")
    ax_f1.text(x[-1] + 0.12, base_comp["macro_f1"][-1], f"{base_comp['macro_f1'][-1]:.3f}",
               color=PALETA["baseline"], fontsize=8.5, fontweight="bold", va="center", ha="left")

    # Anotaciones Panel AUC Macro
    ax_auc.axhline(0.50, color="#9CA3AF", linestyle="-.", linewidth=0.95, alpha=0.8, zorder=0)
    ax_auc.text(0.05, 0.455, "Nivel de azar (AUC = 0.50)", color="#6B7280", fontsize=8.0, ha="left", va="top")

    ax_auc.text(idx_k10, prop_data["auc"][idx_k10] + 0.065, f"{prop_data['auc'][idx_k10]:.3f}",
                color=PALETA["propuesto"], fontsize=8.5, fontweight="bold", ha="center", va="bottom")
    ax_auc.text(x[-1] + 0.12, prop_data["auc"][-1], f"{prop_data['auc'][-1]:.3f}",
                color=PALETA["propuesto"], fontsize=8.6, fontweight="bold", va="center", ha="left")
    ax_auc.text(x[-1] + 0.12, base_comp["auc"][-1], f"{base_comp['auc'][-1]:.3f}",
                color=PALETA["baseline"], fontsize=8.5, fontweight="bold", va="center", ha="left")

    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(etiquetas_x)
    ax_auc.set_xlim(-0.30, len(VENTANAS) - 0.25)
    ax_auc.set_xlabel("Límite máximo de mensajes leídos por usuario (entre paréntesis: media real)")

    # Leyenda compacta en 1 sola fila
    banda_proxy = patches.Patch(
        color=PALETA["propuesto"], alpha=0.22, label="±1 DE (10 folds)"
    )
    handles, labels = ax_f1.get_legend_handles_labels()
    handles.append(banda_proxy)
    labels.append("±1 DE (10 folds)")

    fig.suptitle(
        "RQ2: Detección temprana según historial de mensajes (Test, n = 75)",
        fontsize=11.4,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.940),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
    )

    fig.subplots_adjust(top=0.865, bottom=0.105, left=0.14, right=0.95)
    guardar_figura(fig, os.path.join(out_dir, "rq2_deteccion_temprana"))
    plt.close(fig)

def _dibujar_heatmap_clinico(ax, cm, clases, titulo, subtitulo, show_ylabel=True, cbar_ax=None):
    """
        Dibuja sobre unos ejes dados una matriz de confusión normalizada por filas (recall %)
        en orden clínico, resaltando la diagonal de aciertos exactos.
        Precondición:
            - `cm` debe ser una matriz NumPy 2D cuadrada de enteros no negativos con forma
              `(len(clases), len(clases))`.
            - `ax` debe ser un objeto `matplotlib.axes.Axes` válido (y `cbar_ax` también si no es `None`).
        Entradas:
            ax (matplotlib.axes.Axes): Ejes donde se dibuja el mapa de calor.
            cm (np.ndarray): Matriz de confusión en conteos absolutos.
            clases (list[str]): Etiquetas de clase en orden clínico creciente de riesgo.
            titulo (str): Título principal del subgráfico.
            subtitulo (str): Línea(s) de métricas resumen mostradas bajo el título.
            show_ylabel (bool): Indica si se muestran las etiquetas del eje vertical.
            cbar_ax (matplotlib.axes.Axes | None): Ejes opcionales para la barra de color compartida.
        Salidas:
            None.
        Postcondición:
            Modifica `ax` (y `cbar_ax` si se proporciona) dibujando el mapa de calor en escala
            0%–100%, las anotaciones de porcentaje y conteo absoluto, y los recuadros de la diagonal.
    """
    totales = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), totales, out=np.zeros_like(cm, dtype=float), where=totales != 0
    )

    anot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            anot[i, j] = f"{100.0 * cm_norm[i, j]:.0f}%\n({cm[i, j]})"

    etiquetas_y = [f"{c} (n={int(t)})" for c, t in zip(clases, totales.ravel())] if show_ylabel else False

    cbar_kws = (
        {
            "label": "Recall por clase real (%)",
            "format": mtick.PercentFormatter(xmax=1.0, decimals=0),
            "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
        if cbar_ax is not None
        else None
    )

    hm = sns.heatmap(
        cm_norm,
        annot=anot,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        linewidths=1.1,
        linecolor="white",
        square=True,
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        annot_kws={"fontsize": 8.8, "fontweight": "bold"},
        xticklabels=clases,
        yticklabels=etiquetas_y,
        ax=ax,
    )

    for texto, valor in zip(hm.texts, cm_norm.ravel()):
        texto.set_color("white" if valor >= 0.52 else PALETA["texto"])

    for i in range(len(clases)):
        ax.add_patch(
            patches.Rectangle((i, i), 1, 1, fill=False, edgecolor="#0F172A", linewidth=1.7, zorder=4)
        )

    ax.set_title(titulo, fontsize=10.8, fontweight="bold", pad=27)
    ax.text(
        0.5,
        1.018,
        subtitulo,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#374151",
        linespacing=1.28,
    )
    ax.set_xlabel("Nivel de riesgo predicho (menor → mayor riesgo)", fontsize=9.4, labelpad=7)
    if show_ylabel:
        ax.set_ylabel("Nivel de riesgo real (escala C-SSRS)", fontsize=9.4, labelpad=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="right")


def generar_matriz_individual(cm, clases, resumen, out_dir):
    """
        Genera y guarda la figura individual de la matriz de confusión clínica del modelo propuesto.
        Precondición:
            - `cm` debe tener dimensiones `(len(clases), len(clases))`.
            - `resumen` debe contener las claves numéricas `"macro_f1"`, `"graded_f1"`, `"qwk"`
              y `"acc_pm1"`.
        Entradas:
            cm (np.ndarray): Matriz de confusión del modelo propuesto en Test.
            clases (list[str]): Lista de nombres de clases en orden clínico.
            resumen (dict[str, float]): Métricas agregadas mostradas en el subtítulo.
            out_dir (str): Carpeta destino donde se guardará la figura.
        Salidas:
            None.
        Postcondición:
            Exporta `rq1_matriz_propuesto.pdf` y `rq1_matriz_propuesto.png` en `out_dir`
            y cierra la figura para liberar memoria.
    """
    fig = plt.figure(figsize=(6.6, 5.2))
    gs = GridSpec(1, 2, width_ratios=[1.0, 0.055], wspace=0.06, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    subtitulo = (
        f"Macro F1: {resumen['macro_f1']:.3f}  |  Graded F1: {resumen['graded_f1']:.3f}\n"
        f"Kappa cuadrático (QWK): {resumen['qwk']:.3f}  |  Exacto o ±1 nivel: {resumen['acc_pm1']:.1f}%"
    )
    _dibujar_heatmap_clinico(
        ax,
        cm,
        clases,
        titulo=LBL_PROPUESTO,
        subtitulo=subtitulo,
        show_ylabel=True,
        cbar_ax=cax,
    )

    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.20, right=0.88)
    guardar_figura(fig, os.path.join(out_dir, "rq1_matriz_propuesto"))
    plt.close(fig)


def generar_matrices_comparativa(cm_base, res_base, cm_prop, res_prop, clases, out_dir):
    """
        Genera y guarda la comparativa lado a lado entre la matriz de confusión del baseline
        y la del modelo propuesto con barra de color compartida.
        Precondición:
            - `cm_base` y `cm_prop` deben ser matrices cuadradas `(len(clases), len(clases))`.
            - `res_base` y `res_prop` deben incluir las claves `"macro_f1"`, `"graded_f1"`,
              `"qwk"` y `"acc_pm1"`.
        Entradas:
            cm_base (np.ndarray): Matriz de confusión del baseline en Test.
            res_base (dict[str, float]): Métricas resumen del baseline.
            cm_prop (np.ndarray): Matriz de confusión del modelo propuesto en Test.
            res_prop (dict[str, float]): Métricas resumen del modelo propuesto.
            clases (list[str]): Lista de clases en orden clínico.
            out_dir (str): Carpeta destino donde se guardará la figura.
        Salidas:
            None.
        Postcondición:
            Exporta `rq1_matrices_comparativa.pdf` y `rq1_matrices_comparativa.png` en `out_dir`
            y cierra la figura para liberar memoria.
    """
    fig = plt.figure(figsize=(11.0, 5.1))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.14, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    sub_base = (
        f"Macro F1: {res_base['macro_f1']:.3f}  |  Graded F1: {res_base['graded_f1']:.3f}\n"
        f"QWK: {res_base['qwk']:.3f}  |  Exacto o ±1 nivel clínico: {res_base['acc_pm1']:.1f}%"
    )
    sub_prop = (
        f"Macro F1: {res_prop['macro_f1']:.3f}  |  Graded F1: {res_prop['graded_f1']:.3f}\n"
        f"QWK: {res_prop['qwk']:.3f}  |  Exacto o ±1 nivel clínico: {res_prop['acc_pm1']:.1f}%"
    )

    _dibujar_heatmap_clinico(
        ax1, cm_base, clases, titulo=f"(a) {LBL_BASELINE}", subtitulo=sub_base, show_ylabel=True
    )
    _dibujar_heatmap_clinico(
        ax2, cm_prop, clases, titulo=f"(b) {LBL_PROPUESTO}", subtitulo=sub_prop, show_ylabel=False, cbar_ax=cax
    )

    fig.suptitle(
        "RQ1: Comparativa de matrices de confusión en orden clínico (Test, n = 75)",
        fontsize=11.6,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.12, right=0.91)
    guardar_figura(fig, os.path.join(out_dir, "rq1_matrices_comparativa"))
    plt.close(fig)


# PUNTO DE ENTRADA

def parse_args():
    """
        Procesa los argumentos opcionales de línea de comandos para el script de gráficas.

        Precondición:
            Los argumentos presentes en `sys.argv` deben ser compatibles con las opciones
            `--csv`, `--run_id` y `--out_dir`.
        Entradas:
            Ninguna por parámetro (lee directamente `sys.argv`).
        Salidas:
            argparse.Namespace: Objeto con los atributos `csv` (str | None), `run_id` (str | None)
            y `out_dir` (str).
        Postcondición:
            Devuelve el `Namespace` parseado o finaliza el programa si hay error de sintaxis
            o se solicita `--help`.
    """
    parser = argparse.ArgumentParser(description="Generador de gráficas vectoriales para póster DIN-A3")
    parser.add_argument(
        "--csv",
        default=None,
        help="Opcional: ruta a output/resultados_modelos.csv para cargar el Ensemble desde disco",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Opcional: run_id concreto a leer del CSV (por defecto el último)",
    )
    parser.add_argument(
        "--out_dir",
        default="output/graficas",
        help="Carpeta de salida para las figuras PDF/PNG",
    )
    return parser.parse_args()


def main():
    """
        Orquesta la configuración del estilo gráfico, la carga opcional de métricas desde CSV
        y la generación de las tres figuras definitivas para el póster.

        Precondición:
            - Si se pasa `--csv`, el archivo debe existir y cumplir el esquema esperado por
              `cargar_propuesto_desde_csv`.
            - Debe existir permiso de escritura en `--out_dir`.
        Entradas:
            Ninguna por parámetro (lee opciones de consola y constantes globales del módulo).
        Salidas:
            None.
        Postcondición:
            Genera en `--out_dir` los archivos PDF y PNG (300 DPI) correspondientes a
            `rq2_deteccion_temprana`, `rq1_matriz_propuesto` y `rq1_matrices_comparativa`.
    """
    args = parse_args()
    configurar_estilo()

    prop_data = (
        cargar_propuesto_desde_csv(args.csv, run_id=args.run_id)
        if args.csv is not None
        else PROPUESTO_RQ2
    )

    generar_grafica_rq2(prop_data, PROPUESTO_STD_RQ2, BASE_COMPLETO_RQ2, args.out_dir)
    generar_matriz_individual(CM_PROPUESTO, CLASES, RESUMEN_PROPUESTO, args.out_dir)
    generar_matrices_comparativa(
        CM_BASE_COMPLETO, RESUMEN_BASE_COMPLETO, CM_PROPUESTO, RESUMEN_PROPUESTO, CLASES, args.out_dir
    )


if __name__ == "__main__":
    main()