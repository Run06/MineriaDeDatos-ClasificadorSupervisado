import argparse
import json
import pandas as pd

from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from eval import predecir_por_usuario
from main import cargar_csv
from preproceso import (
    dividir_mensajes,
    vectorizar_embeddings_y_vader,
)
from re_etiquetado import cascading_self_training


def emparejar_vader_con_embeddings(
    X_tr_emb, X_seed_emb, vader_seed, X_reddit_emb, vader_train
):
    """
    - Precondiciones: Las matrices de embeddings tienen las mismas columnas D y los vectores VADER tienen 4 columnas.
    - Entradas: X_tr_emb (np.ndarray con embeddings aumentados), X_seed_emb (np.ndarray de semillas),
      vader_seed (np.ndarray VADER de semillas), X_reddit_emb (np.ndarray del train) y vader_train (np.ndarray VADER del train).
    - Salidas: np.ndarray de forma (M, 4) en float32 alineado fila a fila con X_tr_emb.
    - Postcondiciones: Devuelve la matriz VADER exacta correspondiente a cada fila de X_tr_emb.
    """
    #Construimos un diccionario por bytes del embedding para recuperar su fila VADER
    mapa_vader = {}
    for emb_row, vad_row in zip(
        np.asarray(X_seed_emb, dtype=np.float32),
        np.asarray(vader_seed, dtype=np.float32),
    ):
        mapa_vader.setdefault(emb_row.tobytes(), vad_row)

    for emb_row, vad_row in zip(
        np.asarray(X_reddit_emb, dtype=np.float32),
        np.asarray(vader_train, dtype=np.float32),
    ):
        mapa_vader.setdefault(emb_row.tobytes(), vad_row)

    #Recorremos el conjunto aumentado buscando cada vector
    filas = []
    for i, emb_row in enumerate(np.asarray(X_tr_emb, dtype=np.float32)):
        clave = emb_row.tobytes()
        if clave not in mapa_vader:
            raise KeyError(
                f"Falta el vector VADER para la fila {i} del conjunto aumentado."
            )
        filas.append(mapa_vader[clave])

    X_tr_vader = np.asarray(filas, dtype=np.float32)
    assert len(X_tr_vader) == len(X_tr_emb), (
        f"No cuadran las filas de VADER ({len(X_tr_vader)}) y embeddings ({len(X_tr_emb)})"
    )
    return X_tr_vader


def entrenar_modelo(tipo_modelo, X_tr, y_tr, usar_pesos=True, random_state=42, ruta_config="config.json"):
    """
    - Precondiciones: tipo_modelo es 'svm', 'xgb' o 'logistic', X_tr es matriz 2D y len(X_tr) == len(y_tr) > 0.
    - Entradas: tipo_modelo (str), X_tr (np.ndarray), y_tr (array de etiquetas), usar_pesos (bool) y random_state (int).
    - Salidas: tupla (clf, clases_unicas) con el modelo entrenado y las etiquetas ordenadas.
    - Postcondiciones: El clasificador queda ajustado sobre X_tr e y_tr listo para predecir.
    """
    with open(ruta_config, "r", encoding="utf-8") as f:
        cfg_modelos = json.load(f)["modelos"]
    #Convertimos etiquetas de texto a indices enteros
    clases_unicas = np.unique(y_tr)
    idx_map = {c: i for i, c in enumerate(clases_unicas)}
    y_idx = np.array([idx_map[v] for v in y_tr], dtype=int)
    pesos = compute_sample_weight(class_weight="balanced", y=y_idx) if usar_pesos else None

    if tipo_modelo == "svm":
        cfg_svm = cfg_modelos["svm"]
        clf = SVC(
            kernel=cfg_svm["kernel"],
            C=cfg_svm["C"],
            gamma=cfg_svm["gamma"],
            class_weight="balanced" if usar_pesos else None,
            random_state=random_state,
        )
        clf.fit(X_tr, y_idx)
        return clf, clases_unicas

    if tipo_modelo == "xgb":
        cfg_xgb = cfg_modelos["xgb"]
        num_clases = len(clases_unicas)
        clf = XGBClassifier(
            n_estimators=cfg_xgb["n_estimators"],
            max_depth=cfg_xgb["max_depth"],
            learning_rate=cfg_xgb["learning_rate"],
            subsample=cfg_xgb["subsample"],
            colsample_bytree=cfg_xgb["colsample_bytree"],
            eval_metric="mlogloss" if num_clases > 2 else "logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_idx, sample_weight=pesos)
        return clf, clases_unicas

    raise ValueError(f"Modelo no reconocido: {tipo_modelo}")


def obtener_puntuaciones(clf, X_mat):
    """
    - Precondiciones: clf esta entrenado y X_mat tiene las mismas columnas que en entrenamiento.
    - Entradas: clf (estimador entrenado) y X_mat (np.ndarray 2D).
    - Salidas: np.ndarray 2D de forma (N, C) con scores o probabilidades por clase.
    - Postcondiciones: Devuelve una matriz 2D donde cada columna j corresponde a la clase j.
    """
    #Si el modelo tiene decision_function (como SVM), usamos sus distancias al hiperplano
    if hasattr(clf, "decision_function"):
        scores = np.asarray(clf.decision_function(X_mat))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores
    return np.asarray(clf.predict_proba(X_mat))


def buscar_mejor_k_dev(scores_dev, clases, dev_msgs, col_label, candidatos_k):
    """
    - Precondiciones: candidatos_k no esta vacio y len(scores_dev) == len(dev_msgs).
    - Entradas: scores_dev (np.ndarray), clases (np.ndarray), dev_msgs (DataFrame), col_label (str) y candidatos_k (iterable).
    - Salidas: tupla (mejor_k, f1_dev_por_k) con el mejor k en Dev y el diccionario de resultados por k.
    - Postcondiciones: Devuelve el numero de mensajes k que maximiza el Macro-F1 en Dev sin tocar Test.
    """
    f1_dev_por_k = {}
    mejor_k = None
    mejor_f1 = -1.0

    #Probamos cada ventana k en validacion y nos quedamos con la mejor
    for k in candidatos_k:
        y_dev_u, preds_dev_u, _ = predecir_por_usuario(
            scores_dev, clases, dev_msgs, col_label, max_mensajes=k
        )
        f1_dev = float(f1_score(y_dev_u, preds_dev_u, average="macro", zero_division=0))
        f1_dev_por_k[k] = f1_dev
        if f1_dev > mejor_f1:
            mejor_f1 = f1_dev
            mejor_k = k

    return mejor_k, f1_dev_por_k


def ver_peso_vader_xgb(clf_xgb, n_emb):
    """
    - Precondiciones: clf_xgb es un XGBoost ajustado y tiene al menos n_emb + 4 caracteristicas.
    - Entradas: clf_xgb (XGBClassifier) y n_emb (int con el numero de columnas de embeddings).
    - Salidas: dict con el porcentaje de importancia de embeddings, de VADER y el detalle por columna.
    - Postcondiciones: Muestra por pantalla el peso relativo de VADER frente a los embeddings y devuelve el resumen.
    """
    imp = np.asarray(clf_xgb.feature_importances_, dtype=float)
    imp_emb = imp[:n_emb]
    imp_vader = imp[n_emb : n_emb + 4]
    nombres_vader = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]

    total_imp = imp.sum() if imp.sum() > 0 else 1.0
    pct_vader = float(imp_vader.sum() / total_imp * 100.0)
    pct_emb = float(imp_emb.sum() / total_imp * 100.0)

    print("\nPeso de variables en XGBoost (Embeddings + VADER):")
    print(f"  Embeddings ({n_emb} cols): {pct_emb:.2f}%")
    print(f"  VADER (4 cols):          {pct_vader:.2f}%")
    for nombre, val in zip(nombres_vader, imp_vader):
        print(f"    * {nombre:15s}: {val:.6f}")

    return {
        "pct_embeddings": round(pct_emb, 2),
        "pct_vader": round(pct_vader, 2),
        "por_columna_vader": dict(zip(nombres_vader, np.round(imp_vader, 6))),
    }


def evaluar_usuario_desbalanceado(y_true, y_pred, clase_minoritaria="Attempt"):
    """
    - Precondiciones: len(y_true) == len(y_pred) > 0.
    - Entradas: y_true (etiquetas reales por usuario), y_pred (etiquetas predichas) y clase_minoritaria (str).
    - Salidas: dict con macro_f1, weighted_f1, kappa y f1_minoritaria.
    - Postcondiciones: Devuelve un diccionario con las metricas clave para clases desbalanceadas.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    f1_por_clase = f1_score(
        y_true, y_pred, labels=[clase_minoritaria], average=None, zero_division=0
    )
    f1_min = float(f1_por_clase[0]) if len(f1_por_clase) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "f1_minoritaria": f1_min,
    }


import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLOR_MACRO = "#A8D5A2"    # verde claro
COLOR_ATTEMPT = "#E8D5A9"  # beige

# Etiquetas cortas (mismas que las tablas del poster) para poder usar letra grande
ETIQUETAS_CORTAS = {
    "Baseline (Embeddings + SVM)": "SVM\nEmb (base)",
    "Embeddings + VADER (SVM)": "SVM\nEmb+VADER",
    "Embeddings (XGBoost)": "XGB\nEmb",
    "VADER solo (XGBoost)": "XGB\nVADER",
    "Embeddings + VADER (XGBoost)": "XGB\nEmb+VADER",
}


def guardar_grafico_comparativa(
    df_multi_run, conjunto="full", out_dir="poster_figuras", df_runs=None,
    ymax=None, figsize=(10, 4.8),
):
    """
    - Precondiciones: df_multi_run no esta vacio y contiene las columnas de medias y desviaciones en Test.
    - Entradas: df_multi_run (DataFrame resumen), conjunto (str con 'sin_pseudo', 'clean' o 'full'),
      out_dir (str), df_runs (DataFrame con una fila por run y modelo, opcional), ymax (float o None; si None
      el eje se ajusta a los datos) y figsize (tupla en pulgadas; usar el MISMO ancho con el que se incluye en el poster).
    - Salidas: ruta_png (str con la ruta del archivo guardado o None si ocurre un error).
    - Postcondiciones: Guarda un PNG con letra grande (pensado para imprimir en A3) con barras (media),
      barras de error (± StDev entre runs), un punto por run si se pasa df_runs y la media y la StDev escritas.
    """
    if df_multi_run is None or len(df_multi_run) == 0:
        return None

    try:
        os.makedirs(out_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")

        df_plot = df_multi_run.copy()
        nombres = df_plot["Configuracion"].tolist()
        x = np.arange(len(df_plot))
        ancho = 0.36

        macro_m = df_plot["Test_MacroF1_mean"].to_numpy()
        macro_s = df_plot["Test_MacroF1_std"].to_numpy()
        att_m = df_plot["Test_F1_Attempt_mean"].to_numpy()
        att_s = df_plot["Test_F1_Attempt_std"].to_numpy()

        # La barra de error inferior no baja de 0 (un F1 no puede ser negativo)
        yerr_macro = [np.minimum(macro_m, macro_s), macro_s]
        yerr_att = [np.minimum(att_m, att_s), att_s]

        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.set_axisbelow(True)
        borde = {"edgecolor": "#5A5A5A", "linewidth": 1.0}
        err = {"elinewidth": 1.8, "alpha": 0.9, "capthick": 1.8}
        barras_macro = ax.bar(
            x - ancho / 2, macro_m, width=ancho, color=COLOR_MACRO,
            yerr=yerr_macro, capsize=5, error_kw=err, label="Macro-F1", **borde,
        )
        barras_att = ax.bar(
            x + ancho / 2, att_m, width=ancho, color=COLOR_ATTEMPT,
            yerr=yerr_att, capsize=5, error_kw=err, label="F1 Attempt", **borde,
        )

        #Un punto por run: con solo 3 runs es mas honesto que una barra de error sola
        if df_runs is not None and len(df_runs) > 0:
            for i, nombre in enumerate(nombres):
                sub = df_runs[df_runs["Configuracion"] == nombre]
                for col, desp in [("Test_MacroF1", -ancho / 2), ("Test_F1_Attempt", ancho / 2)]:
                    ys = sub[col].to_numpy()
                    xs = np.full(len(ys), i + desp) + np.linspace(-0.05, 0.05, len(ys))
                    ax.scatter(xs, ys, s=40, color="black", zorder=5, alpha=0.75)

        #Si un F1 vale 0 no se ve barra, rombo para que no parezca un hueco
        for i, h in enumerate(att_m):
            if h == 0:
                ax.plot(i + ancho / 2, 0, marker="D", ms=10, markerfacecolor=COLOR_ATTEMPT,
                        markeredgecolor="#5A5A5A", clip_on=False, zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [ETIQUETAS_CORTAS.get(n, textwrap.fill(n, 12)) for n in nombres], fontsize=15
        )
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylabel("F1", fontsize=17)
        ax.xaxis.grid(False)
        sns.despine(ax=ax, top=True, right=True)

        tope = float(np.max(np.concatenate([macro_m + macro_s, att_m + att_s])))
        techo = tope + 0.11
        ax.set_ylim(0.0, techo if ymax is None else max(ymax, techo))

        #Encima de cada barra: la MEDIA (negrita) y debajo la desviacion (±StDev)
        for barras, medias, stds in [(barras_macro, macro_m, macro_s), (barras_att, att_m, att_s)]:
            for b, h, s in zip(barras, medias, stds):
                ax.annotate(
                    f"{h:.2f}\n±{s:.2f}",
                    xy=(b.get_x() + b.get_width() / 2, h + float(s)),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    linespacing=1.0,
                    fontweight="bold",
                )

        #Leyenda encima del grafico
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=16)

        plt.tight_layout()
        ruta_png = os.path.join(out_dir, f"comparativa_{conjunto}.png")
        plt.savefig(ruta_png, bbox_inches="tight")
        plt.close(fig)
        print(f"\nGrafico guardado en: {ruta_png}")
        return ruta_png
    except Exception as exc:
        print(f"\nNo se pudo generar el grafico: {exc}")
        return None


def medir_falsos_positivos_graves(clf, clases, x_test, test_msgs, col_label, orden_gravedad):
    """
    - Precondiciones: Todas las etiquetas reales y predichas existen dentro de orden_gravedad.
    - Entradas: clf (modelo), clases (np.ndarray), x_test (np.ndarray), test_msgs (DataFrame),
      col_label (str) y orden_gravedad (lista de menor a mayor gravedad).
    - Salidas: tupla (fp_grave_tasa, gap_medio) con la tasa de usuarios sobreestimados y el exceso medio de gravedad.
    - Postcondiciones: Devuelve la proporcion de usuarios con prediccion maxima > real maxima y su salto medio.
    """
    etiquetas_presentes = set(test_msgs[col_label].unique())
    if not etiquetas_presentes.issubset(set(orden_gravedad)):
        fuera = sorted(etiquetas_presentes - set(orden_gravedad))
        raise ValueError(f"Hay etiquetas que no estan en --orden-gravedad: {fuera}")

    rank = {etiqueta: idx for idx, etiqueta in enumerate(orden_gravedad)}
    preds_msg = predecir_etiquetas_mensajes(clf, clases, x_test)
    preds_set = set(preds_msg)
    if not preds_set.issubset(set(orden_gravedad)):
        fuera_p = sorted(preds_set - set(orden_gravedad))
        raise ValueError(f"Hay predicciones que no estan en --orden-gravedad: {fuera_p}")

    #Agrupamos por usuario tomando el nivel maximo real y el nivel maximo predicho
    aux = test_msgs[["user_id", col_label]].copy()
    aux["_rank_real"] = aux[col_label].map(rank)
    aux["_rank_pred"] = [rank[p] for p in preds_msg]

    por_u = (
        aux.groupby("user_id", as_index=False)
        .agg(
            real_rank=("_rank_real", "max"),
            pred_rank=("_rank_pred", "max"),
        )
    )
    total_u = len(por_u)
    fp_grave_tasa = float((por_u["pred_rank"] > por_u["real_rank"]).sum()) / max(total_u, 1)
    exceso = np.maximum(0, por_u["pred_rank"].to_numpy() - por_u["real_rank"].to_numpy())
    gap_medio = float(exceso.mean()) if len(exceso) > 0 else 0.0
    return fp_grave_tasa, gap_medio


def predecir_etiquetas_mensajes(clf, clases, X_mat):
    """
    - Precondiciones: clf esta entrenado y X_mat es compatible con sus columnas.
    - Entradas: clf (estimador), clases (np.ndarray) y X_mat (np.ndarray 2D).
    - Salidas: np.ndarray 1D con los nombres de clase predichos para cada fila.
    - Postcondiciones: Devuelve las etiquetas originales correspondientes a los indices predichos.
    """
    idx_pred = np.asarray(clf.predict(X_mat), dtype=int)
    return clases[idx_pred]


def probar_sensibilidad_ruido(
    baseline_tuple,
    modelo_aporte_tuple,
    pares_frases,
    usa_vader_en_aporte=True,
    etiqueta_base="Baseline",
    etiqueta_aporte="Aporte",
):
    """
    - Precondiciones: Ambos modelos estan entrenados y pares_frases contiene tuplas de dos textos no vacios.
    - Entradas: baseline_tuple, modelo_aporte_tuple, pares_frases, usa_vader_en_aporte (bool),
      etiqueta_base (str) y etiqueta_aporte (str).
    - Salidas: df_res (DataFrame con las predicciones antes y despues de variar la frase).
    - Postcondiciones: Imprime la tasa de cambio ante ruido superficial y devuelve el DataFrame detallado.
    """
    clf_base, clases_base, _ = baseline_tuple
    clf_app, clases_app, _ = modelo_aporte_tuple

    #Vectorizamos cada par (frase limpia vs frase con jerga/erratas) y comparamos predicciones
    registros = []
    for orig, var in pares_frases:
        emb, _, concat = vectorizar_embeddings_y_vader([orig, var], cache_dir="cache")
        p_base = predecir_etiquetas_mensajes(clf_base, clases_base, emb)
        x_app = concat if usa_vader_en_aporte else emb
        p_app = predecir_etiquetas_mensajes(clf_app, clases_app, x_app)

        registros.append({
            "original": orig,
            "variada": var,
            "base_orig": p_base[0],
            "base_var": p_base[1],
            "cambio_base": bool(p_base[0] != p_base[1]),
            "aporte_orig": p_app[0],
            "aporte_var": p_app[1],
            "cambio_aporte": bool(p_app[0] != p_app[1]),
        })

    df_res = pd.DataFrame(registros)
    if len(df_res) > 0:
        print("\nPrueba RQ1 - Sensibilidad a cambios superficiales (nivel mensaje):")
        print(f"  Tasa de cambio {etiqueta_base}: {df_res['cambio_base'].mean():.4f}")
        print(f"  Tasa de cambio {etiqueta_aporte}: {df_res['cambio_aporte'].mean():.4f}")
        print(df_res.to_string(index=False))
    return df_res


def analizar_falsos_positivos_usuarios(
    test_msgs,
    col_label,
    baseline_tuple,
    modelo_aporte_tuple,
    orden_gravedad,
    etiqueta_base="Baseline",
    etiqueta_aporte="Aporte",
):
    """
    - Precondiciones: Las etiquetas de test_msgs y de ambos modelos estan incluidas en orden_gravedad.
    - Entradas: test_msgs (DataFrame), col_label (str), baseline_tuple, modelo_aporte_tuple,
      orden_gravedad (lista), etiqueta_base (str) y etiqueta_aporte (str).
    - Salidas: resumen (DataFrame con el rango real y predicho por cada usuario).
    - Postcondiciones: Imprime la comparativa de falsos positivos graves e intervalo bootstrap y devuelve resumen.
    """
    etiquetas_presentes = set(test_msgs[col_label].unique())
    if not etiquetas_presentes.issubset(set(orden_gravedad)):
        fuera = sorted(etiquetas_presentes - set(orden_gravedad))
        raise ValueError(f"Hay etiquetas que no estan en --orden-gravedad: {fuera}")

    rank = {etiqueta: idx for idx, etiqueta in enumerate(orden_gravedad)}
    clf_base, clases_base, x_test_base = baseline_tuple
    clf_app, clases_app, x_test_app = modelo_aporte_tuple

    pred_msg_base = predecir_etiquetas_mensajes(clf_base, clases_base, x_test_base)
    pred_msg_app = predecir_etiquetas_mensajes(clf_app, clases_app, x_test_app)

    aux = test_msgs[["user_id", col_label]].copy()
    aux["_rank_real"] = aux[col_label].map(rank)
    aux["_rank_base"] = [rank[p] for p in pred_msg_base]
    aux["_rank_app"] = [rank[p] for p in pred_msg_app]

    resumen = (
        aux.groupby("user_id", as_index=False)
        .agg(
            real_rank=("_rank_real", "max"),
            base_rank=("_rank_base", "max"),
            app_rank=("_rank_app", "max"),
            n_mensajes=("_rank_real", "count"),
        )
    )
    resumen["fp_grave_base"] = resumen["base_rank"] > resumen["real_rank"]
    resumen["fp_grave_app"] = resumen["app_rank"] > resumen["real_rank"]

    total_u = len(resumen)
    fp_b = int(resumen["fp_grave_base"].sum())
    fp_a = int(resumen["fp_grave_app"].sum())

    #Calculamos intervalo de confianza por bootstrap sobre la diferencia por usuario
    dif_u = resumen["fp_grave_base"].astype(int).to_numpy() - resumen["fp_grave_app"].astype(int).to_numpy()
    rng = np.random.default_rng(42)
    n_boot = 2000
    medias_boot = np.array([
        rng.choice(dif_u, size=len(dif_u), replace=True).mean() if len(dif_u) > 0 else 0.0
        for _ in range(n_boot)
    ])
    ic_low, ic_high = np.percentile(medias_boot, [2.5, 97.5]) if len(dif_u) > 0 else (0.0, 0.0)

    print("\nPrueba RQ2 - Falsos positivos graves por usuario:")
    print(f"  {etiqueta_base}:  {fp_b} / {total_u} usuarios ({fp_b / max(total_u, 1):.2%})")
    print(f"  {etiqueta_aporte}: {fp_a} / {total_u} usuarios ({fp_a / max(total_u, 1):.2%})")
    print(
        f"  Diferencia ({etiqueta_base} - {etiqueta_aporte}): {dif_u.mean():+.4f} "
        f"[IC95%: {ic_low:+.4f}, {ic_high:+.4f}]"
    )
    return resumen


def leer_argumentos():
    """
    - Precondiciones: Ninguna.
    - Entradas: Argumentos recibidos por linea de comandos (sys.argv).
    - Salidas: argparse.Namespace con las rutas y opciones configuradas.
    - Postcondiciones: Devuelve los parametros listos para lanzar los experimentos.
    """
    parser = argparse.ArgumentParser(
        description="Comparativa baseline vs VADER/Embeddings y analisis RQ1/RQ2"
    )
    parser.add_argument("-f", "--file", required=True, help="CSV de usuarios/mensajes")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiqueta")
    parser.add_argument("-s", "--seed", default="semilla.csv", help="CSV de semillas")
    parser.add_argument("--seed-target", default=None, help="Columna de texto en semilla.csv (por defecto usa 'Post' si existe, si no --target)",)
    parser.add_argument("--conjunto-train", choices=["sin_pseudo", "clean", "full"], default="full", help="Elige entre 'sin_pseudo', 'clean' o 'full'",)
    parser.add_argument("--orden-gravedad", nargs="+", default=["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"], help="Etiquetas ordenadas de menor a mayor gravedad",)
    parser.add_argument("--runs", type=int, default=3, help="Numero de repeticiones con distintas semillas",)
    return parser.parse_args()


def ejecutar_experimento_semilla(raw_data, semillas_df, args, seed_run):
    """
    - Precondiciones: raw_data y semillas_df contienen las columnas indicadas en args.
    - Entradas: raw_data (DataFrame), semillas_df (DataFrame), args (Namespace) y seed_run (int).
    - Salidas: tupla (df_metricas, modelos, test_msgs, n_emb) con resultados del run, modelos entrenados y datos de test.
    - Postcondiciones: Particiona por usuario, vectoriza, entrena los 5 modelos y devuelve sus metricas en Dev y Test.
    """
    #Particionamos primero por usuario para no mezclar mensajes del mismo autor entre splits
    col_user = "User" if "User" in raw_data.columns else ("user_id" if "user_id" in raw_data.columns else None)
    if col_user is not None:
        usuarios_unicos = raw_data.drop_duplicates(col_user)[[col_user, args.label]]
        tr_u, te_u = train_test_split(
            usuarios_unicos, test_size=0.15, random_state=seed_run, stratify=usuarios_unicos[args.label]
        )
        tr_u, dev_u = train_test_split(
            tr_u, test_size=(15 / 85), random_state=seed_run, stratify=tr_u[args.label]
        )
        train_users = raw_data[raw_data[col_user].isin(tr_u[col_user])].copy()
        dev_users = raw_data[raw_data[col_user].isin(dev_u[col_user])].copy()
        test_users = raw_data[raw_data[col_user].isin(te_u[col_user])].copy()
    else:
        train_users, test_users = train_test_split(
            raw_data, test_size=0.15, random_state=seed_run, stratify=raw_data[args.label]
        )
        train_users, dev_users = train_test_split(
            train_users, test_size=(15 / 85), random_state=seed_run, stratify=train_users[args.label]
        )

    #Separamos las listas de posts en mensajes individuales
    train_msgs = dividir_mensajes(train_users, col_post=args.target)
    dev_msgs = dividir_mensajes(dev_users, col_post=args.target)
    test_msgs = dividir_mensajes(test_users, col_post=args.target)

    col_seed_texto = args.seed_target if args.seed_target else ("Post" if "Post" in semillas_df.columns else args.target)
    X_seed = semillas_df[col_seed_texto]
    y_seed = semillas_df[args.label].values

    X_train = train_msgs[args.target]
    y_train = train_msgs[args.label]

    #Calculamos embeddings y caracteristicas VADER
    X_train_comb = pd.concat([X_seed, X_train], ignore_index=True)
    X_train_comb_emb, X_train_comb_vader, _ = vectorizar_embeddings_y_vader(X_train_comb)
    X_dev_emb, X_dev_vader, _ = vectorizar_embeddings_y_vader(dev_msgs[args.target])
    X_test_emb, X_test_vader, _ = vectorizar_embeddings_y_vader(test_msgs[args.target])

    n_sem = len(X_seed)
    X_seed_emb = X_train_comb_emb[:n_sem]
    X_reddit_emb = X_train_comb_emb[n_sem:]
    X_seed_vader = X_train_comb_vader[:n_sem]
    X_train_vader = X_train_comb_vader[n_sem:]

    #Aplicamos auto-entrenamiento en cascada sobre los embeddings
    X_clean_emb, y_clean, X_full_emb, y_full = cascading_self_training(
        X_seed=X_seed_emb,
        y_seed=y_seed,
        X_unlabeled=X_reddit_emb,
        y_unlabeled_orig=y_train.values,
        textos_unlabeled=X_train.values
    )

    #Elegimos el conjunto de entrenamiento pedido por consola
    if args.conjunto_train == "sin_pseudo":
        X_tr_emb = np.vstack([X_seed_emb, X_reddit_emb])
        y_tr = np.concatenate([np.asarray(y_seed), np.asarray(y_train.values)])
    elif args.conjunto_train == "clean":
        X_tr_emb, y_tr = X_clean_emb, y_clean
    else:
        X_tr_emb, y_tr = X_full_emb, y_full

    X_tr_vader = emparejar_vader_con_embeddings(
        X_tr_emb=X_tr_emb,
        X_seed_emb=X_seed_emb,
        vader_seed=X_seed_vader,
        X_reddit_emb=X_reddit_emb,
        vader_train=X_train_vader,
    )
    ventanas = (1, 3, 5, 10, 15, None)

    configs_info = [
        ("Baseline (Embeddings + SVM)", "svm", X_tr_emb, X_dev_emb, X_test_emb),
        (
            "Embeddings + VADER (SVM)",
            "svm",
            np.hstack([X_tr_emb, X_tr_vader]),
            np.hstack([X_dev_emb, X_dev_vader]),
            np.hstack([X_test_emb, X_test_vader]),
        ),
        ("Embeddings (XGBoost)", "xgb", X_tr_emb, X_dev_emb, X_test_emb),
        ("VADER solo (XGBoost)", "xgb", X_tr_vader, X_dev_vader, X_test_vader),
        (
            "Embeddings + VADER (XGBoost)",
            "xgb",
            np.hstack([X_tr_emb, X_tr_vader]),
            np.hstack([X_dev_emb, X_dev_vader]),
            np.hstack([X_test_emb, X_test_vader]),
        ),
    ]

    modelos = {}
    metricas_run = []
    for nombre, tipo, x_tr, x_dev, x_test in configs_info:
        clf, clases = entrenar_modelo(tipo, x_tr, y_tr, random_state=seed_run)
        s_dev = obtener_puntuaciones(clf, x_dev)
        s_test = obtener_puntuaciones(clf, x_test)
        mejor_k_dev, f1_dev_por_k = buscar_mejor_k_dev(
            s_dev, clases, dev_msgs, args.label, ventanas
        )
        y_test_u, preds_test_u, _ = predecir_por_usuario(
            s_test, clases, test_msgs, args.label, max_mensajes=mejor_k_dev
        )
        extra = evaluar_usuario_desbalanceado(y_test_u, preds_test_u, clase_minoritaria="Attempt")
        modelos[nombre] = (clf, clases, x_test)
        fp_tasa_u, gap_u = medir_falsos_positivos_graves(
            clf, clases, x_test, test_msgs, args.label, args.orden_gravedad
        )
        metricas_run.append({
            "run": seed_run,
            "Configuracion": nombre,
            "Mejor_k_Dev": "Todos" if mejor_k_dev is None else mejor_k_dev,
            "Dev_MacroF1": f1_dev_por_k[mejor_k_dev],
            "Test_MacroF1": extra["macro_f1"],
            "Test_WeightedF1": extra["weighted_f1"],
            "Test_Kappa": extra["kappa"],
            "Test_F1_Attempt": extra["f1_minoritaria"],
            "FP_grave_tasa": fp_tasa_u,
            "Gap_gravedad": gap_u,
        })

    return pd.DataFrame(metricas_run), modelos, test_msgs, X_tr_emb.shape[1]


def main():
    """
    - Precondiciones: Existen los ficheros CSV pasados en --file y --seed.
    - Entradas: Ninguna directa (lee parametros desde leer_argumentos).
    - Salidas: Ninguna.
    - Postcondiciones: Ejecuta todos los runs, guarda los CSV y el grafico en poster_figuras/ e imprime RQ1 y RQ2.
    """
    args = leer_argumentos()
    raw_data = cargar_csv(args.file)
    semillas_df = cargar_csv(args.seed)

    n_runs = max(1, int(args.runs))
    semillas_runs = [42 + i * 10 for i in range(n_runs)]

    dfs_runs = []
    modelo_base_42 = None
    test_msgs_42 = None
    n_emb_dim_42 = None

    #Lanzamos cada repeticion y guardamos el run 42 como referencia para las pruebas cualitativas
    for s_run in semillas_runs:
        print(f"\n--- Ejecutando run con semilla {s_run} ---")
        df_r, modelos_r, test_msgs_r, n_emb_r = ejecutar_experimento_semilla(raw_data, semillas_df, args, s_run)
        df_r["Conjunto"] = args.conjunto_train
        dfs_runs.append(df_r)
        if s_run == 42 or modelo_base_42 is None:
            modelo_base_42 = modelos_r
            test_msgs_42 = test_msgs_r
            n_emb_dim_42 = n_emb_r

    #Agrupamos resultados entre runs para sacar medias y desviaciones tipicas
    df_all = pd.concat(dfs_runs, ignore_index=True)
    resumen_stats = (
        df_all.groupby(["Conjunto", "Configuracion"], as_index=False, sort=False)
        .agg(
            Mejor_k_Dev=("Mejor_k_Dev", lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.iloc[0]),
            Dev_MacroF1_mean=("Dev_MacroF1", "mean"),
            Dev_MacroF1_std=("Dev_MacroF1", "std"),
            Test_MacroF1_mean=("Test_MacroF1", "mean"),
            Test_MacroF1_std=("Test_MacroF1", "std"),
            Test_F1_Attempt_mean=("Test_F1_Attempt", "mean"),
            Test_F1_Attempt_std=("Test_F1_Attempt", "std"),
            Test_Kappa_mean=("Test_Kappa", "mean"),
            Test_WeightedF1_mean=("Test_WeightedF1", "mean"),
            FP_grave_tasa_mean=("FP_grave_tasa", "mean"),
            FP_grave_tasa_std=("FP_grave_tasa", "std"),
            Gap_gravedad_mean=("Gap_gravedad", "mean"),
            Gap_gravedad_std=("Gap_gravedad", "std"),
        )
    )
    for col_std in [
        "Dev_MacroF1_std",
        "Test_MacroF1_std",
        "Test_F1_Attempt_std",
        "FP_grave_tasa_std",
        "Gap_gravedad_std",
    ]:
        resumen_stats[col_std] = resumen_stats[col_std].fillna(0.0)

    os.makedirs("poster_figuras", exist_ok=True)
    df_all.to_csv(os.path.join("poster_figuras", f"runs_{args.conjunto_train}.csv"), index=False)
    resumen_stats.to_csv(os.path.join("poster_figuras", f"resumen_{args.conjunto_train}.csv"), index=False)

    print(f"\nResumen ({n_runs} runs | conjunto: {args.conjunto_train}):")
    for _, row in resumen_stats.iterrows():
        print(
            f"  {row['Configuracion']:28s} | k={str(row['Mejor_k_Dev']):>5s} | "
            f"Macro-F1: {row['Test_MacroF1_mean']:.4f}±{row['Test_MacroF1_std']:.4f} | "
            f"Attempt: {row['Test_F1_Attempt_mean']:.4f}±{row['Test_F1_Attempt_std']:.4f} | "
            f"FP_tasa: {row['FP_grave_tasa_mean']:.3f}±{row['FP_grave_tasa_std']:.3f} | "
            f"Gap: {row['Gap_gravedad_mean']:.3f}"
        )

    guardar_grafico_comparativa(resumen_stats, conjunto=args.conjunto_train)

    #Bateria de frases para probar estabilidad frente a ruido superficial (RQ1)
    pares_ejemplo_rq1 = [
        (
            "Thanks for supporting me today, I really appreciate it.",
            "Thx for supporting me rn, lol XD!!!",
        ),
        (
            "I feel overwhelmed with exams and don't know where to start.",
            "I feel overwelmed with exams idk tbh???",
        ),
        (
            "Everything is going fine this week and I feel calm.",
            "EVERYTHING IS FINE THIS WEEK LOL!!",
        ),
        (
            "I am exhausted and tired of arguing with everyone lately.",
            "im so exhasted n tired of argung w/ everyone smh...",
        ),
        (
            "Thank you so much for listening to me when I was down.",
            "thx so much for listening when i was down :)!!",
        ),
        (
            "I cannot sleep properly and feel stressed about my future.",
            "cant sleep right n feel super stressed abt my future tbh??",
        ),
    ]
    probar_sensibilidad_ruido(
        baseline_tuple=modelo_base_42["Baseline (Embeddings + SVM)"],
        modelo_aporte_tuple=modelo_base_42["Embeddings + VADER (XGBoost)"],
        pares_frases=pares_ejemplo_rq1,
        usa_vader_en_aporte=True,
        etiqueta_base="Baseline (Emb+SVM)",
        etiqueta_aporte="Emb+VADER (XGB)",
    )
    print("\nControl RQ1 (Embeddings XGBoost vs Embeddings + VADER XGBoost):")
    probar_sensibilidad_ruido(
        baseline_tuple=modelo_base_42["Embeddings (XGBoost)"],
        modelo_aporte_tuple=modelo_base_42["Embeddings + VADER (XGBoost)"],
        pares_frases=pares_ejemplo_rq1,
        usa_vader_en_aporte=True,
        etiqueta_base="Embeddings (XGB)",
        etiqueta_aporte="Emb+VADER (XGB)",
    )

    clf_xgb_mix, _, _ = modelo_base_42["Embeddings + VADER (XGBoost)"]
    ver_peso_vader_xgb(clf_xgb_mix, n_emb=n_emb_dim_42)

    #Comprobacion de falsos positivos graves en la particion base (RQ2)
    etiquetas_presentes = set(test_msgs_42[args.label].unique())
    if not etiquetas_presentes.issubset(set(args.orden_gravedad)):
        fuera = sorted(etiquetas_presentes - set(args.orden_gravedad))
        raise ValueError(f"Hay etiquetas que no estan en --orden-gravedad: {fuera}")

    analizar_falsos_positivos_usuarios(
        test_msgs=test_msgs_42,
        col_label=args.label,
        baseline_tuple=modelo_base_42["Baseline (Embeddings + SVM)"],
        modelo_aporte_tuple=modelo_base_42["Embeddings + VADER (XGBoost)"],
        orden_gravedad=args.orden_gravedad,
        etiqueta_base="Baseline (Emb+SVM)",
        etiqueta_aporte="Emb+VADER (XGB)",
    )
    print("\nControl RQ2 (Embeddings XGBoost vs Embeddings + VADER XGBoost):")
    analizar_falsos_positivos_usuarios(
        test_msgs=test_msgs_42,
        col_label=args.label,
        baseline_tuple=modelo_base_42["Embeddings (XGBoost)"],
        modelo_aporte_tuple=modelo_base_42["Embeddings + VADER (XGBoost)"],
        orden_gravedad=args.orden_gravedad,
        etiqueta_base="Embeddings (XGB)",
        etiqueta_aporte="Emb+VADER (XGB)",
    )


if __name__ == "__main__":
    main()