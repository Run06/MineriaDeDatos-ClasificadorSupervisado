import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import SVC

try:
    import resource
except ImportError:
    resource = None

ORDEN_CLINICO = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]


def calcular_metricas_graded(y_true_ord, y_pred_ord):
    """Misma fórmula de Graded Precision/Recall/F1 que el modelo jerárquico."""
    y_true_ord = np.asarray(y_true_ord, dtype=np.int64)
    y_pred_ord = np.asarray(y_pred_ord, dtype=np.int64)

    tp = int(np.sum(y_true_ord == y_pred_ord))
    fp_ord = int(np.sum(np.maximum(0, y_pred_ord - y_true_ord)))
    fn_ord = int(np.sum(np.maximum(0, y_true_ord - y_pred_ord)))

    gp = tp / (tp + fp_ord) if (tp + fp_ord) > 0 else 0.0
    gr = tp / (tp + fn_ord) if (tp + fn_ord) > 0 else 0.0
    gf1 = (2.0 * gp * gr / (gp + gr)) if (gp + gr) > 0 else 0.0
    return {"graded_p": gp, "graded_r": gr, "graded_f1": gf1}


def _ram_pico_mb():
    if resource is None:
        return float("nan")
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss / (1024.0 ** 2) if sys.platform == "darwin" else max_rss / 1024.0


def predecir_por_usuario(scores_mensajes, clases, df_mensajes, col_label, max_mensajes=None):
    """
    Recorre hasta `max_mensajes` de cada usuario y aplica Max-Pooling sobre los scores
    del clasificador para no diluir los mensajes críticos entre mensajes neutros.
    """
    df_eval = df_mensajes[["user_id", col_label]].copy()
    df_eval["idx_fila"] = np.arange(len(df_eval))

    y_true_users = []
    y_pred_users = []
    scores_users = []
    mensajes_leidos = []

    for _, grupo in df_eval.groupby("user_id", sort=False):
        y_true_users.append(grupo[col_label].iloc[0])

        indices = grupo["idx_fila"].values
        if max_mensajes is not None:
            indices = indices[:max_mensajes]

        scores_usuario = scores_mensajes[indices]
        # Max-Pooling: conserva la señal más fuerte de cada clase en la ventana
        max_scores = scores_usuario.max(axis=0)
        pred_final = clases[np.argmax(max_scores)]

        y_pred_users.append(pred_final)
        scores_users.append(max_scores)
        mensajes_leidos.append(len(indices))

    return y_true_users, y_pred_users, np.vstack(scores_users), float(np.mean(mensajes_leidos))


def _evaluar_metricas_completas(y_true_str, y_pred_str, scores_users, clases_svc, orden_clinico):
    """Calcula todas las métricas alineadas con el modelo jerárquico."""
    mapa_ord = {c: i for i, c in enumerate(orden_clinico)}
    y_true_ord = np.array([mapa_ord[y] for y in y_true_str], dtype=np.int64)
    y_pred_ord = np.array([mapa_ord[y] for y in y_pred_str], dtype=np.int64)

    mf1 = f1_score(y_true_ord, y_pred_ord, average="macro", zero_division=0)
    wf1 = f1_score(y_true_ord, y_pred_ord, average="weighted", zero_division=0)
    graded = calcular_metricas_graded(y_true_ord, y_pred_ord)
    qwk = cohen_kappa_score(y_true_ord, y_pred_ord, weights="quadratic")
    kappa = cohen_kappa_score(y_true_ord, y_pred_ord)
    dist = np.abs(y_true_ord - y_pred_ord)
    mae = float(np.mean(dist))
    acc_pm1 = float(100.0 * np.mean(dist <= 1))

    # AUC OvR por clase directamente desde los margins de decision_function
    lista_svc = list(clases_svc)
    aucs_por_clase = {}
    for c in orden_clinico:
        col_idx = lista_svc.index(c)
        y_bin = (y_true_ord == mapa_ord[c]).astype(int)
        try:
            aucs_por_clase[c] = float(roc_auc_score(y_bin, scores_users[:, col_idx]))
        except ValueError:
            aucs_por_clase[c] = float("nan")
    auc_macro = float(np.nanmean(list(aucs_por_clase.values())))

    return {
        "y_true_ord": y_true_ord,
        "y_pred_ord": y_pred_ord,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "graded_p": graded["graded_p"],
        "graded_r": graded["graded_r"],
        "graded_f1": graded["graded_f1"],
        "auc": auc_macro,
        "aucs_por_clase": aucs_por_clase,
        "qwk": qwk,
        "kappa": kappa,
        "mae": mae,
        "acc_pm1": acc_pm1,
    }


def entrenar_y_evaluar(
    titulo,
    X_tr,
    y_tr,
    X_dev_vec,
    dev_msgs,
    X_test_vec,
    test_msgs,
    col_label,
    ventanas_k=(1, 3, 5, 10, 15, None),
):
    print(f"MODELO: {titulo} (Train: {len(y_tr)} mensajes)")
    t0 = time.perf_counter()
    svc_model = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )
    svc_model.fit(X_tr, y_tr)
    tiempo_fit = time.perf_counter() - t0
    clases = svc_model.classes_

    # Orden clínico exacto presente en el dataset
    unicas = set(clases)
    orden = [c for c in ORDEN_CLINICO if c in unicas]
    orden += [c for c in sorted(unicas) if c not in orden]

    scores_dev = svc_model.decision_function(X_dev_vec)
    scores_test = svc_model.decision_function(X_test_vec)

    print(
        f"{'Ventana':>8} | {'Msgs':>5} | {'Dev F1':>8} | {'Test F1':>8} | "
        f"{'W-F1':>8} | {'GradedF1':>8} | {'Test AUC':>8} | {'Test QWK':>8} | {'Kappa':>8}"
    )
    print("-" * 92)

    ultimo_test = None
    for k in ventanas_k:
        y_dev_u, preds_dev_u, scores_dev_u, _ = predecir_por_usuario(
            scores_dev, clases, dev_msgs, col_label, max_mensajes=k
        )
        y_test_u, preds_test_u, scores_test_u, media_test = predecir_por_usuario(
            scores_test, clases, test_msgs, col_label, max_mensajes=k
        )

        m_dev = _evaluar_metricas_completas(y_dev_u, preds_dev_u, scores_dev_u, clases, orden)
        m_test = _evaluar_metricas_completas(y_test_u, preds_test_u, scores_test_u, clases, orden)
        ultimo_test = (y_test_u, preds_test_u, m_test)

        etiqueta_k = "Todos" if k is None else str(k)
        print(
            f"{etiqueta_k:>8} | {media_test:5.1f} | {m_dev['macro_f1']:8.4f} | "
            f"{m_test['macro_f1']:8.4f} | {m_test['weighted_f1']:8.4f} | "
            f"{m_test['graded_f1']:8.4f} | {m_test['auc']:8.4f} | "
            f"{m_test['qwk']:8.4f} | {m_test['kappa']:8.4f}"
        )

    y_test_u, preds_test_u, m_test = ultimo_test
    print(f"\nReporte de clasificación Test (Nivel Usuario, k={ventanas_k[-1]})")
    print(classification_report(y_test_u, preds_test_u, labels=orden, zero_division=0))

    aucs_str = " | ".join(f"{c}:{m_test['aucs_por_clase'][c]:.3f}" for c in orden)
    print(f"AUC OvR por clase (Test, k=Todos): {aucs_str}")

    cm = confusion_matrix(m_test["y_true_ord"], m_test["y_pred_ord"], labels=list(range(len(orden))))
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real:{c[:4]}" for c in orden],
        columns=[f"Pred:{c[:4]}" for c in orden],
    )
    print("\nMatriz de confusión Test (orden clínico):")
    print(df_cm.to_string())

    print(
        f"\nGraded P: {m_test['graded_p']:.4f} | Graded R: {m_test['graded_r']:.4f} | "
        f"Graded F1: {m_test['graded_f1']:.4f} | QWK: {m_test['qwk']:.4f} | "
        f"MAE: {m_test['mae']:.3f} | Exacto o ±1: {m_test['acc_pm1']:.1f}%"
    )
    print(f"Tiempo entrenamiento SVC: {tiempo_fit:.2f}s | RAM pico proceso: {_ram_pico_mb():.1f} MB")