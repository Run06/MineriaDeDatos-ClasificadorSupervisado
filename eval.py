from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
import numpy as np

def predecir_por_usuario(scores_mensajes, clases,df_mensajes,col_label,max_mensajes=None):
    """
    Recorre hasta `max_mensajes` de cada usuario y aplica Max-Pooling sobre los scores
    del clasificador para no diluir los mensajes críticos entre mensajes neutros.
    """
    df_eval = df_mensajes[["user_id", col_label]].copy()
    df_eval["idx_fila"] = np.arange(len(df_eval))

    y_true_users = []
    y_pred_users = []
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
        mensajes_leidos.append(len(indices))

    return y_true_users, y_pred_users, float(np.mean(mensajes_leidos))

def entrenar_y_evaluar(titulo, X_tr, y_tr, X_dev_vec, dev_msgs, X_test_vec, test_msgs, col_label,ventanas_k=(1,3,5,10,None)):
    print(f"MODELO: {titulo} (Train: {len(y_tr)} mensajes)")
    svc_model = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )
    svc_model.fit(X_tr, y_tr)
    clases = svc_model.classes_

    scores_dev = svc_model.decision_function(X_dev_vec)
    scores_test = svc_model.decision_function(X_test_vec)

    for k in ventanas_k:
        y_dev_u, preds_dev_u, media_dev = predecir_por_usuario(
            scores_dev, clases, dev_msgs, col_label, max_mensajes=k
        )
        y_test_u, preds_test_u, media_test = predecir_por_usuario(
            scores_test, clases, test_msgs, col_label, max_mensajes=k
        )

        f1_dev = f1_score(y_dev_u, preds_dev_u, average="macro", zero_division=0)
        f1_test = f1_score(y_test_u, preds_test_u, average="macro", zero_division=0)
        etiqueta_k = "Todos" if k is None else f"{k:>5}"
        print(
            f"k={etiqueta_k} | Dev F1: {f1_dev:.4f} (msgs: {media_dev:4.1f}) "
            f"| Test F1: {f1_test:.4f} (msgs: {media_test:4.1f})"
        )

    print(f"\nReporte de clasificación Test (Nivel Usuario, k={ventanas_k[-1]})")
    print(classification_report(y_test_u, preds_test_u, zero_division=0))
