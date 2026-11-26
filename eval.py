import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, accuracy_score, cohen_kappa_score,
                             confusion_matrix)
from sklearn.svm import SVC, LinearSVC
from medicion import medir
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


def crear_clasificador(nombre):
    if nombre == 'svm_linear':
        return LinearSVC(C=1.0, class_weight='balanced', max_iter=10000, random_state=42)
    if nombre == 'svm_rbf':
        return SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', random_state=42)
    if nombre == 'rf':
        return RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=300,
                                      min_samples_split=5, min_samples_leaf=3,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
    if nombre == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(256,), activation='relu', alpha=1e-4,
                             early_stopping=True, n_iter_no_change=10, max_iter=300,
                             random_state=42)
    raise ValueError(f'Modelo no soportado: {nombre}')


def _scores(modelo, X):
    if hasattr(modelo, 'decision_function'):
        scores = modelo.decision_function(X)
        # SVM binaria: margen positivo corresponde a classes_[1].
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        return scores
    return modelo.predict_proba(X)


def predecir_por_usuario(scores_mensajes, clases, df_mensajes, col_label, max_mensajes=None):
    df_eval = df_mensajes[['user_id', col_label]].copy()
    df_eval['idx_fila'] = np.arange(len(df_eval))
    y_true_users, y_pred_users, mensajes_leidos = [], [], []
    for _, grupo in df_eval.groupby('user_id', sort=False):
        y_true_users.append(grupo[col_label].iloc[0])
        indices = grupo['idx_fila'].values
        if max_mensajes is not None:
            indices = indices[:max_mensajes]
        pred_final = clases[np.argmax(scores_mensajes[indices].max(axis=0))]
        y_pred_users.append(pred_final)
        mensajes_leidos.append(len(indices))
    return y_true_users, y_pred_users, float(np.mean(mensajes_leidos))


def entrenar_y_evaluar(titulo, nombre_modelo, X_tr, y_tr, X_dev_vec, dev_msgs,
                      X_test_vec, test_msgs, col_label, ventanas_k=(1, 3, 5, 10, None),
                      clases_eval=None):
    print('\n' + '=' * 50)
    print(f'MODELO: {titulo} (Train: {len(y_tr)} mensajes)')
    modelo = crear_clasificador(nombre_modelo)
    with medir(f'Entrenamiento [{titulo}]'):
        if nombre_modelo == 'mlp':
            modelo.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
        else:
            modelo.fit(X_tr, y_tr)
    clases = modelo.classes_
    if clases_eval is None:
        clases_eval = np.unique(np.concatenate([clases, dev_msgs[col_label].values,
                                                test_msgs[col_label].values]))
    clases_eval = np.asarray(clases_eval)
    with medir(f'Predicción dev+test [{titulo}]'):
        scores_dev = _scores(modelo, X_dev_vec)
        scores_test = _scores(modelo, X_test_vec)

    metricas, por_clase, matrices = [], [], []
    with medir(f'Agregación y métricas [{titulo}]'):
        for k in ventanas_k:
            etiqueta_k = 'Todos' if k is None else str(k)
            for particion, scores, df in [('Dev', scores_dev, dev_msgs), ('Test', scores_test, test_msgs)]:
                real, pred, media = predecir_por_usuario(scores, clases, df, col_label, k)
                opts = dict(labels=clases_eval, average='macro', zero_division=0)
                fila = dict(estrategia=titulo, particion=particion, k=etiqueta_k,
                            usuarios=len(real), mensajes_media=media, train_mensajes=len(y_tr),
                            f1_macro=f1_score(real, pred, **opts),
                            precision_macro=precision_score(real, pred, **opts),
                            recall_macro=recall_score(real, pred, **opts),
                            accuracy=accuracy_score(real, pred),
                            kappa=cohen_kappa_score(real, pred, labels=clases_eval))
                metricas.append(fila)
                print(f'k={etiqueta_k:>5} | {particion} F1: {fila["f1_macro"]:.4f} (msgs: {media:.1f})')
                # Usar claves numéricas evita colisiones con nombres como "accuracy".
                reporte = classification_report(real, pred, labels=clases_eval,
                          target_names=[str(i) for i in range(len(clases_eval))],
                          output_dict=True, zero_division=0)
                for i, clase in enumerate(clases_eval):
                    r = reporte[str(i)]
                    por_clase.append(dict(estrategia=titulo, particion=particion, k=etiqueta_k,
                                         clase=str(clase), precision=r['precision'], recall=r['recall'],
                                         f1=r['f1-score'], soporte=int(r['support'])))
                matriz = confusion_matrix(real, pred, labels=clases_eval)
                for i, a in enumerate(clases_eval):
                    for j, b in enumerate(clases_eval):
                        matrices.append(dict(estrategia=titulo, particion=particion, k=etiqueta_k,
                                             real=str(a), predicha=str(b), cantidad=int(matriz[i, j])))
                if particion == 'Test' and k == ventanas_k[-1]:
                    print(classification_report(real, pred, labels=clases_eval, zero_division=0))
    # Desempate: preferir la primera ventana (orden creciente en main.py).
    mejor = max((r for r in metricas if r['particion'] == 'Dev'), key=lambda r: r['f1_macro'])['k']
    for fila in metricas + por_clase + matrices:
        fila['k_elegido_dev'] = fila['k'] == mejor
    return metricas, por_clase, matrices
