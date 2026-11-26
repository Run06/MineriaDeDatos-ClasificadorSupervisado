import hashlib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

def cascading_self_training(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled_orig,
    textos_unlabeled,
    umbrales=(0.95, 0.90, 0.85, 0.80, 0.75),
    max_iter_interna=10,
    factor_balanceo=2,
    min_palabras=8,
    random_state=42,
    cache_dir="cache",
    devolver_historial=False,
):
    """
    Ejecuta un bucle de pseudo-etiquetado en cascada con balanceo relativo y filtro de longitud.
    """
    historial = []
    # --- CACHÉ ---
    ruta_cache = None
    if cache_dir is not None:
        hasher = hashlib.md5()
        firma = f"historial_v1|{tuple(umbrales)}|{max_iter_interna}|{factor_balanceo}|{min_palabras}|{random_state}"
        hasher.update(firma.encode("utf-8"))
        hasher.update(np.array([len(str(t).split()) >= min_palabras for t in textos_unlabeled], dtype=bool).tobytes())
        hasher.update(np.ascontiguousarray(X_seed).tobytes())
        hasher.update(np.ascontiguousarray(X_unlabeled).tobytes())
        hasher.update("\x1f".join(map(str, y_seed)).encode("utf-8"))
        hasher.update("\x1f".join(map(str, y_unlabeled_orig)).encode("utf-8"))
        hash_id = hasher.hexdigest()[:12]

        os.makedirs(cache_dir, exist_ok=True)
        ruta_cache = os.path.join(cache_dir, f"reetiquetado_{hash_id}.npz")
        if os.path.exists(ruta_cache):
            print(f"\nCargando re-etiquetado desde caché: {ruta_cache}")
            with np.load(ruta_cache, allow_pickle=True) as data:
                resultado = (data["X_clean"], data["y_clean"], data["X_full"], data["y_full"])
                historial = data["historial"].tolist()
            return (*resultado, historial) if devolver_historial else resultado
    n_clases_seed = len(np.unique(y_seed))
    X_current_train = X_seed
    y_current_train = y_seed

    X_unl = X_unlabeled
    y_unl_orig = y_unlabeled_orig
    valid_len_unl = np.array([len(str(t).split()) >= min_palabras for t in textos_unlabeled])

    X_pseudo_bloques = []
    y_pseudo_bloques = []

    modelo_maestro = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=500,
        C=1.0,
        solver="lbfgs",
    )

    necesita_reentrenar = True
    max_probs = None
    preds = None

    print("\n" + "=" * 50)
    print("INICIANDO RE-ETIQUETADO")
    print("=" * 50)

    for i, umbral in enumerate(umbrales):
        if X_unl.shape[0] == 0:
            break

        print(f"\n--- NIVEL {i + 1} | Umbral: {umbral} ---")

        for iteracion_interna in range(1, max_iter_interna + 1):
            if X_unl.shape[0] == 0:
                break

            if necesita_reentrenar:
                modelo_maestro.fit(X_current_train, y_current_train)
                probs = modelo_maestro.predict_proba(X_unl)
                max_probs = np.max(probs, axis=1)
                preds = modelo_maestro.classes_[np.argmax(probs, axis=1)]
                necesita_reentrenar = False

            mask_confianza = (max_probs >= umbral) & valid_len_unl
            indices_candidatos = np.flatnonzero(mask_confianza)

            if indices_candidatos.size > 0 and factor_balanceo is not None:
                y_candidatos = preds[indices_candidatos]
                clases, conteos = np.unique(y_candidatos, return_counts=True)
                limite = max(1, np.min(conteos) * factor_balanceo)

                indices_elegidos = []
                for c in clases:
                    idx_clase = indices_candidatos[y_candidatos == c]
                    if idx_clase.size > limite:
                        np.random.seed(random_state)
                        idx_clase = np.random.choice(idx_clase, size=limite, replace=False)
                    indices_elegidos.append(idx_clase)

                mask_confianza = np.zeros(X_unl.shape[0], dtype=bool)
                mask_confianza[np.concatenate(indices_elegidos)] = True

            nuevos_etiquetados = int(mask_confianza.sum())
            historial.append(dict(umbral=umbral, iteracion=iteracion_interna,
                                  nuevos=nuevos_etiquetados,
                                  pendientes=int(X_unl.shape[0] - nuevos_etiquetados),
                                  confianza_media=float(max_probs[mask_confianza].mean()) if nuevos_etiquetados else None,
                                  cambios=int(np.sum(preds[mask_confianza] != y_unl_orig[mask_confianza]))))
            if nuevos_etiquetados == 0:
                if iteracion_interna == 1:
                    print(f"  > Confianza máxima actual: {max_probs.max():.4f}. Ninguno supera {umbral}.")
                else:
                    print(f"  > Umbral {umbral} agotado. Bajando de nivel...")
                break

            print(f"  > [Iteración {iteracion_interna}] Mensajes extraídos: {nuevos_etiquetados}")

            X_new = X_unl[mask_confianza]
            y_new = preds[mask_confianza]

            X_pseudo_bloques.append(X_new)
            y_pseudo_bloques.append(y_new)

            keep_mask = ~mask_confianza
            X_unl = X_unl[keep_mask]
            y_unl_orig = y_unl_orig[keep_mask]
            valid_len_unl = valid_len_unl[keep_mask]

            if i == 0 and iteracion_interna == 1 and len(np.unique(y_new)) == n_clases_seed:
                print("    -> Regla aplicada: se sustituye la semilla inicial.")
                X_current_train = X_new
                y_current_train = y_new
            else:
                X_current_train = np.vstack((X_current_train, X_new))
                y_current_train = np.concatenate((y_current_train, y_new))
            necesita_reentrenar = True
        else:
            print(f"  > Límite de {max_iter_interna} iteraciones alcanzado. Bajando de nivel...")

    # Reconstrucción del dataset final
    if X_pseudo_bloques:
        X_pseudo_acumulado = np.vstack(X_pseudo_bloques)
        y_pseudo_acumulado = np.concatenate(y_pseudo_bloques)
        X_clean = np.vstack((X_seed, X_pseudo_acumulado))
        y_clean = np.concatenate((y_seed, y_pseudo_acumulado))
        pseudo_count = X_pseudo_acumulado.shape[0]
    else:
        X_clean = X_seed
        y_clean = y_seed
        pseudo_count = 0

    X_full = np.vstack((X_clean, X_unl))
    y_full = np.concatenate((y_clean, y_unl_orig))

    print("\n" + "=" * 50)
    print(f"1. Semilla manual (Anclaje): {X_seed.shape[0]}")
    print(f"2. Mensajes re-etiquetados: {pseudo_count}")
    print(f"3. Mensajes con etiqueta original (Fallback): {X_unl.shape[0]}")
    print(f"TOTAL SOLO RE-ETIQUETADO (+ semillas): {X_clean.shape[0]}")
    print(f"TOTAL COMPLETO (con fallback): {X_full.shape[0]}")
    print("=" * 50)

    if ruta_cache is not None:
        np.savez(ruta_cache, X_clean=X_clean, y_clean=y_clean, X_full=X_full, y_full=y_full, historial=np.array(historial, dtype=object))
        print(f"Re-etiquetado guardado en caché: {ruta_cache}")

    resultado = (X_clean, y_clean, X_full, y_full)
    return (*resultado, historial) if devolver_historial else resultado