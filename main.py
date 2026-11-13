import argparse
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np

from preproceso import  dividir_mensajes, tfidf

def parse_args():
    """
    - Precondición: El archivo CSV de entrada especificado debe existir en el sistema.
    - Inputs: Argumentos recibidos por la línea de comandos y el archivo opcional clasificador.json.
    - Outputs: Objeto con los argumentos analizados.
    - Postcondición: Analiza los argumentos proporcionados desde la línea de comandos e integra los valores leídos desde el archivo clasificador.json en el objeto resultante.
    """
    parser = argparse.ArgumentParser(description="Clasificación no supervisada K-Means")
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-t", "--target", default='Post', help="Columna de la tabla a clasificar")
    parser.add_argument("-l", "--label", default='Label', help="Columna del Label para métricas externas")
    parser.add_argument("-s", "--seed", default='semilla', help="Archivo csv con instancias semilla para re etiquetado")

    return parser.parse_args()


def cascading_self_training(X_seed, y_seed, X_unlabeled, y_unlabeled_orig, umbrales=[0.85, 0.80, 0.75, 0.70, 0.65],
                            max_iter_interna=10):
    """
    Ejecuta un bucle de pseudo-etiquetado con umbrales en cascada y efecto 'bola de nieve' balanceado.
    Exprime cada umbral iterativamente con un límite de seguridad antes de bajar al siguiente nivel.
    """
    X_current_train = X_seed
    y_current_train = y_seed

    X_unl = X_unlabeled
    y_unl_orig = y_unlabeled_orig

    X_pseudo_acumulado = None
    y_pseudo_acumulado = []

    print("\n" + "=" * 50)
    print("INICIANDO RE-ETIQUETADO (CASCADA + BOLA DE NIEVE BALANCEADA)")
    print("=" * 50)

    for i, umbral in enumerate(umbrales):
        if X_unl.shape[0] == 0:
            break

        print(f"\n--- NIVEL {i + 1} | Bajando umbral a: {umbral} ---")

        # BUCLE INTERNO
        iteracion_interna = 1
        while True:
            #Condición de parada por falta de datos
            if X_unl.shape[0] == 0:
                break

            #Condición de parada por límite de iteraciones
            if iteracion_interna > max_iter_interna:
                print(f"  > [Info] Límite de {max_iter_interna} iteraciones alcanzado. Bajando de nivel...")
                break
            modelo_maestro = LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000, C=10.0,
                                                solver='lbfgs')
            modelo_maestro.fit(X_current_train, y_current_train)
            probs = modelo_maestro.predict_proba(X_unl)
            max_probs = np.max(probs, axis=1)
            preds = modelo_maestro.classes_[np.argmax(probs, axis=1)]
            mask_confianza = max_probs >= umbral
            y_candidatos = preds[mask_confianza]

            if len(y_candidatos) > 0:
                # Contamos cuántos mensajes hay de cada clase
                clases, conteos = np.unique(y_candidatos, return_counts=True)

                #Buscamos la clase minoritaria para usarla de límite
                limite_minoritario = max(1, np.min(conteos))

                indices_balanceados = []
                for c in np.unique(preds):
                    idx_clase = np.where(y_candidatos == c)[0]
                    if len(idx_clase) > 0:
                        #Si hay muchos mensajes, cogemos aleatoriamente solo hasta el límite minoritario
                        if len(idx_clase) > limite_minoritario:
                            np.random.seed(42)
                            idx_elegidos = np.random.choice(idx_clase, limite_minoritario, replace=False)
                            indices_balanceados.extend(idx_elegidos)
                        else:
                            indices_balanceados.extend(idx_clase)

                # Actualizamos la máscara de confianza para que SOLO contenga los índices balanceados
                mask_balanceada_absoluta = np.zeros(X_unl.shape[0], dtype=bool)
                indices_originales = np.where(mask_confianza)[0][indices_balanceados]
                mask_balanceada_absoluta[indices_originales] = True

                mask_confianza = mask_balanceada_absoluta

            nuevos_etiquetados = mask_confianza.sum()
            if nuevos_etiquetados == 0:
                if iteracion_interna == 1:
                    print(f"  > [Info] Confianza máxima actual: {max_probs.max():.4f}. Ninguno supera {umbral}.")
                else:
                    print(f"  > [Info] El umbral {umbral} se ha agotado. Bajando de nivel...")
                break

            print(f"  > [Iteración {iteracion_interna}] Mensajes extraídos (Balanceados): {nuevos_etiquetados}")

            X_new = X_unl[mask_confianza]
            y_new = preds[mask_confianza]

            if X_pseudo_acumulado is None:
                X_pseudo_acumulado = X_new
            else:
                X_pseudo_acumulado = sp.vstack((X_pseudo_acumulado, X_new))
            y_pseudo_acumulado.extend(y_new)

            X_unl = X_unl[~mask_confianza]
            y_unl_orig = y_unl_orig[~mask_confianza]

            # Verificación de seguridad para no perder clases
            if len(np.unique(y_new)) == len(np.unique(y_seed)):
                if i == 0 and iteracion_interna == 1:
                    print("    -> Regla Aplicada: Se elimina la semilla inicial.")
                    X_current_train = X_new
                    y_current_train = y_new
                else:
                    X_current_train = sp.vstack((X_current_train, X_new))
                    y_current_train = np.concatenate((y_current_train, y_new))
            else:
                X_current_train = sp.vstack((X_current_train, X_new))
                y_current_train = np.concatenate((y_current_train, y_new))

            iteracion_interna += 1

    # Reconstrucción del dataset final
    if X_pseudo_acumulado is not None:
        X_train_final = sp.vstack((X_seed, X_pseudo_acumulado, X_unl))
        y_train_final = np.concatenate((y_seed, np.array(y_pseudo_acumulado), y_unl_orig))
        pseudo_count = X_pseudo_acumulado.shape[0]
    else:
        X_train_final = sp.vstack((X_seed, X_unl))
        y_train_final = np.concatenate((y_seed, y_unl_orig))
        pseudo_count = 0

    print("\n" + "=" * 50)
    print(f"1. Semilla manual e híbrida (Anclaje): {X_seed.shape[0]}")
    print(f"2. Mensajes re-etiquetados (Cascada Balanceada): {pseudo_count}")
    print(f"3. Mensajes con etiqueta original (Fallback): {X_unl.shape[0]}")
    print(f"TOTAL ENTRENAMIENTO RF: {X_train_final.shape[0]}")
    print("=" * 50)

    return X_train_final, y_train_final
if __name__ == "__main__":
    args = parse_args()
    #Cargar csv
    try:
        raw_data = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: El archivo '{args.file}' no existe o la ruta es incorrecta.")
        sys.exit(-1)

    #Separar lista de textos
    data = dividir_mensajes(raw_data)
    X = data[args.target]
    y = data[args.label]

    print(f"Dimensiones del dataset cargado: {data.shape}")

    #Partición (Train / Dev / Test)
    #Train (85%) Test (15%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    #train_full Train (80%) Dev (20%)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=(15 / 85), random_state=42, stratify=y_train_full
    )

    print("Tamaños de las particiones (70% / 15% / 15%)")
    print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

    #Mensajes semilla
    try:
        mensajes_semilla = pd.read_csv(args.seed)
    except FileNotFoundError:
        print("Error: No se encuentra el archivo 'semillas.csv'.")
        sys.exit(-1)

    print(f"Semillas manuales cargadas: {len(mensajes_semilla)}")
    X_train_seed_real, X_train_unlabeled, y_train_seed_real, y_train_unlabeled = train_test_split(
        X_train, y_train, test_size=0.90, random_state=42, stratify=y_train
    )

    X_seed_combined = pd.concat([mensajes_semilla[args.target], X_train_seed_real], ignore_index=True)
    y_seed = np.concatenate([mensajes_semilla[args.label].values, y_train_seed_real.values])

    X_reddit_unlabeled = X_train_unlabeled
    y_reddit_orig = y_train_unlabeled.values

    X_train_combinado_total = pd.concat([X_seed_combined, X_reddit_unlabeled], ignore_index=True)
    X_train_combinado_vec, X_dev_vec, X_test_vec = tfidf(X_train_combinado_total, X_dev, X_test)

    num_semillas_total = len(X_seed_combined)
    X_seed_vec = X_train_combinado_vec[:num_semillas_total]
    X_reddit_vec = X_train_combinado_vec[num_semillas_total:]
    #Funcion auxiliar
    X_train_final, y_train_final = cascading_self_training(
        X_seed=X_seed_vec,
        y_seed=y_seed,
        X_unlabeled=X_reddit_vec,
        y_unlabeled_orig=y_reddit_orig,
        umbrales=[ 0.90,0.85, 0.80, 0.75, 0.70,0.65],
        max_iter_interna=5
    )

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_features='sqrt',
        max_depth=300,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        criterion='gini'
    )
    rf_model.fit(X_train_final, y_train_final)
    print("-"*50)

    #Evaluación sobre Dev
    preds_dev = rf_model.predict(X_dev_vec)
    print(f"Macro F1-Score - Dev: {f1_score(y_dev, preds_dev, average='macro', zero_division=0):.4f}")

    preds_test = rf_model.predict(X_test_vec)
    print(f"Macro F1-Score - Test: {f1_score(y_test, preds_test, average='macro', zero_division=0):.4f}")

    print("\nReporte de clasificación Test")
    print(classification_report(y_test, preds_test, zero_division=0))
