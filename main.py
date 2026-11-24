import argparse
import os
import shutil
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from eval import entrenar_y_evaluar
from medicion import imprimir_resumen, medir, ram_mb
from preproceso import dividir_mensajes, vectorizar
from re_etiquetado import cascading_self_training

CACHE_DIR = "cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificación supervisada con re-etiquetado en cascada y embeddings"
    )
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto a clasificar")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas")
    parser.add_argument("-s", "--seed", default="semilla.csv", help="Archivo CSV con instancias semilla")
    parser.add_argument("-m", "--model", choices=["svm_linear", "svm_rbf", "rf"],
                        default="svm_linear", help="Clasificador final")
    parser.add_argument("-v", "--vectorizer", choices=["fasttext", "embeddings"],
                        default="embeddings", help="Vectorización")
    parser.add_argument("--limpiar-cache", action="store_true",
                        help="Borra la carpeta 'cache' antes de ejecutar (medición en frío)")
    return parser.parse_args()


def cargar_csv(ruta):
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)


def main():
    args = parse_args()
    t_total = time.perf_counter()

    if args.limpiar_cache and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Caché '{CACHE_DIR}' eliminada.")

    print(f"[RAM] Inicio: {ram_mb():.0f} MB")
    print(f"Configuración -> vectorizador: {args.vectorizer} | modelo: {args.model}")

    with medir("Carga, partición y división en mensajes"):
        raw_data = cargar_csv(args.file)
        print(f"Usuarios totales cargados: {len(raw_data)}")
        train_users, test_users = train_test_split(
            raw_data, test_size=0.15, random_state=42, stratify=raw_data[args.label]
        )
        train_users, dev_users = train_test_split(
            train_users, test_size=(15 / 85), random_state=42, stratify=train_users[args.label]
        )

        print("Partición de USUARIOS (70% / 15% / 15%):")
        print(f"  Train: {len(train_users)} usuarios | Dev: {len(dev_users)} usuarios | Test: {len(test_users)} usuarios")

        # Explotar mensajes por separado (cero fuga de datos entre particiones)
        train_msgs = dividir_mensajes(train_users, col_post=args.target)
        dev_msgs = dividir_mensajes(dev_users, col_post=args.target)
        test_msgs = dividir_mensajes(test_users, col_post=args.target)

        print("Partición de MENSAJES:")
        print(f"  Train: {len(train_msgs)} | Dev: {len(dev_msgs)} | Test: {len(test_msgs)}")

        mensajes_semilla = cargar_csv(args.seed)
        print(f"Semillas manuales cargadas: {len(mensajes_semilla)}")

        X_seed = mensajes_semilla[args.target]
        y_seed = mensajes_semilla[args.label].values

        X_train = train_msgs[args.target]
        y_train = train_msgs[args.label]

    # Vectorización
    X_train_combinado = pd.concat([X_seed, X_train], ignore_index=True)
    print(f"\nVectorizando con: {args.vectorizer}")
    with medir(f"Vectorización ({args.vectorizer})"):
        X_train_combinado_vec, X_dev_vec, X_test_vec = vectorizar(
            args.vectorizer, X_train_combinado, dev_msgs[args.target], test_msgs[args.target]
        )

    num_semillas = len(X_seed)
    X_seed_vec = X_train_combinado_vec[:num_semillas]
    X_reddit_vec = X_train_combinado_vec[num_semillas:]

    # Re-etiquetado en cascada
    with medir("Re-etiquetado en cascada"):
        X_train_clean, y_train_clean, X_train_full, y_train_full = cascading_self_training(
            X_seed=X_seed_vec,
            y_seed=y_seed,
            X_unlabeled=X_reddit_vec,
            y_unlabeled_orig=y_train.values,
            textos_unlabeled=X_train.values,
            umbrales=(0.95, 0.90, 0.85, 0.80, 0.75, 0.70),
            max_iter_interna=10,
            factor_balanceo=2,
            cache_dir=CACHE_DIR,
        )

    ventanas_k = (1, 3, 5, 10, 15, None)
    entrenar_y_evaluar(
        "SOLO RE-ETIQUETADOS (+ SEMILLAS)", args.model,
        X_train_clean, y_train_clean, X_dev_vec, dev_msgs, X_test_vec, test_msgs, args.label,
        ventanas_k=ventanas_k,
    )
    entrenar_y_evaluar(
        "COMPLETO (RE-ETIQUETADOS + FALLBACK)", args.model,
        X_train_full, y_train_full, X_dev_vec, dev_msgs, X_test_vec, test_msgs, args.label,
        ventanas_k=ventanas_k,
    )

    imprimir_resumen(time.perf_counter() - t_total)


if __name__ == "__main__":
    main()