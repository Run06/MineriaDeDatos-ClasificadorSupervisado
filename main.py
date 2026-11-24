import argparse
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager

import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from preproceso import dividir_mensajes, vectorizar
from re_etiquetado import cascading_self_training

CACHE_DIR = "cache"
PROCESO = psutil.Process(os.getpid())
RESUMEN = []  # (fase, segundos, ram_inicio, ram_fin, ram_pico)


# --------------------------------------------------------------------------
# Medición de tiempo y memoria
# --------------------------------------------------------------------------
def ram_mb():
    """RAM residente (RSS) actual del proceso, en MB."""
    return PROCESO.memory_info().rss / 1024 ** 2


@contextmanager
def medir(nombre):
    """
    Mide tiempo y RAM de una fase. El pico se obtiene muestreando el RSS
    del proceso cada 50 ms desde un hilo auxiliar.
    """
    parar = threading.Event()
    pico = [ram_mb()]

    def muestrear():
        while not parar.is_set():
            pico[0] = max(pico[0], ram_mb())
            parar.wait(0.05)

    hilo = threading.Thread(target=muestrear, daemon=True)
    ram_ini = ram_mb()
    t0 = time.perf_counter()
    hilo.start()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        parar.set()
        hilo.join()
        ram_fin = ram_mb()
        pico[0] = max(pico[0], ram_fin)
        RESUMEN.append((nombre, dt, ram_ini, ram_fin, pico[0]))
        print(f"[MEDIDA] {nombre}: {dt:.1f} s | RAM {ram_ini:.0f} -> {ram_fin:.0f} MB | pico {pico[0]:.0f} MB")


def imprimir_resumen(t_total):
    print("\n" + "=" * 78)
    print("RESUMEN DE TIEMPO Y MEMORIA")
    print("=" * 78)
    print(f"{'Fase':<46}{'Tiempo (s)':>12}{'RAM fin (MB)':>14}{'Pico (MB)':>10}")
    print("-" * 78)
    for nombre, dt, _, ram_fin, pico in RESUMEN:
        print(f"{nombre:<46}{dt:>12.1f}{ram_fin:>14.0f}{pico:>10.0f}")
    print("-" * 78)
    print(f"{'TOTAL':<46}{t_total:>12.1f}")
    pico_global = max(r[4] for r in RESUMEN)
    print(f"Pico de RAM (muestreado): {pico_global:.0f} MB")
    pico_so = getattr(PROCESO.memory_info(), "peak_wset", None)  # solo Windows
    if pico_so is not None:
        print(f"Pico de RAM (Windows, peak_wset): {pico_so / 1024 ** 2:.0f} MB")
    print("=" * 78)


# --------------------------------------------------------------------------
# Argumentos y utilidades
# --------------------------------------------------------------------------
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


def crear_clasificador(nombre):
    if nombre == "svm_linear":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=42)
    if nombre == "svm_rbf":
        return SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42)
    if nombre == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_features="sqrt", max_depth=300,
            min_samples_split=5, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1)
    raise ValueError(f"Modelo no soportado: {nombre}")


def entrenar_y_evaluar(titulo, nombre_modelo, X_tr, y_tr, X_dev_vec, y_dev, X_test_vec, y_test):
    print("\n" + "=" * 50)
    print(f"MODELO: {titulo} (Train: {len(y_tr)} mensajes)")
    print("=" * 50)

    modelo = crear_clasificador(nombre_modelo)

    with medir(f"Entrenamiento [{titulo}]"):
        modelo.fit(X_tr, y_tr)

    with medir(f"Predicción dev+test [{titulo}]"):
        preds_dev = modelo.predict(X_dev_vec)
        preds_test = modelo.predict(X_test_vec)

    print(f"Macro F1-Score - Dev:  {f1_score(y_dev, preds_dev, average='macro', zero_division=0):.4f}")
    print(f"Macro F1-Score - Test: {f1_score(y_test, preds_test, average='macro', zero_division=0):.4f}")

    print("\nReporte de clasificación Test")
    print(classification_report(y_test, preds_test, zero_division=0))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    t_total = time.perf_counter()

    if args.limpiar_cache and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Caché '{CACHE_DIR}' eliminada.")

    print(f"[RAM] Inicio: {ram_mb():.0f} MB")
    print(f"Configuración -> vectorizador: {args.vectorizer} | modelo: {args.model}")

    with medir("Carga y partición de datos"):
        raw_data = cargar_csv(args.file)
        data = dividir_mensajes(raw_data)
        X = data[args.target]
        y = data[args.label]

        print(f"Dimensiones del dataset cargado: {data.shape}")

        # Partición estratificada: 70% Train / 15% Dev / 15% Test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train_full, y_train_full, test_size=(15 / 85), random_state=42, stratify=y_train_full
        )

        print("Tamaños de las particiones (70% / 15% / 15%)")
        print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

        mensajes_semilla = cargar_csv(args.seed)
        print(f"Semillas manuales cargadas: {len(mensajes_semilla)}")

        X_seed = mensajes_semilla[args.target]
        y_seed = mensajes_semilla[args.label].values

    # Vectorización
    X_train_combinado = pd.concat([X_seed, X_train], ignore_index=True)
    print(f"\nVectorizando con: {args.vectorizer}")
    with medir(f"Vectorización ({args.vectorizer})"):
        X_train_combinado_vec, X_dev_vec, X_test_vec = vectorizar(
            args.vectorizer, X_train_combinado, X_dev, X_test)

    num_semillas = len(X_seed)
    X_seed_vec = X_train_combinado_vec[:num_semillas]
    X_reddit_vec = X_train_combinado_vec[num_semillas:]

    # Re-etiquetado en cascada
    with medir("Re-etiquetado en cascada"):
        X_train_clean, y_train_clean, X_train_fb, y_train_fb = cascading_self_training(
            X_seed=X_seed_vec,
            y_seed=y_seed,
            X_unlabeled=X_reddit_vec,
            y_unlabeled_orig=y_train.values,
            textos_unlabeled=X_train.values,
            umbrales=(0.95, 0.90, 0.85, 0.80, 0.75),
            max_iter_interna=10,
            factor_balanceo=2,
            cache_dir=CACHE_DIR,
        )

    entrenar_y_evaluar("SOLO RE-ETIQUETADOS (+ SEMILLAS)", args.model,
                       X_train_clean, y_train_clean, X_dev_vec, y_dev, X_test_vec, y_test)
    entrenar_y_evaluar("COMPLETO (RE-ETIQUETADOS + FALLBACK)", args.model,
                       X_train_fb, y_train_fb, X_dev_vec, y_dev, X_test_vec, y_test)

    imprimir_resumen(time.perf_counter() - t_total)


if __name__ == "__main__":
    main()