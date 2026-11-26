import argparse
import os
import shutil
import sys
import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from eval import entrenar_y_evaluar
from medicion import imprimir_resumen, medir, ram_mb, exportar_mediciones, RESUMEN
from graficas import caracterizar, graficas_evaluacion, grafica_recursos, graficas_cascada
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
    parser.add_argument("-m", "--model", choices=["svm_linear", "svm_rbf", "rf", "mlp"],
                        default="svm_linear", help="Clasificador final")
    parser.add_argument("-v", "--vectorizer", choices=["fasttext", "embeddings"],
                        default="embeddings", help="Vectorización")
    parser.add_argument("--limpiar-cache", action="store_true",
                        help="Borra la carpeta 'cache' antes de ejecutar (medición en frío)")
    parser.add_argument("--resultados", default="resultados", help="Carpeta de exportación")
    parser.add_argument("--baseline-original", action="store_true", help="Añade control con etiquetas originales y las mismas semillas")
    return parser.parse_args()


def cargar_csv(ruta):
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)


def main():
    args = parse_args()
    RESUMEN.clear()
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
        X_train_clean, y_train_clean, X_train_full, y_train_full, historial = cascading_self_training(
            X_seed=X_seed_vec,
            y_seed=y_seed,
            X_unlabeled=X_reddit_vec,
            y_unlabeled_orig=y_train.values,
            textos_unlabeled=X_train.values,
            umbrales=(0.95, 0.90, 0.85, 0.80, 0.75, 0.70),
            max_iter_interna=10,
            factor_balanceo=2,
            cache_dir=CACHE_DIR,
            devolver_historial=True,
        )

    ventanas_k = (1, 3, 5, 10, 15, None)
    estrategias = [("Solo pseudo", X_train_clean, y_train_clean),
                   ("Con fallback", X_train_full, y_train_full)]
    if args.baseline_original:
        estrategias.append(("Etiquetas originales", X_train_combinado_vec,
                            np.concatenate([y_seed, y_train.values])))
    clases_eval = np.unique(np.concatenate([raw_data[args.label].values, y_seed]))
    metricas, por_clase, matrices = [], [], []
    for titulo, X, y in estrategias:
        m, c, cm = entrenar_y_evaluar(titulo, args.model, X, y, X_dev_vec, dev_msgs,
                                      X_test_vec, test_msgs, args.label,
                                      ventanas_k=ventanas_k, clases_eval=clases_eval)
        metricas.extend(m)
        por_clase.extend(c)
        matrices.extend(cm)

    segundos = time.perf_counter() - t_total
    imprimir_resumen(segundos)
    # Exportar fuera de las mediciones del pipeline: no penalizar al modelo por dibujar.
    salida = Path(args.resultados) / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") +
                                     f"_{args.vectorizer}_{args.model}")
    salida.mkdir(parents=True, exist_ok=False)
    met, pc, cm = map(pd.DataFrame, (metricas, por_clase, matrices))
    for nombre, df in [("metricas", met), ("metricas_por_clase", pc), ("confusiones", cm)]:
        df.to_csv(salida / f"{nombre}.csv", index=False)
    exportar_mediciones(salida / "recursos.csv")
    caracterizar([("Train", train_users, train_msgs), ("Dev", dev_users, dev_msgs),
                  ("Test", test_users, test_msgs)], args.target, args.label, salida)
    graficas_evaluacion(met, pc, cm, salida)
    grafica_recursos(pd.read_csv(salida / "recursos.csv"), salida)
    graficas_cascada(historial, len(X_train), (0.95, 0.90, 0.85, 0.80, 0.75, 0.70), salida)
    firma = hashlib.sha256()
    for ruta in (args.file, args.seed):
        firma.update(hashlib.sha256(Path(ruta).read_bytes()).digest())
    firma.update(f"{args.target}|{args.label}|split42_70_15_15".encode())
    config = dict(vectorizer=args.vectorizer, model=args.model,
                  modo_cache="frio" if args.limpiar_cache else "normal",
                  baseline_original=args.baseline_original, firma_datos=firma.hexdigest(),
                  segundos_pipeline=segundos, semillas=len(y_seed),
                  mensajes_train_original=len(y_train), pseudoetiquetados=len(y_train_clean)-len(y_seed),
                  fallback=len(y_train_full)-len(y_train_clean))
    (salida / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Gráficas y CSV guardados en: {salida}")


if __name__ == "__main__":
    main()