import argparse
import sys
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preproceso import dividir_mensajes, vectorizar_embeddings
from re_etiquetado import cascading_self_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificación supervisada con re-etiquetado en cascada y embeddings"
    )
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto a clasificar")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas")
    parser.add_argument("-s", "--seed", default="semilla.csv", help="Archivo CSV con instancias semilla")
    return parser.parse_args()


def cargar_csv(ruta):
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)
def entrenar_y_evaluar(titulo, X_tr, y_tr, X_dev_vec, y_dev, X_test_vec, y_test):
    print("\n" + "=" * 50)
    print(f"MODELO: {titulo} (Train: {len(y_tr)} mensajes)")
    print("=" * 50)

    svc_model = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )
    svc_model.fit(X_tr, y_tr)

    preds_dev = svc_model.predict(X_dev_vec)
    print(f"Macro F1-Score - Dev:  {f1_score(y_dev, preds_dev, average='macro', zero_division=0):.4f}")

    preds_test = svc_model.predict(X_test_vec)
    print(f"Macro F1-Score - Test: {f1_score(y_test, preds_test, average='macro', zero_division=0):.4f}")

    print("\nReporte de clasificación Test")
    print(classification_report(y_test, preds_test, zero_division=0))


def main():
    args = parse_args()

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
    print("\nCalculando embeddings para el conjunto de entrenamiento combinado...")
    X_train_combinado_vec = vectorizar_embeddings(X_train_combinado)

    print("Calculando embeddings para Dev...")
    X_dev_vec = vectorizar_embeddings(X_dev)

    print("Calculando embeddings para Test...")
    X_test_vec = vectorizar_embeddings(X_test)

    num_semillas = len(X_seed)
    X_seed_vec = X_train_combinado_vec[:num_semillas]
    X_reddit_vec = X_train_combinado_vec[num_semillas:]

    # Re-etiquetado en cascada
    X_train_clean, y_train_clean, X_train_full, y_train_full = cascading_self_training(
        X_seed=X_seed_vec,
        y_seed=y_seed,
        X_unlabeled=X_reddit_vec,
        y_unlabeled_orig=y_train.values,
        textos_unlabeled=X_train.values,
        umbrales=(0.95, 0.90, 0.85, 0.80, 0.75),
        max_iter_interna=10,
        factor_balanceo=2,
    )

    entrenar_y_evaluar(
        "SOLO RE-ETIQUETADOS (+ SEMILLAS)",
        X_train_clean, y_train_clean, X_dev_vec, y_dev, X_test_vec, y_test
    )
    entrenar_y_evaluar(
        "COMPLETO (RE-ETIQUETADOS + FALLBACK)",
        X_train_full, y_train_full, X_dev_vec, y_dev, X_test_vec, y_test
    )


if __name__ == "__main__":
    main()