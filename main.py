import argparse
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from preproceso import dividir_mensajes, vectorizar_embeddings
from re_etiquetado import cascading_self_training
from eval import entrenar_y_evaluar


def parse_args():
    """
    - Precondiciones: Ninguna.
    - Entradas: Argumentos recibidos por linea de comandos (sys.argv).
    - Salidas: argparse.Namespace con las rutas de entrada y columnas configuradas.
    - Postcondiciones: Devuelve los parametros listos para ejecutar la clasificacion.
    """
    parser = argparse.ArgumentParser(
        description="Clasificación supervisada con re-etiquetado en cascada y embeddings"
    )
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto a clasificar")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas")
    parser.add_argument("-s", "--seed", default="semilla.csv", help="Archivo CSV con instancias semilla")
    return parser.parse_args()


def cargar_csv(ruta):
    """
    - Precondiciones: ruta es una cadena o Path valida.
    - Entradas: ruta (str o Path del archivo CSV a cargar).
    - Salidas: pd.DataFrame con los datos leidos del CSV.
    - Postcondiciones: Devuelve el DataFrame cargado o finaliza el programa con codigo 1 si no existe.
    """
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)

def main():
    """
    - Precondiciones: Existen los ficheros CSV indicados en --file y --seed y contienen las columnas target y label.
    - Entradas: Ninguna directa (lee parametros desde parse_args).
    - Salidas: Ninguna.
    - Postcondiciones: Divide por usuario sin fuga, vectoriza, ejecuta el re-etiquetado en cascada y evalua los conjuntos.
    """
    args = parse_args()

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
    #Explotar mensajes por separado (cero fuga de datos entre particiones)
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
    print("\nCalculando embeddings para el conjunto de entrenamiento combinado...")
    X_train_combinado_vec = vectorizar_embeddings(X_train_combinado)

    print("Calculando embeddings para Dev...")
    X_dev_vec = vectorizar_embeddings(dev_msgs[args.target])

    print("Calculando embeddings para Test...")
    X_test_vec = vectorizar_embeddings(test_msgs[args.target])

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
        umbrales=(0.95, 0.90, 0.85, 0.80, 0.75,0.70),
        max_iter_interna=10,
        factor_balanceo=2,
    )

    entrenar_y_evaluar(
        "SOLO RE-ETIQUETADOS (+ SEMILLAS)",
        X_train_clean, y_train_clean, X_dev_vec, dev_msgs, X_test_vec, test_msgs, args.label,
        ventanas_k=(1, 3, 5, 10, 15, None),
    )
    entrenar_y_evaluar(
        "COMPLETO (RE-ETIQUETADOS + FALLBACK)",
        X_train_full, y_train_full, X_dev_vec, dev_msgs, X_test_vec, test_msgs, args.label,
        ventanas_k=(1, 3, 5, 10, 15, None),
    )

if __name__ == "__main__":
    main()