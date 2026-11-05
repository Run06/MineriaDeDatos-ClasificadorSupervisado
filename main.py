import argparse
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

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

    return parser.parse_args()

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
    #Train (80%) Test (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #train_full Train (80%) Dev (20%)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print("Tamaños de las particiones")
    print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

    #Vectorización baseline - TF-IDF
    X_train_vec, X_dev_vec, X_test_vec = tfidf(X_train, X_dev, X_test)

    #Algoritmo baseline - RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    print("-"*50)

    #Evaluación sobre Dev
    preds_dev = model.predict(X_dev_vec)
    print(f"Macro F1-Score - Dev: {f1_score(y_dev, preds_dev, average='macro'):.4f}")

    #Evaluación sobre Test
    preds_test = model.predict(X_test_vec)
    print(f"Macro F1-Score - Test: {f1_score(y_test, preds_test, average='macro'):.4f}")
    print("Reporte de clasificación Test")
    print(classification_report(y_test, preds_test))