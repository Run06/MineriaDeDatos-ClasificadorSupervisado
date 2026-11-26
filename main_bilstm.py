import argparse
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preproceso import DEVICE, dividir_mensajes, vectorizar_embeddings
from modelo_bilstm import (
    construir_secuencias_por_usuario,
    entrenar_bilstm,
    evaluar_modelo,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificación a nivel de usuario con BiLSTM + Attention"
    )
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto a clasificar")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimensión oculta de la BiLSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Tasa de dropout")
    parser.add_argument("--epochs", type=int, default=40, help="Épocas máximas por ejecución")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()

def cargar_csv(ruta):
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)

def main():
    args = parse_args()
    device = torch.device(DEVICE)
    print(f"Dispositivo de entrenamiento: {device}")

    raw_data = cargar_csv(args.file)
    print(f"Usuarios totales cargados: {len(raw_data)}")

    # 1. Misma partición estratificada por usuario (70% / 15% / 15%) que el baseline
    train_users, test_users = train_test_split(
        raw_data, test_size=0.15, random_state=42, stratify=raw_data[args.label]
    )
    train_users, dev_users = train_test_split(
        train_users, test_size=(15 / 85), random_state=42, stratify=train_users[args.label]
    )

    print("Partición de USUARIOS (70% / 15% / 15%):")
    print(f"  Train: {len(train_users)} | Dev: {len(dev_users)} | Test: {len(test_users)}")

    # 2. Explotar mensajes por separado (cero fuga de datos)
    train_msgs = dividir_mensajes(train_users, col_post=args.target)
    dev_msgs = dividir_mensajes(dev_users, col_post=args.target)
    test_msgs = dividir_mensajes(test_users, col_post=args.target)

    print("Partición de MENSAJES:")
    print(f"  Train: {len(train_msgs)} | Dev: {len(dev_msgs)} | Test: {len(test_msgs)}")

    # 3. Vectorización (reutiliza automáticamente la caché de Dev y Test)
    print("\nCalculando embeddings para Train...")
    X_train_vec = vectorizar_embeddings(train_msgs[args.target])

    print("Calculando embeddings para Dev...")
    X_dev_vec = vectorizar_embeddings(dev_msgs[args.target])

    print("Calculando embeddings para Test...")
    X_test_vec = vectorizar_embeddings(test_msgs[args.target])

    # 4. Agrupar embeddings en secuencias por usuario
    clases = sorted(raw_data[args.label].unique())
    label_to_idx = {c: i for i, c in enumerate(clases)}

    train_seqs, train_labels, _ = construir_secuencias_por_usuario(
        X_train_vec, train_msgs, args.label, label_to_idx
    )
    dev_seqs, dev_labels, _ = construir_secuencias_por_usuario(
        X_dev_vec, dev_msgs, args.label, label_to_idx
    )
    test_seqs, test_labels, _ = construir_secuencias_por_usuario(
        X_test_vec, test_msgs, args.label, label_to_idx
    )

    input_dim = X_train_vec.shape[1]
    num_classes = len(clases)
    ventanas_k = (1, 3, 5, 10, 15, None)
    semillas = (42, 100, 2024, 7, 99)

    # Acumuladores para media y desviación típica por ventana k
    resultados_dev = {k: {"f1": [], "auc": [], "kappa": [], "msgs": []} for k in ventanas_k}
    resultados_test = {k: {"f1": [], "auc": [], "kappa": [], "msgs": []} for k in ventanas_k}

    mejor_dev_global = -1.0
    mejor_reporte_test = None

    print("\n" + "=" * 65)
    print(f"ENTRENANDO BiLSTM + ATTENTION ({len(semillas)} semillas | dim={input_dim})")
    print("=" * 65)

    for seed in semillas:
        model, mejor_f1_dev = entrenar_bilstm(
            train_seqs=train_seqs,
            train_labels=train_labels,
            dev_seqs=dev_seqs,
            dev_labels=dev_labels,
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            seed=seed,
        )

        for k in ventanas_k:
            res_d = evaluar_modelo(model, dev_seqs, dev_labels, device, max_mensajes=k)
            res_t = evaluar_modelo(model, test_seqs, test_labels, device, max_mensajes=k)

            resultados_dev[k]["f1"].append(res_d["macro_f1"])
            resultados_dev[k]["auc"].append(res_d["auc"])
            resultados_dev[k]["kappa"].append(res_d["kappa"])
            resultados_dev[k]["msgs"].append(res_d["media_msgs"])

            resultados_test[k]["f1"].append(res_t["macro_f1"])
            resultados_test[k]["auc"].append(res_t["auc"])
            resultados_test[k]["kappa"].append(res_t["kappa"])
            resultados_test[k]["msgs"].append(res_t["media_msgs"])

            if k is None and mejor_f1_dev > mejor_dev_global:
                mejor_dev_global = mejor_f1_dev
                mejor_reporte_test = (seed, res_t["y_true"], res_t["y_pred"])

        print(f"  > Semilla {seed:>4} completada | Mejor Dev Macro F1: {mejor_f1_dev:.4f}")

    # 5. Tabla resumen (Media ± StDev) para RQ1 y RQ2
    print("\n" + "=" * 85)
    print("RESULTADOS BiLSTM + ATTENTION (Media ± StDev sobre 5 semillas)")
    print("=" * 85)
    print(
        f"{'Ventana':>8} | {'Msgs':>5} | {'Dev Macro F1':>15} | "
        f"{'Test Macro F1':>15} | {'Test AUC':>15} | {'Test Kappa':>15}"
    )
    print("-" * 85)

    for k in ventanas_k:
        etiqueta_k = "Todos" if k is None else str(k)
        msgs = np.mean(resultados_test[k]["msgs"])
        d_f1_m, d_f1_s = np.mean(resultados_dev[k]["f1"]), np.std(resultados_dev[k]["f1"])
        t_f1_m, t_f1_s = np.mean(resultados_test[k]["f1"]), np.std(resultados_test[k]["f1"])
        t_auc_m, t_auc_s = np.mean(resultados_test[k]["auc"]), np.std(resultados_test[k]["auc"])
        t_kap_m, t_kap_s = np.mean(resultados_test[k]["kappa"]), np.std(resultados_test[k]["kappa"])

        print(
            f"{etiqueta_k:>8} | {msgs:5.1f} | {d_f1_m:.4f} ± {d_f1_s:.4f} | "
            f"{t_f1_m:.4f} ± {t_f1_s:.4f} | {t_auc_m:.4f} ± {t_auc_s:.4f} | "
            f"{t_kap_m:.4f} ± {t_kap_s:.4f}"
        )

    # 6. Reporte detallado por clase de la mejor semilla en Dev (k=Todos)
    if mejor_reporte_test is not None:
        best_seed, y_true, y_pred = mejor_reporte_test
        print(f"\nReporte de clasificación Test (Mejor semilla en Dev: {best_seed}, k=Todos)")
        print(classification_report(y_true, y_pred, target_names=clases, zero_division=0))


if __name__ == "__main__":
    main()