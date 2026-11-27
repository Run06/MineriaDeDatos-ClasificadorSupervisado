import argparse
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from preproceso import DEVICE, dividir_mensajes
from modelo_jerarquico import (
    ORDEN_CLINICO,
    calcular_metricas_graded,
    construir_textos_por_usuario,
    entrenar_jerarquico,
    evaluar_secuencias_emb,
    extraer_secuencias_con_encoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modelo jerárquico Softmax: MentalRoBERTa (últimas 2 capas) + BiLSTM + Attention + MaxPool"
    )
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de usuarios")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas")
    parser.add_argument("--model_name", default="mental/mental-roberta-base")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Capas finales a descongelar")
    parser.add_argument("--max_length", type=int, default=256, help="Tokens máximos por mensaje")
    parser.add_argument("--max_msgs_train", type=int, default=24, help="Tope de mensajes/usuario en Train")
    parser.add_argument("--batch_size", type=int, default=2, help="Usuarios por micro-batch")
    parser.add_argument("--accum_steps", type=int, default=8, help="Pasos de acumulación (batch=16)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr_encoder", type=float, default=3e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_power", type=float, default=0.70, help="Exponente de pesos de clase")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024, 7, 99])
    return parser.parse_args()


def cargar_csv(ruta):
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)


def imprimir_diagnostico_test(y_true, y_pred, y_prob, clases):
    """Imprime AUC por clase, métricas ordinales/Graded, matriz clínica y radiografía de Attempt."""
    aucs_clase = []
    for i, c in enumerate(clases):
        try:
            auc_c = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            aucs_clase.append(f"{c}:{auc_c:.3f}")
        except ValueError:
            aucs_clase.append(f"{c}:nan")
    print("\nAUC OvR por clase (Test, k=Todos): " + " | ".join(aucs_clase))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(clases))))
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real:{c[:4]}" for c in clases],
        columns=[f"Pred:{c[:4]}" for c in clases],
    )
    print("\nMatriz de confusión Test (orden clínico de riesgo, k=Todos):")
    print(df_cm.to_string())

    graded = calcular_metricas_graded(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    distancias = np.abs(y_true - y_pred)
    print(
        f"\nGraded Precision: {graded['graded_p']:.4f} | "
        f"Graded Recall: {graded['graded_r']:.4f} | "
        f"Graded F1: {graded['graded_f1']:.4f}"
    )
    print(
        f"QWK (Kappa cuadrático): {qwk:.4f} | "
        f"MAE ordinal (0-4): {distancias.mean():.3f} | "
        f"Exacto o ±1 nivel: {100 * np.mean(distancias <= 1):.1f}%"
    )

    if "Attempt" in clases:
        idx_att = clases.index("Attempt")
        indices_att = np.where(y_true == idx_att)[0]
        cabecera = "  ".join(f"{c[:4]:>6}" for c in clases)
        print(f"\nRadiografía de los {len(indices_att)} usuarios reales de Attempt:")
        print(f"  {'Usr':>3} | {'Pred':<10} | {cabecera}")
        print("  " + "-" * (18 + 8 * len(clases)))
        for pos in indices_att:
            pred_nombre = clases[y_pred[pos]]
            probs_str = "  ".join(f"{y_prob[pos, i]:6.3f}" for i in range(len(clases)))
            print(f"  {pos:3d} | {pred_nombre:<10} | {probs_str}")


def main():
    args = parse_args()
    device = torch.device(DEVICE)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(
        f"Dispositivo: {device} | Encoder: {args.model_name} "
        f"(últimas {args.unfreeze_layers} capas | weight_power={args.weight_power})"
    )

    raw_data = cargar_csv(args.file)
    print(f"Usuarios totales cargados: {len(raw_data)}")

    # 1. Misma partición estratificada por usuario (70% / 15% / 15%) que baseline y BiLSTM
    train_users, test_users = train_test_split(
        raw_data, test_size=0.15, random_state=42, stratify=raw_data[args.label]
    )
    train_users, dev_users = train_test_split(
        train_users, test_size=(15 / 85), random_state=42, stratify=train_users[args.label]
    )

    train_msgs = dividir_mensajes(train_users, col_post=args.target)
    dev_msgs = dividir_mensajes(dev_users, col_post=args.target)
    test_msgs = dividir_mensajes(test_users, col_post=args.target)

    print(
        f"Usuarios -> Train: {len(train_users)} | Dev: {len(dev_users)} | Test: {len(test_users)}"
    )
    print(
        f"Mensajes -> Train: {len(train_msgs)} | Dev: {len(dev_msgs)} | Test: {len(test_msgs)}"
    )

    # Orden clínico fijo (0=Supportive .. 4=Attempt) para que Graded F1 y QWK sean exactos
    unicas = set(raw_data[args.label].unique())
    clases = [c for c in ORDEN_CLINICO if c in unicas]
    clases += [c for c in sorted(unicas) if c not in clases]
    label_to_idx = {c: i for i, c in enumerate(clases)}
    print(f"Orden clínico de clases: {clases}")

    train_texts, train_labels, _ = construir_textos_por_usuario(
        train_msgs, args.target, args.label, label_to_idx
    )
    dev_texts, dev_labels, _ = construir_textos_por_usuario(
        dev_msgs, args.target, args.label, label_to_idx
    )
    test_texts, test_labels, _ = construir_textos_por_usuario(
        test_msgs, args.target, args.label, label_to_idx
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ventanas_k = (1, 3, 5, 10, 15, None)

    resultados_dev = {k: {"f1": [], "gf1": [], "auc": [], "msgs": []} for k in ventanas_k}
    resultados_test = {
        k: {"f1": [], "wf1": [], "gf1": [], "auc": [], "kappa": [], "qwk": [], "msgs": []}
        for k in ventanas_k
    }
    probs_test_ensemble = {k: [] for k in ventanas_k}

    print("\n" + "=" * 85)
    print(f"ENTRENAMIENTO JERÁRQUICO SOFTMAX END-TO-END ({len(args.seeds)} semillas)")
    print("=" * 85)

    for seed in args.seeds:
        print(f"\n--- Semilla {seed} ---")
        model, mejor_f1_dev, mejor_gf1_dev, mejor_auc_dev = entrenar_jerarquico(
            train_texts=train_texts,
            train_labels=train_labels,
            dev_texts=dev_texts,
            dev_labels=dev_labels,
            clases=clases,
            tokenizer=tokenizer,
            device=device,
            model_name=args.model_name,
            unfreeze_layers=args.unfreeze_layers,
            max_length=args.max_length,
            max_msgs_train=args.max_msgs_train,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            lr_encoder=args.lr_encoder,
            lr_head=args.lr_head,
            weight_power=args.weight_power,
            epochs=args.epochs,
            patience=args.patience,
            seed=seed,
        )

        dev_embs = extraer_secuencias_con_encoder(
            model, dev_texts, tokenizer, device, max_length=args.max_length, use_bf16=use_bf16
        )
        test_embs = extraer_secuencias_con_encoder(
            model, test_texts, tokenizer, device, max_length=args.max_length, use_bf16=use_bf16
        )

        for k in ventanas_k:
            res_d = evaluar_secuencias_emb(model, dev_embs, dev_labels, device, max_mensajes=k)
            res_t = evaluar_secuencias_emb(model, test_embs, test_labels, device, max_mensajes=k)

            resultados_dev[k]["f1"].append(res_d["macro_f1"])
            resultados_dev[k]["gf1"].append(res_d["graded_f1"])
            resultados_dev[k]["auc"].append(res_d["auc"])
            resultados_dev[k]["msgs"].append(res_d["media_msgs"])

            resultados_test[k]["f1"].append(res_t["macro_f1"])
            resultados_test[k]["wf1"].append(res_t["weighted_f1"])
            resultados_test[k]["gf1"].append(res_t["graded_f1"])
            resultados_test[k]["auc"].append(res_t["auc"])
            resultados_test[k]["kappa"].append(res_t["kappa"])
            resultados_test[k]["qwk"].append(res_t["qwk"])
            resultados_test[k]["msgs"].append(res_t["media_msgs"])

            probs_test_ensemble[k].append(res_t["y_prob"])

        print(
            f"  > Semilla {seed} cerrada | Mejor Dev AUC: {mejor_auc_dev:.4f} | "
            f"Dev F1: {mejor_f1_dev:.4f} (GF1: {mejor_gf1_dev:.4f}) | "
            f"Test F1: {resultados_test[None]['f1'][-1]:.4f} "
            f"(GF1: {resultados_test[None]['gf1'][-1]:.4f})"
        )

        del model, dev_embs, test_embs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 1. Tabla Media ± StDev
    print("\n" + "=" * 118)
    print(f"RESULTADOS JERÁRQUICO SOFTMAX (Media ± StDev sobre {len(args.seeds)} semillas)")
    print("=" * 118)
    print(
        f"{'Ventana':>8} | {'Msgs':>5} | {'Dev Macro F1':>15} | {'Test Macro F1':>15} | "
        f"{'Test Graded F1':>15} | {'Test AUC':>15} | {'Test Kappa':>15}"
    )
    print("-" * 118)

    for k in ventanas_k:
        etiqueta_k = "Todos" if k is None else str(k)
        msgs = np.mean(resultados_test[k]["msgs"])
        d_m, d_s = np.mean(resultados_dev[k]["f1"]), np.std(resultados_dev[k]["f1"])
        t_m, t_s = np.mean(resultados_test[k]["f1"]), np.std(resultados_test[k]["f1"])
        g_m, g_s = np.mean(resultados_test[k]["gf1"]), np.std(resultados_test[k]["gf1"])
        a_m, a_s = np.mean(resultados_test[k]["auc"]), np.std(resultados_test[k]["auc"])
        k_m, k_s = np.mean(resultados_test[k]["kappa"]), np.std(resultados_test[k]["kappa"])

        print(
            f"{etiqueta_k:>8} | {msgs:5.1f} | {d_m:.4f} ± {d_s:.4f} | {t_m:.4f} ± {t_s:.4f} | "
            f"{g_m:.4f} ± {g_s:.4f} | {a_m:.4f} ± {a_s:.4f} | {k_m:.4f} ± {k_s:.4f}"
        )

    # 2. Tabla Ensemble (promedio de probabilidades de todas las semillas)
    if len(args.seeds) > 1:
        print("\n" + "=" * 105)
        print(f"ENSEMBLE ({len(args.seeds)} semillas promediadas) - TEST")
        print("=" * 105)
        print(
            f"{'Ventana':>8} | {'Macro F1':>10} | {'Weighted F1':>11} | "
            f"{'Graded F1':>10} | {'Test AUC':>10} | {'Test QWK':>10} | {'Test Kappa':>10}"
        )
        print("-" * 105)

        for k in ventanas_k:
            etiqueta_k = "Todos" if k is None else str(k)
            prob_ens = np.mean(probs_test_ensemble[k], axis=0)
            pred_ens = np.argmax(prob_ens, axis=1)
            mf1 = f1_score(test_labels, pred_ens, average="macro", zero_division=0)
            wf1 = f1_score(test_labels, pred_ens, average="weighted", zero_division=0)
            gf1 = calcular_metricas_graded(test_labels, pred_ens)["graded_f1"]
            auc = roc_auc_score(test_labels, prob_ens, multi_class="ovr", average="macro")
            qwk = cohen_kappa_score(test_labels, pred_ens, weights="quadratic")
            kap = cohen_kappa_score(test_labels, pred_ens)
            print(
                f"{etiqueta_k:>8} | {mf1:10.4f} | {wf1:11.4f} | "
                f"{gf1:10.4f} | {auc:10.4f} | {qwk:10.4f} | {kap:10.4f}"
            )

    # 3. Reporte detallado y diagnóstico final (k=Todos)
    prob_final = np.mean(probs_test_ensemble[None], axis=0)
    pred_final = np.argmax(prob_final, axis=1)
    titulo_rep = "Ensemble" if len(args.seeds) > 1 else f"Semilla {args.seeds[0]}"
    print(f"\nReporte de clasificación Test ({titulo_rep}, k=Todos)")
    print(classification_report(test_labels, pred_final, target_names=clases, zero_division=0))
    imprimir_diagnostico_test(test_labels, pred_final, prob_final, clases)


if __name__ == "__main__":
    main()