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
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer

from preproceso import DEVICE, dividir_mensajes
from modelo_jerarquico import (
    ORDEN_CLINICO,
    calcular_metricas_graded,
    construir_textos_por_usuario,
    entrenar_jerarquico_mil,
    evaluar_secuencias_emb,
    extraer_secuencias_con_encoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MentalRoBERTa-LoRA + MIL (LogSumExp) + Multi-Task Semillas + 5-Fold CV"
    )
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de usuarios")
    parser.add_argument("-s", "--seed_file", default="semilla.csv", help="CSV de semillas limpias")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas")
    parser.add_argument("--model_name", default="mental/mental-roberta-base")
    parser.add_argument("--lora_layers", type=int, default=6, help="Capas finales con LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="Rango LoRA")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Escala LoRA")
    parser.add_argument("--seed_alpha", type=float, default=0.35, help="Peso Multi-Task semillas (0 desactiva)")
    parser.add_argument("--max_length", type=int, default=256, help="Tokens máximos por mensaje")
    parser.add_argument("--max_msgs_train", type=int, default=32, help="Tope de mensajes/usuario en Train")
    parser.add_argument("--msg_dropout", type=float, default=0.10, help="Dropout de mensajes en Train")
    parser.add_argument("--batch_size", type=int, default=2, help="Usuarios por micro-batch")
    parser.add_argument("--accum_steps", type=int, default=8, help="Pasos de acumulación (batch=16)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr_lora", type=float, default=3e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.0, help="Escalar Balanced Softmax")
    parser.add_argument("--sord_sigma", type=float, default=0.45, help="Sigma SORD ordinal")
    parser.add_argument("--n_folds", type=int, default=5, help="Número de folds CV sobre Train+Dev")
    parser.add_argument("--max_folds", type=int, default=None, help="Ejecutar solo N folds (para prueba rápida)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cargar_csv(ruta):
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta}' no existe o la ruta es incorrecta.")
        sys.exit(1)


def filtrar_semillas_sin_fuga(seed_texts, seed_labels, textos_prohibidos_norm):
    """Elimina cualquier semilla cuyo texto aparezca en Val o Test."""
    t_limpios, y_limpios = [], []
    for texto, etiqueta in zip(seed_texts, seed_labels):
        if str(texto).strip().lower() not in textos_prohibidos_norm:
            t_limpios.append(texto)
            y_limpios.append(etiqueta)
    return t_limpios, np.array(y_limpios, dtype=np.int64)


def imprimir_diagnostico_test(y_true, y_pred, y_prob, clases, titulo="Test, k=Todos"):
    """Imprime AUC por clase, métricas ordinales/Graded, matriz clínica y radiografía de Attempt."""
    aucs_clase = []
    for i, c in enumerate(clases):
        try:
            auc_c = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            aucs_clase.append(f"{c}:{auc_c:.3f}")
        except ValueError:
            aucs_clase.append(f"{c}:nan")
    print(f"\nAUC OvR por clase ({titulo}): " + " | ".join(aucs_clase))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(clases))))
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real:{c[:4]}" for c in clases],
        columns=[f"Pred:{c[:4]}" for c in clases],
    )
    print(f"\nMatriz de confusión ({titulo}):")
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
        print(f"\nRadiografía de los {len(indices_att)} usuarios reales de Attempt ({titulo}):")
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
        f"| LoRA(capas={args.lora_layers}, r={args.lora_r}) | tau={args.tau}"
    )

    raw_data = cargar_csv(args.file)
    print(f"Usuarios totales cargados: {len(raw_data)}")

    # 1. Exactamente el mismo Test congelado (15% = 75 usuarios) que el baseline y ejecuciones previas
    train_dev_users, test_users = train_test_split(
        raw_data, test_size=0.15, random_state=42, stratify=raw_data[args.label]
    )
    test_msgs = dividir_mensajes(test_users, col_post=args.target)

    print(
        f"Partición principal -> Train+Dev (CV): {len(train_dev_users)} usuarios | "
        f"Test congelado: {len(test_users)} usuarios ({len(test_msgs)} mensajes)"
    )

    # Orden clínico fijo: Supportive(0) < Indicator(1) < Ideation(2) < Behavior(3) < Attempt(4)
    unicas = set(raw_data[args.label].unique())
    clases = [c for c in ORDEN_CLINICO if c in unicas]
    clases += [c for c in sorted(unicas) if c not in clases]
    label_to_idx = {c: i for i, c in enumerate(clases)}
    print(f"Orden clínico de clases: {clases}")

    test_texts, test_labels, _ = construir_textos_por_usuario(
        test_msgs, args.target, args.label, label_to_idx
    )
    test_msgs_norm = {str(t).strip().lower() for seq in test_texts for t in seq}

    # 2. Carga y limpieza de semillas manuales (eliminando cualquier coincidencia con Test)
    seed_texts_base, seed_labels_base = [], np.array([], dtype=np.int64)
    if args.seed_alpha > 0 and args.seed_file:
        df_seed = cargar_csv(args.seed_file).dropna(subset=[args.target, args.label])
        df_seed = df_seed[df_seed[args.label].isin(label_to_idx)]
        raw_s_texts = df_seed[args.target].astype(str).tolist()
        raw_s_labels = np.array([label_to_idx[c] for c in df_seed[args.label]], dtype=np.int64)
        seed_texts_base, seed_labels_base = filtrar_semillas_sin_fuga(
            raw_s_texts, raw_s_labels, test_msgs_norm
        )
        print(
            f"Semillas Multi-Task cargadas: {len(seed_texts_base)} / {len(raw_s_texts)} "
            f"(tras filtrar coincidencias con Test)"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ventanas_k = (1, 3, 5, 10, 15, None)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    resultados_val = {k: {"f1": [], "gf1": [], "auc": [], "msgs": []} for k in ventanas_k}
    resultados_test = {
        k: {"f1": [], "wf1": [], "gf1": [], "auc": [], "kappa": [], "qwk": [], "msgs": []}
        for k in ventanas_k
    }
    probs_test_ensemble = {k: [] for k in ventanas_k}
    oof_true_list, oof_prob_list = [], []

    n_ejecutar = args.max_folds if args.max_folds is not None else args.n_folds
    print("\n" + "=" * 90)
    print(f"ENTRENAMIENTO MIL + LoRA + MULTI-TASK ({n_ejecutar}/{args.n_folds} Folds sobre 425 usuarios)")
    print("=" * 90)

    for fold_idx, (tr_idx, va_idx) in enumerate(
        skf.split(train_dev_users, train_dev_users[args.label]), start=1
    ):
        if fold_idx > n_ejecutar:
            break

        print(f"\n--- Fold {fold_idx}/{args.n_folds} (Train={len(tr_idx)} | Val={len(va_idx)}) ---")
        fold_train_msgs = dividir_mensajes(train_dev_users.iloc[tr_idx], col_post=args.target)
        fold_val_msgs = dividir_mensajes(train_dev_users.iloc[va_idx], col_post=args.target)

        tr_texts, tr_labels, _ = construir_textos_por_usuario(
            fold_train_msgs, args.target, args.label, label_to_idx
        )
        va_texts, va_labels, _ = construir_textos_por_usuario(
            fold_val_msgs, args.target, args.label, label_to_idx
        )

        # Filtramos también cualquier semilla perteneciente a Val de este fold
        val_msgs_norm = {str(t).strip().lower() for seq in va_texts for t in seq}
        fold_seed_texts, fold_seed_labels = filtrar_semillas_sin_fuga(
            seed_texts_base, seed_labels_base, val_msgs_norm
        )

        model, mejor_f1_val, mejor_gf1_val, mejor_auc_val = entrenar_jerarquico_mil(
            train_texts=tr_texts,
            train_labels=tr_labels,
            val_texts=va_texts,
            val_labels=va_labels,
            clases=clases,
            tokenizer=tokenizer,
            device=device,
            seed_texts=fold_seed_texts,
            seed_labels=fold_seed_labels,
            seed_alpha=args.seed_alpha,
            model_name=args.model_name,
            lora_layers=args.lora_layers,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            max_msgs_train=args.max_msgs_train,
            msg_dropout=args.msg_dropout,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            lr_lora=args.lr_lora,
            lr_head=args.lr_head,
            tau=args.tau,
            sord_sigma=args.sord_sigma,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + fold_idx * 100,
        )

        val_embs = extraer_secuencias_con_encoder(
            model, va_texts, tokenizer, device, max_length=args.max_length, use_bf16=use_bf16
        )
        test_embs = extraer_secuencias_con_encoder(
            model, test_texts, tokenizer, device, max_length=args.max_length, use_bf16=use_bf16
        )

        for k in ventanas_k:
            res_v = evaluar_secuencias_emb(model, val_embs, va_labels, device, max_mensajes=k)
            res_t = evaluar_secuencias_emb(model, test_embs, test_labels, device, max_mensajes=k)

            resultados_val[k]["f1"].append(res_v["macro_f1"])
            resultados_val[k]["gf1"].append(res_v["graded_f1"])
            resultados_val[k]["auc"].append(res_v["auc"])
            resultados_val[k]["msgs"].append(res_v["media_msgs"])

            resultados_test[k]["f1"].append(res_t["macro_f1"])
            resultados_test[k]["wf1"].append(res_t["weighted_f1"])
            resultados_test[k]["gf1"].append(res_t["graded_f1"])
            resultados_test[k]["auc"].append(res_t["auc"])
            resultados_test[k]["kappa"].append(res_t["kappa"])
            resultados_test[k]["qwk"].append(res_t["qwk"])
            resultados_test[k]["msgs"].append(res_t["media_msgs"])

            probs_test_ensemble[k].append(res_t["y_prob"])
            if k is None:
                oof_true_list.append(va_labels)
                oof_prob_list.append(res_v["y_prob"])

        print(
            f"  > Fold {fold_idx} cerrado | Mejor Val AUC: {mejor_auc_val:.4f} | "
            f"Val F1: {mejor_f1_val:.4f} (GF1: {mejor_gf1_val:.4f}) | "
            f"Test F1: {resultados_test[None]['f1'][-1]:.4f} "
            f"(GF1: {resultados_test[None]['gf1'][-1]:.4f})"
        )

        del model, val_embs, test_embs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 1. Resumen Out-Of-Fold (validación real sobre hasta 425 usuarios)
    oof_true = np.concatenate(oof_true_list)
    oof_prob = np.vstack(oof_prob_list)
    oof_pred = np.argmax(oof_prob, axis=1)
    oof_f1 = f1_score(oof_true, oof_pred, average="macro", zero_division=0)
    oof_gf1 = calcular_metricas_graded(oof_true, oof_pred)["graded_f1"]
    oof_auc = roc_auc_score(oof_true, oof_prob, multi_class="ovr", average="macro")
    oof_qwk = cohen_kappa_score(oof_true, oof_pred, weights="quadratic")

    print("\n" + "=" * 100)
    print(
        f"VALIDACIÓN OUT-OF-FOLD ({len(oof_true)} usuarios, k=Todos): "
        f"Macro F1={oof_f1:.4f} | Graded F1={oof_gf1:.4f} | AUC={oof_auc:.4f} | QWK={oof_qwk:.4f}"
    )
    print("=" * 100)

    # 2. Tabla Media ± StDev entre folds
    print("\n" + "=" * 118)
    print(f"RESULTADOS MIL + LoRA + MULTI-TASK (Media ± StDev sobre {n_ejecutar} folds)")
    print("=" * 118)
    print(
        f"{'Ventana':>8} | {'Msgs':>5} | {'Val Macro F1':>15} | {'Test Macro F1':>15} | "
        f"{'Test Graded F1':>15} | {'Test AUC':>15} | {'Test Kappa':>15}"
    )
    print("-" * 118)

    for k in ventanas_k:
        etiqueta_k = "Todos" if k is None else str(k)
        msgs = np.mean(resultados_test[k]["msgs"])
        v_m, v_s = np.mean(resultados_val[k]["f1"]), np.std(resultados_val[k]["f1"])
        t_m, t_s = np.mean(resultados_test[k]["f1"]), np.std(resultados_test[k]["f1"])
        g_m, g_s = np.mean(resultados_test[k]["gf1"]), np.std(resultados_test[k]["gf1"])
        a_m, a_s = np.mean(resultados_test[k]["auc"]), np.std(resultados_test[k]["auc"])
        k_m, k_s = np.mean(resultados_test[k]["kappa"]), np.std(resultados_test[k]["kappa"])

        print(
            f"{etiqueta_k:>8} | {msgs:5.1f} | {v_m:.4f} ± {v_s:.4f} | {t_m:.4f} ± {t_s:.4f} | "
            f"{g_m:.4f} ± {g_s:.4f} | {a_m:.4f} ± {a_s:.4f} | {k_m:.4f} ± {k_s:.4f}"
        )

    # 3. Tabla Ensemble (promedio de los N folds sobre el Test congelado)
    if n_ejecutar > 1:
        print("\n" + "=" * 105)
        print(f"ENSEMBLE {n_ejecutar}-FOLD CV ({len(train_dev_users)} usuarios vistos) - TEST CONGELADO")
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

    # 4. Reporte detallado y diagnóstico final (Test, k=Todos)
    prob_final = np.mean(probs_test_ensemble[None], axis=0)
    pred_final = np.argmax(prob_final, axis=1)
    titulo_rep = f"Ensemble {n_ejecutar}-Fold" if n_ejecutar > 1 else "Fold 1"
    print(f"\nReporte de clasificación Test ({titulo_rep}, k=Todos)")
    print(classification_report(test_labels, pred_final, target_names=clases, zero_division=0))
    imprimir_diagnostico_test(test_labels, pred_final, prob_final, clases, titulo=f"{titulo_rep}, k=Todos")


if __name__ == "__main__":
    main()