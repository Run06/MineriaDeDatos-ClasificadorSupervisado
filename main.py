import argparse
from datetime import datetime
import time
import numpy as np
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, roc_auc_score
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
    guardar_modelo_mil,
)
from utilidades import (
    anexar_fila_csv,
    cargar_csv,
    cargar_parametros,
    construir_fila_metricas,
    filtrar_semillas_sin_fuga,
    formatear_duracion,
    imprimir_diagnostico_test,
    imprimir_resumen_recursos,
    imprimir_tabla_individuales,
    obtener_consumo_recursos,
    obtener_mejor_f1_en_disco,
    reiniciar_estadisticas_memoria,
)


def parse_args():
    """
        Define y procesa los argumentos pasados por línea de comandos.
        Precondición:
            Los argumentos recibidos en `sys.argv` deben respetar los tipos definidos
            (`--n_folds` y `--n_repeats`, si se indican, deben ser enteros válidos).
        Entradas:
            Ninguna por parámetro (lee directamente `sys.argv`).
        Salidas:
            argparse.Namespace: Objeto con los campos `params` (str), `file` (str | None),
            `seed_file` (str | None), `n_folds` (int | None) y `n_repeats` (int | None).
        Postcondición:
            Devuelve el `Namespace` parseado. Si los argumentos son inválidos o se
            invoca `-h`/`--help`, el proceso finaliza mostrando la ayuda de uso.
    """
    parser = argparse.ArgumentParser(
        description="MentalRoBERTa-LoRA + MIL + Multi-Task + CV (configurado desde parametros.json)"
    )
    parser.add_argument(
        "-p", "--params", default="parametros.json", help="Ruta al archivo JSON de hiperparámetros"
    )
    parser.add_argument("-f", "--file", default=None, help="Sobreescribe datos.file del JSON")
    parser.add_argument("-s", "--seed_file", default=None, help="Sobreescribe datos.seed_file del JSON")
    parser.add_argument("--n_folds", type=int, default=None, help="Sobreescribe validacion.n_folds")
    parser.add_argument("--n_repeats", type=int, default=None, help="Sobreescribe validacion.n_repeats")
    return parser.parse_args()


def main():
    """
        Orquesta el flujo completo de carga de datos, validación cruzada estratificada,
        evaluación Out-Of-Fold (OOF), evaluación del Ensemble en Test y guardado de resultados.
        Precondición:
            - Debe existir el archivo JSON indicado en `--params` con las secciones
              `datos`, `modelo`, `entrenamiento`, `validacion` y `salidas`.
            - Debe existir el CSV de usuarios (`datos.file`) con las columnas `target_col`
              y `label_col` y suficientes muestras por clase para estratificar el split
              de Test y los `n_folds`.
            - Si `seed_alpha > 0` y `datos.seed_file` está definido, dicho CSV debe existir.
            - Debe haber acceso al tokenizer y pesos preentrenados de `modelo.model_name`.
        Entradas:
            Ninguna por parámetro (lee la configuración desde línea de comandos, el JSON
            de parámetros y los archivos CSV configurados).
        Salidas:
            None.
        Postcondición:
            - Se entrenan `n_repeats * n_folds` modelos y se libera la memoria GPU/CPU al
              finalizar cada fold.
            - Se anexan de forma incremental las filas de tipo `modelo_fold`, `oof` y
              `ensemble` (por ventana k) en `salidas.metrics_csv`.
            - Se guarda o sobrescribe `salidas.best_model_path` si algún fold supera el
              mejor `Test Macro F1` registrado.
            - Se imprimen por salida estándar los reportes de clasificación, el diagnóstico
              detallado en Test y el resumen de tiempo y consumo de RAM/VRAM.
    """
    t_inicio_run = time.perf_counter()
    args = parse_args()
    params = cargar_parametros(args.params)

    cfg_dat = params["datos"]
    cfg_mod = params["modelo"]
    cfg_ent = params["entrenamiento"]
    cfg_val = params["validacion"]
    cfg_out = params["salidas"]

    if args.file is not None:
        cfg_dat["file"] = args.file
    if args.seed_file is not None:
        cfg_dat["seed_file"] = args.seed_file
    if args.n_folds is not None:
        cfg_val["n_folds"] = args.n_folds
    if args.n_repeats is not None:
        cfg_val["n_repeats"] = args.n_repeats

    device = torch.device(DEVICE)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_modelos = cfg_val["n_repeats"] * cfg_val["n_folds"]

    nombre_hw = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"Run ID: {run_id} | Parámetros: '{args.params}' | Dispositivo: {device} ({nombre_hw})\n"
        f"Encoder: {cfg_mod['model_name']} | LoRA(capas={cfg_mod['lora_layers']}, r={cfg_mod['lora_r']}) "
        f"| max_len={cfg_ent['max_length']} | tau={cfg_ent['tau']} | sord={cfg_ent['sord_sigma']}"
    )

    raw_data = cargar_csv(cfg_dat["file"])
    col_post = cfg_dat["target_col"]
    col_label = cfg_dat["label_col"]
    print(f"Usuarios totales cargados: {len(raw_data)}")

    # 1. Test congelado estratificado (mismo split que el baseline)
    train_dev_users, test_users = train_test_split(
        raw_data,
        test_size=cfg_dat["test_size"],
        random_state=cfg_dat["split_seed"],
        stratify=raw_data[col_label],
    )
    test_msgs = dividir_mensajes(test_users, col_post=col_post)
    print(
        f"Partición principal -> Train+Dev (CV): {len(train_dev_users)} usuarios | "
        f"Test congelado: {len(test_users)} usuarios ({len(test_msgs)} mensajes)"
    )

    unicas = set(raw_data[col_label].unique())
    clases = [c for c in ORDEN_CLINICO if c in unicas]
    clases += [c for c in sorted(unicas) if c not in clases]
    label_to_idx = {c: i for i, c in enumerate(clases)}
    print(f"Orden clínico de clases: {clases}")

    test_texts, test_labels, _ = construir_textos_por_usuario(
        test_msgs, col_post, col_label, label_to_idx
    )
    test_msgs_norm = {str(t).strip().lower() for seq in test_texts for t in seq}

    # 2. Semillas manuales limpias (sin fuga hacia Test)
    seed_texts_base, seed_labels_base = [], np.array([], dtype=np.int64)
    if cfg_ent["seed_alpha"] > 0 and cfg_dat.get("seed_file"):
        df_seed = cargar_csv(cfg_dat["seed_file"]).dropna(subset=[col_post, col_label])
        df_seed = df_seed[df_seed[col_label].isin(label_to_idx)]
        raw_s_texts = df_seed[col_post].astype(str).tolist()
        raw_s_labels = np.array([label_to_idx[c] for c in df_seed[col_label]], dtype=np.int64)
        seed_texts_base, seed_labels_base = filtrar_semillas_sin_fuga(
            raw_s_texts, raw_s_labels, test_msgs_norm
        )
        print(
            f"Semillas Multi-Task cargadas: {len(seed_texts_base)} / {len(raw_s_texts)} "
            f"(tras filtrar coincidencias con Test)"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg_mod["model_name"])
    ventanas_k = tuple(cfg_val["ventanas_k"])

    resultados_val = {k: {"f1": [], "gf1": [], "auc": [], "msgs": []} for k in ventanas_k}
    resultados_test = {
        k: {"f1": [], "wf1": [], "gf1": [], "auc": [], "kappa": [], "qwk": [], "msgs": []}
        for k in ventanas_k
    }
    probs_test_ensemble = {k: [] for k in ventanas_k}
    oof_por_usuario = {}

    tiempos_folds = []
    vram_pico_global_asig = 0.0
    vram_pico_global_res = 0.0

    best_model_path = cfg_out["best_model_path"]
    metrics_csv = cfg_out["metrics_csv"]
    mejor_test_f1_guardado = (
        obtener_mejor_f1_en_disco(best_model_path)
        if cfg_out.get("solo_guardar_si_supera_disco", True)
        else -1.0
    )
    if mejor_test_f1_guardado >= 0:
        print(
            f"Modelo previo detectado en '{best_model_path}' "
            f"(Test Macro F1 = {mejor_test_f1_guardado:.4f}). Solo se sobrescribirá si se supera."
        )

    config_modelo_ckpt = {
        "model_name": cfg_mod["model_name"],
        "lora_layers": cfg_mod["lora_layers"],
        "lora_r": cfg_mod["lora_r"],
        "lora_alpha": cfg_mod["lora_alpha"],
        "lora_dropout": cfg_mod["lora_dropout"],
        "lora_include_out": cfg_mod["lora_include_out"],
        "hidden_dim": cfg_mod["hidden_dim"],
        "num_classes": len(clases),
        "dropout": cfg_mod["dropout"],
        "multi_sample_dropout_n": cfg_mod["multi_sample_dropout_n"],
    }

    print("\n" + "=" * 95)
    print(
        f"ENTRENAMIENTO MIL + LoRA ({cfg_val['n_repeats']}x{cfg_val['n_folds']} = "
        f"{total_modelos} modelos sobre {len(train_dev_users)} usuarios)"
    )
    print("=" * 95)

    modelo_num = 0
    for rep in range(1, cfg_val["n_repeats"] + 1):
        skf = StratifiedKFold(
            n_splits=cfg_val["n_folds"],
            shuffle=True,
            random_state=cfg_val["seed"] + (rep - 1) * 1000,
        )
        for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(train_dev_users, train_dev_users[col_label]), start=1
        ):
            modelo_num += 1
            print(
                f"\n--- Rep {rep}/{cfg_val['n_repeats']} | Fold {fold_idx}/{cfg_val['n_folds']} "
                f"(Modelo {modelo_num}/{total_modelos} | Train={len(tr_idx)} | Val={len(va_idx)}) ---"
            )
            fold_train_msgs = dividir_mensajes(train_dev_users.iloc[tr_idx], col_post=col_post)
            fold_val_msgs = dividir_mensajes(train_dev_users.iloc[va_idx], col_post=col_post)

            tr_texts, tr_labels, _ = construir_textos_por_usuario(
                fold_train_msgs, col_post, col_label, label_to_idx
            )
            va_texts, va_labels, va_uids = construir_textos_por_usuario(
                fold_val_msgs, col_post, col_label, label_to_idx
            )

            val_msgs_norm = {str(t).strip().lower() for seq in va_texts for t in seq}
            fold_seed_texts, fold_seed_labels = filtrar_semillas_sin_fuga(
                seed_texts_base, seed_labels_base, val_msgs_norm
            )

            # Reinicio de métricas de memoria y cronómetro del fold
            reiniciar_estadisticas_memoria(device)
            t_inicio_fold = time.perf_counter()

            model, mejor_f1_val, mejor_gf1_val, mejor_auc_val, mejor_epoca = entrenar_jerarquico_mil(
                train_texts=tr_texts,
                train_labels=tr_labels,
                val_texts=va_texts,
                val_labels=va_labels,
                clases=clases,
                tokenizer=tokenizer,
                device=device,
                seed_texts=fold_seed_texts,
                seed_labels=fold_seed_labels,
                seed_alpha=cfg_ent["seed_alpha"],
                seed_alpha_decay=cfg_ent["seed_alpha_decay"],
                sord_seed_sigma=cfg_ent["sord_seed_sigma"],
                seed_batch_size=cfg_ent["seed_batch_size"],
                model_name=cfg_mod["model_name"],
                lora_layers=cfg_mod["lora_layers"],
                lora_r=cfg_mod["lora_r"],
                lora_alpha=cfg_mod["lora_alpha"],
                lora_dropout=cfg_mod["lora_dropout"],
                lora_include_out=cfg_mod["lora_include_out"],
                grad_checkpointing=cfg_mod["grad_checkpointing"],
                hidden_dim=cfg_mod["hidden_dim"],
                dropout=cfg_mod["dropout"],
                multi_sample_dropout_n=cfg_mod["multi_sample_dropout_n"],
                max_length=cfg_ent["max_length"],
                max_msgs_train=cfg_ent["max_msgs_train"],
                msg_dropout=cfg_ent["msg_dropout"],
                batch_size=cfg_ent["batch_size"],
                accum_steps=cfg_ent["accum_steps"],
                lr_lora=cfg_ent["lr_lora"],
                lr_head=cfg_ent["lr_head"],
                lora_weight_decay=cfg_ent["lora_weight_decay"],
                head_weight_decay=cfg_ent["head_weight_decay"],
                warmup_ratio=cfg_ent["warmup_ratio"],
                tau=cfg_ent["tau"],
                sord_sigma=cfg_ent["sord_sigma"],
                early_stop_auc_weight=cfg_ent["early_stop_auc_weight"],
                use_top2_soup=cfg_ent["use_top2_soup"],
                epochs=cfg_ent["epochs"],
                patience=cfg_ent["patience"],
                seed=cfg_val["seed"] + rep * 1000 + fold_idx * 100,
            )

            tiempo_fold_seg = time.perf_counter() - t_inicio_fold
            recursos_fold = obtener_consumo_recursos(device)
            tiempos_folds.append(tiempo_fold_seg)
            vram_pico_global_asig = max(vram_pico_global_asig, recursos_fold["vram_pico_asignada_mb"])
            vram_pico_global_res = max(vram_pico_global_res, recursos_fold["vram_pico_reservada_mb"])

            val_embs = extraer_secuencias_con_encoder(
                model, va_texts, tokenizer, device, max_length=cfg_ent["max_length"], use_bf16=use_bf16
            )
            test_embs = extraer_secuencias_con_encoder(
                model, test_texts, tokenizer, device, max_length=cfg_ent["max_length"], use_bf16=use_bf16
            )

            res_t_todos = None
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
                    res_t_todos = res_t
                    for uid, y_u, p_u in zip(va_uids, va_labels, res_v["y_prob"]):
                        if uid not in oof_por_usuario:
                            oof_por_usuario[uid] = {"label": int(y_u), "probs": []}
                        oof_por_usuario[uid]["probs"].append(p_u)

            if res_t_todos is not None:
                fila_modelo = construir_fila_metricas(
                    run_id=run_id,
                    tipo="modelo_fold",
                    conjunto_eval="test",
                    rep=rep,
                    fold=fold_idx,
                    epoca_sel=mejor_epoca,
                    ventana="Todos",
                    clases=clases,
                    y_true=test_labels,
                    y_pred=res_t_todos["y_pred"],
                    y_prob=res_t_todos["y_prob"],
                    cfg_mod=cfg_mod,
                    cfg_ent=cfg_ent,
                    dev_fold_auc=mejor_auc_val,
                    dev_fold_f1=mejor_f1_val,
                    dev_fold_gf1=mejor_gf1_val,
                    tiempo_seg=tiempo_fold_seg,
                    recursos=recursos_fold,
                )
                anexar_fila_csv(metrics_csv, fila_modelo)

                if res_t_todos["macro_f1"] > mejor_test_f1_guardado:
                    mejor_test_f1_guardado = float(res_t_todos["macro_f1"])
                    guardar_modelo_mil(
                        model=model,
                        ruta=best_model_path,
                        config_modelo=config_modelo_ckpt,
                        clases=clases,
                        metricas={
                            "run_id": run_id,
                            "rep": rep,
                            "fold": fold_idx,
                            "epoca_sel": mejor_epoca,
                            "dev_fold_auc": mejor_auc_val,
                            "dev_fold_macro_f1": mejor_f1_val,
                            "test_macro_f1": res_t_todos["macro_f1"],
                            "test_weighted_f1": res_t_todos["weighted_f1"],
                            "test_graded_f1": res_t_todos["graded_f1"],
                            "test_auc": res_t_todos["auc"],
                            "test_qwk": res_t_todos["qwk"],
                            "tiempo_entrenamiento_seg": tiempo_fold_seg,
                            "ram_pico_mb": recursos_fold["ram_pico_mb"],
                            "vram_pico_asignada_mb": recursos_fold["vram_pico_asignada_mb"],
                        },
                        hiperparametros=params,
                    )
                    print(
                        f"  *** NUEVO MEJOR MODELO GUARDADO en '{best_model_path}' "
                        f"(Test Macro F1: {mejor_test_f1_guardado:.4f} | GF1: {res_t_todos['graded_f1']:.4f}) ***"
                    )

            vram_txt = (
                f" | VRAM pico: {recursos_fold['vram_pico_asignada_mb'] / 1024.0:.2f} GB"
                if device.type == "cuda"
                else ""
            )
            print(
                f"  > Modelo {modelo_num}/{total_modelos} cerrado | Dev AUC: {mejor_auc_val:.4f} | "
                f"Dev F1: {mejor_f1_val:.4f} (GF1: {mejor_gf1_val:.4f}) | "
                f"Test F1: {resultados_test[None]['f1'][-1]:.4f} "
                f"(GF1: {resultados_test[None]['gf1'][-1]:.4f}) | "
                f"Tiempo: {formatear_duracion(tiempo_fold_seg)} | "
                f"RAM pico: {recursos_fold['ram_pico_mb'] / 1024.0:.2f} GB{vram_txt}"
            )

            del model, val_embs, test_embs
            reiniciar_estadisticas_memoria(device)

    # Consolidación de recursos globales del run
    recursos_globales = obtener_consumo_recursos(device)
    recursos_globales["vram_pico_asignada_mb"] = vram_pico_global_asig
    recursos_globales["vram_pico_reservada_mb"] = vram_pico_global_res
    tiempo_ent_acumulado = float(np.sum(tiempos_folds))

    # 3. Validación Out-Of-Fold (425 usuarios -> conjunto_eval='dev_oof')
    oof_uids = list(oof_por_usuario.keys())
    oof_true = np.array([oof_por_usuario[u]["label"] for u in oof_uids], dtype=np.int64)
    oof_prob = np.vstack([np.mean(oof_por_usuario[u]["probs"], axis=0) for u in oof_uids])
    oof_pred = np.argmax(oof_prob, axis=1)

    oof_f1 = f1_score(oof_true, oof_pred, average="macro", zero_division=0)
    oof_gf1 = calcular_metricas_graded(oof_true, oof_pred)["graded_f1"]
    oof_auc = roc_auc_score(oof_true, oof_prob, multi_class="ovr", average="macro")
    oof_qwk = cohen_kappa_score(oof_true, oof_pred, weights="quadratic")

    anexar_fila_csv(
        metrics_csv,
        construir_fila_metricas(
            run_id=run_id,
            tipo="oof",
            conjunto_eval="dev_oof",
            rep=cfg_val["n_repeats"],
            fold="ALL",
            epoca_sel="-",
            ventana="Todos",
            clases=clases,
            y_true=oof_true,
            y_pred=oof_pred,
            y_prob=oof_prob,
            cfg_mod=cfg_mod,
            cfg_ent=cfg_ent,
            tiempo_seg=tiempo_ent_acumulado,
            recursos=recursos_globales,
        ),
    )

    print("\n" + "=" * 100)
    print(
        f"VALIDACIÓN OUT-OF-FOLD ({len(oof_true)} usuarios, {cfg_val['n_repeats']} reps): "
        f"Macro F1={oof_f1:.4f} | Graded F1={oof_gf1:.4f} | AUC={oof_auc:.4f} | QWK={oof_qwk:.4f}"
    )
    print("=" * 100)

    # 4. Tablas resumen (individuales y ensemble por ventana k)
    imprimir_tabla_individuales(ventanas_k, resultados_val, resultados_test, total_modelos)

    print("\n" + "=" * 105)
    print(f"ENSEMBLE {cfg_val['n_repeats']}x{cfg_val['n_folds']}-FOLD ({total_modelos} modelos) - TEST CONGELADO")
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

        anexar_fila_csv(
            metrics_csv,
            construir_fila_metricas(
                run_id=run_id,
                tipo="ensemble",
                conjunto_eval="test",
                rep=cfg_val["n_repeats"],
                fold="ALL",
                epoca_sel="-",
                ventana=etiqueta_k,
                clases=clases,
                y_true=test_labels,
                y_pred=pred_ens,
                y_prob=prob_ens,
                cfg_mod=cfg_mod,
                cfg_ent=cfg_ent,
                tiempo_seg=tiempo_ent_acumulado,
                recursos=recursos_globales,
            ),
        )

    # 5. Reporte detallado final (Test, k=Todos) y resumen computacional
    prob_final = np.mean(probs_test_ensemble[None], axis=0)
    pred_final = np.argmax(prob_final, axis=1)
    print(f"\nReporte de clasificación Test (Ensemble {total_modelos} modelos, k=Todos)")
    print(classification_report(test_labels, pred_final, target_names=clases, zero_division=0))
    imprimir_diagnostico_test(
        test_labels, pred_final, prob_final, clases, titulo=f"Ensemble {total_modelos}, k=Todos"
    )

    tiempo_total_run = time.perf_counter() - t_inicio_run
    imprimir_resumen_recursos(tiempos_folds, tiempo_total_run, recursos_globales, device)

    print(f"\nMétricas registradas en: '{metrics_csv}'")
    print(f"Mejor modelo individual en '{best_model_path}' (Test Macro F1 = {mejor_test_f1_guardado:.4f})")


if __name__ == "__main__":
    main()