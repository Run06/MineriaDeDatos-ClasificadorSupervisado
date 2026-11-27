from contextlib import nullcontext
from itertools import cycle
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


ORDEN_CLINICO = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]


def fijar_semilla(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construir_textos_por_usuario(df_mensajes, col_post, col_label, label_to_idx):
    """Agrupa los textos crudos en una lista cronológica por usuario."""
    secuencias_texto, etiquetas, user_ids = [], [], []
    for uid, grupo in df_mensajes.groupby("user_id", sort=False):
        secuencias_texto.append(grupo[col_post].astype(str).tolist())
        etiquetas.append(label_to_idx[grupo[col_label].iloc[0]])
        user_ids.append(uid)
    return secuencias_texto, np.array(etiquetas, dtype=np.int64), user_ids


def calcular_metricas_graded(y_true, y_pred):
    """
    Calcula Graded Precision, Graded Recall y Graded F1 sobre etiquetas en orden clínico (0..4).
    - TP: aciertos exactos (y_pred == y_true)
    - FP_ord: suma de distancias en sobreestimaciones (y_pred mayor que y_true)
    - FN_ord: suma de distancias en subestimaciones (y_pred menor que y_true)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum(y_true == y_pred))
    fp_ord = int(np.sum(np.maximum(0, y_pred - y_true)))
    fn_ord = int(np.sum(np.maximum(0, y_true - y_pred)))

    gp = tp / (tp + fp_ord) if (tp + fp_ord) > 0 else 0.0
    gr = tp / (tp + fn_ord) if (tp + fn_ord) > 0 else 0.0
    gf1 = (2.0 * gp * gr / (gp + gr)) if (gp + gr) > 0 else 0.0
    return {"graded_p": gp, "graded_r": gr, "graded_f1": gf1}


def crear_targets_sord(num_classes, sigma=0.45, device="cpu"):
    """Soft Ordinal Targets (SORD): reparte masa suave solo a clases clínicas contiguas."""
    idx = torch.arange(num_classes, dtype=torch.float32, device=device)
    dist_sq = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
    return torch.softmax(-dist_sq / (2.0 * sigma ** 2), dim=1)


class LoRALinear(nn.Module):
    """Adaptador LoRA ligero en PyTorch puro sobre una capa nn.Linear congelada."""

    def __init__(self, base_layer, r=8, alpha=16.0, dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


class MessageDataset(Dataset):
    """Dataset de mensajes individuales limpios (semilla.csv) para Multi-Task."""

    def __init__(self, textos, etiquetas):
        self.textos = list(textos)
        self.etiquetas = list(etiquetas)

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, idx):
        return str(self.textos[idx]), int(self.etiquetas[idx])


class CollateMessages:
    """Tokeniza un mini-lote de mensajes individuales."""

    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        textos, etiquetas = zip(*batch)
        enc = self.tokenizer(
            list(textos),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(etiquetas, dtype=torch.long),
        )


class UserTextDataset(Dataset):
    """Dataset de secuencias por usuario con Message Dropout leve en entrenamiento."""

    def __init__(self, secuencias_texto, etiquetas, max_msgs=32, msg_dropout=0.10, training=True):
        self.secuencias_texto = secuencias_texto
        self.etiquetas = etiquetas
        self.max_msgs = max_msgs
        self.msg_dropout = msg_dropout
        self.training = training

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, idx):
        mensajes = list(self.secuencias_texto[idx])

        # Message Dropout leve solo en historiales largos para regularizar sin borrar posts únicos
        if self.training and self.msg_dropout > 0 and len(mensajes) >= 8:
            filtrados = [m for m in mensajes if random.random() >= self.msg_dropout]
            if len(filtrados) >= 4:
                mensajes = filtrados

        if self.max_msgs is not None and len(mensajes) > self.max_msgs:
            if self.training:
                indices = sorted(random.sample(range(len(mensajes)), self.max_msgs))
                mensajes = [mensajes[i] for i in indices]
            else:
                mensajes = mensajes[: self.max_msgs]

        return mensajes, int(self.etiquetas[idx])


class CollateUserTexts:
    """Aplana los mensajes de los usuarios del batch y los tokeniza de golpe."""

    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        listas_mensajes, etiquetas = zip(*batch)
        longitudes = torch.tensor([len(m) for m in listas_mensajes], dtype=torch.long)
        mensajes_planos = [msg for lista in listas_mensajes for msg in lista]

        enc = self.tokenizer(
            mensajes_planos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        etiquetas = torch.tensor(etiquetas, dtype=torch.long)
        return enc["input_ids"], enc["attention_mask"], longitudes, etiquetas


class HierarchicalMentalRoBERTaMIL(nn.Module):
    """
    MentalRoBERTa con LoRA (últimas capas) + Token Attention Pooling
    + Proyección + BiLSTM residual + Clasificador a nivel de mensaje (compartido con semillas)
    + Agregación MIL (LogSumExp por clase + Media de logits).
    """

    def __init__(
        self,
        model_name="mental/mental-roberta-base",
        lora_layers=6,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        proj_dim=64,
        hidden_dim=32,
        num_classes=5,
        dropout=0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_name)
        self._aplicar_lora(lora_layers, lora_r, lora_alpha, lora_dropout)

        emb_dim = self.encoder.config.hidden_size

        # Atención sobre tokens inicializada a 0 (arranca exactamente como mean_pool)
        self.token_attn = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.token_attn.weight)
        nn.init.zeros_(self.token_attn.bias)

        # proj_dim == hidden_dim * 2 para conexión residual limpia con la BiLSTM
        seq_dim = hidden_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, seq_dim),
            nn.LayerNorm(seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.seq_norm = nn.LayerNorm(seq_dim)
        self.head_dropout = nn.Dropout(dropout)

        # Clasificador a nivel de mensaje (compartido entre secuencias de usuario y semillas)
        self.msg_classifier = nn.Linear(seq_dim, num_classes)

        # Temperatura aprendible por clase para LogSumExp MIL (beta inicial ~3.58)
        self.raw_beta = nn.Parameter(torch.full((1, 1, num_classes), 2.5))

        # Fusión ligera (55 params): combina evidencia pico (LogSumExp) y tono medio (Mean)
        self.user_combiner = nn.Linear(num_classes * 2, num_classes)
        with torch.no_grad():
            self.user_combiner.weight.zero_()
            self.user_combiner.bias.zero_()
            eye = torch.eye(num_classes)
            self.user_combiner.weight[:, :num_classes] = 0.7 * eye
            self.user_combiner.weight[:, num_classes:] = 0.3 * eye

    def _aplicar_lora(self, lora_layers, r, alpha, dropout):
        for p in self.encoder.parameters():
            p.requires_grad = False

        if lora_layers > 0:
            for capa in self.encoder.encoder.layer[-lora_layers:]:
                capa.attention.self.query = LoRALinear(
                    capa.attention.self.query, r=r, alpha=alpha, dropout=dropout
                )
                capa.attention.self.value = LoRALinear(
                    capa.attention.self.value, r=r, alpha=alpha, dropout=dropout
                )

    def encode_messages(self, input_ids, attention_mask, use_bf16=True):
        """Vectoriza mensajes con LoRA-MentalRoBERTa y atención sobre tokens reales."""
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and input_ids.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state

        hidden_f32 = hidden.float()
        scores = self.token_attn(hidden_f32).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_f32 * weights).sum(dim=1)

    def _clasificar_representaciones_msg(self, h):
        """Aplica el clasificador por mensaje con Multi-Sample Dropout en entrenamiento."""
        if self.training:
            return torch.stack(
                [self.msg_classifier(self.head_dropout(h)) for _ in range(5)], dim=0
            ).mean(dim=0)
        return self.msg_classifier(h)

    def forward_messages(self, input_ids, attention_mask, use_bf16=True):
        """Rama Multi-Task a nivel de mensaje individual (para semilla.csv)."""
        msg_embs = self.encode_messages(input_ids, attention_mask, use_bf16=use_bf16)
        x = self.proj(msg_embs).unsqueeze(1)  # (B, 1, seq_dim)
        lstm_out, _ = self.bilstm(x)
        h = self.seq_norm(lstm_out + x).squeeze(1)
        return self._clasificar_representaciones_msg(h)

    def forward_from_embeddings(self, x_pad, longitudes):
        """Clasifica secuencias ya vectorizadas mediante MIL LogSumExp + Media."""
        x = self.proj(x_pad)

        packed = pack_padded_sequence(
            x, longitudes.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        h = self.seq_norm(lstm_out + x)

        # Logits de riesgo por cada mensaje individual del usuario: (B, T, 5)
        msg_logits = self._clasificar_representaciones_msg(h)

        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < longitudes.unsqueeze(1)
        long_f = longitudes.float().clamp(min=1.0).view(-1, 1)

        # 1. MIL Softmax-Weighted Pooling por clase (invariante a la longitud del historial)
        beta = nn.functional.softplus(self.raw_beta) + 1.0  # (1, 1, 5)
        scores_mil = (msg_logits * beta).masked_fill(~mask.unsqueeze(-1), -1e9)
        weights_mil = torch.softmax(scores_mil, dim=1)
        mil_peak = (weights_mil * msg_logits).sum(dim=1)

        # 2. Media enmascarada de logits (captura tono global del historial)
        mask_f = mask.unsqueeze(-1).float()
        mil_mean = (msg_logits * mask_f).sum(dim=1) / long_f

        user_logits = self.user_combiner(torch.cat([mil_peak, mil_mean], dim=1))
        return user_logits, msg_logits

    def forward(self, input_ids, attention_mask, longitudes, use_bf16=True):
        msg_embs = self.encode_messages(input_ids, attention_mask, use_bf16=use_bf16)
        splits = torch.split(msg_embs, longitudes.tolist(), dim=0)
        x_pad = pad_sequence(splits, batch_first=True, padding_value=0.0)
        return self.forward_from_embeddings(x_pad, longitudes)


@torch.no_grad()
def extraer_secuencias_con_encoder(
    model, secuencias_texto, tokenizer, device, max_length=256, msg_batch_size=64, use_bf16=True
):
    """Vectoriza una sola vez todos los mensajes de Val/Test con el encoder actual."""
    model.eval()
    longitudes = [len(s) for s in secuencias_texto]
    todos_mensajes = [msg for s in secuencias_texto for msg in s]

    bloques = []
    for i in range(0, len(todos_mensajes), msg_batch_size):
        lote = todos_mensajes[i : i + msg_batch_size]
        enc = tokenizer(
            lote,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        embs = model.encode_messages(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            use_bf16=use_bf16,
        )
        bloques.append(embs.cpu())

    todos_embs = torch.cat(bloques, dim=0)
    return list(torch.split(todos_embs, longitudes, dim=0))


@torch.no_grad()
def evaluar_secuencias_emb(
    model, secuencias_emb, etiquetas, device, max_mensajes=None, batch_size=32
):
    """Evalúa la cabeza MIL truncando a los primeros `max_mensajes` (RQ2)."""
    model.eval()
    sec_eval = (
        [s[:max_mensajes] for s in secuencias_emb]
        if max_mensajes is not None
        else secuencias_emb
    )

    all_probs = []
    for i in range(0, len(sec_eval), batch_size):
        lote = sec_eval[i : i + batch_size]
        longitudes = torch.tensor([len(s) for s in lote], dtype=torch.long, device=device)
        x_pad = pad_sequence(lote, batch_first=True, padding_value=0.0).to(device)
        logits, _ = model.forward_from_embeddings(x_pad, longitudes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    y_true = np.array(etiquetas, dtype=np.int64)
    y_prob = np.vstack(all_probs)
    y_pred = np.argmax(y_prob, axis=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    graded = calcular_metricas_graded(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    media_msgs = float(np.mean([len(s) for s in sec_eval]))
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "graded_p": graded["graded_p"],
        "graded_r": graded["graded_r"],
        "graded_f1": graded["graded_f1"],
        "kappa": kappa,
        "qwk": qwk,
        "auc": auc,
        "media_msgs": media_msgs,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def entrenar_jerarquico_mil(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    clases,
    tokenizer,
    device,
    seed_texts=None,
    seed_labels=None,
    seed_alpha=0.35,
    seed_batch_size=16,
    model_name="mental/mental-roberta-base",
    lora_layers=6,
    lora_r=8,
    lora_alpha=16.0,
    hidden_dim=32,
    dropout=0.4,
    max_length=256,
    max_msgs_train=32,
    msg_dropout=0.10,
    batch_size=2,
    accum_steps=8,
    lr_lora=3e-4,
    lr_head=1e-3,
    weight_decay=2e-2,
    tau=1.0,
    sord_sigma=0.45,
    epochs=25,
    patience=6,
    seed=42,
):
    """Entrena MentalRoBERTa-LoRA + MIL LogSumExp + Multi-Task con semillas limpias."""
    fijar_semilla(seed)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    num_classes = len(clases)

    model = HierarchicalMentalRoBERTaMIL(
        model_name=model_name,
        lora_layers=lora_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    usar_semillas = seed_texts is not None and len(seed_texts) > 0 and seed_alpha > 0
    print(
        f"Parámetros entrenables: {train_p:,} / {total_p:,} ({100 * train_p / total_p:.2f}%) | "
        f"LoRA(capas={lora_layers}, r={lora_r}) | Multi-Task semillas="
        f"{len(seed_texts) if usar_semillas else 0} (alpha={seed_alpha if usar_semillas else 0.0})"
    )

    train_loader = DataLoader(
        UserTextDataset(
            train_texts,
            train_labels,
            max_msgs=max_msgs_train,
            msg_dropout=msg_dropout,
            training=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CollateUserTexts(tokenizer, max_length=max_length),
    )

    # Balanced Softmax + SORD para usuarios
    conteos = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    priors = conteos / conteos.sum()
    log_priors = torch.tensor(np.log(np.clip(priors, 1e-12, 1.0)), dtype=torch.float32, device=device)
    sord_user = crear_targets_sord(num_classes, sigma=sord_sigma, device=device)

    # Loader auxiliar de mensajes semilla para entrenamiento Multi-Task
    seed_iter = None
    log_seed_priors = None
    sord_seed = None
    if usar_semillas:
        seed_loader = DataLoader(
            MessageDataset(seed_texts, seed_labels),
            batch_size=seed_batch_size,
            shuffle=True,
            collate_fn=CollateMessages(tokenizer, max_length=max_length),
        )
        seed_iter = cycle(seed_loader)
        conteos_s = np.bincount(seed_labels, minlength=num_classes).astype(np.float32)
        priors_s = conteos_s / conteos_s.sum()
        log_seed_priors = torch.tensor(
            np.log(np.clip(priors_s, 1e-12, 1.0)), dtype=torch.float32, device=device
        )
        sord_seed = crear_targets_sord(num_classes, sigma=0.40, device=device)

    lora_params = [
        p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad
    ]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": 0.01})
    param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(param_groups)

    steps_por_epoca = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_por_epoca * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    mejor_score = -1.0
    mejor_auc_real = 0.0
    mejor_f1_val = 0.0
    mejor_gf1_val = 0.0
    mejor_estado = None
    sin_mejora = 0

    for epoca in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_user_acum = 0.0
        loss_seed_acum = 0.0
        n_seed_steps = 0
        alpha_ep = seed_alpha * (1.0 - 0.75 * (epoca - 1) / max(1, epochs - 1))

        for step, (input_ids, attention_mask, longitudes, y) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            longitudes = longitudes.to(device)
            y = y.to(device)

            logits, _ = model(input_ids, attention_mask, longitudes, use_bf16=use_bf16)
            log_probs = nn.functional.log_softmax(logits + tau * log_priors, dim=1)
            loss_u = -(sord_user[y] * log_probs).sum(dim=1).mean() / accum_steps
            loss_u.backward()
            loss_user_acum += loss_u.item() * accum_steps

            if step % accum_steps == 0 or step == len(train_loader):
                # Paso auxiliar Multi-Task con 1 mini-lote de semillas limpias por actualización
                if usar_semillas:
                    s_ids, s_mask, s_y = next(seed_iter)
                    s_ids = s_ids.to(device)
                    s_mask = s_mask.to(device)
                    s_y = s_y.to(device)

                    s_logits = model.forward_messages(s_ids, s_mask, use_bf16=use_bf16)
                    s_log_probs = nn.functional.log_softmax(s_logits + tau * log_seed_priors, dim=1)
                    loss_s = -(sord_seed[s_y] * s_log_probs).sum(dim=1).mean()
                    (alpha_ep * loss_s).backward()
                    loss_seed_acum += loss_s.item()
                    n_seed_steps += 1

                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_embs = extraer_secuencias_con_encoder(
            model, val_texts, tokenizer, device, max_length=max_length, use_bf16=use_bf16
        )
        res_val = evaluar_secuencias_emb(model, val_embs, val_labels, device, max_mensajes=None)
        score_actual = (
            0.5 * res_val["auc"] + 0.5 * res_val["macro_f1"]
            if not np.isnan(res_val["auc"])
            else res_val["macro_f1"]
        )

        seed_msg = (
            f" | LossS: {loss_seed_acum / max(1, n_seed_steps):.4f}" if usar_semillas else ""
        )
        print(
            f"    Época {epoca:02d}/{epochs} | LossU: {loss_user_acum / len(train_loader):.4f}"
            f"{seed_msg} | Val AUC: {res_val['auc']:.4f} | Val F1: {res_val['macro_f1']:.4f} "
            f"| Val GF1: {res_val['graded_f1']:.4f}"
        )

        if score_actual > mejor_score:
            mejor_score = score_actual
            mejor_auc_real = res_val["auc"]
            mejor_f1_val = res_val["macro_f1"]
            mejor_gf1_val = res_val["graded_f1"]
            mejor_estado = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if k in trainable_keys
            }
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= patience:
                print(f"    Early stopping en época {epoca} (sin mejora en {patience} épocas).")
                break

    if mejor_estado is not None:
        model.load_state_dict(mejor_estado, strict=False)
        del mejor_estado

    return model, mejor_f1_val, mejor_gf1_val, mejor_auc_real