from contextlib import nullcontext
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
    - FP_ord: suma de distancias en sobreestimaciones (y_pred > y_true)
    - FN_ord: suma de distancias en subestimaciones (y_pred < y_true)
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
    """
    Soft Ordinal Targets (SORD): mantiene ~92% de masa en extremos y ~86% en centro,
    cediendo ~7-8% solo a vecinos inmediatos y 0% a clases lejanas.
    """
    idx = torch.arange(num_classes, dtype=torch.float32, device=device)
    dist_sq = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
    return torch.softmax(-dist_sq / (2.0 * sigma ** 2), dim=1)


class UserTextDataset(Dataset):
    """Dataset de secuencias de mensajes crudos por usuario."""

    def __init__(self, secuencias_texto, etiquetas, max_msgs=24, training=True):
        self.secuencias_texto = secuencias_texto
        self.etiquetas = etiquetas
        self.max_msgs = max_msgs
        self.training = training

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, idx):
        mensajes = self.secuencias_texto[idx]
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


class HierarchicalMentalRoBERTa(nn.Module):
    """
    Vectorizador MentalRoBERTa (últimas capas descongeladas) + Token Attention Pooling
    + Proyección + BiLSTM + Fusión (Atención aditiva + Max-Pooling por mensaje)
    + Multi-Sample Dropout.
    """

    def __init__(
        self,
        model_name="mental/mental-roberta-base",
        unfreeze_layers=2,
        proj_dim=64,
        hidden_dim=32,
        num_classes=5,
        dropout=0.4,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self._configurar_capas_entrenables(unfreeze_layers)

        emb_dim = self.encoder.config.hidden_size
        # Inicializado a 0 -> en época 1 equivale a mean_pool exacto y se especializa sin desestabilizar
        self.token_attn = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.token_attn.weight)
        nn.init.zeros_(self.token_attn.bias)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def _configurar_capas_entrenables(self, unfreeze_layers):
        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_layers > 0:
            for capa in self.encoder.encoder.layer[-unfreeze_layers:]:
                for p in capa.parameters():
                    p.requires_grad = True

    def encode_messages(self, input_ids, attention_mask, use_bf16=True):
        """Vectoriza mensajes con atención aprendida sobre tokens reales (arranca como mean-pool)."""
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

    def forward_from_embeddings(self, x_pad, longitudes):
        """Clasifica secuencias ya vectorizadas usando Multi-Sample Dropout en Train."""
        x = self.proj(x_pad)

        packed = pack_padded_sequence(
            x, longitudes.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < longitudes.unsqueeze(1)

        # 1. Contexto por Atención aditiva entre mensajes
        scores = self.attn_score(torch.tanh(self.attn_proj(lstm_out))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context_attn = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        # 2. Contexto por Max-Pooling sobre mensajes
        lstm_masked = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        context_max = lstm_masked.max(dim=1).values

        context = torch.cat([context_attn, context_max], dim=1)

        # Multi-Sample Dropout (5 máscaras en paralelo, 0 parámetros extra)
        if self.training:
            logits = torch.stack(
                [self.classifier(self.head_dropout(context)) for _ in range(5)], dim=0
            ).mean(dim=0)
        else:
            logits = self.classifier(context)

        return logits, attn_weights

    def forward(self, input_ids, attention_mask, longitudes, use_bf16=True):
        msg_embs = self.encode_messages(input_ids, attention_mask, use_bf16=use_bf16)
        splits = torch.split(msg_embs, longitudes.tolist(), dim=0)
        x_pad = pad_sequence(splits, batch_first=True, padding_value=0.0)
        return self.forward_from_embeddings(x_pad, longitudes)


@torch.no_grad()
def extraer_secuencias_con_encoder(
    model, secuencias_texto, tokenizer, device, max_length=256, msg_batch_size=64, use_bf16=True
):
    """Vectoriza una sola vez todos los mensajes de Dev/Test con el encoder actual."""
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
    """Evalúa la cabeza BiLSTM + Attention truncando a los primeros `max_mensajes` (RQ2)."""
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


def entrenar_jerarquico(
    train_texts,
    train_labels,
    dev_texts,
    dev_labels,
    clases,
    tokenizer,
    device,
    model_name="mental/mental-roberta-base",
    unfreeze_layers=2,
    proj_dim=64,
    hidden_dim=32,
    dropout=0.4,
    max_length=256,
    max_msgs_train=24,
    batch_size=2,
    accum_steps=8,
    lr_encoder=3e-5,
    lr_head=1e-3,
    weight_decay=2e-2,
    weight_power=1.0,
    sord_sigma=0.45,
    epochs=25,
    patience=6,
    seed=42,
):
    """Entrena MentalRoBERTa (parcial) + Token Attn + BiLSTM + Balanced Softmax + SORD."""
    fijar_semilla(seed)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model = HierarchicalMentalRoBERTa(
        model_name=model_name,
        unfreeze_layers=unfreeze_layers,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        num_classes=len(clases),
        dropout=dropout,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Parámetros entrenables: {train_p:,} / {total_p:,} "
        f"({100 * train_p / total_p:.1f}%) | bfloat16={use_bf16} | tau={weight_power} | sord_sigma={sord_sigma}"
    )

    train_loader = DataLoader(
        UserTextDataset(train_texts, train_labels, max_msgs=max_msgs_train, training=True),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CollateUserTexts(tokenizer, max_length=max_length),
    )

    # Balanced Softmax + Soft Ordinal Targets (SORD)
    conteos = np.bincount(train_labels, minlength=len(clases)).astype(np.float32)
    priors = conteos / conteos.sum()
    log_priors = torch.tensor(np.log(np.clip(priors, 1e-12, 1.0)), dtype=torch.float32, device=device)
    tau = weight_power
    sord_matrix = crear_targets_sord(len(clases), sigma=sord_sigma, device=device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder, "weight_decay": 0.01},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
    )

    steps_por_epoca = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_por_epoca * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    mejor_score = -1.0
    mejor_auc_real = 0.0
    mejor_f1_dev = 0.0
    mejor_gf1_dev = 0.0
    mejor_estado = None
    sin_mejora = 0

    for epoca in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acum = 0.0

        for step, (input_ids, attention_mask, longitudes, y) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            longitudes = longitudes.to(device)
            y = y.to(device)

            logits, _ = model(input_ids, attention_mask, longitudes, use_bf16=use_bf16)
            log_probs = nn.functional.log_softmax(logits + tau * log_priors, dim=1)
            y_soft = sord_matrix[y]
            loss = -(y_soft * log_probs).sum(dim=1).mean() / accum_steps
            loss.backward()
            loss_acum += loss.item() * accum_steps

            if step % accum_steps == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        dev_embs = extraer_secuencias_con_encoder(
            model, dev_texts, tokenizer, device, max_length=max_length, use_bf16=use_bf16
        )
        res_dev = evaluar_secuencias_emb(model, dev_embs, dev_labels, device, max_mensajes=None)
        score_actual = (
            0.5 * res_dev["auc"] + 0.5 * res_dev["macro_f1"]
            if not np.isnan(res_dev["auc"])
            else res_dev["macro_f1"]
        )

        print(
            f"    Época {epoca:02d}/{epochs} | Loss: {loss_acum / len(train_loader):.4f} "
            f"| Dev AUC: {res_dev['auc']:.4f} | Dev F1: {res_dev['macro_f1']:.4f} "
            f"| Dev Graded F1: {res_dev['graded_f1']:.4f}"
        )

        if score_actual > mejor_score:
            mejor_score = score_actual
            mejor_auc_real = res_dev["auc"]
            mejor_f1_dev = res_dev["macro_f1"]
            mejor_gf1_dev = res_dev["graded_f1"]
            mejor_estado = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= patience:
                print(f"    Early stopping en época {epoca} (sin mejora en {patience} épocas).")
                break

    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)
        del mejor_estado

    return model, mejor_f1_dev, mejor_gf1_dev, mejor_auc_real