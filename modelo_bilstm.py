import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, roc_auc_score

def fijar_semilla(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
class UserSequenceDataset(Dataset):
    """Agrupa embeddings de mensajes en una secuencia por usuario"""
    def __init__(self, secuencias, etiquetas):
        self.secuencias = [torch.tensor(np.array(s), dtype=torch.float32) for s in secuencias]
        self.etiquetas = torch.tensor(etiquetas, dtype=torch.long)
    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, idx):
        return self.secuencias[idx], self.etiquetas[idx]
def collate_secuencias(batch):
    """Aplica padding dinámico al batch y devuelve las longitudes reales"""
    secuencias, etiquetas = zip(*batch)
    longitudes = torch.tensor([len(s) for s in secuencias], dtype=torch.long)
    secuencias_pad = pad_sequence(secuencias, batch_first=True, padding_value=0.0)
    etiquetas = torch.stack(etiquetas)
    return secuencias_pad, longitudes, etiquetas

class BiLSTMAttention(nn.Module):
    """
        BiLSTM + Mecanismo de Atención aditiva sobre secuencias de mensajes de usuario.
        Aprende automáticamente qué mensajes del historial tienen señal de riesgo.
    """
    def __init__(self, input_dim, proj_dim=64,hidden_dim=32, num_classes=5, num_layers=1, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        #Capa de atención aditiva
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x, longitudes):
        x = self.proj(x)

        #Empaquetar para que LSTM ignore el padding
        packed = pack_padded_sequence(x, longitudes.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        #Máscara booleana, true en mensajes reales, false en padding
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < longitudes.unsqueeze(1)

        #Contexto por Atención
        scores = self.attn_score(torch.tanh(self.attn_proj(lstm_out))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context_attn = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        #Contexto por Max-Pooling enmascarado
        lstm_masked = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        context_max = lstm_masked.max(dim=1).values

        #Fusión: 64 + 64 = 128 dimensiones
        context = torch.cat([context_attn, context_max], dim=1)
        logits = self.classifier(self.head_dropout(context))
        return logits, attn_weights

def construir_secuencias_por_usuario(embeddings, df_mensajes, col_label, label_to_idx):
    """Convierte la matriz de embeddings de mensajes en listas de secuencias por usuario."""
    df_tmp = df_mensajes[["user_id", col_label]].copy()
    df_tmp["idx_fila"] = np.arange(len(df_tmp))

    secuencias, etiquetas, user_ids = [], [], []
    for uid, grupo in df_tmp.groupby("user_id",sort=False):
        idxs = grupo["idx_fila"].values
        secuencias.append(embeddings[idxs])
        etiquetas.append(label_to_idx[grupo[col_label].iloc[0]])
        user_ids.append(uid)
    return secuencias, np.array(etiquetas, dtype=np.int64), user_ids

def evaluar_modelo(model, secuencias, etiquetas, device, max_mensajes=None, batch_size=32):
    """Evalúa el modelo truncando cada usuario a sus primeros `max_mensajes` (RQ2)."""
    if max_mensajes is not None:
        sec_eval = [s[:max_mensajes] for s in secuencias]
    else:
        sec_eval = secuencias

    loader = DataLoader(
        UserSequenceDataset(sec_eval, etiquetas),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_secuencias,
    )

    model.eval()
    all_preds, all_probs, all_true = [], [], []

    with torch.no_grad():
        for x_pad, longitudes, y in loader:
            x_pad, longitudes = x_pad.to(device), longitudes.to(device)
            logits, _ = model(x_pad, longitudes)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_prob = np.vstack(all_probs)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    media_msgs = float(np.mean([len(s) for s in sec_eval]))
    return {
        "macro_f1": macro_f1,
        "kappa": kappa,
        "auc": auc,
        "media_msgs": media_msgs,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def entrenar_bilstm(
        train_seqs,
        train_labels,
        dev_seqs,
        dev_labels,
        input_dim,
        num_classes,
        device,
        hidden_dim=32,
        dropout=0.4,
        lr=5e-4,
        weight_decay=2e-2,
        batch_size=16,
        epochs=80,
        patience=15,
        seed=42
):
    """Entrena BiLSTM + Attention con pesos de clase y Early Stopping sobre Dev Macro F1."""
    fijar_semilla(seed)
    train_loader = DataLoader(
        UserSequenceDataset(train_seqs, train_labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_secuencias,
    )

    conteos = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    pesos = np.sqrt(len(train_labels) / (num_classes * np.maximum(conteos, 1.0)))
    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(pesos, device=device), label_smoothing=0.05
    )
    niveles_riesgo = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], device=device)
    model = BiLSTMAttention(
        input_dim=input_dim,
        proj_dim=64,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mejor_score = -1.0
    mejor_f1_dev = 0.0
    mejor_estado = None
    sin_mejora = 0

    for _ in range(epochs):
        model.train()
        for x_pad, longitudes, y in train_loader:
            x_pad, longitudes, y = x_pad.to(device), longitudes.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_pad, longitudes)

            loss_ce = criterion_ce(logits, y)
            # Pérdida ordinal: distancia entre riesgo esperado y riesgo real
            probs = torch.softmax(logits, dim=1)
            riesgo_pred = (probs * niveles_riesgo).sum(dim=1)
            riesgo_real = niveles_riesgo[y]
            loss_ord = nn.functional.mse_loss(riesgo_pred, riesgo_real)

            loss = loss_ce + 0.15 * loss_ord
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        res_dev = evaluar_modelo(model, dev_seqs, dev_labels, device)
        score_actual = res_dev["auc"] if not np.isnan(res_dev["auc"]) else res_dev["macro_f1"]
        if score_actual > mejor_score:
            mejor_score = score_actual
            mejor_f1_dev = res_dev["macro_f1"]
            mejor_estado = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= patience:
                break

    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)

    return model, mejor_f1_dev