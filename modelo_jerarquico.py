from contextlib import nullcontext
from itertools import cycle
import math
import os
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
    """
        Fija la semilla aleatoria en Python, NumPy y PyTorch (CPU y CUDA) para reproducibilidad.
        Precondición:
            `seed` debe ser un número entero no negativo.
        Entradas:
            seed (int): Semilla base para los generadores pseudoaleatorios.
        Salidas:
            None.
        Postcondición:
            Los estados de los generadores de `random`, `numpy.random`, `torch` y
            `torch.cuda` (si hay GPU disponible) quedan inicializados con `seed`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construir_textos_por_usuario(df_mensajes, col_post, col_label, label_to_idx):
    """
    Agrupa los mensajes crudos de un DataFrame en secuencias cronológicas por usuario.
    Precondición:
        - `df_mensajes` debe contener las columnas `"user_id"`, `col_post` y `col_label`.
        - Cada usuario debe tener al menos un mensaje y su etiqueta en `col_label` debe
          existir como clave en `label_to_idx`.
    Entradas:
        df_mensajes (pd.DataFrame): DataFrame expandido con un mensaje por fila.
        col_post (str): Nombre de la columna que contiene el texto del mensaje.
        col_label (str): Nombre de la columna con la etiqueta clínica del usuario.
        label_to_idx (dict[str, int]): Mapeo de nombre de clase a índice entero ordenado.
    Salidas:
        tuple[list[list[str]], np.ndarray, list]:
            - `secuencias_texto`: lista de listas de textos por usuario.
            - `etiquetas`: array NumPy `int64` con el índice de clase de cada usuario.
            - `user_ids`: lista de identificadores de usuario en el mismo orden.
    Postcondición:
        Devuelve tres colecciones alineadas de longitud igual al número de usuarios
        únicos presentes en `df_mensajes`, preservando el orden de aparición.
    """
    secuencias_texto, etiquetas, user_ids = [], [], []
    for uid, grupo in df_mensajes.groupby("user_id", sort=False):
        secuencias_texto.append(grupo[col_post].astype(str).tolist())
        etiquetas.append(label_to_idx[grupo[col_label].iloc[0]])
        user_ids.append(uid)
    return secuencias_texto, np.array(etiquetas, dtype=np.int64), user_ids


def calcular_metricas_graded(y_true, y_pred):
    """
    Calcula Graded Precision, Graded Recall y Graded F1 sobre etiquetas en orden clínico (0..4).
    Precondición:
        - `y_true` e `y_pred` deben tener la misma longitud y contener enteros
          correspondientes a niveles ordinales crecientes de riesgo.
    Entradas:
        y_true (array-like[int]): Etiquetas reales en orden clínico.
        y_pred (array-like[int]): Etiquetas predichas en orden clínico.
    Salidas:
        dict[str, float]: Diccionario con claves `"graded_p"`, `"graded_r"` y `"graded_f1"`,
        todas en el rango `[0.0, 1.0]`.
    Postcondición:
        Penaliza las sobreestimaciones (`y_pred > y_true`) en precisión y las
        subestimaciones (`y_pred < y_true`) en recall proporcionalmente a la distancia ordinal.
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


def crear_targets_sord(num_classes, sigma=0.38, device="cpu"):
    """
    Genera la matriz de objetivos ordinales suaves SORD (Soft Ordinal Targets).
    Precondición:
        `num_classes` debe ser un entero >= 2 y `sigma` un flotante estrictamente positivo.
    Entradas:
        num_classes (int): Número total de clases clínicas ordenadas.
        sigma (float): Desviación estándar del kernel gaussiano ordinal.
        device (str | torch.device): Dispositivo destino del tensor resultante.
    Salidas:
        torch.Tensor: Matriz flotante de forma `(num_classes, num_classes)` en `device`,
        donde la fila `i` es la distribución de probabilidad suave para la clase `i`.
    Postcondición:
        Cada fila del tensor devuelto suma exactamente 1.0 y concentra la mayor masa
        en la diagonal principal y las clases clínicamente contiguas.
    """
    idx = torch.arange(num_classes, dtype=torch.float32, device=device)
    dist_sq = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
    return torch.softmax(-dist_sq / (2.0 * sigma ** 2), dim=1)


class LoRALinear(nn.Module):
    """
        Inicializa un adaptador LoRA sobre una capa `nn.Linear` congelada.

        Precondición:
            `base_layer` debe ser una instancia de `nn.Linear`, `r >= 1`, `alpha > 0`
            y `0.0 <= dropout < 1.0`.
        Entradas:
            base_layer (nn.Linear): Capa lineal preentrenada original.
            r (int): Rango de las matrices de descomposición de bajo rango.
            alpha (float): Factor de escala LoRA.
            dropout (float): Probabilidad de dropout aplicada a la entrada de la rama LoRA.
        Salidas:
            None.
        Postcondición:
            Todos los parámetros de `base_layer` quedan congelados (`requires_grad=False`),
            `lora_A` se inicializa con Kaiming uniforme y `lora_B` a ceros para que la
            capa inicie como función identidad respecto a `base_layer`.
    """

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
        """
        Calcula la transformación lineal base sumada a la corrección de bajo rango LoRA.
         Precondición:
               La última dimensión de `x` debe coincidir con `base_layer.in_features`.
         Entradas:
            x (torch.Tensor): Tensor de entrada de forma `(..., in_features)`.
         Salidas:
             torch.Tensor: Tensor de salida de forma `(..., out_features)`.
         Postcondición:
             Devuelve `base_layer(x) + (alpha / r) * lora_B(lora_A(dropout(x)))` sin
             alterar los pesos congelados de `base_layer`.
        """
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


class MessageDataset(Dataset):
    """Dataset de mensajes individuales limpios (semilla.csv) para Multi-Task."""

    def __init__(self, textos, etiquetas):
        """
        Inicializa el dataset de mensajes individuales anotados (`semilla.csv`).
        Precondición:
           `textos` y `etiquetas` deben ser secuencias iterables de igual longitud.
        Entradas:
            textos (Sequence[str]): Lista de mensajes individuales.
            etiquetas (Sequence[int]): Índices de clase correspondientes a cada mensaje.
        Salidas:
            None.
        Postcondición:
            Almacena copias en lista de `textos` y `etiquetas` listas para indexación.
        """
        self.textos = list(textos)
        self.etiquetas = list(etiquetas)

    def __len__(self):
        """
        Devuelve el número total de mensajes del dataset.
        Precondición:
            El objeto `MessageDataset` debe estar inicializado.
        Entradas:
            Ninguna.
        Salidas:
           int: Cantidad de ejemplos almacenados.
        Postcondición:
            Devuelve un entero no negativo sin modificar el estado del dataset.
        """
        return len(self.etiquetas)

    def __getitem__(self, idx):
        """
        Recupera el mensaje y la etiqueta entera en la posición `idx`.
        Precondición:
            `0 <= idx < len(self)`.
        Entradas:
            idx (int): Índice del ejemplo a recuperar.
        Salidas:
            tuple[str, int]: Par `(texto, etiqueta)`.
        Postcondición:
            Devuelve el texto convertido a `str` y la etiqueta convertida a `int`.
        """
        return str(self.textos[idx]), int(self.etiquetas[idx])


class CollateMessages:

    def __init__(self, tokenizer, max_length=384):
        """
        Configura la función de empaquetado y tokenización para lotes de mensajes individuales.
        Precondición:
            `tokenizer` debe ser un tokenizador compatible con Hugging Face y `max_length > 0`.
        Entradas:
            tokenizer: Tokenizador preentrenado del encoder.
            max_length (int): Número máximo de tokens por mensaje.
        Salidas:
            None.
        Postcondición:
            Guarda `tokenizer` y `max_length` como atributos de la instancia invocable.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        Tokeniza y empaqueta un mini-lote de pares `(texto, etiqueta)`.
        Precondición:
            `batch` debe ser una lista no vacía de tuplas `(str, int)`.
        Entradas:
            batch (list[tuple[str, int]]): Mini-lote entregado por `DataLoader`.
        Salidas:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            `(input_ids, attention_mask, etiquetas)` de longitud `len(batch)`.
        Postcondición:
            Devuelve tensores con padding dinámico y truncamiento a `self.max_length`.
        """
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

    def __init__(self, secuencias_texto, etiquetas, max_msgs=32, msg_dropout=0.10, training=True):
        """
        Inicializa el dataset de secuencias de mensajes por usuario.
        Precondición:
            - `secuencias_texto` y `etiquetas` deben tener igual longitud.
            - Cada elemento de `secuencias_texto` debe ser una lista no vacía de textos.
            - `0.0 <= msg_dropout < 1.0` y `max_msgs` es `None` o un entero >= 1.
        Entradas:
            secuencias_texto (list[list[str]]): Mensajes cronológicos por usuario.
            etiquetas (array-like[int]): Etiqueta clínica por usuario.
            max_msgs (int | None): Tope máximo de mensajes conservados por usuario.
            msg_dropout (float): Probabilidad de descartar cada mensaje en entrenamiento.
            training (bool): Activa el muestreo aleatorio y el dropout de mensajes.
        Salidas:
            None.
        Postcondición:
            Almacena las secuencias y configuración de muestreo en la instancia.
        """
        self.secuencias_texto = secuencias_texto
        self.etiquetas = etiquetas
        self.max_msgs = max_msgs
        self.msg_dropout = msg_dropout
        self.training = training

    def __len__(self):
        """
        Devuelve el número total de usuarios en el dataset.
        Precondición:
            El objeto `UserTextDataset` debe estar inicializado.
        Entradas:
            Ninguna.
        Salidas:
            int: Cantidad de usuarios almacenados.
        Postcondición:
            Devuelve un entero no negativo sin alterar el estado interno.
        """
        return len(self.etiquetas)

    def __getitem__(self, idx):
        """
        Obtiene la secuencia de mensajes (con dropout y submuestreo opcionales) y la etiqueta del usuario `idx`.

        Precondición:
            `0 <= idx < len(self)` y `secuencias_texto[idx]` contiene al menos 1 mensaje.
        Entradas:
            idx (int): Índice del usuario.
        Salidas:
            tuple[list[str], int]: Par `(mensajes, etiqueta)`.
        Postcondición:
            Devuelve una lista no vacía de mensajes respetando el orden cronológico relativo
            y acotada a `max_msgs` si está definido, sin mutar la lista original.
        """
        mensajes = list(self.secuencias_texto[idx])

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
    def __init__(self, tokenizer, max_length=384):
        """
        Configura la función de empaquetado para mini-lotes de secuencias de usuarios.
        Precondición:
            `tokenizer` debe ser un tokenizador compatible con Hugging Face y `max_length > 0`.
        Entradas:
            tokenizer: Tokenizador preentrenado del encoder.
            max_length (int): Límite máximo de tokens por mensaje.
        Salidas:
            None.
        Postcondición:
            Almacena el tokenizador y la longitud máxima en la instancia.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        Aplana todos los mensajes de los usuarios del lote y los tokeniza en una sola llamada.
        Precondición:
            `batch` debe ser una lista no vacía de pares `(lista_mensajes, etiqueta)`,
            donde cada `lista_mensajes` tiene al menos un texto.
        Entradas:
            batch (list[tuple[list[str], int]]): Mini-lote de usuarios.
        Salidas:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                `(input_ids, attention_mask, longitudes, etiquetas)` donde
                `input_ids.size(0) == longitudes.sum().item()`.
        Postcondición:
            Devuelve los tensores tokenizados con padding dinámico y el vector de
            longitudes necesario para reconstruir cada usuario tras el encoder.
        """
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

    def __init__(
        self,
        model_name="mental/mental-roberta-base",
        lora_layers=12,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        lora_include_out=True,
        grad_checkpointing=False,
        hidden_dim=32,
        num_classes=5,
        dropout=0.4,
        multi_sample_dropout_n=5,
    ):
        """
        Construye la arquitectura jerárquica MentalRoBERTa-LoRA + Token Attention + BiLSTM + MIL.
        Precondición:
            - `model_name` debe corresponder a un modelo RoBERTa/Transformer accesible
              localmente o en Hugging Face.
            - `0 <= lora_layers <= num_capas_encoder`, `lora_r >= 1`, `hidden_dim >= 1`,
              `num_classes >= 2` y `0.0 <= dropout < 1.0`.
        Entradas:
            model_name (str): Identificador del encoder base.
            lora_layers (int): Número de capas finales del encoder adaptadas con LoRA.
            lora_r (int): Rango LoRA.
            lora_alpha (float): Factor de escala LoRA.
            lora_dropout (float): Dropout en la rama LoRA.
            lora_include_out (bool): Si es `True`, aplica LoRA también a `attention.output.dense`.
            grad_checkpointing (bool): Activa recomputación de activaciones para ahorrar VRAM.
            hidden_dim (int): Dimensión oculta por dirección de la BiLSTM.
            num_classes (int): Número de clases clínicas de salida.
            dropout (float): Tasa de dropout en proyección y cabeza clasificadora.
            multi_sample_dropout_n (int): Número de máscaras en Multi-Sample Dropout.
        Salidas:
            None.
        Postcondición:
            El encoder queda congelado salvo los adaptadores LoRA, `token_attn` inicia a
            ceros (equivalente a mean-pooling) y `user_combiner` inicia con pesos 0.7/0.3
            sobre `mil_peak` y `mil_mean`.
        """
        super().__init__()
        self.num_classes = num_classes
        self.multi_sample_dropout_n = max(1, int(multi_sample_dropout_n))
        self.encoder = AutoModel.from_pretrained(model_name)
        self._aplicar_lora(lora_layers, lora_r, lora_alpha, lora_dropout, lora_include_out)

        if grad_checkpointing and lora_layers > 0:
            self._activar_checkpointing()

        emb_dim = self.encoder.config.hidden_size

        # Atención sobre tokens inicializada a 0 (arranca exactamente como mean_pool)
        self.token_attn = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.token_attn.weight)
        nn.init.zeros_(self.token_attn.bias)

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

        # Clasificador a nivel de mensaje (compartido con semillas Multi-Task)
        self.msg_classifier = nn.Linear(seq_dim, num_classes)

        # Temperatura aprendible por clase para MIL Softmax-Weighted Pooling
        self.raw_beta = nn.Parameter(torch.full((1, 1, num_classes), 2.5))

        # Fusión ligera: combina evidencia pico (MIL) y tono medio (Mean)
        self.user_combiner = nn.Linear(num_classes * 2, num_classes)
        with torch.no_grad():
            self.user_combiner.weight.zero_()
            self.user_combiner.bias.zero_()
            eye = torch.eye(num_classes)
            self.user_combiner.weight[:, :num_classes] = 0.7 * eye
            self.user_combiner.weight[:, num_classes:] = 0.3 * eye

    def _activar_checkpointing(self):
        """
        Activa el checkpointing de activaciones/gradientes en el encoder Transformer.
        Precondición:
            `self.encoder` debe estar inicializado y soportar `activation_checkpointing_enable`
            o `gradient_checkpointing_enable`.
        Entradas:
            Ninguna.
        Salidas:
            None.
        Postcondición:
            Desactiva `use_cache` si existe, activa el checkpointing no reentrante en el
            encoder y habilita gradientes en los embeddings de entrada para compatibilidad con LoRA.
        """
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False
        if hasattr(self.encoder, "activation_checkpointing_enable"):
            try:
                self.encoder.activation_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.encoder.activation_checkpointing_enable()
        else:
            try:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.encoder.gradient_checkpointing_enable()
        self.encoder.enable_input_require_grads()

    def _aplicar_lora(self, lora_layers, r, alpha, dropout, include_out=True):
        """
        Congela todos los parámetros del encoder e inyecta capas `LoRALinear` en las últimas capas.
        Precondición:
            `self.encoder` debe tener la estructura `encoder.layer` con subcapas
            `attention.self.query`, `attention.self.value` y `attention.output.dense`.
        Entradas:
            lora_layers (int): Número de capas finales donde inyectar LoRA.
            r (int): Rango LoRA.
            alpha (float): Escala LoRA.
            dropout (float): Dropout interno de LoRA.
            include_out (bool): Indica si se envuelve también `attention.output.dense`.
        Salidas:
            None.
        Postcondición:
            Únicamente los pesos `lora_A` y `lora_B` de las últimas `lora_layers` capas
            mantienen `requires_grad=True` dentro de `self.encoder`.
        """
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
                if include_out:
                    capa.attention.output.dense = LoRALinear(
                        capa.attention.output.dense, r=r, alpha=alpha, dropout=dropout
                    )

    def trainable_state_dict(self):
        """
        Extrae un diccionario clonado en CPU únicamente con los parámetros entrenables del modelo.
        Precondición:
            El modelo debe estar instanciado con los flags `requires_grad` correctamente configurados.
        Entradas:
            Ninguna.
        Salidas:
            dict[str, torch.Tensor]: Subconjunto de `state_dict()` limitado a parámetros entrenables.
        Postcondición:
            Devuelve tensores desacoplados del grafo (`detach`) y copiados en CPU (~2 MB),
            sin modificar el dispositivo ni los pesos del modelo activo.
        """
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k in trainable_keys
        }

    def encode_messages(self, input_ids, attention_mask, use_bf16=True):
        """
        Codifica un lote de mensajes mediante MentalRoBERTa-LoRA y Token Attention Pooling.
        Precondición:
            `input_ids` y `attention_mask` deben tener forma `(N, L)` en el dispositivo del modelo.
        Entradas:
            input_ids (torch.Tensor): Identificadores de tokens `(N, L)`.
            attention_mask (torch.Tensor): Máscara de tokens válidos `(N, L)`.
            use_bf16 (bool): Activa `autocast` en `bfloat16` si el tensor está en CUDA.
        Salidas:
            torch.Tensor: Embeddings densos por mensaje de forma `(N, emb_dim)` en `float32`.
        Postcondición:
            Pondera cada token válido mediante `token_attn` (enmascarando el padding) y
            devuelve un único vector por mensaje.
        """
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
        """
        Codifica un lote de mensajes mediante MentalRoBERTa-LoRA y Token Attention Pooling.
        Precondición:
            `input_ids` y `attention_mask` deben tener forma `(N, L)` en el dispositivo del modelo.
        Entradas:
            input_ids (torch.Tensor): Identificadores de tokens `(N, L)`.
            attention_mask (torch.Tensor): Máscara de tokens válidos `(N, L)`.
            use_bf16 (bool): Activa `autocast` en `bfloat16` si el tensor está en CUDA.
        Salidas:
            torch.Tensor: Embeddings densos por mensaje de forma `(N, emb_dim)` en `float32`.
        Postcondición:
            Pondera cada token válido mediante `token_attn` (enmascarando el padding) y
            devuelve un único vector por mensaje.
        """
        if self.training and self.multi_sample_dropout_n > 1:
            return torch.stack(
                [self.msg_classifier(self.head_dropout(h)) for _ in range(self.multi_sample_dropout_n)],
                dim=0,
            ).mean(dim=0)
        if self.training:
            return self.msg_classifier(self.head_dropout(h))
        return self.msg_classifier(h)

    def forward_messages(self, input_ids, attention_mask, use_bf16=True):
        """
        Ejecuta la rama Multi-Task a nivel de mensaje individual (para `semilla.csv`).
        Precondición:
            `input_ids` y `attention_mask` deben tener forma `(B, L)` en el dispositivo del modelo.
        Entradas:
            input_ids (torch.Tensor): Identificadores de tokens de mensajes semilla `(B, L)`.
            attention_mask (torch.Tensor): Máscara de tokens válidos `(B, L)`.
            use_bf16 (bool): Activa `autocast` en `bfloat16` en CUDA.
        Salidas:
            torch.Tensor: Logits por mensaje de forma `(B, num_classes)`.
        Postcondición:
            Reutiliza el encoder, proyección, BiLSTM residual y `msg_classifier` compartidos
            con la rama de usuario.
        """
        msg_embs = self.encode_messages(input_ids, attention_mask, use_bf16=use_bf16)
        x = self.proj(msg_embs).unsqueeze(1)
        lstm_out, _ = self.bilstm(x)
        h = self.seq_norm(lstm_out + x).squeeze(1)
        return self._clasificar_representaciones_msg(h)

    def forward_from_embeddings(self, x_pad, longitudes):
        """
        Procesa secuencias ya vectorizadas de usuarios mediante BiLSTM residual y agregación MIL.
        Precondición:
            - `x_pad` debe tener forma `(U, M_max, emb_dim)` en el dispositivo del modelo.
            - `longitudes` debe ser un tensor 1D de enteros positivos con longitud `U` y
              valores `<= M_max`.
        Entradas:
            x_pad (torch.Tensor): Secuencias de embeddings con padding `(U, M_max, emb_dim)`.
            longitudes (torch.Tensor): Número real de mensajes por usuario `(U,)`.
        Salidas:
            tuple[torch.Tensor, torch.Tensor]:
                - `user_logits`: logits a nivel de usuario `(U, num_classes)`.
                - `msg_logits`: logits por mensaje `(U, M_max, num_classes)`.
        Postcondición:
            Combina evidencia pico (MIL Softmax-Weighted Pooling con temperatura aprendible)
            y media enmascarada ignorando cualquier posición de padding.
        """
        x = self.proj(x_pad)

        packed = pack_padded_sequence(
            x, longitudes.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        h = self.seq_norm(lstm_out + x)

        msg_logits = self._clasificar_representaciones_msg(h)

        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < longitudes.unsqueeze(1)
        long_f = longitudes.float().clamp(min=1.0).view(-1, 1)

        # 1. MIL Softmax-Weighted Pooling por clase (invariante a la longitud del historial)
        beta = nn.functional.softplus(self.raw_beta) + 1.0
        scores_mil = (msg_logits * beta).masked_fill(~mask.unsqueeze(-1), -1e9)
        weights_mil = torch.softmax(scores_mil, dim=1)
        mil_peak = (weights_mil * msg_logits).sum(dim=1)

        # 2. Media enmascarada de logits (captura tono global del historial)
        mask_f = mask.unsqueeze(-1).float()
        mil_mean = (msg_logits * mask_f).sum(dim=1) / long_f

        user_logits = self.user_combiner(torch.cat([mil_peak, mil_mean], dim=1))
        return user_logits, msg_logits

    def forward(self, input_ids, attention_mask, longitudes, use_bf16=True):
        """
        Ejecuta el paso hacia adelante completo desde tokens planos hasta logits de usuario y mensaje.
        Precondición:
            - `input_ids` y `attention_mask` tienen forma `(N_total, L)` en el dispositivo del modelo.
            - `longitudes` contiene enteros positivos cuya suma es exactamente `N_total`.
        Entradas:
            input_ids (torch.Tensor): Tokens de todos los mensajes del lote `(N_total, L)`.
            attention_mask (torch.Tensor): Máscara de tokens válidos `(N_total, L)`.
            longitudes (torch.Tensor): Mensajes por cada usuario del lote `(U,)`.
            use_bf16 (bool): Activa `autocast` en `bfloat16` en CUDA.
        Salidas:
            tuple[torch.Tensor, torch.Tensor]: `(user_logits, msg_logits)`.
        Postcondición:
            Agrupa los embeddings codificados por usuario respetando `longitudes` y
            devuelve las predicciones jerárquicas completas.
        """
        msg_embs = self.encode_messages(input_ids, attention_mask, use_bf16=use_bf16)
        splits = torch.split(msg_embs, longitudes.tolist(), dim=0)
        x_pad = pad_sequence(splits, batch_first=True, padding_value=0.0)
        return self.forward_from_embeddings(x_pad, longitudes)


def guardar_modelo_mil(model, ruta, config_modelo, clases, metricas, hiperparametros=None):
    """
        Codifica un lote de mensajes mediante MentalRoBERTa-LoRA y Token Attention Pooling.
        Precondición:
            `input_ids` y `attention_mask` deben tener forma `(N, L)` en el dispositivo del modelo.
        Entradas:
            input_ids (torch.Tensor): Identificadores de tokens `(N, L)`.
            attention_mask (torch.Tensor): Máscara de tokens válidos `(N, L)`.
            use_bf16 (bool): Activa `autocast` en `bfloat16` si el tensor está en CUDA.
        Salidas:
            torch.Tensor: Embeddings densos por mensaje de forma `(N, emb_dim)` en `float32`.
        Postcondición:
            Pondera cada token válido mediante `token_attn` (enmascarando el padding) y
            devuelve un único vector por mensaje.
    """
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    payload = {
        "trainable_state_dict": model.trainable_state_dict(),
        "model_config": dict(config_modelo),
        "clases": list(clases),
        "metricas": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metricas.items()},
        "hiperparametros": hiperparametros or {},
    }
    torch.save(payload, ruta)


def cargar_modelo_mil(ruta, device="cpu"):
    """
    Ejecuta el paso hacia adelante completo desde tokens planos hasta logits de usuario y mensaje.
    Precondición:
        - `input_ids` y `attention_mask` tienen forma `(N_total, L)` en el dispositivo del modelo.
        - `longitudes` contiene enteros positivos cuya suma es exactamente `N_total`.
    Entradas:
        input_ids (torch.Tensor): Tokens de todos los mensajes del lote `(N_total, L)`.
        attention_mask (torch.Tensor): Máscara de tokens válidos `(N_total, L)`.
        longitudes (torch.Tensor): Mensajes por cada usuario del lote `(U,)`.
        use_bf16 (bool): Activa `autocast` en `bfloat16` en CUDA.
    Salidas:
        tuple[torch.Tensor, torch.Tensor]: `(user_logits, msg_logits)`.
    Postcondición:
        Agrupa los embeddings codificados por usuario respetando `longitudes` y
        devuelve las predicciones jerárquicas completas.
    """
    try:
        ckpt = torch.load(ruta, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ruta, map_location="cpu")

    cfg = dict(ckpt["model_config"])
    cfg["grad_checkpointing"] = False
    model = HierarchicalMentalRoBERTaMIL(**cfg).to(device)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model.eval()
    return model, ckpt["clases"], ckpt.get("metricas", {}), ckpt.get("hiperparametros", {})


@torch.no_grad()
def extraer_secuencias_con_encoder(
    model, secuencias_texto, tokenizer, device, max_length=384, msg_batch_size=64, use_bf16=True
):
    """
    Vectoriza en modo inferencia todos los mensajes de un conjunto de usuarios en una sola pasada.
    Precondición:
        - `secuencias_texto` debe ser una lista no vacía de secuencias no vacías de textos.
        - `model` debe encontrarse en `device` y `msg_batch_size >= 1`.
    Entradas:
        model (HierarchicalMentalRoBERTaMIL): Modelo con el encoder a utilizar.
        secuencias_texto (list[list[str]]): Mensajes cronológicos por usuario.
        tokenizer: Tokenizador compatible con el encoder del modelo.
        device (torch.device): Dispositivo de inferencia.
        max_length (int): Longitud máxima de tokens por mensaje.
        msg_batch_size (int): Tamaño de lote de mensajes por pasada del encoder.
        use_bf16 (bool): Activa `autocast` en `bfloat16` en CUDA.
    Salidas:
        list[torch.Tensor]: Lista de tensores en CPU de forma `(n_msgs_u, emb_dim)`,
        uno por cada usuario en el mismo orden que `secuencias_texto`.
    Postcondición:
        Deja `model` en modo evaluación sin construir grafo de gradientes y devuelve los
        embeddings listos para evaluar múltiples ventanas `k` sin recodificar textos.
    """
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
    """
    Evalúa el modelo sobre secuencias de embeddings ya extraídas, aplicando un corte opcional
    a los primeros `max_mensajes` (ventana estática `[:k]`) por usuario.
    Precondición:
        - `secuencias_emb` y `etiquetas` deben ser no vacíos y tener la misma longitud.
        - `model` debe estar en `device` y `batch_size >= 1`.
    Entradas:
        model (HierarchicalMentalRoBERTaMIL): Modelo jerárquico a evaluar.
        secuencias_emb (list[torch.Tensor]): Lista de tensores `(n_msgs_u, emb_dim)` por usuario.
        etiquetas (array-like[int]): Etiquetas reales en orden clínico.
        device (torch.device): Dispositivo de inferencia.
        max_mensajes (int | None): Límite máximo de mensajes leídos por usuario (`None` = todos).
        batch_size (int): Número de usuarios por lote de evaluación.
    Salidas:
        dict[str, object]: Diccionario con `macro_f1`, `weighted_f1`, `graded_p`, `graded_r`,
        `graded_f1`, `kappa`, `qwk`, `auc`, `media_msgs`, `y_true`, `y_pred` e `y_prob`.
    Postcondición:
        Deja `model` en modo evaluación sin construir grafo de gradientes y devuelve todas
        las métricas agregadas junto con las predicciones y probabilidades por usuario.
    """
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
    seed_alpha=0.20,
    seed_alpha_decay=0.75,
    sord_seed_sigma=0.40,
    seed_batch_size=16,
    model_name="mental/mental-roberta-base",
    lora_layers=12,
    lora_r=8,
    lora_alpha=16.0,
    lora_dropout=0.1,
    lora_include_out=True,
    grad_checkpointing=False,
    hidden_dim=32,
    dropout=0.4,
    multi_sample_dropout_n=5,
    max_length=384,
    max_msgs_train=32,
    msg_dropout=0.10,
    batch_size=2,
    accum_steps=8,
    lr_lora=2e-4,
    lr_head=1e-3,
    lora_weight_decay=1e-2,
    head_weight_decay=2e-2,
    warmup_ratio=0.10,
    tau=0.75,
    sord_sigma=0.38,
    early_stop_auc_weight=0.5,
    use_top2_soup=True,
    epochs=25,
    patience=6,
    seed=42,
):
    """
    Entrena un modelo `HierarchicalMentalRoBERTaMIL` con Balanced Softmax + SORD,
    Multi-Task opcional con semillas, early stopping compuesto y Top-2 Model Soup.
    Precondición:
        - `train_texts`/`train_labels` y `val_texts`/`val_labels` deben ser no vacíos y
          alineados en longitud, con etiquetas enteras en `[0, len(clases) - 1]`.
        - Si se pasan `seed_texts` y `seed_labels` con `seed_alpha > 0`, deben tener igual
          longitud y contener etiquetas válidas en `[0, len(clases) - 1]`.
        - `epochs >= 1`, `patience >= 1`, `batch_size >= 1`, `accum_steps >= 1` y
          `0.0 <= early_stop_auc_weight <= 1.0`.
    Entradas:
        train_texts (list[list[str]]): Secuencias de mensajes de entrenamiento por usuario.
        train_labels (np.ndarray): Etiquetas de entrenamiento en orden clínico.
        val_texts (list[list[str]]): Secuencias de mensajes de validación por usuario.
        val_labels (np.ndarray): Etiquetas de validación en orden clínico.
        clases (list[str]): Lista de nombres de clases en orden clínico.
        tokenizer: Tokenizador del encoder preentrenado.
        device (torch.device): Dispositivo de entrenamiento (`cuda` o `cpu`).
        seed_texts (list[str] | None): Mensajes individuales limpios para la rama Multi-Task.
        seed_labels (np.ndarray | None): Etiquetas de los mensajes semilla.
        seed_alpha (float): Peso inicial de la pérdida auxiliar Multi-Task.
        seed_alpha_decay (float): Fracción de decaimiento lineal de `seed_alpha` a lo largo de las épocas.
        sord_seed_sigma (float): Sigma SORD aplicado a la rama de mensajes semilla.
        seed_batch_size (int): Tamaño de lote para los mensajes semilla.
        model_name (str): Identificador del encoder base.
        lora_layers (int): Capas finales del encoder con adaptadores LoRA.
        lora_r (int): Rango LoRA.
        lora_alpha (float): Factor de escala LoRA.
        lora_dropout (float): Dropout de LoRA.
        lora_include_out (bool): Incluye proyección de salida de atención en LoRA.
        grad_checkpointing (bool): Activa checkpointing de activaciones en el encoder.
        hidden_dim (int): Dimensión oculta por dirección de la BiLSTM.
        dropout (float): Dropout de la cabeza jerárquica.
        multi_sample_dropout_n (int): Número de máscaras en Multi-Sample Dropout.
        max_length (int): Longitud máxima en tokens por mensaje.
        max_msgs_train (int): Tope de mensajes por usuario durante entrenamiento.
        msg_dropout (float): Probabilidad de descartar mensajes en entrenamiento.
        batch_size (int): Usuarios por micro-lote.
        accum_steps (int): Pasos de acumulación de gradientes.
        lr_lora (float): Tasa de aprendizaje para los parámetros LoRA.
        lr_head (float): Tasa de aprendizaje para la cabeza secuencial MIL.
        lora_weight_decay (float): Decaimiento de pesos para LoRA.
        head_weight_decay (float): Decaimiento de pesos para la cabeza MIL.
        warmup_ratio (float): Proporción de pasos iniciales de warmup del scheduler coseno.
        tau (float): Escalar de ajuste de priors en Balanced Softmax.
        sord_sigma (float): Sigma del suavizado ordinal SORD a nivel de usuario.
        early_stop_auc_weight (float): Peso del AUC en la métrica compuesta de parada temprana.
        use_top2_soup (bool): Si es `True`, evalúa el promedio de pesos de las 2 mejores épocas.
        epochs (int): Número máximo de épocas de entrenamiento.
        patience (int): Épocas consecutivas sin mejora antes de detener el entrenamiento.
        seed (int): Semilla aleatoria del fold.
    Salidas:
        tuple[HierarchicalMentalRoBERTaMIL, float, float, float, str]:
            `(model, mejor_f1_val, mejor_gf1_val, mejor_auc_real, mejor_epoca)`.
    Postcondición:
        Devuelve el modelo en `device` cargado con los pesos de la mejor época individual
        o del promedio Top-2 Soup (solo si iguala o mejora el score compuesto en validación)
        y libera la lista temporal de estados guardados.
    """
    fijar_semilla(seed)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    num_classes = len(clases)

    model = HierarchicalMentalRoBERTaMIL(
        model_name=model_name,
        lora_layers=lora_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_include_out=lora_include_out,
        grad_checkpointing=grad_checkpointing,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        multi_sample_dropout_n=multi_sample_dropout_n,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    usar_semillas = seed_texts is not None and len(seed_texts) > 0 and seed_alpha > 0
    modulos_str = "Q+V+Out" if lora_include_out else "Q+V"
    print(
        f"Parámetros entrenables: {train_p:,} / {total_p:,} ({100 * train_p / total_p:.2f}%) | "
        f"LoRA(capas={lora_layers}, r={lora_r}, {modulos_str}) | max_len={max_length} | "
        f"grad_ckpt={grad_checkpointing} | sord={sord_sigma} | semillas={len(seed_texts) if usar_semillas else 0}"
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

    conteos = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    priors = conteos / conteos.sum()
    log_priors = torch.tensor(np.log(np.clip(priors, 1e-12, 1.0)), dtype=torch.float32, device=device)
    sord_user = crear_targets_sord(num_classes, sigma=sord_sigma, device=device)

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
        sord_seed = crear_targets_sord(num_classes, sigma=sord_seed_sigma, device=device)

    lora_params = [
        p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad
    ]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": lora_weight_decay})
    param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": head_weight_decay})
    optimizer = torch.optim.AdamW(param_groups)

    steps_por_epoca = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_por_epoca * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(warmup_ratio * total_steps)),
        num_training_steps=total_steps,
    )

    mejor_score = -1.0
    sin_mejora = 0
    top_checkpoints = []

    for epoca in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_user_acum = 0.0
        loss_seed_acum = 0.0
        n_seed_steps = 0
        alpha_ep = seed_alpha * (1.0 - seed_alpha_decay * (epoca - 1) / max(1, epochs - 1))

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
        w_auc = early_stop_auc_weight
        score_actual = (
            w_auc * res_val["auc"] + (1.0 - w_auc) * res_val["macro_f1"]
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

        max_keep = 2 if use_top2_soup else 1
        if len(top_checkpoints) < max_keep or score_actual > top_checkpoints[-1]["score"]:
            top_checkpoints.append(
                {
                    "score": score_actual,
                    "auc": res_val["auc"],
                    "f1": res_val["macro_f1"],
                    "gf1": res_val["graded_f1"],
                    "epoca": epoca,
                    "state": model.trainable_state_dict(),
                }
            )
            top_checkpoints.sort(key=lambda x: x["score"], reverse=True)
            top_checkpoints = top_checkpoints[:max_keep]

        if score_actual > mejor_score:
            mejor_score = score_actual
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= patience:
                print(f"    Early stopping en época {epoca} (sin mejora en {patience} épocas).")
                break

    best_cp = top_checkpoints[0]
    mejor_auc_real = best_cp["auc"]
    mejor_f1_val = best_cp["f1"]
    mejor_gf1_val = best_cp["gf1"]
    mejor_epoca = str(best_cp["epoca"])

    if use_top2_soup and len(top_checkpoints) == 2:
        cp1, cp2 = top_checkpoints[0], top_checkpoints[1]
        soup_state = {k: 0.5 * (cp1["state"][k] + cp2["state"][k]) for k in cp1["state"]}
        model.load_state_dict(soup_state, strict=False)
        val_embs_soup = extraer_secuencias_con_encoder(
            model, val_texts, tokenizer, device, max_length=max_length, use_bf16=use_bf16
        )
        res_soup = evaluar_secuencias_emb(
            model, val_embs_soup, val_labels, device, max_mensajes=None
        )
        w_auc = early_stop_auc_weight
        score_soup = (
            w_auc * res_soup["auc"] + (1.0 - w_auc) * res_soup["macro_f1"]
            if not np.isnan(res_soup["auc"])
            else res_soup["macro_f1"]
        )

        if score_soup >= cp1["score"]:
            mejor_auc_real = res_soup["auc"]
            mejor_f1_val = res_soup["macro_f1"]
            mejor_gf1_val = res_soup["graded_f1"]
            mejor_epoca = f"{cp1['epoca']}+{cp2['epoca']}"
            print(
                f"    [Top-2 Soup] Épocas {mejor_epoca} aceptado | "
                f"Val AUC: {mejor_auc_real:.4f} | Val F1: {mejor_f1_val:.4f} | Val GF1: {mejor_gf1_val:.4f}"
            )
        else:
            model.load_state_dict(cp1["state"], strict=False)
            print(
                f"    [Top-2 Soup] Se conserva mejor época individual ({cp1['epoca']}) | "
                f"Val AUC: {mejor_auc_real:.4f} | Val F1: {mejor_f1_val:.4f}"
            )
    else:
        model.load_state_dict(best_cp["state"], strict=False)

    del top_checkpoints
    return model, mejor_f1_val, mejor_gf1_val, mejor_auc_real, mejor_epoca