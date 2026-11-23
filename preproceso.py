import ast
import hashlib
import os
import re
import numpy as np
import torch
from nltk import SnowballStemmer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_PALABRAS = 80   # ~100-110 tokens, por debajo del límite de 128
MAX_CHUNKS = 50       # Tope de trozos por instancia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Expresiones regulares precompiladas
_RE_CORCHETE_INI = re.compile(r"^\[\s*['\"]")
_RE_CORCHETE_FIN = re.compile(r"['\"]\s*\]\s*$")
_RE_SPLIT_LISTA = re.compile(r"['\"]\s*,\s*['\"]")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NO_LETRAS = re.compile(r"[^a-záéíóúñ\s]")
_RE_ESPACIOS = re.compile(r"\s+")

_STEMMER = SnowballStemmer("english")
_MODELOS_CACHE = {}


def _obtener_modelo(nombre_modelo):
    if nombre_modelo not in _MODELOS_CACHE:
        from sentence_transformers import SentenceTransformer

        print(f"Cargando modelo '{nombre_modelo}' en dispositivo: {DEVICE}...")
        _MODELOS_CACHE[nombre_modelo] = SentenceTransformer(nombre_modelo, device=DEVICE)
    return _MODELOS_CACHE[nombre_modelo]


def dividir_mensajes(df, col_post="Post"):
    """Expande listas de mensajes en filas individuales, conservando el ID del usuario."""
    def parsear(crudo):
        try:
            lista = ast.literal_eval(crudo)
            return [str(m) for m in lista] if isinstance(lista, (list, tuple)) else [str(lista)]
        except (ValueError, SyntaxError):
            s = str(crudo).strip()
            s = _RE_CORCHETE_INI.sub("", s)
            s = _RE_CORCHETE_FIN.sub("", s)
            return _RE_SPLIT_LISTA.split(s)

    out = df.copy()
    out["user_id"] = out.index
    out[col_post] = out[col_post].apply(parsear)
    out = out.explode(col_post, ignore_index=True).dropna(subset=[col_post])
    out[col_post] = out[col_post].astype(str).str.strip()
    return out[out[col_post] != ""].reset_index(drop=True)


def clean_text(text):
    """Limpia ruido y aplica stemming conservando stop words (pronombres y conectores)."""
    text = _RE_URL.sub("", str(text).lower())
    limpio = _RE_ESPACIOS.sub(" ", _RE_NO_LETRAS.sub("", text)).strip()
    tokens = word_tokenize(limpio, language="english")
    return " ".join(_STEMMER.stem(w) for w in tokens)


def tfidf(X_train, X_dev, X_test):
    """Vectoriza las particiones usando TF-IDF sobre texto limpio."""
    vectorizer = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), min_df=5, max_df=0.85, stop_words=None
    )
    X_train_vec = vectorizer.fit_transform(X_train.apply(clean_text))
    X_dev_vec = vectorizer.transform(X_dev.apply(clean_text))
    X_test_vec = vectorizer.transform(X_test.apply(clean_text))
    return X_train_vec, X_dev_vec, X_test_vec


def trocear(texto, chunk_palabras, max_chunks):
    """Divide un texto en trozos de `chunk_palabras` palabras (muestreo uniforme si supera `max_chunks`)."""
    palabras = texto.split()
    if not palabras:
        return [""]
    chunks = [
        " ".join(palabras[i : i + chunk_palabras])
        for i in range(0, len(palabras), chunk_palabras)
    ]
    if max_chunks and len(chunks) > max_chunks:
        idx = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in idx]
    return chunks


def _vectorizar_por_trozos(
    textos, firma, prefijo, cache_dir, chunk_palabras, max_chunks, codificar, normalizar=True
):
    hash_datos = hashlib.md5((firma + "\x1f".join(textos)).encode("utf-8")).hexdigest()[:12]

    os.makedirs(cache_dir, exist_ok=True)
    ruta_cache = os.path.join(cache_dir, f"{prefijo}_{hash_datos}.npy")
    if os.path.exists(ruta_cache):
        print(f"Cargando vectores desde caché: {ruta_cache}")
        return np.load(ruta_cache, mmap_mode="r")

    todos_chunks = []
    inicios = []
    for t in textos:
        inicios.append(len(todos_chunks))
        todos_chunks.extend(trocear(t, chunk_palabras, max_chunks))

    print(f"{len(textos)} mensajes -> {len(todos_chunks)} trozos")
    emb = codificar(todos_chunks)

    # Promediado vectorizado por mensaje
    conteos = np.diff(inicios + [len(todos_chunks)])
    X = np.add.reduceat(emb, inicios, axis=0) / conteos[:, None]

    if normalizar:
        normas = np.linalg.norm(X, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        X = X / normas

    np.save(ruta_cache, X)
    print(f"Vectores guardados en {ruta_cache} con forma {X.shape}")
    return X


def vectorizar_embeddings(
    text_cols, cache_dir="cache", chunk_palabras=CHUNK_PALABRAS, max_chunks=MAX_CHUNKS
):
    """Genera (o carga de caché) embeddings L2-normalizados por mensaje."""
    textos = [str(t) for t in text_cols]
    firma = f"{MODELO_EMBEDDINGS}|msg|{chunk_palabras}|{max_chunks}||"

    def codificar(chunks):
        modelo = _obtener_modelo(MODELO_EMBEDDINGS)
        print("Vectorizando textos...")
        return modelo.encode(
            chunks, batch_size=128, show_progress_bar=True, normalize_embeddings=True
        )

    return _vectorizar_por_trozos(
        textos, firma, "vectores", cache_dir, chunk_palabras, max_chunks, codificar, normalizar=True
    )