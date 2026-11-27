import ast
import hashlib
import os
import re
import numpy as np
import torch
from nltk import SnowballStemmer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    """
    - Precondiciones: nombre_modelo es un identificador valido de SentenceTransformer y DEVICE esta definido.
    - Entradas: nombre_modelo (str con el nombre del modelo de embeddings).
    - Salidas: SentenceTransformer cargado en el dispositivo configurado.
    - Postcondiciones: Cachea y devuelve la instancia del modelo lista para codificar textos.
    """
    if nombre_modelo not in _MODELOS_CACHE:
        from sentence_transformers import SentenceTransformer

        print(f"Cargando modelo '{nombre_modelo}' en dispositivo: {DEVICE}...")
        _MODELOS_CACHE[nombre_modelo] = SentenceTransformer(nombre_modelo, device=DEVICE)
    return _MODELOS_CACHE[nombre_modelo]


def dividir_mensajes(df, col_post="Post"):
    """
    - Entradas:
        df: DataFrame con una fila por usuario y una lista de mensajes en `col_post`.
        col_post: str con el nombre de la columna que contiene los mensajes.
    - Salidas:
        DataFrame expandido con una fila por mensaje no vacío y columna `user_id`.
    - Precondición:
        `col_post` existe en `df`.
    - Postcondición:
        Conserva el identificador original del usuario en `user_id` y elimina mensajes vacíos.
    """
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
    out["user_id"] = out["User"] if "User" in out.columns else out.index
    out[col_post] = out[col_post].apply(parsear)
    out = out.explode(col_post, ignore_index=True).dropna(subset=[col_post])
    out[col_post] = out[col_post].astype(str).str.strip()
    return out[out[col_post] != ""].reset_index(drop=True)


def clean_text(text):
    """
    - Precondiciones: text es convertible a cadena.
    - Entradas: text (texto de entrada en bruto).
    - Salidas: str con el texto en minusculas, sin URLs ni caracteres no alfabeticos y con stemming en ingles.
    - Postcondiciones: Devuelve una cadena limpia con tokens stemizados conservando palabras funcionales.
    """
    """Limpia ruido y aplica stemming conservando stop words (pronombres y conectores)."""
    text = _RE_URL.sub("", str(text).lower())
    limpio = _RE_ESPACIOS.sub(" ", _RE_NO_LETRAS.sub("", text)).strip()
    tokens = word_tokenize(limpio, language="english")
    return " ".join(_STEMMER.stem(w) for w in tokens)

def trocear(texto, chunk_palabras, max_chunks):
    """
    - Precondiciones: chunk_palabras > 0 y max_chunks es None o entero positivo.
    - Entradas: texto (str), chunk_palabras (int) y max_chunks (int o None).
    - Salidas: list[str] con los fragmentos de texto resultantes.
    - Postcondiciones: Devuelve al menos un fragmento y limita a max_chunks mediante muestreo uniforme si se excede.
    """
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


def _vectorizar_por_trozos(textos, firma, prefijo, cache_dir, chunk_palabras, max_chunks, codificar, normalizar=True):
    """
    - Precondiciones: textos es iterable, chunk_palabras > 0 y codificar es invocable sobre una lista de cadenas.
    - Entradas: textos (iterable de str), firma (str), prefijo (str), cache_dir (str),
      chunk_palabras (int), max_chunks (int), codificar (callable) y normalizar (bool).
    - Salidas: np.ndarray 2D con el vector agregado por cada texto de entrada.
    - Postcondiciones: Recupera o calcula y guarda en cache la matriz de vectores por mensaje (normalizada si normalizar=True).
    """
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
    """
    Entradas:
        text_cols: secuencia/Series de textos de longitud N.
        cache_dir: str con el directorio de caché.
        chunk_palabras: int máximo de palabras por fragmento.
        max_chunks: int máximo de fragmentos por mensaje.
    Salidas:
        np.ndarray de forma (N, D_emb) con embeddings L2-normalizados por mensaje.
    Precondición:
        `text_cols` es iterable y `chunk_palabras > 0`.
    Postcondición:
        Devuelve (y guarda en caché) una matriz de embeddings normalizados alineada con `text_cols`.
    """
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


def vectorizar_vader(text_cols, cache_dir="cache"):
    """
    Entradas:
        text_cols: secuencia/Series de textos originales (sin stemming ni borrado de puntuación), longitud N.
        cache_dir: str con el directorio donde guardar/leer caché `.npy` (o None para no cachear).
    Salidas:
        np.ndarray de forma (N, 4) y tipo float32 con [neg, neu, pos, compound] por mensaje.
    Precondición:
        `text_cols` es iterable y `vaderSentiment` está instalado.
    Postcondición:
        Devuelve una matriz 2D `(len(text_cols), 4)` en `float32` alineada fila a fila con `text_cols`.
    """
    textos = [str(t) for t in text_cols]
    ruta_cache = None

    if cache_dir is not None:
        hash_datos = hashlib.md5(("vader_v1|" + "\x1f".join(textos)).encode("utf-8")).hexdigest()[:12]
        os.makedirs(cache_dir, exist_ok=True)
        ruta_cache = os.path.join(cache_dir, f"vader_{hash_datos}.npy")
        if os.path.exists(ruta_cache):
            print(f"Cargando vectores VADER desde caché: {ruta_cache}")
            return np.load(ruta_cache)

    analizador = SentimentIntensityAnalyzer()
    filas = []
    for t in textos:
        s = analizador.polarity_scores(t)
        filas.append([s["neg"], s["neu"], s["pos"], s["compound"]])

    X_vader = np.asarray(filas, dtype=np.float32)
    if ruta_cache is not None:
        np.save(ruta_cache, X_vader)
        print(f"Vectores VADER guardados en {ruta_cache} con forma {X_vader.shape}")
    return X_vader


def vectorizar_embeddings_y_vader(
    text_cols, cache_dir="cache", chunk_palabras=CHUNK_PALABRAS, max_chunks=MAX_CHUNKS
):
    """
    Entradas:
        text_cols: secuencia/Series de textos originales de longitud N.
        cache_dir: str con el directorio de caché.
        chunk_palabras: int máximo de palabras por trozo para embeddings.
        max_chunks: int máximo de trozos por mensaje para embeddings.
    Salidas:
        Tupla `(X_emb, X_vader, X_concat)` donde:
            - X_emb: np.ndarray (N, D_emb)
            - X_vader: np.ndarray (N, 4)
            - X_concat: np.ndarray (N, D_emb + 4)
    Precondición:
        `text_cols` es iterable; `vectorizar_embeddings` y `vectorizar_vader` operan sobre los mismos textos.
    Postcondición:
        Devuelve por separado ambos bloques y su concatenación horizontal lista para clasificadores.
    """
    X_emb = np.asarray(
        vectorizar_embeddings(
            text_cols, cache_dir=cache_dir, chunk_palabras=chunk_palabras, max_chunks=max_chunks
        )
    )
    X_vader = vectorizar_vader(text_cols, cache_dir=cache_dir)
    X_concat = np.hstack([X_emb, X_vader])
    return X_emb, X_vader, X_concat