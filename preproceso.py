import ast
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords


def dividir_mensajes(df, col_post="Post"):
    def parsear(crudo):
        try:
            lista = ast.literal_eval(crudo)                      # "['a', 'b']" -> ['a', 'b']
            return [str(m) for m in lista] if isinstance(lista, (list, tuple)) else [str(lista)]
        except (ValueError, SyntaxError):
            # Lista truncada (cadena sin cerrar): se quitan los corchetes y se parte por "', '"
            s = str(crudo).strip()
            s = re.sub(r"^\[\s*['\"]", "", s)
            s = re.sub(r"['\"]\s*\]\s*$", "", s)
            return re.split(r"['\"]\s*,\s*['\"]", s)

    out = df.copy()
    out[col_post] = out[col_post].apply(parsear)
    out = out.explode(col_post, ignore_index=True)               # una fila por mensaje
    out = out.dropna(subset=[col_post])                          # listas vacías
    out[col_post] = out[col_post].astype(str).str.strip()
    return out[out[col_post] != ""].reset_index(drop=True)       # mensajes en blanco

def tfidf(X_train, X_dev, X_test):
    X_train_limpio = X_train.apply(clean_text)
    X_dev_limpio = X_dev.apply(clean_text)
    X_test_limpio = X_test.apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=5, max_df=0.85,stop_words=None)
    X_train_vec = vectorizer.fit_transform(X_train_limpio)
    X_dev_vec = vectorizer.transform(X_dev_limpio)
    X_test_vec = vectorizer.transform(X_test_limpio)

    return X_train_vec, X_dev_vec, X_test_vec
_STEMMER = SnowballStemmer('english')
_STOP_WORDS = None
def clean_text(text):
    """
    Limpia ruido y unifica raíces (stemming), pero CONSERVA las stop words.
    Vital para análisis psicológico donde los pronombres y conectores forman frases clave.
    """
    text = str(text).lower()
    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Limpiar puntuación (dejamos solo letras y espacios)
    limpio = re.sub(r'[^a-záéíóúñ\s]', '', text)
    limpio = re.sub(r'\s+', ' ', limpio).strip()

    tokens = word_tokenize(limpio, language='english')
    # APLICAMOS STEMMING A TODO, PERO SIN FILTRAR STOP WORDS
    cleaned = [_STEMMER.stem(w) for w in tokens]

    return ' '.join(cleaned)