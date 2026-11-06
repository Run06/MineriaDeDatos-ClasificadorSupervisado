import ast
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_dev_vec, X_test_vec