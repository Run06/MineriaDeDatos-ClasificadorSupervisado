import ast
import re
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Expresiones regulares precompiladas
_RE_CORCHETE_INI = re.compile(r"^\[\s*['\"]")
_RE_CORCHETE_FIN = re.compile(r"['\"]\s*\]\s*$")
_RE_SPLIT_LISTA = re.compile(r"['\"]\s*,\s*['\"]")

def dividir_mensajes(df, col_post="Post"):
    """
    Expande las listas de mensajes de cada usuario en filas individuales, conservando
    el identificador original del usuario y el orden cronológico de sus publicaciones.
    Precondición:
        - `df` debe ser un `pd.DataFrame` válido que contenga la columna `col_post`.
        - El índice de `df` debe identificar a cada usuario de forma consistente.
    Entradas:
        df (pd.DataFrame): DataFrame a nivel de usuario con los historiales de mensajes.
        col_post (str): Nombre de la columna que contiene los mensajes (lista serializada
            como texto o cadena simple).
    Salidas:
        pd.DataFrame: Nuevo DataFrame expandido con una fila por mensaje no vacío, la
        columna `"user_id"` con el índice original del usuario y el índice reiniciado.
    Postcondición:
        No modifica `df` original; elimina mensajes nulos o vacíos tras hacer `strip()`
        y garantiza que todos los valores de `col_post` sean cadenas no vacías.
    """
    def parsear(crudo):
        """
        Convierte el texto crudo de una celda de mensajes en una lista de cadenas.
        Precondición:
            `crudo` debe ser un objeto evaluable como literal de Python o convertible a `str`.
        Entradas:
            crudo (object): Valor crudo de la columna `col_post` para un usuario.
        Salidas:
            list[str]: Lista con los mensajes individuales extraídos como cadenas.
        Postcondición:
            Si `ast.literal_eval` falla por `ValueError` o `SyntaxError`, aplica una
            separación de respaldo con expresiones regulares devolviendo siempre una `list[str]`.
        """
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