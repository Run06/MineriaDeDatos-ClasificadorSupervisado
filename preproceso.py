import ast
import re
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Expresiones regulares precompiladas
_RE_CORCHETE_INI = re.compile(r"^\[\s*['\"]")
_RE_CORCHETE_FIN = re.compile(r"['\"]\s*\]\s*$")
_RE_SPLIT_LISTA = re.compile(r"['\"]\s*,\s*['\"]")

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