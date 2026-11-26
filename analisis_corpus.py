import pandas as pd
import matplotlib.pyplot as plt
from preproceso import dividir_mensajes

# Carga del corpus original
df = pd.read_csv("500_Reddit_users_posts_labels.csv")

numero_usuarios = len(df)

# Separación de las publicaciones de cada usuario
df_mensajes = dividir_mensajes(df, col_post="Post")

# Extracción de los mensajes individuales
textos = df_mensajes["Post"]

# Tokenización por espacios y conversión a minúsculas
tokens = textos.str.lower().str.split()

# Número de palabras por mensaje individual
longitudes = tokens.apply(len)

# Construcción del vocabulario global
vocabulario = set()

for mensaje in tokens:
    vocabulario.update(mensaje)

# Estadísticas del corpus
estadisticas = {
    "Número de usuarios": numero_usuarios,
    "Número de mensajes": len(df_mensajes),
    "Número total de palabras": longitudes.sum(),
    "Tamaño del vocabulario": len(vocabulario),
    "Media de mensajes por usuario": len(df_mensajes) / numero_usuarios,
    "Media de palabras por mensaje": longitudes.mean(),
    "Desviación estándar": longitudes.std(),
    "Mediana": longitudes.median(),
    "Longitud mínima": longitudes.min(),
    "Longitud máxima": longitudes.max(),
    "Primer cuartil": longitudes.quantile(0.25),
    "Tercer cuartil": longitudes.quantile(0.75)
}

print("\n===== ESTADÍSTICAS DEL CORPUS =====")

for nombre, valor in estadisticas.items():
    print(f"{nombre}: {valor:.2f}")

# Exportación de estadísticas
pd.DataFrame([estadisticas]).to_csv(
    "estadisticas_corpus.csv", index=False
)

# Generación del boxplot
fig, ax = plt.subplots(figsize=(8, 3))

ax.boxplot(
    longitudes,
    orientation="horizontal",
    patch_artist=True,
    showmeans=True,
    showfliers=False,
    boxprops=dict(
        facecolor="#5066A2",
        alpha=0.7
    ),
    medianprops=dict(
        color="black",
        linewidth=2
    ),
    meanprops=dict(
        marker="D",
        markerfacecolor="#E68A32",
        markeredgecolor="black"
    )
)

ax.set_title("Message length distribution")
ax.set_xlabel("Number of words per message")
ax.set_yticks([])
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()

# Exportación de gráficas
plt.savefig("longitud_corpus.pdf", bbox_inches="tight")
plt.savefig("longitud_corpus.png", dpi=300, bbox_inches="tight")

plt.show()