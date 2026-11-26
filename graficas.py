"""Gráficas reproducibles para el póster; no requiere entorno gráfico."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORES = {'Solo pseudo': '#007F86', 'Con fallback': '#E38A32', 'Etiquetas originales': '#5268A8'}
ORDEN = ['1', '3', '5', '10', '15', 'Todos']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'figure.dpi': 120})


def guardar(fig, carpeta, nombre):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(carpeta / f'{nombre}.pdf', bbox_inches='tight')
    fig.savefig(carpeta / f'{nombre}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def caracterizar(particiones, col_texto, col_label, salida):
    filas, resumen = [], []
    fig, ax = plt.subplots(figsize=(7, 4))
    largos = []
    for nombre, usuarios, mensajes in particiones:
        for unidad, df in [('Usuario', usuarios), ('Mensaje', mensajes)]:
            for clase, n in df[col_label].value_counts().items():
                filas.append(dict(particion=nombre, unidad=unidad, clase=str(clase), cantidad=n))
        longitudes = mensajes[col_texto].str.split().str.len()
        largos.append(longitudes.to_numpy())
        resumen.append(dict(particion=nombre, usuarios=len(usuarios), usuarios_con_mensajes=mensajes.user_id.nunique(),
                            mensajes=len(mensajes), longitud_media=longitudes.mean(),
                            longitud_std=longitudes.std(), longitud_mediana=longitudes.median(),
                            vocabulario_bruto=len(set(' '.join(mensajes[col_texto].str.lower()).split()))))
    dist = pd.DataFrame(filas)
    dist.to_csv(salida / 'distribucion.csv', index=False)
    pd.DataFrame(resumen).to_csv(salida / 'dataset_resumen.csv', index=False)
    ax.boxplot(largos, showfliers=False)
    ax.set_xticks(range(1, len(particiones)+1), [p[0] for p in particiones])
    ax.set_ylabel('Palabras por mensaje (sin atípicos dibujados)')
    guardar(fig, salida, 'longitudes')
    for unidad in ['Usuario', 'Mensaje']:
        tabla = dist[dist.unidad == unidad].pivot_table(index='clase', columns='particion', values='cantidad', fill_value=0)
        tabla = tabla.reindex(columns=['Train', 'Dev', 'Test'], fill_value=0)
        ax = tabla.plot.bar(figsize=(7, 4), color=['#007F86', '#E38A32', '#5268A8'], rot=0)
        ax.set(xlabel='Clase', ylabel=f'N.º de {unidad.lower()}s')
        guardar(ax.figure, salida, f'distribucion_{unidad.lower()}')


def grafica_recursos(rec, salida):
    nombres = rec.fase.str.replace('Entrenamiento ', 'Entr. ', regex=False).str.replace('Predicción dev+test ', 'Pred. ', regex=False).str.replace('Agregación y métricas ', 'Eval. ', regex=False)
    for col, titulo, archivo in [('segundos', 'Tiempo (s)', 'tiempo_fases'), ('ram_pico_mib', 'Pico RSS (MiB)', 'ram_fases')]:
        fig, ax = plt.subplots(figsize=(8, max(3.5, .39 * len(rec) + 1)))
        y = np.arange(len(rec))
        ax.barh(y, rec[col], color='#007F86')
        ax.set_yticks(y, nombres)
        ax.invert_yaxis()
        ax.set_xlabel(titulo)
        ax.bar_label(ax.containers[0], fmt='%.1f', padding=3, fontsize=9)
        ax.margins(x=.17)
        guardar(fig, salida, archivo)


def graficas_evaluacion(met, clases, matrices, salida):
    # Curvas exploratorias. La ventana se selecciona únicamente con Dev.
    fig, ax = plt.subplots(figsize=(7.7, 4.3))
    for (estrategia, particion), datos in met.groupby(['estrategia', 'particion'], sort=False):
        datos = datos.set_index('k').reindex(ORDEN).dropna(subset=['f1_macro'])
        ax.plot([ORDEN.index(k) for k in datos.index], datos.f1_macro, marker='o',
                color=COLORES.get(estrategia), linestyle='-' if particion == 'Dev' else '--', label=f'{estrategia} · {particion}')
    ax.set_xticks(range(len(ORDEN)), ORDEN)
    ax.set(xlabel='Máximo de mensajes por usuario (k)', ylabel='F1 macro', ylim=(0, 1.04))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=.25)
    guardar(fig, salida, 'f1_ventana')
    elegidos = met[(met.particion == 'Test') & met.k_elegido_dev]
    ax = elegidos.set_index('estrategia')[['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']].plot.bar(figsize=(8, 4.4), rot=0)
    ax.set(xlabel='', ylabel='Puntuación (Test, k elegido en Dev)', ylim=(0, 1.05))
    ax.legend(['Precisión macro', 'Recall macro', 'F1 macro', 'Accuracy'], fontsize=8)
    guardar(ax.figure, salida, 'metricas_test')
    for estrategia, datos in clases[(clases.particion == 'Test') & clases.k_elegido_dev].groupby('estrategia', sort=False):
        slug = estrategia.lower().replace(' ', '_')
        k = datos.k.iloc[0]
        ax = datos.set_index('clase')[['precision', 'recall', 'f1']].plot.bar(figsize=(6.5, 4), rot=0, color=['#5268A8', '#E38A32', '#007F86'])
        ax.set(xlabel='Clase', ylabel='Puntuación', ylim=(0, 1.05), title=f'{estrategia} · Test (k={k})')
        guardar(ax.figure, salida, f'metricas_clase_{slug}')
        sub = matrices[(matrices.estrategia == estrategia) & (matrices.particion == 'Test') & matrices.k_elegido_dev]
        mat = sub.pivot(index='real', columns='predicha', values='cantidad').fillna(0)
        mat = mat.reindex(index=datos.clase, columns=datos.clase)
        counts = mat.to_numpy(dtype=float)
        tot = counts.sum(axis=1, keepdims=True)
        norm = np.divide(counts, tot, out=np.zeros_like(counts), where=tot != 0)
        fig, ax = plt.subplots(figsize=(5.8, 5))
        im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i, j in np.ndindex(norm.shape):
            ax.text(j, i, f'{int(counts[i,j])}\n{norm[i,j]:.0%}', ha='center', va='center', color='white' if norm[i,j] > .55 else 'black', fontsize=9)
        ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=25, ha='right')
        ax.set_yticks(range(len(mat.index)), mat.index)
        ax.set(xlabel='Clase predicha', ylabel='Clase real', title=f'{estrategia} · Test (k={k})')
        fig.colorbar(im, ax=ax, fraction=.045, label='Proporción por clase real')
        guardar(fig, salida, f'confusion_{slug}')


def graficas_cascada(hist, n_original, umbrales, salida):
    h = pd.DataFrame(hist, columns=['umbral','iteracion','nuevos','pendientes','confianza_media','cambios'])
    h.to_csv(salida / 'cascada_iteraciones.csv', index=False)
    agg = h.groupby('umbral')[['nuevos', 'cambios']].sum().reindex(umbrales, fill_value=0)
    agg['acumulados'] = agg.nuevos.cumsum()
    agg['pendientes'] = n_original - agg.acumulados
    agg['cobertura'] = agg.acumulados / max(n_original, 1)
    agg.to_csv(salida / 'cascada_umbrales.csv')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(agg))
    ax.bar(x, agg.nuevos, color='#007F86', label='Nuevas pseudoetiquetas')
    ax.set_xticks(x, [f'{u:.2f}' for u in umbrales])
    ax.set(xlabel='Umbral de confianza (orden de la cascada)', ylabel='Mensajes incorporados')
    ax2 = ax.twinx()
    ax2.plot(x, 100*agg.cobertura, 'o-', color='#E38A32', label='Cobertura acumulada')
    ax2.set(ylabel='Cobertura acumulada (%)', ylim=(0, 105))
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    guardar(fig, salida, 'cascada')


def comparar(raiz, modo):
    raiz = Path(raiz)
    runs = []
    for p in sorted(raiz.glob('*/config.json')):
        config = json.loads(p.read_text(encoding='utf-8'))
        if config['modo_cache'] == modo and (p.parent / 'metricas.csv').exists():
            runs.append((p.parent, config))
    if not runs:
        raise SystemExit(f'No hay ejecuciones completas para modo {modo}.')
    firmas = {c['firma_datos'] for _, c in runs}
    if len(firmas) != 1:
        raise SystemExit('Hay datasets/semillas/columnas distintos; usa una carpeta de resultados por dataset.')
    # Última repetición por configuración, no la que mejor puntúa.
    ultimos = {(c['vectorizer'], c['model'], c['baseline_original']): (p,c) for p,c in runs}
    met, recursos = [], []
    for p, c in ultimos.values():
        m = pd.read_csv(p / 'metricas.csv', dtype={'k': str})
        m['vectorizador'], m['modelo'] = c['vectorizer'], c['model']
        m['ejecucion'] = p.name
        met.append(m)
        r = pd.read_csv(p / 'recursos.csv')
        recursos.append(dict(vectorizador=c['vectorizer'], modelo=c['model'], ejecucion=p.name,
                             baseline_original=c['baseline_original'], segundos=c['segundos_pipeline'],
                             ram_pico_mib=r.ram_pico_mib.max()))
    met = pd.concat(met, ignore_index=True)
    recursos = pd.DataFrame(recursos)
    salida = raiz / f'comparacion_{modo}'
    salida.mkdir(exist_ok=True)
    met.to_csv(salida / 'metricas_comparadas.csv', index=False)
    recursos.to_csv(salida / 'recursos_comparados.csv', index=False)
    # Para evitar mezclar duplicados, distinguir ejecuciones con/sin control original.
    met = met.merge(recursos[['ejecucion','baseline_original']], on='ejecucion')
    for baseline, m in met.groupby('baseline_original'):
        sufijo = 'con_control' if baseline else 'sin_control'
        test = m[(m.particion == 'Test') & m.k_elegido_dev].copy()
        test['configuracion'] = test.vectorizador + ' / ' + test.modelo
        tabla = test.pivot(index='configuracion', columns='estrategia', values='f1_macro')
        ax = tabla.plot.bar(figsize=(9, 4.6), rot=20, color=[COLORES.get(c, '#888888') for c in tabla.columns])
        ax.set(xlabel='', ylabel='F1 macro Test (k elegido en Dev)', ylim=(0, 1.05))
        ax.legend(fontsize=8)
        guardar(ax.figure, salida, f'comparacion_f1_{sufijo}')
        r = recursos[recursos.baseline_original == baseline].copy()
        r['configuracion'] = r.vectorizador + ' / ' + r.modelo
        for col, ylabel in [('segundos', 'Tiempo del pipeline (s)'), ('ram_pico_mib', 'Pico RSS (MiB)')]:
            ax = r.set_index('configuracion')[col].plot.bar(figsize=(8, 4.4), rot=20, color='#007F86')
            ax.set(xlabel='', ylabel=ylabel)
            guardar(ax.figure, salida, f'comparacion_{col}_{sufijo}')
    print(f'Comparación guardada en {salida}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultados', default='resultados')
    parser.add_argument('--modo-cache', choices=['frio', 'normal'], default='frio')
    args = parser.parse_args()
    comparar(args.resultados, args.modo_cache)
