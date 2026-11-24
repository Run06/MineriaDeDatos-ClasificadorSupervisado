import os
import threading
import time
from contextlib import contextmanager

import psutil

PROCESO = psutil.Process(os.getpid())
RESUMEN = []  # (fase, segundos, ram_inicio, ram_fin, ram_pico)


def ram_mb():
    """RAM residente (RSS) actual del proceso, en MB."""
    return PROCESO.memory_info().rss / 1024 ** 2


@contextmanager
def medir(nombre):
    """
    Mide tiempo y RAM de una fase. El pico se obtiene muestreando el RSS
    del proceso cada 50 ms desde un hilo auxiliar.
    """
    parar = threading.Event()
    pico = [ram_mb()]

    def muestrear():
        while not parar.is_set():
            pico[0] = max(pico[0], ram_mb())
            parar.wait(0.05)

    hilo = threading.Thread(target=muestrear, daemon=True)
    ram_ini = ram_mb()
    t0 = time.perf_counter()
    hilo.start()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        parar.set()
        hilo.join()
        ram_fin = ram_mb()
        pico[0] = max(pico[0], ram_fin)
        RESUMEN.append((nombre, dt, ram_ini, ram_fin, pico[0]))
        print(f"[MEDIDA] {nombre}: {dt:.1f} s | RAM {ram_ini:.0f} -> {ram_fin:.0f} MB | pico {pico[0]:.0f} MB")


def imprimir_resumen(t_total):
    print("\n" + "=" * 78)
    print("RESUMEN DE TIEMPO Y MEMORIA")
    print("=" * 78)
    print(f"{'Fase':<46}{'Tiempo (s)':>12}{'RAM fin (MB)':>14}{'Pico (MB)':>10}")
    print("-" * 78)
    for nombre, dt, _, ram_fin, pico in RESUMEN:
        print(f"{nombre:<46}{dt:>12.1f}{ram_fin:>14.0f}{pico:>10.0f}")
    print("-" * 78)
    print(f"{'TOTAL':<46}{t_total:>12.1f}")
    pico_global = max(r[4] for r in RESUMEN)
    print(f"Pico de RAM (muestreado): {pico_global:.0f} MB")
    pico_so = getattr(PROCESO.memory_info(), "peak_wset", None)  # solo Windows
    if pico_so is not None:
        print(f"Pico de RAM (Windows, peak_wset): {pico_so / 1024 ** 2:.0f} MB")
    print("=" * 78)