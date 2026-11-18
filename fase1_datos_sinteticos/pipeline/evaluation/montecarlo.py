"""Error estándar de Monte Carlo, umbral de relevancia práctica y tamaño de la simulación (obs. 8).

Lee salidas/rq2_por_usuario.csv (no vuelve a simular) y, para cada comparación, informa:
  - estimando: reducción relativa de la media frente a la referencia (%) y diferencia absoluta
    de medias por tarea;
  - MCSE: error estándar de Monte Carlo de la reducción (desviación estándar bootstrap sobre
    usuarios, 2000 remuestreos) y de la diferencia absoluta (DE de la diferencia pareada / √n);
  - relevancia práctica: reducción >= 5 % y, para el tiempo, diferencia >= 0.1 s por tarea;
  - n necesario para que el semiancho del IC 95 % de la reducción sea <= 1 punto porcentual.

Salida: salidas/montecarlo.json
Uso:    python montecarlo.py
"""
import csv
from collections import defaultdict

import numpy as np

import comun as C

UMBRAL_PCT, UMBRAL_S = 5.0, 0.1
COMPARACIONES = [  # (escenario, política, referencia)
    ("base", "Proposed", "Static"), ("base", "Score-greedy", "Static"), ("base", "Frequency", "Static"),
    ("base", "A2 solo barra", "Static"), ("base", "A1 sin barra", "Static"),
    ("base", "Proposed", "Score-greedy"), ("base", "Proposed", "A2 solo barra"), ("base", "Proposed", "A3 beta=0"),
    ("deriva", "Proposed", "Static"),
]


def main():
    tab = defaultdict(dict)
    for f in csv.DictReader(open(C.SALIDAS / "rq2_por_usuario.csv", encoding="utf-8")):
        if f["subconjunto"] == "all":
            tab[(f["escenario"], f["condicion"])][f["user"]] = {m: float(f[m]) for m in ("M1", "M2", "M3")}
    out = []
    for esc, pol, ref in COMPARACIONES:
        us = sorted(set(tab[(esc, pol)]) & set(tab[(esc, ref)]))
        n = len(us)
        fila = {"escenario": esc, "politica": pol, "referencia": ref, "n": n}
        for m in ("M1", "M2", "M3"):
            x = np.array([tab[(esc, pol)][u][m] for u in us])
            b = np.array([tab[(esc, ref)][u][m] for u in us])
            red = 100 * (1 - x.mean() / b.mean())
            rng = np.random.default_rng(1)
            boots = []
            for _ in range(2000):
                i = rng.integers(0, n, n)
                boots.append(100 * (1 - x[i].mean() / b[i].mean()))
            mcse = float(np.std(boots, ddof=1))
            d = b - x                                    # positivo = la política cuesta menos
            relevante = abs(red) >= UMBRAL_PCT and (m != "M2" or abs(d.mean()) >= UMBRAL_S)
            fila[m] = {"reduccion_pct": float(red), "mcse_pct": mcse,
                       "diferencia_media": float(d.mean()), "mcse_diferencia": float(d.std(ddof=1) / np.sqrt(n)),
                       "relevante": bool(relevante),
                       "n_para_semiancho_1pp": int(np.ceil(n * (1.96 * mcse / 1.0) ** 2))}
        out.append(fila)
        print(f"{esc:6s} {pol:13s} vs {ref:13s} " + "  ".join(
            f"{m} {fila[m]['reduccion_pct']:6.2f}% (MCSE {fila[m]['mcse_pct']:.2f}, Δ {fila[m]['diferencia_media']:+.3f}, "
            f"{'relevante' if fila[m]['relevante'] else 'no relevante'})" for m in ("M1", "M2", "M3")))
    C.guardar_json("montecarlo.json", {"umbral_pct": UMBRAL_PCT, "umbral_s": UMBRAL_S, "comparaciones": out})


if __name__ == "__main__":
    main()
