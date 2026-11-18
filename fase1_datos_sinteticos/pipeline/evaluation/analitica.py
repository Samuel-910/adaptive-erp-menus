"""Aproximación analítica de la reducción de clics, contrastada con la simulación (obs. 4).

La barra ahorra (d - 1) clics en cada tarea cuyo destino está en ella (d = clics para abrir
la función en el menú estático, 2 o 3) y no cambia las demás tareas, así que

    ΔM1 ≈ q · (d̄ - 1) / M1_static

donde q es la proporción de tareas servidas por la barra y d̄ la profundidad media de las
funciones típicas. Se calculan dos versiones:

  - q_perfil: barra ideal con las k funciones más probables del perfil. Con la regla de no
    repetir la tarea anterior, la cadena de tareas tiene distribución estacionaria
    π_i ∝ w_i (1 - w_i), así que q_perfil = Σ_{i<=k} π_i (fórmula cerrada).
  - q_simulado: proporción medida en la simulación (la barra se estima con las 40 tareas de
    la sesión de entrenamiento, por eso q_simulado < q_perfil).

M1_static se toma de la simulación (incluye las visitas erróneas). Entrada: salidas/
sensibilidad_*.json. Salida: salidas/aproximacion_analitica.json.
Uso: python analitica.py
"""
import json

import numpy as np

import comun as C
import sensibilidad as SE

PARAMS = ["zipf", "funciones", "barra", "t_clic", "t_scan", "error"]


def profundidad_media(n):
    """Clics medios para abrir una función típica en el menú estático (promedio de los 4 roles)."""
    ds = []
    for rol in C.ROLES:
        info = C.ubicaciones(C.menu_estatico(rol))
        ds.append(np.mean([info[r]["clics"] for r in SE.funciones_tipicas(rol, n)]))
    return float(np.mean(ds))


def q_perfil(s, n, k):
    w = np.array([1 / (i + 1) ** s for i in range(n)])
    w /= w.sum()
    pi = w * (1 - w)
    pi /= pi.sum()
    return float(pi[:k].sum())


def main():
    filas = []
    for param in PARAMS:
        d = C.leer_json(f"sensibilidad_{param}.json")
        clave = d["parametro"]
        for r in d["resultados"]:
            dbar = profundidad_media(r["n_funciones"])
            m1s = r["politicas"]["Static"]["M1"]["media"]
            qp = q_perfil(r["s"], r["n_funciones"], r["k_barra"])
            qs = r["tareas_en_barra"]
            fila = {"parametro": param, "valor": r[clave], "profundidad_media": dbar, "M1_static": m1s,
                    "q_perfil": qp, "q_simulado": qs,
                    "dM1_analitica_q_perfil": 100 * qp * (dbar - 1) / m1s,
                    "dM1_analitica_q_simulado": 100 * qs * (dbar - 1) / m1s,
                    "dM1_simulacion": r["politicas"]["Proposed"]["M1"]["reduccion_pct"]}
            filas.append(fila)
            print(f"{param:9s} {r[clave]!s:5s} q_perfil {qp:.3f} q_sim {qs:.3f} | ΔM1 analítica "
                  f"{fila['dM1_analitica_q_perfil']:5.1f} / {fila['dM1_analitica_q_simulado']:5.1f}  simulación {fila['dM1_simulacion']:5.1f}")
    sim = np.array([f["dM1_simulacion"] for f in filas])
    resumen = {}
    for k in ("dM1_analitica_q_perfil", "dM1_analitica_q_simulado"):
        a = np.array([f[k] for f in filas])
        resumen[k] = {"error_medio": float(np.mean(a - sim)), "error_abs_max": float(np.max(np.abs(a - sim))),
                      "r": float(np.corrcoef(a, sim)[0, 1])}
        print(k, {x: round(v, 3) for x, v in resumen[k].items()})
    C.guardar_json("aproximacion_analitica.json", {"resumen": resumen, "filas": filas})


if __name__ == "__main__":
    main()
