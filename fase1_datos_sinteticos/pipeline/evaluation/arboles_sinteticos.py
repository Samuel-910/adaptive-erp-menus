"""Generalización a otros menús: árboles sintéticos de distinta amplitud y profundidad (obs. 10).

Para cada árbol (número de grupos G, funciones por grupo I y proporción q de funciones dentro de
submenús, que necesitan un clic más) se simulan 300 usuarios con el mismo generador que el
artículo: nivel de experiencia de la Tabla 2, ocho funciones típicas elegidas al azar entre las del
árbol, perfil Zipf (s = 1), una sesión de entrenamiento de 40 tareas y 40 tareas de evaluación con
números aleatorios comunes. Se mide M1, M2 y M3 con el modelo de navegación del artículo, y M2 también
con el modelo con memoria espacial de memoria_espacial.py (b = 0.15 s/bit, c = 0.5 s).

El predictor y los agentes DQN dependen del vocabulario y de los tamaños de grupo del banco de
pruebas, así que no se aplican aquí. Por las propiedades P1 y P2, con estos pesos ambos se reducen
al orden por frecuencia, de modo que se comparan:
  Static              menú del árbol sin cambios
  Solo barra          barra con las 3 funciones más usadas en el entrenamiento, menú sin cambios
  Reordenado          grupos e ítems ordenados por frecuencia (submenús como bloques), sin barra
  Reordenado + barra  lo anterior con la barra (equivalente a Score-greedy / Proposed)

Salida: salidas/arboles_sinteticos.json
Uso:    python arboles_sinteticos.py
"""
import math
import random
from collections import Counter

import numpy as np

import comun as C

GRUPOS = [5, 9, 15, 25]
ITEMS = [4, 7, 12]
SUBMENU = [0.0, 0.3, 0.6]
N_USUARIOS = 300
B_HH, C_REAP = 0.15, 0.5          # parámetros centrales del modelo con memoria espacial
POLITICAS = ["Static", "Solo barra", "Reordenado", "Reordenado + barra"]


def arbol(G, I, q, rnd):
    grupos, k = [], 0
    for g in range(G):
        rutas = [f"g{g}/f{i}" for i in range(I)]
        n_sub = int(round(q * I))
        sub, directas = rutas[:n_sub], rutas[n_sub:]
        hijos = [{"route": r} for r in directas]
        for j in range(0, len(sub), 4):                      # submenús de hasta 4 funciones
            hijos.append({"id": f"s{g}_{j}", "children": [{"route": r} for r in sub[j:j + 4]]})
        rnd.shuffle(hijos)
        grupos.append({"id": f"g{g}", "children": hijos})
    return grupos


def reordenar(grupos, f, por="sum"):
    v = lambda r: f.get(r, 0)
    out = []
    for gi, g in enumerate(grupos):
        rutas = C.hojas(g)
        orden = sorted(rutas, key=lambda r: (-v(r), rutas.index(r)))
        hijos = C._ordenar_nodos(g["children"], {r: k for k, r in enumerate(orden)})
        agg = sum(v(r) for r in rutas) if por == "sum" else max(v(r) for r in rutas)
        out.append((-agg, gi, {**g, "children": hijos}))
    return [g for _, _, g in sorted(out, key=lambda x: (x[0], x[1]))]


def evaluar(G, I, q):
    rnd = random.Random(f"arbol-{G}-{I}-{q}")
    grupos = arbol(G, I, q, rnd)
    todas = [r for g in grupos for r in C.hojas(g)]
    niveles = list(C.sim.EXPERIENCIA)
    pesos = [C.sim.EXPERIENCIA[e]["p"] for e in niveles]
    res = {p: {"M1": [], "M2": [], "M3": [], "M2_memoria": []} for p in POLITICAS}
    en_barra = []
    for k in range(N_USUARIOS):
        ru = random.Random(f"usuario-{G}-{I}-{q}-{k}")
        tipicas = ru.sample(todas, 8)
        u = {"user_id": f"A{G}-{I}-{q}-{k}", "experiencia": ru.choices(niveles, weights=pesos)[0],
             "perfil": C.sim.perfil_estudio(tipicas, ru)}
        f = Counter(C.sim.tareas_bloque(u["perfil"], 20, ru) + C.sim.tareas_bloque(u["perfil"], 20, ru))
        barra = [r for r, _ in sorted(f.items(), key=lambda x: (-x[1], todas.index(x[0])))[:3]]
        menus = {"Static": {"bar": [], "groups": grupos},
                 "Solo barra": {"bar": barra, "groups": grupos},
                 "Reordenado": {"bar": [], "groups": reordenar(grupos, f, "sum")},
                 "Reordenado + barra": {"bar": barra, "groups": reordenar(grupos, f, "max")}}
        infos = {n: C.ubicaciones(m) for n, m in menus.items()}
        tareas = C.tareas_sesion2(u)
        en_barra.append(np.mean([r in barra for r, _ in tareas]))
        pos = {n: {r: (gi, ii) for gi, g in enumerate(m["groups"]) for ii, r in enumerate(C.hojas(g))}
               for n, m in menus.items()}
        ex = C.sim.EXPERIENCIA[u["experiencia"]]
        for n in POLITICAS:
            a, mem = [], []
            for r, sem in tareas:
                s = C.sorteos_tarea(sem)
                m1, m2, m3 = C.navegar(r, infos[n], u, s)
                a.append((m1, m2, m3))
                # modelo con memoria espacial (memoria_espacial.py) con b = 0.15 s/bit y c = 0.5 s
                d = infos[n][r]
                t_clic = max(0.2, ex["clic"][0] + ex["clic"][1] * s["z_clic"])
                t_vis = d["scan"] * ex["scan"] * math.exp(0.35 * s["z_scan"])
                resto = m2 - (d["clics"] * t_clic + t_vis) * C.FACTOR_FAMILIAR
                if r in menus[n]["bar"]:
                    t_b = t_vis
                elif pos[n][r] != pos["Static"][r]:
                    t_b = t_vis + C_REAP
                else:
                    e = f.get(r, 0) / (f.get(r, 0) + 1)
                    t_b = (1 - e) * t_vis + e * B_HH * math.log2((40 + 8) / (f.get(r, 0) + 1))
                mem.append((d["clics"] * t_clic + t_b) * C.FACTOR_FAMILIAR + resto)
            a = np.array(a)
            for j, m in enumerate(("M1", "M2", "M3")):
                res[n][m].append(a[:, j].mean())
            res[n]["M2_memoria"].append(np.mean(mem))
    sub = sum(1 for g in grupos for c in g["children"] if "route" not in c for _ in C.hojas(c))
    out = {"G": G, "I": I, "q": q, "funciones": len(todas), "prop_submenu": sub / len(todas),
           "tareas_en_barra": float(np.mean(en_barra)),
           "M1_static": float(np.mean(res["Static"]["M1"])), "M2_static": float(np.mean(res["Static"]["M2"])),
           "reduccion": {n: {m: float(100 * (1 - np.mean(res[n][m]) / np.mean(res["Static"][m])))
                             for m in ("M1", "M2", "M2_memoria")} for n in POLITICAS if n != "Static"}}
    return out


def main():
    filas = []
    for G in GRUPOS:
        for I in ITEMS:
            for q in SUBMENU:
                r = evaluar(G, I, q)
                filas.append(r)
                rd = r["reduccion"]
                print(f"G={G:2d} I={I:2d} q={q:.1f} ({r['funciones']:3d} funciones) static M1 {r['M1_static']:.2f} M2 {r['M2_static']:.1f}s | "
                      f"barra M1 -{rd['Solo barra']['M1']:.1f}% M2 -{rd['Solo barra']['M2']:.1f}% | "
                      f"reord M2 -{rd['Reordenado']['M2']:.1f}% | reord+barra M1 -{rd['Reordenado + barra']['M1']:.1f}% M2 -{rd['Reordenado + barra']['M2']:.1f}% | MEMORIA barra {rd['Solo barra']['M2_memoria']:.1f} reord {rd['Reordenado']['M2_memoria']:.1f} reord+barra {rd['Reordenado + barra']['M2_memoria']:.1f}", flush=True)
    C.guardar_json("arboles_sinteticos.json", {"n_usuarios_por_arbol": N_USUARIOS, "resultados": filas})


if __name__ == "__main__":
    main()
