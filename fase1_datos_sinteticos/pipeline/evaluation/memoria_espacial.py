"""Modelo de evaluación con memoria espacial y costo de reaprendizaje (observación 9 del revisor).

El modelo principal (comun.navegar) aplica búsqueda visual lineal a todos los usuarios. Aquí el
tiempo de búsqueda sigue la transición de búsqueda visual a decisión de Cockburn et al. (2007):

    T_buscar = (1 - e) · T_visual + e · T_decision          si el ítem sigue donde estaba
    T_buscar = T_visual + c_reaprendizaje                    si el ítem cambió de lugar
    T_buscar = T_visual en la barra                          si el ítem está en la barra

  - T_visual: el del modelo principal (ítems escaneados × tiempo por ítem × ruido log-normal);
  - T_decision = b · log2(1 / p): Hick-Hyman, con p la frecuencia del ítem en la sesión de
    entrenamiento del usuario (suavizada: (n+1)/(40+8));
  - e = n / (n + 1): experiencia con el ítem, con n las veces que el usuario lo abrió en la sesión
    de entrenamiento con el menú estático (0 si nunca lo abrió);
  - un ítem "cambió de lugar" si su grupo o su posición dentro del grupo difieren de las del menú
    estático; la barra se añade arriba sin mover nada, así que no invalida la memoria espacial.
Clics, visitas erróneas y su tiempo son los del modelo principal (mismos números aleatorios).

Barre b (0.10, 0.15, 0.20 s/bit) y c_reaprendizaje (0, 0.5, 1.0 s).
Salida: salidas/memoria_espacial.json
Uso:    python memoria_espacial.py
"""
import math
from collections import Counter

import numpy as np

import comun as C
import evaluacion as E

POLITICAS = ["Static", "Frequency", "Score-greedy", "Proposed", "Solo barra", "Sin barra"]
B_HH = [0.10, 0.15, 0.20]
C_REAP = [0.0, 0.5, 1.0]


def posiciones(menu):
    pos = {}
    for gi, g in enumerate(menu["groups"]):
        for ii, r in enumerate(C.hojas(g)):
            pos[r] = (g["id"] if "id" in g else gi, gi, ii)
    return pos


def main():
    usu, seq = C.poblacion(C.SEMILLA_ENTRENAMIENTO)
    net = E.cargar_predictor(C.leer_json("rq1.json")["seleccionado"])
    envs, ags = E.cargar_agentes()
    f = {u: dict(Counter(s)) for u, s in seq.items()}
    prior = {}
    for rol in C.ROLES:
        c = Counter(r for u in usu if u["rol"] == rol for r in seq[u["user_id"]])
        tot = sum(c.values()); prior[rol] = {r: k / tot for r, k in c.items()}

    combos = [(b, c) for b in B_HH for c in C_REAP]
    m2 = {k: {p: [] for p in POLITICAS} for k in combos}
    m2_lineal = {p: [] for p in POLITICAS}
    movidas = {p: [] for p in POLITICAS}
    for u in usu:
        uid, rol = u["user_id"], u["rol"]
        p = E.probs(net, seq[uid]); P = {r: float(p[C.R2I[r]]) for r in C.RUTAS}
        S = C.score(f[uid], P, prior[rol])
        prop = E.menu_propuesto(rol, f[uid], S, envs, ags)
        menus = {"Static": C.menu_estatico(rol),
                 "Frequency": C.menu_adaptado(rol, C.compacto_ordenado(rol, f[uid], barra=False, grupos_por="sum")),
                 "Score-greedy": C.menu_adaptado(rol, C.compacto_ordenado(rol, S)),
                 "Proposed": C.menu_adaptado(rol, prop),
                 "Solo barra": C.menu_adaptado(rol, {"bar": C.compacto_ordenado(rol, S)["bar"], "groups": []}),
                 "Sin barra": C.menu_adaptado(rol, E.menu_propuesto(rol, f[uid], S, envs, ags, barra=False))}
        infos = {n: C.ubicaciones(m) for n, m in menus.items()}
        pos_st = posiciones(menus["Static"])
        pos = {n: posiciones(m) for n, m in menus.items()}
        ex = C.sim.EXPERIENCIA[u["experiencia"]]
        n_tot = len(seq[uid])
        tareas = C.tareas_sesion2(u)
        acc = {k: {n: [] for n in POLITICAS} for k in combos}
        for n in POLITICAS:
            lin, mov = [], []
            for r, sem in tareas:
                s = C.sorteos_tarea(sem)
                _, t_nav, _ = C.navegar(r, infos[n], u, s)
                d = infos[n][r]
                t_clic = max(0.2, ex["clic"][0] + ex["clic"][1] * s["z_clic"])
                t_vis = d["scan"] * ex["scan"] * math.exp(0.35 * s["z_scan"])
                t_resto = t_nav - (d["clics"] * t_clic + t_vis) * C.FACTOR_FAMILIAR   # visitas erróneas
                en_barra = r in menus[n]["bar"]
                movido = (not en_barra) and pos[n][r][1:] != pos_st[r][1:]
                nr = f[uid].get(r, 0)
                e = nr / (nr + 1)
                pr = (nr + 1) / (n_tot + 8)
                lin.append(t_nav); mov.append(movido)
                for (b, c) in combos:
                    if en_barra:
                        t_b = t_vis
                    elif movido:
                        t_b = t_vis + c
                    else:
                        t_b = (1 - e) * t_vis + e * b * math.log2(1 / pr)
                    acc[(b, c)][n].append((d["clics"] * t_clic + t_b) * C.FACTOR_FAMILIAR + t_resto)
            m2_lineal[n].append(np.mean(lin)); movidas[n].append(np.mean(mov))
            for k in combos:
                m2[k][n].append(np.mean(acc[k][n]))

    red = lambda x, base: float(100 * (1 - np.mean(x) / np.mean(base)))
    out = {"modelo_lineal": {n: red(m2_lineal[n], m2_lineal["Static"]) for n in POLITICAS if n != "Static"},
           "tareas_con_item_movido": {n: float(np.mean(movidas[n])) for n in POLITICAS},
           "M2_static_s": {f"b={b}": float(np.mean(m2[(b, 0.0)]["Static"])) for b in B_HH},
           "memoria_espacial": [{"b_hh": b, "c_reaprendizaje": c,
                                 "M2_static_s": float(np.mean(m2[(b, c)]["Static"])),
                                 "reduccion_M2_pct": {n: red(m2[(b, c)][n], m2[(b, c)]["Static"]) for n in POLITICAS if n != "Static"}}
                                for (b, c) in combos]}
    print("lineal:", {k: round(v, 1) for k, v in out["modelo_lineal"].items()})
    print("tareas con ítem movido:", {k: round(v, 2) for k, v in out["tareas_con_item_movido"].items()})
    for x in out["memoria_espacial"]:
        print(f"b={x['b_hh']:.2f} c={x['c_reaprendizaje']:.1f} Static {x['M2_static_s']:.2f}s ",
              {k: round(v, 1) for k, v in x["reduccion_M2_pct"].items()})
    C.guardar_json("memoria_espacial.json", out)


if __name__ == "__main__":
    main()
