"""Sensibilidad del simulador al exponente Zipf s del perfil de uso (observación 4 del revisor).

Para cada s se regenera la población de 600 usuarios (semilla 42) con el perfil
w_i ∝ 1/i^s sobre las ocho funciones típicas del rol (el artículo usa s = 1), se calculan
los menús con los modelos YA entrenados (no se reentrena nada) y se miden M1, M2 y M3 en
las 40 tareas de evaluación (semilla 2026), igual que en evaluacion.py.

Políticas: Static, Frequency, Score-greedy, Proposed y "solo barra" (variante ii de la ablación).
También se informa la proporción de tareas cuyo destino está en la barra de Proposed.

Salida: salidas/sensibilidad_zipf.json
Uso:    python sensibilidad_zipf.py [--s 0.5 0.75 1 1.25 1.5 1.75 2]
"""
import argparse
import time
from collections import Counter

import numpy as np

import comun as C
import evaluacion as E

POLITICAS = ["Static", "Frequency", "Score-greedy", "Proposed", "Solo barra"]


def perfil_zipf(s):
    def perfil_estudio(tareas_rol, rnd):
        rutas = tareas_rol[:]
        rnd.shuffle(rutas)
        w = [1 / (i + 1) ** s for i in range(len(rutas))]
        tot = sum(w)
        return {r: x / tot for r, x in zip(rutas, w)}
    return perfil_estudio


def reduccion_ic(base, pol, n=2000, semilla=0):
    """Reducción relativa de la media (%) frente a Static, con IC bootstrap 95 % pareado por usuario."""
    base, pol = np.asarray(base), np.asarray(pol)
    red = 100 * (1 - pol.mean() / base.mean())
    rng = np.random.default_rng(semilla)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(base), len(base))
        b.append(100 * (1 - pol[i].mean() / base[i].mean()))
    return float(red), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def evaluar(s, net, envs, ags):
    original = C.sim.perfil_estudio
    C.sim.perfil_estudio = perfil_zipf(s)
    try:
        usu, seq = C.poblacion(C.SEMILLA_ENTRENAMIENTO)
    finally:
        C.sim.perfil_estudio = original

    f = {u: dict(Counter(x)) for u, x in seq.items()}
    prior = {}
    for rol in C.ROLES:
        c = Counter(r for u in usu if u["rol"] == rol for r in seq[u["user_id"]])
        tot = sum(c.values())
        prior[rol] = {r: k / tot for r, k in c.items()}

    res = {p: {"M1": [], "M2": [], "M3": []} for p in POLITICAS}
    en_barra = []
    for u in usu:
        uid, rol = u["user_id"], u["rol"]
        p = E.probs(net, seq[uid])
        P = {r: float(p[C.R2I[r]]) for r in C.RUTAS}
        S = C.score(f[uid], P, prior[rol])
        propuesto = E.menu_propuesto(rol, f[uid], S, envs, ags)
        menus = {
            "Static": C.menu_estatico(rol),
            "Frequency": C.menu_adaptado(rol, C.compacto_ordenado(rol, f[uid], barra=False, grupos_por="sum")),
            "Score-greedy": C.menu_adaptado(rol, C.compacto_ordenado(rol, S)),
            "Proposed": C.menu_adaptado(rol, propuesto),
            "Solo barra": C.menu_adaptado(rol, {"bar": C.compacto_ordenado(rol, S)["bar"], "groups": []}),
        }
        infos = {n: C.ubicaciones(m) for n, m in menus.items()}
        tareas = C.tareas_sesion2(u)
        barra = set(menus["Proposed"]["bar"])
        en_barra.append(np.mean([r in barra for r, _ in tareas]))
        for n in POLITICAS:
            a = np.array([C.navegar(r, infos[n], u, C.sorteos_tarea(sem)) for r, sem in tareas])
            for j, m in enumerate(("M1", "M2", "M3")):
                res[n][m].append(a[:, j].mean())

    out = {"s": s, "n_usuarios": len(usu), "tareas_en_barra": float(np.mean(en_barra)), "politicas": {}}
    for n in POLITICAS:
        d = {}
        for m in ("M1", "M2", "M3"):
            d[m] = {"media": float(np.mean(res[n][m]))}
            if n != "Static":
                red, lo, hi = reduccion_ic(res["Static"][m], res[n][m])
                d[m].update({"reduccion_pct": red, "ic95": [lo, hi]})
        out["politicas"][n] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    args = ap.parse_args()

    sel = C.leer_json("rq1.json")["seleccionado"]
    net = E.cargar_predictor(sel)
    envs, ags = E.cargar_agentes()
    filas = []
    for s in args.s:
        t0 = time.time()
        r = evaluar(s, net, envs, ags)
        filas.append(r)
        pp = r["politicas"]
        print(f"s={s:4.2f}  barra={r['tareas_en_barra']:.3f}  "
              f"Proposed M1 -{pp['Proposed']['M1']['reduccion_pct']:.1f}%  M2 -{pp['Proposed']['M2']['reduccion_pct']:.1f}%  "
              f"Solo barra M1 -{pp['Solo barra']['M1']['reduccion_pct']:.1f}%  ({time.time() - t0:.0f} s)", flush=True)
    C.guardar_json("sensibilidad_zipf.json", {"modelo": sel, "resultados": filas})
    print("[OK]", C.SALIDAS / "sensibilidad_zipf.json")


if __name__ == "__main__":
    main()
