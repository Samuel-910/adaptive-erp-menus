"""Sensibilidad del simulador y del menú (observación 4 del revisor).

Varía un parámetro a la vez y deja los demás como en el artículo (s = 1, 8 funciones
típicas por rol, barra de 3):

  zipf       exponente s del perfil de uso w_i ∝ 1/i^s            (0.5 … 2)
  funciones  número de funciones típicas por rol                   (5 … 15)
  barra      número de funciones en la barra de sugerencias        (1 … 5)
  t_clic     factor sobre el tiempo por clic de la Tabla 2 (media y d.e.)   (0.5 … 1.5)
  t_scan     factor sobre el tiempo por ítem escaneado de la Tabla 2        (0.5 … 1.5)
  error      factor sobre la probabilidad de visita errónea de la Tabla 2   (0 … 2)

Los factores de la Tabla 2 se aplican a los tres niveles de experiencia, tanto al generar la
población como al medir las tareas de evaluación.

Para cada valor se regenera la población de 600 usuarios (semilla 42), se calculan los
menús con los modelos YA entrenados (no se reentrena nada) y se miden M1, M2 y M3 en las 40
tareas de evaluación (semilla 2026), igual que en evaluacion.py.

Funciones típicas: con n < 8 se toman las n primeras de la lista del rol; con n > 8 se
agregan funciones autorizadas del rol en un orden aleatorio fijo (semilla por rol).

Políticas: Static, Frequency, Score-greedy, Proposed y "solo barra" (variante ii de la
ablación). También se informa la proporción de tareas cuyo destino está en la barra.

Salidas: salidas/sensibilidad_<parametro>.json
Uso:     python sensibilidad.py                 (todos los parámetros)
         python sensibilidad.py --param barra
"""
import argparse
import random
import time
from collections import Counter

import numpy as np

import comun as C
import evaluacion as E

POLITICAS = ["Static", "Frequency", "Score-greedy", "Proposed", "Solo barra"]
BASE = {"s": 1.0, "n_funciones": 8, "k_barra": 3, "f_clic": 1.0, "f_scan": 1.0, "f_error": 1.0}
BARRIDOS = {
    "zipf": ("s", [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
    "funciones": ("n_funciones", [5, 6, 7, 8, 10, 12, 15]),
    "barra": ("k_barra", [1, 2, 3, 4, 5]),
    "t_clic": ("f_clic", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "t_scan": ("f_scan", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "error": ("f_error", [0.0, 0.5, 1.0, 1.5, 2.0]),
}
_MENU = C.sim.MenuERP(C.sim.CONFIG_ERP)


def funciones_tipicas(rol, n):
    tipicas = list(_MENU.tareas[rol])
    if n <= len(tipicas):
        return tipicas[:n]
    extra = [r for r in C.rutas_rol(rol) if r not in tipicas]
    random.Random(f"extra-{rol}").shuffle(extra)
    return tipicas + extra[:n - len(tipicas)]


def perfil_parcheado(s, n):
    rol_de = {tuple(v): rol for rol, v in _MENU.tareas.items()}

    def perfil_estudio(tareas_rol, rnd):
        rutas = funciones_tipicas(rol_de[tuple(tareas_rol)], n)
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


def evaluar(net, envs, ags, f_clic=1.0, f_scan=1.0, f_error=1.0, **kw):
    original = C.sim.EXPERIENCIA
    C.sim.EXPERIENCIA = {k: {**v, "clic": (v["clic"][0] * f_clic, v["clic"][1] * f_clic),
                             "scan": v["scan"] * f_scan, "error": v["error"] * f_error}
                         for k, v in original.items()}
    try:
        out = _evaluar(net, envs, ags, **kw)
    finally:
        C.sim.EXPERIENCIA = original
    return {**out, "f_clic": f_clic, "f_scan": f_scan, "f_error": f_error}


def _evaluar(net, envs, ags, s=1.0, n_funciones=8, k_barra=3):
    original = C.sim.perfil_estudio
    C.sim.perfil_estudio = perfil_parcheado(s, n_funciones)
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
        barra = [r for r in sorted(C.rutas_rol(rol), key=lambda r: (-S[r], C.R2I[r]))[:k_barra] if S[r] > 0]
        menus = {
            "Static": C.menu_estatico(rol),
            "Frequency": C.menu_adaptado(rol, C.compacto_ordenado(rol, f[uid], barra=False, grupos_por="sum")),
            "Score-greedy": C.menu_adaptado(rol, C.compacto_ordenado(rol, S)),
            "Proposed": C.menu_adaptado(rol, E.menu_propuesto(rol, f[uid], S, envs, ags)),
            "Solo barra": C.menu_estatico(rol),
        }
        for n in ("Score-greedy", "Proposed", "Solo barra"):
            menus[n] = {**menus[n], "bar": barra}
        infos = {n: C.ubicaciones(m) for n, m in menus.items()}
        tareas = C.tareas_sesion2(u)
        en_barra.append(np.mean([r in set(barra) for r, _ in tareas]))
        for n in POLITICAS:
            a = np.array([C.navegar(r, infos[n], u, C.sorteos_tarea(sem)) for r, sem in tareas])
            for j, m in enumerate(("M1", "M2", "M3")):
                res[n][m].append(a[:, j].mean())

    out = {"s": s, "n_funciones": n_funciones, "k_barra": k_barra, "n_usuarios": len(usu),
           "tareas_en_barra": float(np.mean(en_barra)), "politicas": {}}
    for n in POLITICAS:
        d = {}
        for m in ("M1", "M2", "M3"):
            d[m] = {"media": float(np.mean(res[n][m]))}
            if n != "Static" and np.mean(res["Static"][m]) > 0:   # sin errores (f_error = 0) M3 = 0
                red, lo, hi = reduccion_ic(res["Static"][m], res[n][m])
                d[m].update({"reduccion_pct": red, "ic95": [lo, hi]})
            elif n != "Static":
                d[m].update({"reduccion_pct": None, "ic95": None})
        out["politicas"][n] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=list(BARRIDOS), nargs="+", default=list(BARRIDOS))
    args = ap.parse_args()

    sel = C.leer_json("rq1.json")["seleccionado"]
    net = E.cargar_predictor(sel)
    envs, ags = E.cargar_agentes()
    for param in args.param:
        clave, valores = BARRIDOS[param]
        filas = []
        print(f"== {param} ({clave})", flush=True)
        for v in valores:
            t0 = time.time()
            r = evaluar(net, envs, ags, **{**BASE, clave: v})
            filas.append(r)
            pp = r["politicas"]
            print(f"  {clave}={v:<5} barra={r['tareas_en_barra']:.3f}  "
                  f"Proposed M1 -{pp['Proposed']['M1']['reduccion_pct']:.1f}%  M2 -{pp['Proposed']['M2']['reduccion_pct']:.1f}%  "
                  f"Solo barra M1 -{pp['Solo barra']['M1']['reduccion_pct']:.1f}%  ({time.time() - t0:.0f} s)", flush=True)
        C.guardar_json(f"sensibilidad_{param}.json",
                       {"parametro": clave, "base": BASE, "modelo": sel, "resultados": filas})
    print("[OK]", C.SALIDAS)


if __name__ == "__main__":
    main()
