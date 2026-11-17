"""Piezas comunes de la evaluación del artículo (sin entrenar nada).

Usa los modelos YA entrenados del pipeline (fase1_datos_sinteticos/pipeline):
  - predictores OE2 (LSTM, GRU, 1D-CNN, Transformer) entrenados con el corpus simulado
    en modo estudio (600 usuarios, semilla 42) + participante piloto;
  - agentes DQN de OE3 (uno por tamaño de grupo: 2, 3, 4, 6, 7, 10, 12).

Scripts:
  evaluacion.py   RQ1 (predictores + líneas base), desempeño DQN, RQ2/RQ3 (políticas,
                  ablación, pesos, deriva), RQ4 (latencia)  -> salidas/*.json
  tablas.py       tablas .tex y figuras del artículo desde salidas/
"""
import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
UR = AQUI.parent                                   # fase1_datos_sinteticos/pipeline/
RAIZ = UR.parent.parent                            # raíz del repositorio
SALIDAS = AQUI / "salidas"
ARTICULO = AQUI / "articulo"                       # tables/*.tex y figures/* del manuscrito
CATALOGO = RAIZ / "fase2_datos_reales/estudio-menus/app/src/data/catalog.json"
SALIDAS.mkdir(parents=True, exist_ok=True)

SEMILLA_ENTRENAMIENTO = 42      # población con la que se entrenaron los modelos
SEMILLA_EVAL = 2026             # tareas nuevas de evaluación / usuarios nuevos de prueba
USUARIOS_POR_ROL = 150
FACTOR_FAMILIAR = 0.7           # ritmo del bloque 2 del simulador (usuario ya familiarizado)
SEQ_LEN = 10

sys.path.insert(0, str(UR / "00_Generador_Datos"))
import simulador_erp_real as sim  # noqa: E402

CAT = json.loads(CATALOGO.read_text(encoding="utf-8"))
ROLES = list(CAT["roles"])


# ── Menú (misma lógica que app/src/lib/menu.ts) ──────────────────────────────
def hojas(n):
    return [n["route"]] if "route" in n else [r for c in n["children"] for r in hojas(c)]


GRUPO_DE = {r: g["id"] for g in CAT["groups"] for r in hojas(g)}
RUTAS = sorted(GRUPO_DE)                       # 60 rutas = vocabulario de los modelos
R2I = {r: i for i, r in enumerate(RUTAS)}


def rutas_rol(rol):
    ok = set(CAT["roles"][rol]["groups"])
    return [r for r in RUTAS if GRUPO_DE[r] in ok]


def menu_estatico(rol):
    ok = set(CAT["roles"][rol]["groups"])
    return {"bar": [], "groups": [g for g in CAT["groups"] if g["id"] in ok]}


def _ordenar_nodos(nodos, pos):
    big = 10 ** 9
    clave = lambda n: min(pos.get(r, big) for r in hojas(n))
    orden = sorted(enumerate(nodos), key=lambda x: (clave(x[1]), x[0]))
    return [n if "route" in n else {**n, "children": _ordenar_nodos(n["children"], pos)} for _, n in orden]


def menu_adaptado(rol, compacto):
    """compacto = {"bar": [...], "groups": [{"id": "g5", "items": [...]}]} (formato de la app)."""
    base = menu_estatico(rol)
    permitidos = {g["id"] for g in base["groups"]}
    gpos = {g["id"]: i for i, g in enumerate(x for x in compacto.get("groups", []) if x["id"] in permitidos)}
    grupos = []
    for i, g in sorted(enumerate(base["groups"]), key=lambda x: (gpos.get(x[1]["id"], 10 ** 9), x[0])):
        items = [r for x in compacto.get("groups", []) if x["id"] == g["id"] for r in x["items"]]
        grupos.append({**g, "children": _ordenar_nodos(g["children"], {r: k for k, r in enumerate(items)})})
    bar = [r for r in dict.fromkeys(compacto.get("bar", [])) if GRUPO_DE.get(r) in permitidos][:3]
    return {"bar": bar, "groups": grupos}


def ubicaciones(menu):
    """ruta -> clics para abrirla, ítems escaneados antes de ella y vecinos visibles."""
    info, nb = {}, len(menu["bar"])
    for i, r in enumerate(menu["bar"]):
        info[r] = {"clics": 1, "scan": i + 1, "vecinos": [x for x in menu["bar"] if x != r]}
    for gi, g in enumerate(menu["groups"]):
        plano = hojas(g)
        sub = {r for c in g["children"] if "route" not in c for r in hojas(c)}
        for ii, r in enumerate(plano):
            if r in info:
                continue
            vec = sorted((x for x in plano if x != r), key=lambda x: abs(plano.index(x) - ii))
            info[r] = {"clics": 3 if r in sub else 2, "scan": nb + (gi + 1) + (ii + 1), "vecinos": vec}
    return info


# ── Poblaciones simuladas en modo estudio (mismo generador que el entrenamiento) ──
def poblacion(semilla):
    """(usuarios, tareas de la sesión 1) generados exactamente como el corpus de entrenamiento."""
    sim.SALIDA_DIR = Path(tempfile.mkdtemp(prefix="sim_"))     # no tocar datos_sinteticos/
    args = argparse.Namespace(modo="estudio", usuarios_por_rol=USUARIOS_POR_ROL, semilla=semilla,
                              dias=90, bloques=2, tareas_por_bloque=20, practica=3)
    _, tareas, usuarios = sim.simular_estudio(args)
    seq = {}
    for t in tareas:
        seq.setdefault(t["pid"], []).append(t["target_route"])
    return usuarios, seq


# ── Políticas (formato compacto de la app) ───────────────────────────────────
def _orden(rutas, v):
    return sorted(rutas, key=lambda r: (-v(r), R2I[r]))


def compacto_ordenado(rol, valor, barra=True, grupos_por="max"):
    v = lambda r: valor.get(r, 0.0)
    estatico = menu_estatico(rol)["groups"]
    pos_g = {g["id"]: i for i, g in enumerate(estatico)}
    grupos = []
    for g in estatico:
        items = [r for r in hojas(g)]
        items = sorted(items, key=lambda r: (-v(r), items.index(r)))   # empates: orden original
        agg = max(v(r) for r in items) if grupos_por == "max" else sum(v(r) for r in items)
        grupos.append((agg, g["id"], items))
    grupos.sort(key=lambda x: (-x[0], pos_g[x[1]]))
    bar = [r for r in _orden(rutas_rol(rol), v)[:3] if v(r) > 0] if barra else []
    return {"bar": bar, "groups": [{"id": gid, "items": it} for _, gid, it in grupos]}


def score(f_u, p_u, prior, a=0.5, b=0.3, c=0.2):
    return {r: a * f_u.get(r, 0.0) + b * p_u.get(r, 0.0) + c * prior.get(r, 0.0) for r in RUTAS}


# ── Navegación simulada con números aleatorios comunes (modelo del simulador) ───
def sorteos_tarea(semilla):
    r = random.Random(semilla)
    return {"z_clic": r.gauss(0, 1), "z_scan": r.gauss(0, 1),
            "u_err": [r.random() for _ in range(8)], "u_mismo": [r.random() for _ in range(8)],
            "u_elige": [r.random() for _ in range(8)], "t_vuelta": [r.uniform(1.0, 6.0) for _ in range(8)]}


def navegar(objetivo, info, u, sorteo, factor=FACTOR_FAMILIAR):
    """M1 (clics), M2 (s) y M3 (error) al abrir `objetivo` en el menú descrito por `info`.

    Igual que el simulador en modo estudio: tiempo = (clics x t_clic + ítems escaneados x t_scan
    x ruido log-normal) x factor; P(visita errónea) = p_exp x 1.3 si está en submenú x 0.6
    (bloque 2), con decaimiento 0.5 tras cada error; la visita errónea va con prob. 0.65 a un
    ítem vecino visible y si no a otra función del perfil.
    """
    ex = sim.EXPERIENCIA[u["experiencia"]]
    d = info[objetivo]
    t_clic = max(0.2, ex["clic"][0] + ex["clic"][1] * sorteo["z_clic"])
    t = (d["clics"] * t_clic + d["scan"] * ex["scan"] * math.exp(0.35 * sorteo["z_scan"])) * factor
    p_err = ex["error"] * (1.3 if d["clics"] == 3 else 1.0) * 0.6
    perfil = [r for r in u["perfil"] if r != objetivo]
    clics, malas = d["clics"], 0
    while malas < 8 and sorteo["u_err"][malas] < p_err:
        cand = d["vecinos"][:3] if (d["vecinos"] and sorteo["u_mismo"][malas] < 0.65) else perfil
        mala = cand[min(int(sorteo["u_elige"][malas] * len(cand)), len(cand) - 1)]
        clics += info[mala]["clics"]
        t += sorteo["t_vuelta"][malas]
        malas += 1
        p_err *= 0.5
    return clics, t, int(malas > 0)


def tareas_sesion2(u, perfil=None, semilla=SEMILLA_EVAL, bloques=2, n=20):
    """Tareas de una sesión nueva (como la sesión 2 de la app), con semilla distinta."""
    rnd = random.Random(f"{semilla}-{u['user_id']}")
    p = perfil if perfil is not None else u["perfil"]
    out = []
    for b in range(bloques):
        for t, r in enumerate(sim.tareas_bloque(p, n, rnd)):
            out.append((r, f"{semilla}-{u['user_id']}-{b}-{t}"))
    return out


def perfil_con_deriva(u, n_swaps=2, semilla=SEMILLA_EVAL):
    """Deriva del perfil entre sesiones: intercambia los pesos de n pares de funciones."""
    rnd = random.Random(f"deriva-{semilla}-{u['user_id']}")
    p = dict(u["perfil"])
    for _ in range(n_swaps):
        a, b = rnd.sample(list(p), 2)
        p[a], p[b] = p[b], p[a]
    return p


def guardar_json(nombre, obj):
    (SALIDAS / nombre).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def leer_json(nombre):
    return json.loads((SALIDAS / nombre).read_text(encoding="utf-8"))
