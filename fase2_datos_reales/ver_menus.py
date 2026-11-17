"""Vista previa de los menús adaptados ANTES de subirlos a la app.

Para cada participante de OE3_Agente_DQN/output_oe3/menus_app.json muestra, lado a lado,
el menú estático (lo que vio en la sesión 1) y el adaptado (lo que verá en la sesión 2),
armados con la MISMA lógica que la app (staticMenu / adaptedMenu de app/src/lib/menu.ts).

Salida:
  - resumen en consola
  - menus_preview.html (se abre en el navegador)

Uso:
  python ver_menus.py
"""
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent                            # fase2_datos_reales/
PIPE = RAIZ.parent / "fase1_datos_sinteticos" / "pipeline"         # pipeline compartido
CATALOGO = RAIZ / "estudio-menus/app/src/data/catalog.json"
MENUS = PIPE / "OE3_Agente_DQN/output_oe3/menus_app.json"
TAREAS = PIPE / "datos_reales/task_results.csv"
PARTIC = PIPE / "datos_reales/participantes.csv"
SALIDA = RAIZ / "menus_preview.html"

cat = json.loads(CATALOGO.read_text(encoding="utf-8"))


def hojas(n):
    return [n["route"]] if "route" in n else [r for c in n["children"] for r in hojas(c)]


TITULO = {}
GRUPO_DE = {}
for g in cat["groups"]:
    for r in hojas(g):
        GRUPO_DE[r] = g["id"]


def _titulos(n):
    if "route" in n:
        TITULO[n["route"]] = n["title"]
    for c in n.get("children", []):
        _titulos(c)


for g in cat["groups"]:
    _titulos(g)


# ── Misma lógica que app/src/lib/menu.ts ─────────────────────────────────────
def menu_estatico(rol):
    ok = set(cat["roles"][rol]["groups"])
    return {"bar": [], "groups": [g for g in cat["groups"] if g["id"] in ok]}


def ordenar_nodos(nodos, pos):
    big = 10 ** 9
    clave = lambda n: min(pos.get(r, big) for r in hojas(n))
    orden = sorted(enumerate(nodos), key=lambda x: (clave(x[1]), x[0]))
    return [n if "route" in n else {**n, "children": ordenar_nodos(n["children"], pos)} for _, n in orden]


def menu_adaptado(rol, adapted):
    base = menu_estatico(rol)
    permitidos = {g["id"] for g in base["groups"]}
    orden_g = [g["id"] for g in adapted.get("groups", []) if g["id"] in permitidos]
    gpos = {gid: i for i, gid in enumerate(orden_g)}
    grupos = []
    for i, g in sorted(enumerate(base["groups"]), key=lambda x: (gpos.get(x[1]["id"], 10 ** 9), x[0])):
        items = [r for x in adapted.get("groups", []) if x["id"] == g["id"] for r in x["items"]]
        pos = {r: k for k, r in enumerate(items)}
        grupos.append({**g, "children": ordenar_nodos(g["children"], pos)})
    bar = [r for r in dict.fromkeys(adapted.get("bar", [])) if r in GRUPO_DE and GRUPO_DE[r] in permitidos][:3]
    return {"bar": bar, "groups": grupos}


# ── Costo en clics (igual que la app: grupo + [submenú] + ítem; barra = 1) ───
def clics(menu, ruta):
    if ruta in menu["bar"]:
        return 1
    for g in menu["groups"]:
        for c in g["children"]:
            if c.get("route") == ruta:
                return 2
            if ruta in hojas(c):
                return 3
    return None


def costo_medio(menu, uso):
    tot = sum(uso.values())
    return sum(n * clics(menu, r) for r, n in uso.items()) / tot if tot else 0


# ── Datos ─────────────────────────────────────────────────────────────────────
def cargar():
    menus = json.loads(MENUS.read_text(encoding="utf-8"))
    roles, s1_ok = {}, set()
    if PARTIC.exists():
        with open(PARTIC, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("role"):
                    roles[r["pid"]] = r["role"]
                if r.get("session1_done_at"):
                    s1_ok.add(r["pid"])
    # Solo quienes terminaron la sesión 1 (el importador también los omitiría)
    for pid in sorted(set(menus) - s1_ok):
        print(f"{pid}: no terminó la sesión 1, no recibe menú todavía")
    menus = {pid: m for pid, m in menus.items() if pid in s1_ok}
    uso = defaultdict(Counter)
    if TAREAS.exists():
        with open(TAREAS, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["session_no"] == "1":  # todas, incluida la práctica (igual que el modelo)
                    uso[r["pid"]][r["target_route"]] += 1
    return menus, roles, uso


# ── HTML ──────────────────────────────────────────────────────────────────────
def nodo_html(n, uso, nivel=0):
    if "route" in n:
        k = uso.get(n["route"], 0)
        cls = "item usado" if k else "item"
        badge = f'<span class="n">{k}</span>' if k else ""
        return f'<li class="{cls}" style="padding-left:{12 + nivel * 14}px">{html.escape(n["title"])}{badge}</li>'
    hijos = "".join(nodo_html(c, uso, nivel + 1) for c in n["children"])
    return f'<li class="sub" style="padding-left:{12 + nivel * 14}px">▸ {html.escape(n["title"])}</li>{hijos}'


def menu_html(titulo, menu, uso):
    partes = [f"<h3>{titulo}</h3><div class='side'><div class='brand'>ERP Artículos</div>"]
    if menu["bar"]:
        partes.append("<div class='sec'>ACCESOS RÁPIDOS</div><ul>")
        partes += [f"<li class='bar'>★ {html.escape(TITULO[r])}"
                   f"{f'<span class=n>{uso[r]}</span>' if uso.get(r) else ''}</li>" for r in menu["bar"]]
        partes.append("</ul>")
    partes.append("<div class='sec'>MENÚ</div><ul>")
    for g in menu["groups"]:
        usados = sum(uso.get(r, 0) for r in hojas(g))
        partes.append(f"<li class='grp'>{html.escape(g['title'])}{f'<span class=n>{usados}</span>' if usados else ''}</li>")
        partes += [nodo_html(c, uso, 1) for c in g["children"]]
    partes.append("</ul></div>")
    return "".join(partes)


CSS = """
:root{--bg:#f6f7f9;--fg:#1d2430;--mut:#667085;--card:#fff;--side:#1f2a37;--side-fg:#e5e7eb;--acc:#f5c451;--use:#3b82f6}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--fg:#e5e7eb;--mut:#98a2b3;--card:#161d27}}
body{margin:0;font:14px/1.45 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1100px;margin:auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}.lead{color:var(--mut);margin:0 0 24px}
.p{background:var(--card);border-radius:12px;padding:16px 18px;margin-bottom:22px;box-shadow:0 1px 3px #0002}
.p h2{margin:0 0 4px;font-size:18px}.meta{color:var(--mut);margin-bottom:10px}
.kpi{display:inline-block;margin-right:18px}.kpi b{font-size:18px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.cols{grid-template-columns:1fr}}
h3{font-size:13px;color:var(--mut);margin:12px 0 6px;text-transform:uppercase;letter-spacing:.04em}
.side{background:var(--side);color:var(--side-fg);border-radius:10px;padding:10px 0;max-height:560px;overflow:auto}
.brand{font-weight:700;padding:4px 14px 8px;border-bottom:1px solid #ffffff1a}
.sec{font-size:11px;color:#93a4b8;letter-spacing:.06em;padding:10px 14px 4px}
ul{list-style:none;margin:0;padding:0}li{padding:4px 12px;display:flex;justify-content:space-between;gap:8px}
.grp{font-weight:600;margin-top:4px}.sub{color:#cbd5e1}.item{color:#aab6c5}
.usado{color:#fff;font-weight:600}.bar{color:var(--acc)}
.n{background:var(--use);color:#fff;border-radius:9px;padding:0 7px;font-size:11px;font-weight:600}
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="", help="pids separados por coma (p. ej. los que aún no tienen menú)")
    solo = {p.strip().upper() for p in ap.parse_args().solo.split(",") if p.strip()}
    menus, roles, uso = cargar()
    if solo:
        menus = {p: m for p, m in menus.items() if p in solo}
        faltan = solo - set(menus)
        if faltan:
            print("Sin menú generado (no terminaron la sesión 1 o no hay datos):", sorted(faltan))
    tarjetas, filas = [], []
    for pid in sorted(menus):
        rol = roles.get(pid)
        if rol not in cat["roles"]:
            print(f"{pid}: rol desconocido ({rol}), se omite")
            continue
        u = uso.get(pid, Counter())
        est, ada = menu_estatico(rol), menu_adaptado(rol, menus[pid])
        c_est, c_ada = costo_medio(est, u), costo_medio(ada, u)
        red = 100 * (c_est - c_ada) / c_est if c_est else 0
        top = ", ".join(f"{TITULO[r]} ({n})" for r, n in u.most_common(3))
        filas.append((pid, rol, sum(u.values()), c_est, c_ada, red))
        print(f"\n{pid} · {rol} · {sum(u.values())} tareas · más usadas: {top}")
        print(f"  Barra: {' · '.join(TITULO[r] for r in ada['bar'])}")
        print(f"  Grupos: {' > '.join(g['title'] for g in ada['groups'])}")
        print(f"  Clics esperados por tarea: {c_est:.2f} -> {c_ada:.2f}  ({red:.0f} % menos)")
        tarjetas.append(
            f"<section class='p'><h2>{pid} · {html.escape(cat['roles'][rol]['label'])}</h2>"
            f"<div class='meta'>{sum(u.values())} tareas de la sesión 1 · más usadas: {html.escape(top)}</div>"
            f"<div><span class='kpi'>Clics esperados por tarea: <b>{c_est:.2f} → {c_ada:.2f}</b></span>"
            f"<span class='kpi'>Reducción: <b>{red:.0f} %</b></span></div>"
            f"<div class='cols'><div>{menu_html('Sesión 1 · menú estático', est, u)}</div>"
            f"<div>{menu_html('Sesión 2 · menú adaptado', ada, u)}</div></div></section>")

    doc = (f"<!doctype html><html lang='es'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>Vista previa de menús</title>"
           f"<style>{CSS}</style></head><body><main><h1>Vista previa de menús adaptados</h1>"
           f"<p class='lead'>Así quedaría el menú de cada participante en la sesión 2. Los números azules indican "
           f"cuántas veces abrió esa función en la sesión 1 (incluidas las 3 de práctica, igual que el modelo). Nada se ha subido todavía.</p>"
           f"{''.join(tarjetas)}</main></body></html>")
    SALIDA.write_text(doc, encoding="utf-8")
    subir = PIPE / "OE3_Agente_DQN/output_oe3/menus_para_subir.json"
    subir.write_text(json.dumps({pid: menus[pid] for pid, *_ in filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {len(tarjetas)} participantes -> {SALIDA}")
    print(f"     Archivo para subir (solo estos participantes) -> {subir}")


if __name__ == "__main__":
    main()
