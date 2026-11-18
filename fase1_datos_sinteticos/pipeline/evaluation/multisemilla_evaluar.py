"""Variabilidad entre semillas (observación 7 del revisor). Evalúa en CPU las corridas de
multisemilla_entrenar.py y repite la evaluación de los modelos del artículo con otras semillas.

  1. Predictores: Acc@1, Acc@3 y MRR en los 600 usuarios nuevos de la RQ1 (semilla 2026) para
     cada arquitectura y semilla; arquitectura con menor pérdida de validación en cada semilla.
  2. Agentes DQN: % de cabeceras óptimas (como dqn.json) para cada semilla.
  3. Políticas: reducción de M1 y M2 de Proposed y Score-greedy frente a Static usando el LSTM y
     los agentes de cada semilla (tareas de evaluación con la semilla 2026).
  4. Evaluación: con los modelos del artículo, reducción de M1 y M2 con cinco semillas de tareas
     (2026-2030).

Salida: salidas/multisemilla.json
Uso:    python multisemilla_evaluar.py [--semillas 1 2 3 4 5]
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import comun as C
import evaluacion as E
import multisemilla_entrenar as M

NETS = ["lstm", "gru", "cnn1d", "transformer"]


def metricas_rq1(nets, seq_te):
    out = {n: defaultdict(list) for n in nets}
    for uid, s in seq_te.items():
        X = torch.tensor([E.ventana(s[:i]) for i in range(1, len(s))])
        with torch.no_grad():
            P = {n: F.softmax(nets[n](X), dim=1).numpy() for n in nets}
        acc = {n: defaultdict(list) for n in nets}
        for k, i in enumerate(range(1, len(s))):
            y = C.R2I[s[i]]
            for n in nets:
                rk = E.rango_pesimista(P[n][k], y)
                acc[n]["acc1"].append(rk == 1); acc[n]["acc3"].append(rk <= 3); acc[n]["mrr"].append(1 / rk)
        for n in nets:
            for m, v in acc[n].items():
                out[n][m].append(np.mean(v))
    return {n: {m: float(np.mean(v)) for m, v in out[n].items()} for n in nets}


def cargar(sem):
    d = M.SALIDA / f"semilla_{sem}"
    nets = {}
    for n in NETS:
        net = M.modelo(n)
        net.load_state_dict(torch.load(d / f"{n}.pth", map_location="cpu", weights_only=True))
        nets[n] = net.eval()
    envs, ags = E.cargar_agentes()
    for gs, ag in ags.items():
        ag.q_net.load_state_dict(torch.load(d / f"dqn_gs{gs}.pt", map_location="cpu", weights_only=True))
        ag.q_net.eval()
    hist = {n: E.json.loads((d / f"historial_{n}.json").read_text(encoding="utf-8")) for n in NETS}
    return nets, envs, ags, hist


def politicas(usu, seq, net, envs, ags, semilla_tareas):
    f = {u: dict(Counter(s)) for u, s in seq.items()}
    prior = {}
    for rol in C.ROLES:
        c = Counter(r for u in usu if u["rol"] == rol for r in seq[u["user_id"]])
        tot = sum(c.values()); prior[rol] = {r: k / tot for r, k in c.items()}
    res = {p: {"M1": [], "M2": []} for p in ["Static", "Score-greedy", "Proposed"]}
    for u in usu:
        uid, rol = u["user_id"], u["rol"]
        p = E.probs(net, seq[uid]); P = {r: float(p[C.R2I[r]]) for r in C.RUTAS}
        S = C.score(f[uid], P, prior[rol])
        menus = {"Static": C.menu_estatico(rol),
                 "Score-greedy": C.menu_adaptado(rol, C.compacto_ordenado(rol, S)),
                 "Proposed": C.menu_adaptado(rol, E.menu_propuesto(rol, f[uid], S, envs, ags))}
        infos = {n: C.ubicaciones(m) for n, m in menus.items()}
        tareas = C.tareas_sesion2(u, semilla=semilla_tareas)
        for n in menus:
            a = np.array([C.navegar(r, infos[n], u, C.sorteos_tarea(sem)) for r, sem in tareas])
            res[n]["M1"].append(a[:, 0].mean()); res[n]["M2"].append(a[:, 1].mean())
    red = lambda n, m: float(100 * (1 - np.mean(res[n][m]) / np.mean(res["Static"][m])))
    return {n: {m: red(n, m) for m in ("M1", "M2")} for n in ["Score-greedy", "Proposed"]}


def pct_optimo(usu, seq, envs, ags):
    f = {u: dict(Counter(s)) for u, s in seq.items()}
    por = defaultdict(list)
    for u in usu:
        perm = set(C.CAT["roles"][u["rol"]]["groups"])
        for gid, rutas in E.cfg3.GRUPOS_RUTAS.items():
            fr = [f[u["user_id"]].get(r, 0.0) for r in rutas]
            if E.CLAVE[gid] not in perm or len(rutas) < 2 or sum(fr) == 0:
                continue
            _, phi, _ = E.rollout(envs, ags, len(rutas), fr)
            por[len(rutas)].append(phi >= 0.999)
    return {str(gs): float(100 * np.mean(v)) for gs, v in sorted(por.items())}


def resumen(valores):
    a = np.array(valores, dtype=float)
    return {"media": float(a.mean()), "de": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "valores": [float(x) for x in a]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    usu_tr, seq_tr = C.poblacion(C.SEMILLA_ENTRENAMIENTO)
    _, seq_te = C.poblacion(C.SEMILLA_EVAL)

    por_semilla = {}
    for sem in args.semillas:
        nets, envs, ags, hist = cargar(sem)
        r = {"rq1": metricas_rq1(nets, seq_te),
             "val_loss": {n: hist[n]["val_loss_min"] for n in NETS},
             "epocas": {n: f"{hist[n]['mejor_epoca']}/{hist[n]['epocas']}" for n in NETS},
             "seleccionado": min(NETS, key=lambda n: hist[n]["val_loss_min"]),
             "dqn_pct_optimo": pct_optimo(usu_tr, seq_tr, envs, ags),
             "politicas": politicas(usu_tr, seq_tr, nets["lstm"], envs, ags, C.SEMILLA_EVAL)}
        por_semilla[str(sem)] = r
        print(f"semilla {sem}: seleccionado {r['seleccionado']} | LSTM MRR {r['rq1']['lstm']['mrr']:.3f} | "
              f"Proposed M1 -{r['politicas']['Proposed']['M1']:.1f}% | DQN gs12 {r['dqn_pct_optimo'].get('12', 0):.1f}%", flush=True)

    agg = {"rq1": {n: {m: resumen([por_semilla[s]["rq1"][n][m] for s in por_semilla]) for m in ("acc1", "acc3", "mrr")}
                   for n in NETS},
           "val_loss": {n: resumen([por_semilla[s]["val_loss"][n] for s in por_semilla]) for n in NETS},
           "seleccion": dict(Counter(por_semilla[s]["seleccionado"] for s in por_semilla)),
           "dqn_pct_optimo": {gs: resumen([por_semilla[s]["dqn_pct_optimo"][gs] for s in por_semilla])
                              for gs in por_semilla[str(args.semillas[0])]["dqn_pct_optimo"]},
           "politicas": {p: {m: resumen([por_semilla[s]["politicas"][p][m] for s in por_semilla]) for m in ("M1", "M2")}
                         for p in ["Score-greedy", "Proposed"]}}

    # 4. Modelos del artículo con cinco semillas de tareas de evaluación
    net = E.cargar_predictor(C.leer_json("rq1.json")["seleccionado"])
    envs, ags = E.cargar_agentes()
    ev = {str(t): politicas(usu_tr, seq_tr, net, envs, ags, t) for t in range(2026, 2031)}
    agg["evaluacion_modelos_articulo"] = {p: {m: resumen([ev[t][p][m] for t in ev]) for m in ("M1", "M2")}
                                          for p in ["Score-greedy", "Proposed"]}
    print("modelos del artículo, 5 semillas de tareas: Proposed M1", agg["evaluacion_modelos_articulo"]["Proposed"]["M1"])
    C.guardar_json("multisemilla.json", {"semillas": args.semillas, "resumen": agg, "por_semilla": por_semilla,
                                         "evaluacion_por_semilla_de_tareas": ev})
    print("[OK]", C.SALIDAS / "multisemilla.json")


if __name__ == "__main__":
    main()
