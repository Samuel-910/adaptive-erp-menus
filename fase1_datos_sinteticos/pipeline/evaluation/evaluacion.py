"""Evaluación del artículo con los modelos YA entrenados (no entrena nada).

  1. RQ1  predictores (LSTM, GRU, 1D-CNN, Transformer) + líneas base (más frecuente, Markov)
          y techo teórico (perfil verdadero), sobre USUARIOS NUEVOS (semilla 2026).
  2. DQN  desempeño de los agentes por tamaño de grupo (rollout codicioso).
  3. RQ2/RQ3  políticas Static / Frequency / Score-greedy / Proposed, ablación, pesos y
          deriva del perfil: M1, M2 y M3 medidos tarea por tarea con números aleatorios comunes.
  4. RQ4  latencia (t_load, t_gen) en CPU.

Salidas en salidas/: rq1.json, dqn.json, rq2_por_usuario.csv, rq2.json, rq4.json
Uso:  python evaluacion.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # todo en CPU (servidor local sin GPU)

import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats  # noqa: E402

import comun as C  # noqa: E402

torch.set_num_threads(4)
OE2 = C.UR / "OE2_Modelado"
OE3 = C.UR / "OE3_Agente_DQN"

# ── Carga de módulos del pipeline (sin mezclar los config.py de OE1/OE2/OE3) ──
_cwd = os.getcwd()
os.chdir(OE3)
sys.path.insert(0, str(OE3))
import config as cfg3  # noqa: E402  (OE3)
from env import GroupMenuEnv  # noqa: E402
from agent import DQNAgent  # noqa: E402
os.chdir(_cwd)


def _modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V = len(C.RUTAS)
HP = dict(embed=32, hidden=64, layers=2, dropout=0.2, heads=4)


def cargar_predictor(nombre):
    m = _modulo(nombre, OE2 / "models" / f"{nombre}.py")
    if nombre == "lstm":
        net = m.LSTMModel(V, HP["embed"], HP["hidden"], HP["layers"], HP["dropout"], V)
    elif nombre == "gru":
        net = m.GRUModel(V, HP["embed"], HP["hidden"], HP["layers"], HP["dropout"], V)
    elif nombre == "cnn1d":
        net = m.CNN1DModel(V, HP["embed"], HP["hidden"], HP["dropout"], V)
    else:
        net = m.TransformerNavModel(V, HP["embed"], HP["heads"], HP["layers"], HP["dropout"], V)
    net.load_state_dict(torch.load(OE2 / "output_oe2" / f"{nombre}_model.pth", map_location="cpu", weights_only=True))
    return net.eval()


def cargar_agentes():
    tam = sorted({len(v) for v in cfg3.GRUPOS_RUTAS.values()})
    envs = {gs: GroupMenuEnv(gs, top_k=cfg3.TOP_K) for gs in tam}
    ags = {}
    for gs in tam:
        if gs < 2:
            continue
        a = DQNAgent(envs[gs].state_dim, envs[gs].n_actions)
        a.q_net.load_state_dict(torch.load(OE3 / "output_oe3" / f"dqn_agent_gs{gs}.pt", map_location="cpu", weights_only=True))
        a.q_net.eval()
        ags[gs] = a
    return envs, ags


CLAVE = {g["id"]: g["clave"] for g in cfg3.MENU_ERP}          # 1 -> "g1"


def ventana(seq):
    ids = [C.R2I[r] + 1 for r in seq[-C.SEQ_LEN:]]           # +1: 0 = padding
    return [0] * (C.SEQ_LEN - len(ids)) + ids


def probs(net, seq):
    with torch.no_grad():
        return F.softmax(net(torch.tensor([ventana(seq)])), dim=1).numpy()[0]


def rango_pesimista(s, y):
    return int((s >= s[y]).sum())                             # empates en contra


def bootstrap_ic(x, n=2000, semilla=0, fn=np.mean):
    rng = np.random.default_rng(semilla)
    x = np.asarray(x)
    b = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ════════════════════════════════════════════════════════════════════════════
def rq1(usu_tr, seq_tr, usu_te, seq_te, nets):
    print("RQ1: predictores sobre usuarios nuevos...")
    mf = np.zeros(V)
    trans = np.zeros((V, V))
    for s in seq_tr.values():
        for a, b in zip(s, s[1:]):
            trans[C.R2I[a], C.R2I[b]] += 1
        for r in s:
            mf[C.R2I[r]] += 1
    perfil_te = {u["user_id"]: u["perfil"] for u in usu_te}
    rol_te = {u["user_id"]: u["rol"] for u in usu_te}
    rol_tr = {u["user_id"]: u["rol"] for u in usu_tr}
    prior_rol = {}                                            # frecuencia de cada ruta en el corpus del rol
    for rol in C.ROLES:
        c = np.zeros(V)
        for uid, s in seq_tr.items():
            if rol_tr.get(uid) == rol:
                for r in s:
                    c[C.R2I[r]] += 1
        prior_rol[rol] = c / c.sum()

    nombres = ["lstm", "gru", "cnn1d", "transformer", "markov", "mas_frecuente",
               "prior_rol", "conteo_ventana", "conteo_ventana_prior", "oraculo"]
    por_usuario = {n: defaultdict(lambda: defaultdict(list)) for n in nombres}
    for uid, s in seq_te.items():
        X = torch.tensor([ventana(s[:i]) for i in range(1, len(s))])
        with torch.no_grad():
            P = {n: F.softmax(nets[n](X), dim=1).numpy() for n in ["lstm", "gru", "cnn1d", "transformer"]}
        for k, i in enumerate(range(1, len(s))):
            y, prev = C.R2I[s[i]], C.R2I[s[i - 1]]
            orc = np.array([perfil_te[uid].get(r, 0.0) for r in C.RUTAS])
            orc[prev] = 0.0                                   # la app no repite la tarea anterior
            pr = prior_rol[rol_te[uid]].copy()
            pr[prev] = 0.0
            cv = np.zeros(V)                                  # conteo en la ventana del modelo (últimas SEQ_LEN rutas)
            for r in s[max(0, i - C.SEQ_LEN):i]:
                cv[C.R2I[r]] += 1
            cv[prev] = 0.0
            sc = {**{n: P[n][k] for n in P}, "markov": trans[prev] + 1e-9 * mf,
                  "mas_frecuente": mf.copy(), "prior_rol": pr, "conteo_ventana": cv,
                  "conteo_ventana_prior": cv + pr, "oraculo": orc}
            for n, v in sc.items():
                rk = rango_pesimista(v, y)
                d = por_usuario[n][uid]
                d["acc1"].append(rk == 1); d["acc3"].append(rk <= 3)
                d["mrr"].append(1 / rk); d["ndcg5"].append(1 / np.log2(rk + 1) if rk <= 5 else 0.0)

    hist = {}
    for n in ["lstm", "gru", "cnn1d", "transformer"]:
        h = json.loads((OE2 / "output_oe2" / f"historial_{n}.json").read_text())
        i = int(np.argmin(h["val_loss"]))
        hist[n] = {"val_loss": h["val_loss"][i], "val_acc": h["val_acc"][i], "mejor_epoca": i + 1,
                   "epocas": len(h["val_loss"])}

    res = {"n_usuarios_prueba": len(seq_te), "n_muestras": sum(len(s) - 1 for s in seq_te.values()),
           "modelos": {}, "historial": hist}
    usuarios = sorted(seq_te)
    for n in nombres:
        r = {}
        for m in ["acc1", "acc3", "mrr", "ndcg5"]:
            pu = [np.mean(por_usuario[n][u][m]) for u in usuarios]
            r[m] = float(np.mean(pu))
            r[m + "_ic"] = bootstrap_ic(pu)
        res["modelos"][n] = r
    sel = min(hist, key=lambda n: hist[n]["val_loss"])
    res["seleccionado"] = sel
    for base in ["markov", "mas_frecuente"]:
        a = [np.mean(por_usuario[sel][u]["mrr"]) for u in usuarios]
        b = [np.mean(por_usuario[base][u]["mrr"]) for u in usuarios]
        w = stats.wilcoxon(a, b, alternative="greater")
        res[f"wilcoxon_{sel}_vs_{base}"] = {"W": float(w.statistic), "p": float(w.pvalue),
                                           "diff_media_mrr": float(np.mean(a) - np.mean(b))}
    for base in ["prior_rol", "conteo_ventana", "conteo_ventana_prior"]:      # observación 6: bilateral
        a = [np.mean(por_usuario[sel][u]["mrr"]) for u in usuarios]
        b = [np.mean(por_usuario[base][u]["mrr"]) for u in usuarios]
        w = stats.wilcoxon(a, b, alternative="two-sided")
        res[f"wilcoxon_{sel}_vs_{base}"] = {"W": float(w.statistic), "p": float(w.pvalue),
                                           "diff_media_mrr": float(np.mean(a) - np.mean(b)),
                                           "r_rb": r_rb(np.array(a), np.array(b))}
    res["parametros"] = {n: int(sum(p.numel() for p in nets[n].parameters()))
                         for n in ["lstm", "gru", "cnn1d", "transformer"]}
    C.guardar_json("rq1.json", res)
    return sel


# ════════════════════════════════════════════════════════════════════════════
def rollout(envs, ags, gs, freq):
    env = envs[gs]
    s = env.reset(np.asarray(freq, dtype=float))
    pasos = 0
    for _ in range(cfg3.STEPS_PER_EP * 2):
        s, _, done = env.step(ags[gs].act(s, greedy=True))
        pasos += 1
        if done:
            break
    return env.menu[:env.k], env._shaped_reward(), pasos


def desempeno_dqn(usu_tr, f, envs, ags):
    print("DQN: desempeño por tamaño de grupo...")
    por = defaultdict(lambda: {"n": 0, "optimo": 0, "phi": [], "pasos": []})
    for u in usu_tr:
        perm = set(C.CAT["roles"][u["rol"]]["groups"])
        for gid, rutas in cfg3.GRUPOS_RUTAS.items():
            gs = len(rutas)
            fr = [f[u["user_id"]].get(r, 0.0) for r in rutas]
            if CLAVE[gid] not in perm or gs < 2 or sum(fr) == 0:
                continue
            _, phi, pasos = rollout(envs, ags, gs, fr)
            d = por[gs]
            d["n"] += 1; d["optimo"] += phi >= 0.999; d["phi"].append(phi); d["pasos"].append(pasos)
    log = (OE3 / "oe3_reentreno.log").read_text(encoding="utf-8", errors="replace")
    curva = [l.strip() for l in log.splitlines() if l.strip().startswith("Ep ")]
    out = {str(gs): {"grupos_evaluados": d["n"], "pct_optimo": 100 * d["optimo"] / d["n"],
                     "phi_media": float(np.mean(d["phi"])), "phi_min": float(np.min(d["phi"])),
                     "pasos_medios": float(np.mean(d["pasos"]))} for gs, d in sorted(por.items())}
    C.guardar_json("dqn.json", {"por_tamano": out, "curva_entrenamiento": curva})


# ════════════════════════════════════════════════════════════════════════════
def menu_propuesto(rol, f_u, S, envs, ags, barra=True, registro=None):
    perm = set(C.CAT["roles"][rol]["groups"])
    grupos = []
    for gid, rutas in cfg3.GRUPOS_RUTAS.items():
        if CLAVE[gid] not in perm:
            continue
        gs = len(rutas)
        fr = [f_u.get(r, 0.0) for r in rutas]
        cab = []
        if gs > 1 and sum(fr) > 0:
            idx, phi, _ = rollout(envs, ags, gs, fr)
            cab = [rutas[i] for i in idx]
            if registro is not None:
                registro.append(phi >= 0.999)
        resto = sorted((r for r in rutas if r not in cab), key=lambda r: (-S.get(r, 0.0), rutas.index(r)))
        items = cab + resto
        grupos.append((max(S.get(r, 0.0) for r in rutas), CLAVE[gid], items))
    orden = {g["id"]: i for i, g in enumerate(C.menu_estatico(rol)["groups"])}
    grupos.sort(key=lambda x: (-x[0], orden[x[1]]))
    bar = [r for r in sorted(C.rutas_rol(rol), key=lambda r: (-S[r], C.R2I[r]))[:3] if S[r] > 0] if barra else []
    return {"bar": bar, "groups": [{"id": g, "items": it} for _, g, it in grupos]}


PESOS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.3, 0.2), (0.5, 0.5, 0), (0.5, 0, 0.5),
         (0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.7, 0.2, 0.1), (0.2, 0.6, 0.2)]


def rq2(usu_tr, f, P, prior, envs, ags):
    print("RQ2/RQ3: políticas, ablación, pesos y deriva...")
    filas, dqn_ok = [], []
    for k, u in enumerate(usu_tr):
        uid, rol = u["user_id"], u["rol"]
        S = C.score(f[uid], P[uid], prior[rol])
        menus = {
            "Static": C.menu_estatico(rol),
            "Frequency": C.menu_adaptado(rol, C.compacto_ordenado(rol, f[uid], barra=False, grupos_por="sum")),
            "Score-greedy": C.menu_adaptado(rol, C.compacto_ordenado(rol, S)),
            "Proposed": C.menu_adaptado(rol, menu_propuesto(rol, f[uid], S, envs, ags, registro=dqn_ok)),
            "A1 sin barra": C.menu_adaptado(rol, menu_propuesto(rol, f[uid], S, envs, ags, barra=False)),
            "A2 solo barra": C.menu_adaptado(rol, {"bar": C.compacto_ordenado(rol, S)["bar"], "groups": []}),
            "A3 beta=0": C.menu_adaptado(rol, menu_propuesto(
                rol, f[uid], C.score(f[uid], P[uid], prior[rol], 0.625, 0, 0.375), envs, ags)),
        }
        for a, b, c in PESOS:
            menus[f"W {a:.2f}/{b:.2f}/{c:.2f}"] = C.menu_adaptado(
                rol, menu_propuesto(rol, f[uid], C.score(f[uid], P[uid], prior[rol], a, b, c), envs, ags))
        infos = {n: C.ubicaciones(m) for n, m in menus.items()}
        for escenario, perfil in [("base", None), ("deriva", C.perfil_con_deriva(u))]:
            tareas = C.tareas_sesion2(u, perfil=perfil)
            nombres = menus if escenario == "base" else ["Static", "Frequency", "Score-greedy", "Proposed"]
            for n in nombres:
                acc = defaultdict(list)
                for r, sem in tareas:
                    m1, m2, m3 = C.navegar(r, infos[n], u, C.sorteos_tarea(sem))
                    prof = infos["Static"][r]["clics"]
                    for clave in ("all", f"prof{prof}"):
                        acc[clave].append((m1, m2, m3))
                for clave, v in acc.items():
                    a = np.array(v)
                    filas.append({"user": uid, "role": rol, "experiencia": u["experiencia"], "escenario": escenario,
                                  "condicion": n, "subconjunto": clave, "n": len(v),
                                  "M1": a[:, 0].mean(), "M2": a[:, 1].mean(), "M3": a[:, 2].mean()})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(usu_tr)} usuarios")
    with open(C.SALIDAS / "rq2_por_usuario.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    C.guardar_json("rq2_dqn_optimo.json", {"grupos": len(dqn_ok), "pct_optimo": 100 * float(np.mean(dqn_ok))})
    return filas


def r_rb(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    rk = stats.rankdata(np.abs(d))
    wp, wn = rk[d > 0].sum(), rk[d < 0].sum()
    return float((wp - wn) / (wp + wn)) if wp + wn else 0.0


def holm(ps):
    orden = np.argsort(ps)
    adj, m, prev = np.empty(len(ps)), len(ps), 0
    for i, j in enumerate(orden):
        prev = max(prev, min(1.0, (m - i) * ps[j]))
        adj[j] = prev
    return adj.tolist()


def analisis(filas):
    print("Análisis estadístico...")
    tab = defaultdict(dict)                     # (escenario, subconjunto, condicion) -> user -> metrics
    rol_de = {}
    for f_ in filas:
        tab[(f_["escenario"], f_["subconjunto"], f_["condicion"])][f_["user"]] = f_
        rol_de[f_["user"]] = f_["role"]
    principales = ["Static", "Frequency", "Score-greedy", "Proposed"]
    res = {}

    def comparar(esc, sub, conds, usuarios=None):
        us = sorted(set.intersection(*[set(tab[(esc, sub, c)]) for c in conds]))
        if usuarios is not None:
            us = [u for u in us if u in usuarios]
        out = {"n_usuarios": len(us), "metricas": {}}
        for m in ["M1", "M2", "M3"]:
            X = {c: np.array([tab[(esc, sub, c)][u][m] for u in us]) for c in conds}
            base = X["Static"]
            md = {"por_condicion": {}}
            for c in conds:
                x = X[c]
                red = lambda idx, x=x: 100 * (1 - x[idx].mean() / base[idx].mean())
                rng = np.random.default_rng(1)
                boots = [red(rng.integers(0, len(x), len(x))) for _ in range(2000)]
                md["por_condicion"][c] = {
                    "media": float(x.mean()), "mediana": float(np.median(x)),
                    "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
                    "reduccion_vs_static_pct": float(red(np.arange(len(x)))),
                    "ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            if len(conds) >= 3:
                fr = stats.friedmanchisquare(*[X[c] for c in conds])
                md["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue), "gl": len(conds) - 1}
            pares = [(a, b) for i, a in enumerate(conds) for b in conds[i + 1:]]
            ps, det = [], []
            for a, b in pares:
                d = X[a] - X[b]
                if np.allclose(d, 0):
                    det.append({"a": a, "b": b, "Z": 0.0, "p": 1.0, "r_rb": 0.0}); ps.append(1.0); continue
                w = stats.wilcoxon(X[a], X[b], method="approx")
                z = float(getattr(w, "zstatistic", np.nan))
                det.append({"a": a, "b": b, "W": float(w.statistic), "Z": z, "p": float(w.pvalue), "r_rb": r_rb(X[a], X[b])})
                ps.append(float(w.pvalue))
            for d_, p in zip(det, holm(ps)):
                d_["p_holm"] = p
            md["pares"] = det
            # Contraste unilateral planificado: Proposed < Static
            if "Proposed" in conds:
                w1 = stats.wilcoxon(X["Proposed"], X["Static"], alternative="less")
                md["proposed_menor_que_static_p"] = float(w1.pvalue)
            out["metricas"][m] = md
        return out

    res["principal"] = comparar("base", "all", principales)
    res["deriva"] = comparar("deriva", "all", principales)
    res["por_rol"] = {rol: comparar("base", "all", principales, {u for u, r in rol_de.items() if r == rol})
                      for rol in C.ROLES}
    res["por_profundidad"] = {p: comparar("base", f"prof{p}", principales) for p in (2, 3)}
    abl = ["Static", "Proposed", "A1 sin barra", "A2 solo barra", "A3 beta=0", "Score-greedy", "Frequency"]
    res["ablacion"] = comparar("base", "all", abl)
    pesos = ["Static"] + sorted({f_["condicion"] for f_ in filas if f_["condicion"].startswith("W ")})
    res["pesos"] = comparar("base", "all", pesos)
    C.guardar_json("rq2.json", res)


# ════════════════════════════════════════════════════════════════════════════
def rq4(usu_tr, seq_tr, prior):
    print("RQ4: latencia en CPU...")
    t_load = []
    for _ in range(20):
        t0 = time.perf_counter()
        net = cargar_predictor("lstm")
        envs, ags = cargar_agentes()
        json.loads(C.CATALOGO.read_text(encoding="utf-8"))
        t_load.append(time.perf_counter() - t0)
    gen, tam = [], []
    rng = np.random.default_rng(0)
    for i in range(1050):
        u = usu_tr[int(rng.integers(len(usu_tr)))]
        s = seq_tr[u["user_id"]]
        t0 = time.perf_counter()
        f_u = Counter(s)
        p = probs(net, s)
        P_u = {r: float(p[C.R2I[r]]) for r in C.RUTAS}
        S = C.score(f_u, P_u, prior[u["rol"]])
        m = menu_propuesto(u["rol"], f_u, S, envs, ags)
        js = json.dumps(m)
        dt = time.perf_counter() - t0
        if i >= 50:                                   # 50 de calentamiento
            gen.append(dt); tam.append(len(js.encode()))
    tamanos = {"lstm_kb": (OE2 / "output_oe2" / "lstm_model.pth").stat().st_size / 1024,
               "agentes_kb": sum((OE3 / "output_oe3" / f"dqn_agent_gs{g}.pt").stat().st_size
                                 for g in ags) / 1024}
    C.guardar_json("rq4.json", {
        "t_load_ms": {"mediana": 1000 * float(np.median(t_load)), "p95": 1000 * float(np.percentile(t_load, 95)), "n": 20},
        "t_gen_ms": {"mediana": 1000 * float(np.median(gen)), "p95": 1000 * float(np.percentile(gen, 95)), "n": len(gen)},
        "menu_bytes": {"media": float(np.mean(tam)), "max": int(np.max(tam))},
        "modelos": tamanos, "cpu_hilos": torch.get_num_threads(),
    })


def main():
    usu_tr, seq_tr = C.poblacion(C.SEMILLA_ENTRENAMIENTO)
    usu_te, seq_te = C.poblacion(C.SEMILLA_EVAL)
    assert not set(seq_tr) - {u["user_id"] for u in usu_tr}
    nets = {n: cargar_predictor(n) for n in ["lstm", "gru", "cnn1d", "transformer"]}
    sel = rq1(usu_tr, seq_tr, usu_te, seq_te, nets)
    print("  modelo seleccionado (menor pérdida de validación):", sel)

    f = {u: dict(Counter(s)) for u, s in seq_tr.items()}           # 1 sesión por usuario
    P = {}
    for u, s in seq_tr.items():
        p = probs(nets[sel], s)
        P[u] = {r: float(p[C.R2I[r]]) for r in C.RUTAS}
    prior = {}
    for rol in C.ROLES:
        c = Counter(r for u in usu_tr if u["rol"] == rol for r in seq_tr[u["user_id"]])
        tot = sum(c.values())
        prior[rol] = {r: n / tot for r, n in c.items()}

    envs, ags = cargar_agentes()
    desempeno_dqn(usu_tr, f, envs, ags)
    filas = rq2(usu_tr, f, P, prior, envs, ags)
    analisis(filas)
    rq4(usu_tr, seq_tr, prior)
    print("[OK] salidas en", C.SALIDAS)


if __name__ == "__main__":
    main()
