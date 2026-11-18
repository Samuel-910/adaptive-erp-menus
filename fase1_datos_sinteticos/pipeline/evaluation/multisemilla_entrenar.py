"""Reentrenamiento con varias semillas (observación 7 del revisor): predictores y agentes DQN.

No reemplaza a los modelos del artículo (OE2_Modelado/output_oe2, OE3_Agente_DQN/output_oe3), que
son los que generaron los menús del estudio con usuarios. Guarda cada corrida en
evaluation/multisemilla/semilla_<n>/ para que multisemilla_evaluar.py las compare.

Diferencias con el entrenamiento original, todas deliberadas:
  - corpus: solo la población sintética que genera el código del repositorio (600 usuarios,
    semilla 42), sin la sesión piloto, cuyos registros no se publican; las secuencias son las
    funciones objetivo de cada sesión (la misma representación que en la prueba de la RQ1);
  - validación partida por usuario (85 % / 15 % de los usuarios), no por ventana, para que
    ninguna ventana de validación comparta rutas con ventanas de entrenamiento;
  - tope de 300 épocas para las cuatro arquitecturas (con la misma parada temprana de 10
    épocas sin mejora), para que el tope no corte al Transformer.
Los hiperparámetros son los del artículo (Tabla 3 y Tabla de configuración del DQN).

Uso: python multisemilla_entrenar.py [--semillas 1 2 3 4 5]   (usa GPU si hay)
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import comun as C

OE2 = C.UR / "OE2_Modelado"
OE3 = C.UR / "OE3_Agente_DQN"
SALIDA = C.AQUI / "multisemilla"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V = len(C.RUTAS)
HP = dict(embed=32, hidden=64, layers=2, dropout=0.2, heads=4)
LR = {"lstm": 1e-3, "gru": 1e-3, "cnn1d": 1e-3, "transformer": 5e-4}
EPOCAS, PACIENCIA, BATCH = 300, 10, 32

_cwd = os.getcwd()
os.chdir(OE3)
sys.path.insert(0, str(OE3))
import config as cfg3  # noqa: E402
from env import GroupMenuEnv  # noqa: E402
from agent import DQNAgent  # noqa: E402
os.chdir(_cwd)


def fijar_semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def modelo(nombre):
    spec = importlib.util.spec_from_file_location(nombre, OE2 / "models" / f"{nombre}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if nombre == "lstm":
        return m.LSTMModel(V, HP["embed"], HP["hidden"], HP["layers"], HP["dropout"], V)
    if nombre == "gru":
        return m.GRUModel(V, HP["embed"], HP["hidden"], HP["layers"], HP["dropout"], V)
    if nombre == "cnn1d":
        return m.CNN1DModel(V, HP["embed"], HP["hidden"], HP["dropout"], V)
    net = m.TransformerNavModel(V, HP["embed"], HP["heads"], HP["layers"], HP["dropout"], V)
    net._init_weights()
    return net


def ventanas(seqs):
    X, y = [], []
    for s in seqs:
        for i in range(1, len(s)):
            ids = [C.R2I[r] + 1 for r in s[max(0, i - C.SEQ_LEN):i]]
            X.append([0] * (C.SEQ_LEN - len(ids)) + ids)
            y.append(C.R2I[s[i]])
    return torch.tensor(X), torch.tensor(y)


def entrenar_predictor(nombre, tr, va):
    net = modelo(nombre).to(DEV)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=LR[nombre], eps=1e-8, weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-6)
    hist = {"train_loss": [], "val_loss": []}
    mejor, estado, sin_mejora = float("inf"), None, 0
    for ep in range(EPOCAS):
        net.train(); tl, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); loss = crit(net(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            tl += loss.item() * len(yb); n += len(yb)
        net.eval(); vl, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(DEV), yb.to(DEV)
                vl += crit(net(xb), yb).item() * len(yb); m += len(yb)
        vl /= m; sch.step(vl)
        hist["train_loss"].append(tl / n); hist["val_loss"].append(vl)
        if vl < mejor:
            mejor, sin_mejora = vl, 0
            estado = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    hist["mejor_epoca"] = int(np.argmin(hist["val_loss"])) + 1
    hist["epocas"] = len(hist["val_loss"])
    hist["val_loss_min"] = mejor
    return estado, hist


def entrenar_dqn(freqs):
    """Mismo bucle que OE3_Agente_DQN/main.py: un usuario de entrenamiento al azar por episodio."""
    tam = sorted({len(v) for v in cfg3.GRUPOS_RUTAS.values()})
    envs = {gs: GroupMenuEnv(gs, top_k=cfg3.TOP_K) for gs in tam}
    ags = {gs: DQNAgent(envs[gs].state_dim, envs[gs].n_actions) for gs in tam if gs >= 2}
    usuarios = list(freqs)
    for ep in range(cfg3.RL_EPISODES):
        f = freqs[random.choice(usuarios)]
        for rutas in cfg3.GRUPOS_RUTAS.values():
            gs = len(rutas)
            if gs < 2:
                continue
            env, ag = envs[gs], ags[gs]
            s = env.reset(np.array([f.get(r, 0.0) for r in rutas], dtype=float))
            for _ in range(cfg3.STEPS_PER_EP):
                a = ag.act(s)
                ns, r, done = env.step(a)
                ag.buf.push(s, a, r, ns, float(done))
                if ep >= cfg3.WARMUP_EPS:
                    ag.update()
                s = ns
                if done:
                    break
    return {gs: {k: v.detach().cpu().clone() for k, v in ag.q_net.state_dict().items()} for gs, ag in ags.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--solo", choices=["predictores", "dqn"], default=None)
    args = ap.parse_args()

    usu, seq = C.poblacion(C.SEMILLA_ENTRENAMIENTO)
    freqs = {u: dict(Counter(s)) for u, s in seq.items()}
    print(f"Dispositivo: {DEV} | usuarios: {len(usu)}", flush=True)
    for sem in args.semillas:
        dest = SALIDA / f"semilla_{sem}"
        dest.mkdir(parents=True, exist_ok=True)
        if args.solo != "dqn":
            rnd = random.Random(sem)
            ids = sorted(seq); rnd.shuffle(ids)
            n_va = round(0.15 * len(ids))
            val_u, tr_u = set(ids[:n_va]), ids[n_va:]
            (dest / "particion.json").write_text(json.dumps({"validacion": sorted(val_u)}), encoding="utf-8")
            Xtr, ytr = ventanas([seq[u] for u in tr_u])
            Xva, yva = ventanas([seq[u] for u in sorted(val_u)])
            for nombre in ["lstm", "gru", "cnn1d", "transformer"]:
                fijar_semilla(sem)
                tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
                va = DataLoader(TensorDataset(Xva, yva), batch_size=256)
                t0 = time.time()
                estado, hist = entrenar_predictor(nombre, tr, va)
                torch.save(estado, dest / f"{nombre}.pth")
                hist["segundos"] = round(time.time() - t0, 1)
                (dest / f"historial_{nombre}.json").write_text(json.dumps(hist), encoding="utf-8")
                print(f"semilla {sem} {nombre:11s} val_loss {hist['val_loss_min']:.4f} "
                      f"época {hist['mejor_epoca']}/{hist['epocas']} ({hist['segundos']} s)", flush=True)
        if args.solo != "predictores":
            fijar_semilla(sem)
            t0 = time.time()
            for gs, estado in entrenar_dqn(freqs).items():
                torch.save(estado, dest / f"dqn_gs{gs}.pt")
            print(f"semilla {sem} DQN ({time.time() - t0:.0f} s)", flush=True)
    print("[OK]", SALIDA)


if __name__ == "__main__":
    main()
