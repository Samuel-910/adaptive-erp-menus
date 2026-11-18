"""Calibración del simulador: sesión piloto frente a usuarios simulados (observación 4 del revisor).

La sesión piloto la hizo un miembro del equipo de investigación (rol Administrador, sin
experiencia previa, menú estático, 2 bloques de 20 tareas). Sus registros NO se incluyen en el
repositorio: este script lee el archivo task_results.csv exportado de la app (fase 2) y guarda
solo valores agregados en salidas/calibracion.json.

Se compara con los usuarios simulados del corpus de entrenamiento (semilla 42) que tienen el
mismo rol y nivel de experiencia; para cada medida se informa la mediana y el intervalo
p5-p95 de los valores por usuario. Se excluyen las tareas de práctica salvo en su propia fila.

Uso: python calibracion.py --piloto RUTA/task_results.csv [--pid P001]
     (requiere datos_sinteticos/, que genera reproducir_articulo.py)
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

import comun as C

ROL, EXPERIENCIA = "Administrador", "ninguna"


def medidas(filas):
    """Medidas de una sesión (lista de tareas de un usuario)."""
    def bloque(b):
        return [f for f in filas if f["block_no"] == b and f["practice"] != "True"]
    b1, b2 = bloque("1"), bloque("2")
    pr = [f for f in filas if f["practice"] == "True"]
    dur = lambda xs: [float(f["duration_ms"]) / 1000 for f in xs]
    return {
        "clics_b1": float(np.mean([float(f["clicks"]) for f in b1])),
        "clics_b2": float(np.mean([float(f["clicks"]) for f in b2])),
        "error_b1": float(np.mean([int(f["wrong_visits"]) > 0 for f in b1])),
        "error_b2": float(np.mean([int(f["wrong_visits"]) > 0 for f in b2])),
        "tiempo_med_b1": float(np.median(dur(b1))),
        "tiempo_med_b2": float(np.median(dur(b2))),
        "tiempo_practica": float(np.mean(dur(pr))),
        "razon_b2_b1": float(np.median(dur(b2)) / np.median(dur(b1))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", required=True, help="task_results.csv exportado de la app (no versionado)")
    ap.add_argument("--pid", default="P001")
    args = ap.parse_args()

    piloto = [f for f in csv.DictReader(open(args.piloto, encoding="utf-8"))
              if f["pid"] == args.pid and f["session_no"] == "1"]
    assert piloto and {f["condition"] for f in piloto} == {"static"}

    datos = C.UR / "datos_sinteticos"
    part = {p["pid"]: p for p in csv.DictReader(open(datos / "participantes_sinteticos.csv", encoding="utf-8"))}
    por_usuario = defaultdict(list)
    for f in csv.DictReader(open(datos / "task_results_sinteticos.csv", encoding="utf-8")):
        p = part[f["pid"]]
        if p["role"] == ROL and p["experience_level"] == EXPERIENCIA:
            por_usuario[f["pid"]].append(f)
    sim = [medidas(x) for x in por_usuario.values()]

    mp = medidas(piloto)
    out = {"rol": ROL, "experiencia": EXPERIENCIA, "n_tareas_piloto": len(piloto),
           "n_usuarios_simulados": len(sim), "medidas": {}}
    for k, v in mp.items():
        s = [m[k] for m in sim]
        p5, p50, p95 = np.percentile(s, [5, 50, 95])
        out["medidas"][k] = {"piloto": v, "sim_mediana": float(p50), "sim_p5": float(p5), "sim_p95": float(p95)}
        print(f"  {k:16s} piloto {v:7.2f}   simulado {p50:7.2f} [{p5:.2f}, {p95:.2f}]")
    C.guardar_json("calibracion.json", out)
    print("[OK]", C.SALIDAS / "calibracion.json")


if __name__ == "__main__":
    main()
