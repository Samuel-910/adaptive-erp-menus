"""Regenera todas las tablas y figuras del artículo con un solo comando.

Pasos (sin reentrenar; usa los modelos incluidos en OE2_Modelado/output_oe2 y
OE3_Agente_DQN/output_oe3):
  1. 00_Generador_Datos/construir_config_erp.py  menú real del ERP -> configuración
  2. 00_Generador_Datos/simulador_erp_real.py    600 usuarios sintéticos (semilla 42)
  3. evaluation/evaluacion.py                    RQ1-RQ4, DQN, ablación, pesos, deriva -> evaluation/salidas/
  4. evaluation/tablas.py                        evaluation/articulo/tables/*.tex (Tablas 5-12)
                                                 evaluation/articulo/figures/*.png (Figs. 2-3)

Al final compara las salidas nuevas con las del manuscrito (las versionadas en
evaluation/salidas/). La latencia (rq4.json, Tabla 12) depende de la máquina y
no se espera que coincida.

Uso:
  python reproducir_articulo.py
"""
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EVAL = AQUI / "evaluation"
SALIDAS = EVAL / "salidas"
DEPENDE_DE_LA_MAQUINA = {"rq4.json"}

PASOS = [
    ("00_Generador_Datos", "construir_config_erp.py"),
    ("00_Generador_Datos", "simulador_erp_real.py"),
    ("evaluation", "evaluacion.py"),
    ("evaluation", "tablas.py"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    referencia = Path(tempfile.mkdtemp(prefix="salidas_manuscrito_"))
    for f in SALIDAS.iterdir():
        shutil.copy2(f, referencia / f.name)

    t0 = time.time()
    for carpeta, script in PASOS:
        print(f"\n=== {carpeta}/{script}", flush=True)
        r = subprocess.run([sys.executable, script], cwd=AQUI / carpeta, env=env)
        if r.returncode != 0:
            sys.exit(f"[ERROR] {carpeta}/{script} terminó con código {r.returncode}")

    print(f"\n=== Comparación con las salidas del manuscrito ({time.time() - t0:.0f} s)")
    distintos = []
    for f in sorted(referencia.iterdir()):
        igual = filecmp.cmp(f, SALIDAS / f.name, shallow=False)
        nota = " (depende de la máquina)" if f.name in DEPENDE_DE_LA_MAQUINA else ""
        print(f"  {'IGUAL  ' if igual else 'DISTINTO'} {f.name}{nota}")
        if not igual and f.name not in DEPENDE_DE_LA_MAQUINA:
            distintos.append(f.name)
    for f in DEPENDE_DE_LA_MAQUINA:  # se conserva el valor reportado en el manuscrito
        shutil.copy2(referencia / f, SALIDAS / f)
    shutil.rmtree(referencia, ignore_errors=True)

    print(f"\nTablas:  {EVAL / 'articulo' / 'tables'}")
    print(f"Figuras: {EVAL / 'articulo' / 'figures'}")
    if distintos:
        sys.exit(f"[ATENCIÓN] No coinciden con el manuscrito: {', '.join(distintos)}")
    print("[OK] Resultados idénticos a los del manuscrito.")


if __name__ == "__main__":
    main()
