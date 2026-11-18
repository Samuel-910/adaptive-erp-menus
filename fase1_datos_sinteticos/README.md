# Fase 1 — Datos sintéticos

Toda la evidencia de esta fase proviene de **usuarios simulados**.

## `pipeline/` — versión del artículo

| Carpeta | Qué hace |
|---|---|
| `00_Generador_Datos/` | `construir_config_erp.py` arma la configuración desde el menú real del ERP; `simulador_erp_real.py` genera los registros sintéticos (semilla 42). |
| `OE1_Baseline/` | Sesiones, filtrado, perfiles por usuario y costo de navegación del menú estático. |
| `OE2_Modelado/` | Predictores de la siguiente ruta (LSTM, GRU, 1D-CNN, Transformer) y puntaje de relevancia. Incluye los modelos entrenados (`output_oe2/*.pth`). |
| `OE3_Agente_DQN/` | Agentes DQN, uno por tamaño de grupo, con acciones de “mover al inicio”. Incluye los agentes entrenados (`output_oe3/*.pt`). |
| `evaluation/` | Evaluación del artículo: RQ1–RQ4, ablación, sensibilidad de pesos y deriva. `evaluacion.py` escribe `salidas/`; `tablas.py` escribe las tablas y figuras del manuscrito en `articulo/`. |

Las salidas en `evaluation/salidas/` son las que se usaron en el manuscrito. Qué archivo respalda cada tabla y figura, qué significa cada columna y dónde están los registros de entrenamiento: [`pipeline/evaluation/SALIDAS.md`](pipeline/evaluation/SALIDAS.md).

**Para regenerar todas las tablas y figuras del artículo:** `python pipeline/reproducir_articulo.py`
(sin reentrenar; compara el resultado con las salidas del manuscrito).

Análisis de los supuestos del simulador (desde la versión 1.1), en `pipeline/evaluation/`:

- `python sensibilidad.py`: sensibilidad al exponente Zipf, al número de funciones típicas, al tamaño de la barra, a los tiempos y a la probabilidad de error (`salidas/sensibilidad_*.json`, figura `fig_sensitivity.png`);
- `python analitica.py`: aproximación analítica de la reducción de clics frente a la simulación (`salidas/aproximacion_analitica.json`);
- `python calibracion.py --piloto <task_results.csv>`: sesión piloto frente a usuarios simulados (`salidas/calibracion.json`, tabla `tab_calibration.tex`); los registros del piloto no se incluyen, solo los valores agregados.

Análisis agregados en la revisión (versión 1.2):

- `python memoria_espacial.py`: modelo secundario con memoria espacial y costo de reaprendizaje (`salidas/memoria_espacial.json`, tabla `tab_memory.tex`);
- `python arboles_sinteticos.py`: 36 árboles de menú sintéticos de 20 a 300 funciones (`salidas/arboles_sinteticos.json`, tabla `tab_trees.tex`);
- `python montecarlo.py`: error de Monte Carlo y umbral de relevancia (`salidas/montecarlo.json`, columna MCSE de la Tabla 8);
- `python multisemilla_entrenar.py` (GPU, unas 2 h) y `python multisemilla_evaluar.py`: predictores y agentes con cinco semillas, validación partida por usuario (`salidas/multisemilla.json`, tabla `tab_seeds.tex`).

`tablas.py` (y por tanto `reproducir_articulo.py`) arma las tablas y figuras a partir de esos JSON ya guardados.

Opciones de `run_all_pipeline.py`:

- sin opciones: genera los datos sintéticos y reentrena todo;
- `--sin-entrenar`: reutiliza los modelos incluidos;
- `--solo-reales`: usa los datos de la fase 2 (`datos_reales/`, no incluidos) para generar menús adaptados con los modelos ya entrenados.
