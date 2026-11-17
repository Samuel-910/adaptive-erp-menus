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

Las salidas en `evaluation/salidas/` son las que se usaron en el manuscrito.

Opciones de `run_all_pipeline.py`:

- sin opciones: genera los datos sintéticos y reentrena todo;
- `--sin-entrenar`: reutiliza los modelos incluidos;
- `--solo-reales`: usa los datos de la fase 2 (`datos_reales/`, no incluidos) para generar menús adaptados con los modelos ya entrenados.
