# adaptive-erp-menus

Código del trabajo sobre personalización de menús de un ERP con modelos secuenciales
(LSTM, GRU, 1D-CNN, Transformer), un puntaje de relevancia, agentes DQN y una barra de
sugerencias, bajo control de acceso por roles (RBAC).

El trabajo tiene dos fases:

| Fase | Carpeta | Datos | Qué contiene |
|---|---|---|---|
| **1** | [`fase1_datos_sinteticos/`](fase1_datos_sinteticos/) | Usuarios **simulados** | Simulador, pipeline OE1–OE3 (predictores y agentes DQN) y la evaluación que genera las tablas y figuras del artículo. |
| **2** | [`fase2_datos_reales/`](fase2_datos_reales/) | Participantes **reales** | App web del estudio con usuarios (dos sesiones: menú estático y menú adaptado), esquema de la base de datos y scripts para exportar los registros y subir los menús adaptados. |

El menú del ERP (9 grupos, 60 funciones, 4 roles) está definido una sola vez en
`fase2_datos_reales/estudio-menus/app/src/data/catalog.json` y lo usan las dos fases.

## Reproducir los resultados del artículo (fase 1)

```bash
cd fase1_datos_sinteticos/pipeline
python -m venv venv
venv/Scripts/pip install -r requirements.txt      # en Linux/macOS: venv/bin/pip
python run_all_pipeline.py --sin-entrenar          # datos sintéticos + puntajes con los modelos incluidos
cd evaluation
python evaluacion.py                               # salidas/*.json
python tablas.py                                   # articulo/tables/*.tex y articulo/figures/*
```

`python run_all_pipeline.py` (sin opciones) vuelve a generar los datos y reentrena todos los modelos.

## Datos personales

Este repositorio **no** contiene datos de participantes. Las carpetas `datos_reales/` y
`tools/export/`, y el archivo `tools/.env` con la clave del servidor, están excluidos.
