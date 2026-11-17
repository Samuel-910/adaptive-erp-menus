# adaptive-erp-menus

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24322522.svg)](https://doi.org/10.5281/zenodo.24322522)

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
venv/Scripts/python reproducir_articulo.py         # en Linux/macOS: venv/bin/python
```

`reproducir_articulo.py` usa los modelos incluidos (no reentrena), genera los 600 usuarios
sintéticos (semilla 42), corre la evaluación y escribe:

- `evaluation/articulo/tables/*.tex` — Tablas 5–12 del artículo;
- `evaluation/articulo/figures/*.png` — Figuras 2 y 3.

Al terminar compara las salidas con las del manuscrito. Todas deben coincidir, salvo la
latencia (Tabla 12), que depende de la máquina. Tarda alrededor de 2 minutos en CPU.

`python run_all_pipeline.py` (sin opciones) vuelve a generar los datos y reentrena todos los
modelos; los resultados reentrenados pueden diferir de los del artículo.

## Datos personales

Este repositorio **no** contiene datos de participantes. Las carpetas `datos_reales/` y
`tools/export/`, y el archivo `tools/.env` con la clave del servidor, están excluidos.

## Cómo citar

Versión del artículo (v1.0.1): https://doi.org/10.5281/zenodo.24326036 · Todas las versiones: https://doi.org/10.5281/zenodo.24322522.
Los datos de cita están en [`CITATION.cff`](CITATION.cff).
