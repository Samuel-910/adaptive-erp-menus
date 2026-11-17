# Salidas almacenadas y registros de entrenamiento

Guía del material que respalda las tablas y figuras del artículo. Todos los datos son
**simulados** (600 usuarios sintéticos, semilla 42). Rutas relativas a `fase1_datos_sinteticos/pipeline/`.

`python reproducir_articulo.py` vuelve a generar estos archivos y comprueba que coinciden
con los versionados; `evaluation/tablas.py` arma las tablas y figuras solo a partir de ellos.

## Qué archivo respalda cada tabla y figura

| Tabla / figura | Archivo `tablas.py` | Fuente |
|---|---|---|
| Tabla 5 (corpus) | `tab_corpus.tex` | datos sintéticos regenerados (`datos_sinteticos/`) |
| Tabla 6 (RQ1, predictores) | `tab_rq1.tex` | `evaluation/salidas/rq1.json` |
| Tabla 7 (agentes DQN) | `tab_dqn.tex` | `evaluation/salidas/dqn.json` |
| Tabla 8 (costo por política) | `tab_policies.tex` | `evaluation/salidas/rq2.json` → `principal` |
| Tabla 9 (subgrupos y deriva) | `tab_subgroups.tex` | `rq2.json` → `por_rol`, `por_profundidad`, `deriva` |
| Tabla 10 (ablación) | `tab_ablation.tex` | `rq2.json` → `ablacion` |
| Tabla 11 (pesos del puntaje) | `tab_weights.tex` | `rq2.json` → `pesos` |
| Tabla 12 (latencia) | `tab_latency.tex` | `evaluation/salidas/rq4.json` (depende de la máquina) |
| Fig. 2 (curvas de entrenamiento) | `fig_curves.png` | `OE2_Modelado/output_oe2/historial_{lstm,gru,cnn1d,transformer}.json` |
| Fig. 3 (costo por política) | `fig_policies.png` | `rq2.json` |

Las Tablas 8–11 son resúmenes (medias, IC bootstrap, Wilcoxon con Holm, tamaño de efecto)
de `rq2_por_usuario.csv`, que contiene el resultado de **cada usuario con cada política**.

## `evaluation/salidas/`

### `rq2_por_usuario.csv` — resultados por usuario y política

Una fila por usuario × escenario × política × subconjunto de tareas (31 395 filas, 600 usuarios).

| Columna | Significado |
|---|---|
| `user` | Usuario simulado (`SIM0001` … `SIM0600`). |
| `role` | Rol: Ventas, Compras, Almacen, Administrador. |
| `experiencia` | Nivel de experiencia del usuario simulado. |
| `escenario` | `base`: tareas del mismo perfil con el que se calculan los menús. `deriva`: perfil con dos pares de funciones intercambiados entre entrenamiento y evaluación. |
| `condicion` | Política o variante (ver abajo). |
| `subconjunto` | `all`: las 40 tareas de evaluación; `prof2` / `prof3`: solo tareas cuyo destino está a profundidad 2 o 3 en el menú estático. |
| `n` | Número de tareas del subconjunto. |
| `M1` | Clics medios por tarea, incluidos los de funciones equivocadas (Ec. M1). |
| `M2` | Tiempo de búsqueda medio por tarea, en segundos (Ec. M2). |
| `M3` | Proporción de tareas con al menos una función equivocada (tasa de error). |

Valores de `condicion`:

- `Static`, `Frequency`, `Score-greedy`, `Proposed`: políticas comparadas (Tabla 8; con `escenario=deriva`, Tabla 9).
- `A1 sin barra`, `A2 solo barra`, `A3 beta=0`: variantes de la ablación (Tabla 10): sin barra de sugerencias; solo la barra, con el orden estático; puntaje sin el modelo secuencial (β = 0).
- `W a/b/c`: sensibilidad a los pesos (α, β, γ) del puntaje de relevancia (Tabla 11). Los pesos del artículo son `W 0.50/0.30/0.20`.

Todas las políticas se evaluaron con las mismas tareas y los mismos números aleatorios por usuario.

### Otros archivos

| Archivo | Contenido |
|---|---|
| `rq1.json` | RQ1: métricas de prueba de LSTM, GRU, 1D-CNN, Transformer y de las referencias (Markov, más frecuente, oráculo); resumen del entrenamiento de cada modelo (mejor época, pérdida de validación); modelo seleccionado; pruebas de Wilcoxon. |
| `dqn.json` | Calidad de las cabeceras ordenadas por los agentes DQN, por tamaño de grupo (grupos evaluados, % óptimo, φ medio y mínimo, pasos). |
| `rq2.json` | Resúmenes estadísticos de las Tablas 8–11 calculados a partir de `rq2_por_usuario.csv`. |
| `rq2_dqn_optimo.json` | Porcentaje de grupos en los que el agente alcanza el orden óptimo durante la evaluación de políticas. |
| `sensibilidad_zipf.json` | Sensibilidad al exponente Zipf *s* del perfil (0.5–2): para cada *s*, proporción de tareas servidas por la barra y M1–M3 de Static, Frequency, Score-greedy, Proposed y solo barra, con reducción frente a Static e IC 95 %. Lo genera `sensibilidad_zipf.py` (no lo corre `reproducir_articulo.py`). |
| `rq4.json` | Latencia y tamaño (carga de modelos, generación del menú, bytes del menú, tamaño de modelos y agentes). Medido en la laptop del experimento; en otra máquina cambia. |

## Registros de entrenamiento

| Archivo | Contenido |
|---|---|
| `OE2_Modelado/output_oe2/historial_{lstm,gru,cnn1d,transformer}.json` | Pérdida y exactitud de entrenamiento y validación por época de cada predictor (`train_loss`, `val_loss`, `train_acc`, `val_acc`). |
| `OE2_Modelado/output_oe2/metricas_test.json` | Métricas de prueba de los predictores al terminar el entrenamiento. |
| `OE2_Modelado/output_oe2/*.pth` | Pesos de los predictores entrenados que usa la evaluación. |
| `OE3_Agente_DQN/oe3_reentreno.log` | Registro del entrenamiento de los agentes DQN (3000 episodios): ε y calidad de la cabecera por tamaño de grupo cada 500 episodios. `evaluacion.py` copia estas líneas en `dqn.json` (`curva_entrenamiento`). |
| `OE3_Agente_DQN/output_oe3/dqn_agent_gs*.pt` | Agentes DQN entrenados, uno por tamaño de grupo. |

`OE2_Modelado/output_oe2/historial_entrenamiento.json` es de una versión anterior del
pipeline y no se usa en el artículo. `curvas_entrenamiento.png` es la gráfica de los mismos
historiales que hace `OE2_Modelado/plot_benchmark.py` (el artículo usa `fig_curves.png`).
