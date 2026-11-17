# ⚠️ AVISO — Versión anterior del pipeline

Esta carpeta conserva la **primera versión** del pipeline (antes del menú real del ERP y del
artículo). Se mantiene solo como historial del proyecto. **No genera ninguno de los
resultados del artículo** y no debe usarse para reportar resultados.

## Qué NO son mediciones

- **M2 (tiempo) y M3 (errores) “post”** en `OE5_Evaluacion/main.py` **no se miden**: se
  calculan multiplicando la reducción de M1 por 1.2 y 1.4 y sumando ruido aleatorio
  (sección “Proyectar M2 y M3 basados en la Literatura”).
- **La satisfacción** en `OE5_Evaluacion/` **no viene de personas**: es una “Encuesta de
  Satisfacción Sintética” generada por el propio script.
- Los archivos de `OE5_Evaluacion/output_oe5/` (`datos_pre_post.csv`, `satisfaccion.csv`,
  `analisis_estadistico.csv`) son, por tanto, **valores proyectados o generados**.
- Todos los registros de navegación de esta versión son **sintéticos** (simulador de
  `00_Generador_Datos/`); no provienen de un ERP en producción.

## Dónde están los resultados válidos

- Fase 1 (simulación del artículo): [`../pipeline/`](../pipeline/), con la evaluación en
  `../pipeline/evaluation/`.
- Fase 2 (participantes reales): [`../../fase2_datos_reales/`](../../fase2_datos_reales/).
