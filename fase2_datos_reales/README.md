# Fase 2 — Datos reales (estudio con usuarios)

Estudio con participantes: cada persona hace la **sesión 1** con el menú estático del ERP
y, al menos dos días después, la **sesión 2** con su menú adaptado. El menú adaptado se
genera **solo con los datos de su sesión 1**, usando los modelos de la fase 1 sin reentrenar.

| Ruta | Qué contiene |
|---|---|
| `estudio-menus/app/` | App web (Vite + React) del estudio: consentimiento, cuestionario inicial, tareas, cuestionarios SUS y de satisfacción. |
| `estudio-menus/supabase/` | Esquema de la base de datos (tablas, RLS y funciones). |
| `estudio-menus/tools/` | Scripts del investigador: `export_logs.py` (exporta registros), `import_menus.py` (sube menús adaptados). |
| `ver_menus.py` | Vista previa del menú estático frente al adaptado de cada participante antes de subirlo. |

## Generar y subir los menús adaptados

```bash
# 1. Exportar la sesión 1 (requiere estudio-menus/tools/.env con la clave del servidor)
cd estudio-menus/tools
python export_logs.py export --sesion 1
# 2. Copiar export/* a fase1_datos_sinteticos/pipeline/datos_reales/ y generar los menús
cd ../../../fase1_datos_sinteticos/pipeline
python run_all_pipeline.py --solo-reales --excluir P001
# 3. Revisar y subir
cd ../../fase2_datos_reales
python ver_menus.py --solo P0xx,P0yy
cd estudio-menus/tools
python import_menus.py ../../../fase1_datos_sinteticos/pipeline/OE3_Agente_DQN/output_oe3/menus_para_subir.json --modelo lstm-dqn-v1
```

## Datos personales

Los datos de participantes **no** están en este repositorio. Se guardan solo en la base de
datos del estudio y en copias locales del investigador (`datos_reales/`, `tools/export/`),
excluidas por `.gitignore`.
