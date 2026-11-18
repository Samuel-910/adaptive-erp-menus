"""Genera las tablas (.tex) y figuras del artículo a partir de salidas/ (trazabilidad: cada
tabla y figura del manuscrito sale de este script). Uso: python tablas.py"""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import comun as C  # noqa: E402

TAB = C.ARTICULO / "tables"
FIG = C.ARTICULO / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
ROL_EN = {"Ventas": "Sales", "Compras": "Purchasing", "Almacen": "Warehouse", "Administrador": "Administrator"}
MOD_EN = {"lstm": "LSTM", "gru": "GRU", "cnn1d": "1D-CNN", "transformer": "Transformer",
          "markov": "First-order Markov", "mas_frecuente": "Most frequent route", "oraculo": "True profile (ceiling)",
          "prior_rol": "Role prior", "conteo_ventana": "Count in window", "conteo_ventana_prior": "Count in window + role prior"}
POL = ["Static", "Frequency", "Score-greedy", "Proposed"]
POL_TEX = {p: r"\textsc{" + p + "}" for p in POL}


def guardar(nombre, cuerpo):
    (TAB / nombre).write_text("% Generado por fase1_datos_sinteticos/pipeline/evaluation/tablas.py -- no editar a mano\n" + cuerpo,
                              encoding="utf-8")


def fp(p):
    """Valores p secundarios (observación 8): por debajo de 0.001 solo se informa '< 0.001'."""
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}"


def tabla_corpus():
    t = list(csv.DictReader(open(C.UR / "datos_sinteticos/task_results_sinteticos.csv", encoding="utf-8")))
    p = {r["pid"]: r for r in csv.DictReader(open(C.UR / "datos_sinteticos/participantes_sinteticos.csv", encoding="utf-8"))}
    logs = [json.loads(l) for l in open(C.UR / "datos_sinteticos/logs_sinteticos_erp.ndjson", encoding="utf-8")]
    filas, tot = [], Counter()
    for rol in C.ROLES:
        tt = [x for x in t if p[x["pid"]]["role"] == rol]
        ll = [x for x in logs if x["role"] == rol]
        d = {"u": len({x["pid"] for x in tt}), "t": len(tt), "e": len(ll),
             "w": sum(x.get("es_objetivo") is False for x in ll), "r": len({x["target_route"] for x in tt}),
             "a": len(C.rutas_rol(rol))}
        tot.update({k: v for k, v in d.items() if k in "utew"})
        filas.append(f"{ROL_EN[rol]} & {d['u']} & {d['t']:,} & {d['e']:,} & {d['w']:,} & {d['r']} of {d['a']} \\\\")
    filas.append(r"\midrule")
    n_obj = len({x["target_route"] for x in t})
    filas.append(f"Total & {tot['u']} & {tot['t']:,} & {tot['e']:,} & {tot['w']:,} & {n_obj} of 60 \\\\")
    guardar("tab_corpus.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Training corpus (simulated session~1 of the user study, seed 42). Events are menu navigations; wrong visits are navigations to a function other than the task target. Target functions: the eight typical functions of each role.}\label{tab:dataset}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Role} & \textbf{Users} & \textbf{Tasks} & \textbf{Events} & \textbf{Wrong visits} & \textbf{Target functions} \\
\midrule
""" + "\n".join(filas).replace(",", "{,}") + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_rq1():
    r = C.leer_json("rq1.json")
    filas = []
    par = r.get("parametros", {})
    for n in ["lstm", "gru", "cnn1d", "transformer", "markov", "mas_frecuente", "prior_rol", "conteo_ventana",
              "conteo_ventana_prior", "oraculo"]:
        m = r["modelos"][n]
        h = r["historial"].get(n)
        vl = f"{h['val_loss']:.4f}" if h else "--"
        ep = f"{h['mejor_epoca']}/{h['epocas']}" if h else "--"
        nom = MOD_EN[n] + (" (selected)" if n == r["seleccionado"] else "")
        cel = [f"{m[k]:.3f} [{m[k + '_ic'][0]:.3f}, {m[k + '_ic'][1]:.3f}]" for k in ["acc1", "acc3", "mrr", "ndcg5"]]
        if n == "markov":
            filas.append(r"\midrule")
        if n == "oraculo":
            filas.append(r"\midrule")
        pp = f"{par[n]:,}".replace(",", "{,}") if n in par else "--"
        filas.append(f"{nom} & {pp} & {vl} & {ep} & " + " & ".join(cel) + r" \\")
    guardar("tab_rq1.tex", r"""\begin{table}[htbp]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Next-route prediction on """ + str(r["n_usuarios_prueba"]) + r""" new simulated users not seen in training (""" + f"{r['n_muestras']:,}".replace(",", "{,}") + r""" predictions; mean over users with bootstrapped 95\% CI). Params: trainable parameters; Val.\ loss: lowest validation cross-entropy (selection criterion); Epoch: best/total epochs (limit 50). Count in window: occurrences of each route in the last ten routes of the session; role prior: frequency of each route in the training corpus of the role; the previous route is excluded in both, as the task protocol never repeats it. Ties ranked pessimistically. The true-profile row is a theoretical ceiling, not a model.}\label{tab:seq}
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{Val.\ loss} & \textbf{Epoch} & \textbf{Acc@1} & \textbf{Acc@3} & \textbf{MRR} & \textbf{NDCG@5} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_dqn():
    d = C.leer_json("dqn.json")["por_tamano"]
    filas = [f"{gs} & {v['grupos_evaluados']} & {v['pct_optimo']:.1f} & {v['phi_media']:.4f} & {v['phi_min']:.3f} & {v['pasos_medios']:.1f} \\\\"
             for gs, v in sorted(d.items(), key=lambda x: int(x[0]))]
    guardar("tab_dqn.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Greedy performance of the trained DQN agents on the groups of the 600 training users (one group instance per user and authorized group with at least one used item). $\phi$: normalized DCG of the group header (1 = frequency-optimal header). Single training run (seed 42).}\label{tab:dqn}
\begin{tabular}{rrrrrr}
\toprule
\textbf{Group size} & \textbf{Instances} & \textbf{Optimal header (\%)} & \textbf{Mean $\phi$} & \textbf{Min.\ $\phi$} & \textbf{Mean steps} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def _celda(md, c):
    v = md["por_condicion"][c]
    return v


def tabla_politicas(clave="principal", nombre="tab_policies.tex", etiqueta="tab:cost", titulo=None):
    r = C.leer_json("rq2.json")[clave]
    mc = {}
    if (C.SALIDAS / "montecarlo.json").exists():
        for f in C.leer_json("montecarlo.json")["comparaciones"]:
            if f["escenario"] == "base" and f["referencia"] == "Static":
                mc[f["politica"]] = f
    filas = []
    unidades = {"M1": "$M_1$ (clicks)", "M2": "$M_2$ (s)", "M3": "$M_3$ (error rate)"}
    for m in ["M1", "M2", "M3"]:
        md = r["metricas"][m]
        pares = {(p["a"], p["b"]): p for p in md["pares"]}
        for i, c in enumerate(POL):
            v = md["por_condicion"][c]
            dec = 3
            med = f"{v['mediana']:.{dec}f} [{v['q1']:.{dec}f}, {v['q3']:.{dec}f}]"
            if c == "Static":
                red, se, ph, rr = "--", "--", "--", "--"
            else:
                red = f"{v['reduccion_vs_static_pct']:.1f} [{v['ic95'][0]:.1f}, {v['ic95'][1]:.1f}]"
                se = f"{mc[c][m]['mcse_pct']:.2f}" if c in mc else "--"
                p = pares[("Static", c)]
                ph, rr = fp(p["p_holm"]), f"{p['r_rb']:.2f}"
            filas.append(f"{unidades[m] if i == 0 else ''} & {POL_TEX[c]} & {v['media']:.3f} & {med} & {red} & {se} & {ph} & {rr} \\\\")
        fr = md["friedman"]
        filas.append(rf"\multicolumn{{8}}{{l}}{{\footnotesize Friedman $\chi^2({fr['gl']})={fr['chi2']:.1f}$, $p$ {fp(fr['p'])}; \textsc{{Proposed}} vs.\ \textsc{{Score-greedy}}: $p_{{\mathrm{{Holm}}}}$ = {fp(pares[('Score-greedy', 'Proposed')]['p_holm'])}, $r_{{rb}}$ = {pares[('Score-greedy', 'Proposed')]['r_rb']:.2f}}} \\")
        if m != "M3":
            filas.append(r"\midrule")
    titulo = titulo or (r"Navigation cost measured on the evaluation tasks for each policy ($n=" + str(r["n_usuarios"]) +
                        r"$ users; per-user means). Reduction: relative reduction of the mean with respect to \textsc{Static} with bootstrapped 95\% CI; MCSE: Monte Carlo standard error of the reduction (percentage points; bootstrap over users); $p_{\mathrm{Holm}}$ (secondary, see Section~\ref{sec:stats}): two-sided Wilcoxon signed-rank test against \textsc{Static} with Holm correction over the six pairwise comparisons; $r_{rb}$: matched-pairs rank-biserial correlation (positive = lower cost than \textsc{Static}).")
    guardar(nombre, r"""\begin{table}[htbp]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{""" + titulo + r"""}\label{""" + etiqueta + r"""}
\begin{tabular}{llcccccc}
\toprule
\textbf{Metric} & \textbf{Policy} & \textbf{Mean} & \textbf{Median [IQR]} & \textbf{Reduction \% [95\% CI]} & \textbf{MCSE} & \textbf{$p_{\mathrm{Holm}}$} & \textbf{$r_{rb}$} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_ablacion():
    r = C.leer_json("rq2.json")
    a = r["ablacion"]["metricas"]
    nombres = [("Proposed", r"\textsc{Proposed} (full model)"),
               ("A1 sin barra", r"(i) without suggestion bar"),
               ("A2 solo barra", r"(ii) suggestion bar only (static order)"),
               ("A3 beta=0", r"(iii) score without sequential model ($\beta=0$)"),
               ("Score-greedy", r"(iv) without DQN agents (\textsc{Score-greedy})"),
               ("Frequency", r"\textsc{Frequency} (reference)")]
    filas = [f"{t} & " + " & ".join(f"{a[m]['por_condicion'][c]['reduccion_vs_static_pct']:.1f} [{a[m]['por_condicion'][c]['ic95'][0]:.1f}, {a[m]['por_condicion'][c]['ic95'][1]:.1f}]"
                                     for m in ["M1", "M2", "M3"]) + r" \\" for c, t in nombres]
    w = r["pesos"]["metricas"]
    pesos = sorted((c for c in w["M1"]["por_condicion"] if c.startswith("W ")),
                   key=lambda c: tuple(-float(x) for x in c[2:].split("/")))
    fw = []
    for c in pesos:
        a_, b_, c_ = (float(x) for x in c[2:].split("/"))
        fw.append(f"({a_:.2f}, {b_:.2f}, {c_:.2f}) & " + " & ".join(
            f"{w[m]['por_condicion'][c]['reduccion_vs_static_pct']:.1f}" for m in ["M1", "M2", "M3"]) + r" \\")
    guardar("tab_ablation.tex", r"""\begin{table}[htbp]
\centering\scriptsize
\caption{Ablation study: relative reduction (\%) of each metric with respect to \textsc{Static}, with bootstrapped 95\% CI ($n=600$ users).}\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{$\Delta M_1$} & \textbf{$\Delta M_2$} & \textbf{$\Delta M_3$} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    guardar("tab_weights.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Sensitivity of \textsc{Proposed} to the weights $(\alpha,\beta,\gamma)$ of the relevance score (Equation~\ref{eq:score}): relative reduction (\%) with respect to \textsc{Static}.}\label{tab:weights}
\begin{tabular}{lccc}
\toprule
\textbf{$(\alpha, \beta, \gamma)$} & \textbf{$\Delta M_1$} & \textbf{$\Delta M_2$} & \textbf{$\Delta M_3$} \\
\midrule
""" + "\n".join(fw) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_subgrupos():
    r = C.leer_json("rq2.json")
    filas = []
    for etiqueta, d in [*((ROL_EN[k], v) for k, v in r["por_rol"].items()),
                        ("Depth 2 targets", r["por_profundidad"]["2"]), ("Depth 3 targets", r["por_profundidad"]["3"]),
                        ("Profile drift (all users)", r["deriva"])]:
        if etiqueta == "Depth 2 targets" or etiqueta.startswith("Profile"):
            filas.append(r"\midrule")
        celdas = []
        for m in ["M1", "M2"]:
            for c in ["Frequency", "Proposed"]:
                celdas.append(f"{d['metricas'][m]['por_condicion'][c]['reduccion_vs_static_pct']:.1f}")
        filas.append(f"{etiqueta} & {d['n_usuarios']} & " + " & ".join(celdas) + r" \\")
    guardar("tab_subgroups.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Relative reduction (\%) with respect to \textsc{Static} by role, by original depth of the target function and under profile drift (two swapped weights of each user's profile between training and evaluation).}\label{tab:subgroups}
\begin{tabular}{lrcccc}
\toprule
 & & \multicolumn{2}{c}{\textbf{$\Delta M_1$}} & \multicolumn{2}{c}{\textbf{$\Delta M_2$}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Subset} & \textbf{Users} & \textsc{Frequency} & \textsc{Proposed} & \textsc{Frequency} & \textsc{Proposed} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_latencia():
    r = C.leer_json("rq4.json")
    guardar("tab_latency.tex", rf"""\begin{{table}}[htbp]
\centering\small
\caption{{Computational cost on the CPU of the experimental laptop ({r['cpu_hilos']} threads, GPU disabled).}}\label{{tab:latency}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Measure}} & \textbf{{Value}} \\
\midrule
$t_{{\text{{load}}}}$: load predictor, agents and menu catalog (median / p95, $n={r['t_load_ms']['n']}$) & {r['t_load_ms']['mediana']:.1f} / {r['t_load_ms']['p95']:.1f} ms \\
$t_{{\text{{gen}}}}$: generate one personalized menu (median / p95, $n={r['t_gen_ms']['n']}$) & {r['t_gen_ms']['mediana']:.2f} / {r['t_gen_ms']['p95']:.2f} ms \\
Size of the menu description (JSON, mean / max) & {r['menu_bytes']['media']:.0f} / {r['menu_bytes']['max']} bytes \\
Size of the selected predictor / all agents & {r['modelos']['lstm_kb']:.0f} / {r['modelos']['agentes_kb']:.0f} KB \\
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def figura_curvas():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    colores = {"lstm": "#1f5fae", "gru": "#d95f02", "cnn1d": "#1b9e77", "transformer": "#7570b3"}
    for n, col in colores.items():
        h = json.loads((C.UR / "OE2_Modelado/output_oe2" / f"historial_{n}.json").read_text())
        ep = np.arange(1, len(h["val_loss"]) + 1)
        ax[0].plot(ep, h["val_loss"], color=col, lw=2, label=f"{MOD_EN[n]} (validation)")
        ax[0].plot(ep, h["train_loss"], color=col, lw=1, ls="--", alpha=0.6)
        ax[1].plot(ep, h["val_acc"], color=col, lw=2, label=MOD_EN[n])
        ax[1].plot(ep, h["train_acc"], color=col, lw=1, ls="--", alpha=0.6)
    ax[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss (solid: validation, dashed: training)")
    ax[1].set(xlabel="Epoch", ylabel="Top-1 accuracy", title="Top-1 accuracy (solid: validation, dashed: training)")
    for a in ax:
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_curves.png", dpi=200)
    plt.close(fig)


def figura_politicas():
    filas = list(csv.DictReader(open(C.SALIDAS / "rq2_por_usuario.csv", encoding="utf-8")))
    datos = {m: {p: [float(f[m]) for f in filas if f["escenario"] == "base" and f["subconjunto"] == "all" and f["condicion"] == p]
                 for p in POL} for m in ["M1", "M2"]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, m, lab in [(ax[0], "M1", "Clicks per task ($M_1$)"), (ax[1], "M2", "Search time per task, s ($M_2$)")]:
        a.boxplot([datos[m][p] for p in POL], tick_labels=POL, showfliers=False)
        a.set_ylabel(lab)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-user navigation cost on the evaluation tasks (n = 600)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_policies.png", dpi=200)
    plt.close(fig)


CALIB = [("clics_b1", "Clicks per task, block 1", "{:.2f}"), ("clics_b2", "Clicks per task, block 2", "{:.2f}"),
         ("error_b1", "Tasks with a wrong visit, block 1 (\\%)", "pct"), ("error_b2", "Tasks with a wrong visit, block 2 (\\%)", "pct"),
         ("tiempo_med_b1", "Median time per task, block 1 (s)", "{:.1f}"), ("tiempo_med_b2", "Median time per task, block 2 (s)", "{:.1f}"),
         ("tiempo_practica", "Mean time per practice task (s)", "{:.1f}"), ("razon_b2_b1", "Median time, block 2 / block 1", "{:.2f}")]


def tabla_calibracion():
    """Sesión piloto frente a usuarios simulados del mismo rol y experiencia (salidas/calibracion.json)."""
    ruta = C.SALIDAS / "calibracion.json"
    if not ruta.exists():
        return
    d = json.loads(ruta.read_text(encoding="utf-8"))
    f = lambda v, fmt: f"{100 * v:.0f}" if fmt == "pct" else fmt.format(v)
    filas = []
    for k, etiqueta, fmt in CALIB:
        m = d["medidas"][k]
        filas.append(f"{etiqueta} & {f(m['piloto'], fmt)} & {f(m['sim_mediana'], fmt)} [{f(m['sim_p5'], fmt)}, {f(m['sim_p95'], fmt)}] \\\\")
    guardar("tab_calibration.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Calibration check: pilot session of a member of the research team (Administrator role, no prior experience, static menu; practice tasks excluded except in their own row) and the """ + str(d["n_usuarios_simulados"]) + r""" simulated users with the same role and experience level (median and 5th--95th percentile of the per-user values).}\label{tab:calibration}
\begin{tabular}{lcc}
\toprule
\textbf{Measure} & \textbf{Pilot} & \textbf{Simulated, median [p5, p95]} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


SENS = [("zipf", "s", "Zipf exponent $s$ of the usage profile"),
        ("funciones", "n_funciones", "Typical functions per role"),
        ("barra", "k_barra", "Functions in the suggestion bar"),
        ("t_clic", "f_clic", "Time per click (factor)"),
        ("t_scan", "f_scan", "Time per scanned item (factor)"),
        ("error", "f_error", "Probability of a wrong visit (factor)")]


def figura_sensibilidad():
    """Reducción de M1 y M2 de Proposed frente a Static al variar un parámetro a la vez (salidas/sensibilidad_*.json)."""
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
    for k, (a, (nombre, clave, titulo)) in enumerate(zip(axs.flat, SENS)):
        d = json.loads((C.SALIDAS / f"sensibilidad_{nombre}.json").read_text(encoding="utf-8"))
        x = [r[clave] for r in d["resultados"]]
        for m, col, ls, lab in [("M1", "#1f5fae", "-", "Clicks ($M_1$)"), ("M2", "#d95f02", "--", "Search time ($M_2$)")]:
            p = [r["politicas"]["Proposed"][m] for r in d["resultados"]]
            a.plot(x, [q["reduccion_pct"] for q in p], color=col, ls=ls, marker="o", ms=3.5, lw=1.6, label=lab)
            a.fill_between(x, [q["ic95"][0] for q in p], [q["ic95"][1] for q in p], color=col, alpha=0.18, lw=0)
        a.axvline(d["base"][clave], color="0.4", ls=":", lw=1)
        a.set_title(f"({'abcdef'[k]}) {titulo}", fontsize=9)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
    for a in axs[:, 0]:
        a.set_ylabel("Reduction vs. Static (%)", fontsize=9)
    axs[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensitivity.png", dpi=200)
    plt.close(fig)


def tabla_memoria():
    """Reducción de M2 con memoria espacial y costo de reaprendizaje (salidas/memoria_espacial.json)."""
    ruta = C.SALIDAS / "memoria_espacial.json"
    if not ruta.exists():
        return
    d = json.loads(ruta.read_text(encoding="utf-8"))
    cols = [("Solo barra", r"Bar only"), ("Proposed", r"\textsc{Proposed}"), ("Score-greedy", r"\textsc{Score-greedy}"),
            ("Sin barra", r"Without bar"), ("Frequency", r"\textsc{Frequency}")]
    lin = d["modelo_lineal"]
    filas = [r"\multicolumn{3}{l}{Linear visual search (main model)} & " + " & ".join(f"{lin[c]:.1f}" for c, _ in cols) + r" \\",
             r"\midrule"]
    for x in d["memoria_espacial"]:
        filas.append(f"{x['b_hh']:.2f} & {x['c_reaprendizaje']:.1f} & {x['M2_static_s']:.2f} & "
                     + " & ".join(f"{x['reduccion_M2_pct'][c]:.1f}" for c, _ in cols) + r" \\")
    mov = d["tareas_con_item_movido"]
    guardar("tab_memory.tex", r"""\begin{table}[htbp]
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Reduction (\%) of the mean search time $M_2$ with respect to \textsc{Static} when the evaluation models spatial memory and relearning ($n=600$ users; negative values: slower than \textsc{Static}). $b$: Hick--Hyman slope (s/bit); $c$: relearning cost per task whose target was moved (s); $M_2^{\textsc{Static}}$: mean search time of the static menu (s). The target had been moved in """ +
          f"{100 * mov['Proposed']:.0f}" + r"""\% of the tasks with \textsc{Proposed} and in """ + f"{100 * mov['Sin barra']:.0f}" + r"""\% without the bar; the bar alone moves no item. Clicks are not affected by this model.}\label{tab:memory}
\begin{tabular}{ccc""" + "c" * len(cols) + r"""}
\toprule
$b$ & $c$ & $M_2^{\textsc{Static}}$ & """ + " & ".join(rf"\textbf{{{t}}}" for _, t in cols) + r""" \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_arboles():
    """Generalización a árboles sintéticos (salidas/arboles_sinteticos.json): rangos por número de grupos."""
    ruta = C.SALIDAS / "arboles_sinteticos.json"
    if not ruta.exists():
        return
    d = json.loads(ruta.read_text(encoding="utf-8"))
    res = d["resultados"]
    rg = lambda xs: f"[{min(xs):.1f}, {max(xs):.1f}]"
    filas = []
    for G in sorted({x["G"] for x in res}):
        xs = [x for x in res if x["G"] == G]
        v = lambda pol, m: rg([x["reduccion"][pol][m] for x in xs])
        filas.append(f"{G} & {min(x['funciones'] for x in xs)}--{max(x['funciones'] for x in xs)} & {v('Solo barra', 'M1')} & "
                     f"{v('Solo barra', 'M2')} & {v('Reordenado + barra', 'M2')} & {v('Solo barra', 'M2_memoria')} & "
                     f"{v('Reordenado + barra', 'M2_memoria')} & {v('Reordenado', 'M2_memoria')} \\\\")
    guardar("tab_trees.tex", r"""\begin{table}[htbp]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Reduction (\%) with respect to the static menu on synthetic menu trees ([minimum, maximum] over 4, 7 or 12 functions per group and 0, 30 or 60\% of the functions in submenus; """ + str(d["n_usuarios_por_arbol"]) + r""" simulated users per tree; negative values: slower than the static menu). Bar: suggestion bar above the unchanged menu; Reordered: groups and items sorted by frequency, without bar; Reord.\,+\,bar: both, equivalent to \textsc{Score-greedy} and \textsc{Proposed} by properties P1 and P2. Spatial memory: model of Section~\ref{sec:memory} with $b=0.15$~s/bit and $c=0.5$~s.}\label{tab:trees}
\begin{tabular}{cccccccc}
\toprule
 & & \multicolumn{1}{c}{$M_1$} & \multicolumn{2}{c}{$M_2$, linear search} & \multicolumn{3}{c}{$M_2$, spatial memory} \\
\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}
\textbf{Groups} & \textbf{Functions} & \textbf{Bar} & \textbf{Bar} & \textbf{Reord.\,+\,bar} & \textbf{Bar} & \textbf{Reord.\,+\,bar} & \textbf{Reordered} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tabla_semillas():
    """Variabilidad entre semillas (salidas/multisemilla.json, observación 7)."""
    ruta = C.SALIDAS / "multisemilla.json"
    if not ruta.exists():
        return
    d = json.loads(ruta.read_text(encoding="utf-8"))
    a, n = d["resumen"], len(d["semillas"])
    md = lambda x, dec=3: f"{x['media']:.{dec}f} $\\pm$ {x['de']:.{dec}f}"
    filas = []
    for m, nom in [("lstm", "LSTM"), ("gru", "GRU"), ("cnn1d", "1D-CNN"), ("transformer", "Transformer")]:
        eps = [d["por_semilla"][s]["epocas"][m].split("/")[0] for s in d["por_semilla"]]
        filas.append(f"{nom} & {md(a['val_loss'][m], 4)} & {min(map(int, eps))}--{max(map(int, eps))} & {a['seleccion'].get(m, 0)}/{n} & "
                     f"{md(a['rq1'][m]['acc1'])} & {md(a['rq1'][m]['acc3'])} & {md(a['rq1'][m]['mrr'])} \\\\")
    dq = a["dqn_pct_optimo"]
    dqn = "; ".join(f"{gs}: {v['media']:.1f} [{v['min']:.1f}, {v['max']:.1f}]" for gs, v in sorted(dq.items(), key=lambda x: int(x[0])) if v["min"] < 100)
    pol = a["politicas"]["Proposed"]; ev = a["evaluacion_modelos_articulo"]["Proposed"]
    guardar("tab_seeds.tex", r"""\begin{table}[htbp]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Variability across """ + str(n) + r""" training seeds (mean $\pm$ standard deviation). Predictors retrained on the simulated corpus generated by the released code, without the pilot session, with validation split by user and early stopping (patience 10) with a limit of 300 epochs that no run reached; test on the 600 new users of RQ1. Best epoch: range over seeds; Selected: seeds in which the architecture had the lowest validation loss. DQN agents retrained with the same seeds: groups with less than 100\% optimal headers (size: mean [min, max] \%), """ + dqn + r"""; with the LSTM and agents of each seed, \textsc{Proposed} reduced clicks by """ + md(pol["M1"], 2) + r"""\% and search time by """ + md(pol["M2"], 2) + r"""\%. With the models of the main analysis and five seeds of evaluation tasks (2026--2030), the reductions were """ + md(ev["M1"], 2) + r"""\% and """ + md(ev["M2"], 2) + r"""\%.}\label{tab:seeds}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Val.\ loss} & \textbf{Best epoch} & \textbf{Selected} & \textbf{Acc@1} & \textbf{Acc@3} & \textbf{MRR} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    tabla_corpus(); tabla_rq1(); tabla_dqn(); tabla_politicas(); tabla_ablacion(); tabla_subgrupos(); tabla_latencia(); tabla_calibracion()
    tabla_memoria(); tabla_arboles(); tabla_semillas()
    figura_curvas(); figura_politicas(); figura_sensibilidad()
    print("[OK]", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))
