"""
TDA Analysis — ANES Feeling Thermometers
Analiza la evolución topológica por año y bloque de variables.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from ripser import ripser

sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")


class Tee:
    """Escribe simultáneamente en consola y en un fichero de texto."""
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout
    def write(self, data):
        self.terminal.write(data)
        self.file.write(data)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

# ── Configuración ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "anes_timeseries_cdf_csv_20220916.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "resultados_tda")
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_COL      = "VCF0004"
INVALID_CODES = {97, 98, 99}
MIN_POINTS    = 80    # mínimo de encuestados válidos por año

# Años representativos para los gráficos de barcode y diagrama
KEY_YEARS = [1992, 2000, 2008, 2012, 2016, 2020]

BLOCKS = {
    "Bloque 1: Grupos raciales": [
        "VCF0206", "VCF0207", "VCF0217", "VCF0227",
    ],
    "Bloque 2: + Socioeconómico": [
        "VCF0206", "VCF0207", "VCF0217", "VCF0227",
        "VCF0223", "VCF0220",
    ],
    "Bloque 3: + Partidos políticos": [
        "VCF0206", "VCF0207", "VCF0217", "VCF0227",
        "VCF0223", "VCF0220", "VCF0218", "VCF0224",
    ],
}

COL_LABELS = {
    "VCF0206": "Negros",
    "VCF0207": "Blancos",
    "VCF0217": "Hispanos",
    "VCF0227": "Asiát.-amer.",
    "VCF0223": "Pobres",
    "VCF0220": "Subsidios",
    "VCF0218": "Partido Dem.",
    "VCF0224": "Partido Rep.",
}

BLOCK_COLORS = {
    "Bloque 1: Grupos raciales":        "#6366f1",
    "Bloque 2: + Socioeconómico":       "#f59e0b",
    "Bloque 3: + Partidos políticos":   "#10b981",
}

# ── Carga de datos ────────────────────────────────────────────────────────────

print("Cargando datos...")
df_raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)

needed_cols = [YEAR_COL] + list({c for cols in BLOCKS.values() for c in cols})
df = df_raw[needed_cols].copy()

df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
for col in needed_cols:
    if col != YEAR_COL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(INVALID_CODES), col] = np.nan

years_all = sorted(df[YEAR_COL].dropna().unique().astype(int))
print(f"  {len(df):,} filas | años: {years_all[0]}–{years_all[-1]}")

# ── Funciones TDA ─────────────────────────────────────────────────────────────

def barcode_stats(dgm: np.ndarray) -> dict:
    """
    Resumen estadístico descriptivo de un barcode.
    Siempre elimina barras con muerte infinita antes de calcular.
    """
    d = dgm[np.isfinite(dgm[:, 1])]
    if len(d) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0,
                "max": 0.0, "sum": 0.0, "entropy": 0.0}
    pers = d[:, 1] - d[:, 0]
    pers = pers[pers > 1e-10]
    if len(pers) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0,
                "max": 0.0, "sum": 0.0, "entropy": 0.0}
    L = pers.sum()
    p = pers / L
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return {
        "count":   int(len(pers)),
        "mean":    float(pers.mean()),
        "std":     float(pers.std()),
        "max":     float(pers.max()),
        "sum":     float(L),
        "entropy": entropy,
    }


def run_tda(df_year: pd.DataFrame, cols: list):
    """
    Filtra y corre Ripser. Sin normalización (todas las variables en escala 0-100)
    ni submuestreo. Devuelve (dgm0, dgm1, n_puntos) o (None, None, 0) si hay pocos datos.
    """
    sub = df_year[cols].dropna()
    if len(sub) < MIN_POINTS:
        return None, None, 0
    X = sub.values.astype(float)
    res = ripser(X, maxdim=1)
    return res["dgms"][0], res["dgms"][1], len(sub)


def plot_barcode(ax, dgm0, dgm1, year, n):
    """
    Dibuja el barcode completo (H0 azul + H1 naranja) como barras horizontales
    ordenadas de mayor a menor persistencia.
    """
    bars0 = dgm0[np.isfinite(dgm0[:, 1])]
    bars1 = dgm1[np.isfinite(dgm1[:, 1])]

    # Ordenar por persistencia descendente
    if len(bars0):
        order = np.argsort(bars0[:, 1] - bars0[:, 0])[::-1]
        bars0 = bars0[order]
    if len(bars1):
        order = np.argsort(bars1[:, 1] - bars1[:, 0])[::-1]
        bars1 = bars1[order]

    y = 0
    # H0
    for b, d in bars0:
        ax.barh(y, d - b, left=b, height=0.7, color="#6366f1", alpha=0.85)
        y += 1
    sep = y + 0.3
    y += 1
    # H1
    for b, d in bars1:
        ax.barh(y, d - b, left=b, height=0.7, color="#f59e0b", alpha=0.85)
        y += 1

    all_deaths = ([d for _, d in bars0] + [d for _, d in bars1])
    x_max = max(all_deaths) * 1.05 if all_deaths else 1.0

    ax.axhline(sep, color="gray", linewidth=0.9, linestyle="--", alpha=0.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, max(y, 2))
    ax.set_xlabel("Escala ε", fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"{year}  (n={n})", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    p0 = mpatches.Patch(color="#6366f1", alpha=0.85,
                         label=f"H₀  ({len(bars0)} barras)")
    p1 = mpatches.Patch(color="#f59e0b", alpha=0.85,
                         label=f"H₁  ({len(bars1)} barras)")
    ax.legend(handles=[p0, p1], fontsize=7, loc="lower right")


# ── Bucle principal de análisis ───────────────────────────────────────────────

records = []

_log_file  = open(os.path.join(OUT_DIR, "resultados.txt"), "w", encoding="utf-8")
sys.stdout = Tee(_log_file)

for block_name, cols in BLOCKS.items():
    print(f"\n{'-'*60}")
    print(f"  {block_name}")
    print(f"  Columnas: {', '.join(COL_LABELS[c] for c in cols)}")
    print()

    for year in years_all:
        dgm0, dgm1, n = run_tda(df[df[YEAR_COL] == year], cols)
        if dgm0 is None:
            continue

        s0 = barcode_stats(dgm0)
        s1 = barcode_stats(dgm1)

        records.append({
            "bloque": block_name, "año": year, "n": n,
            "h0_count":   s0["count"], "h0_mean":    s0["mean"],
            "h0_std":     s0["std"],   "h0_max":     s0["max"],
            "h0_sum":     s0["sum"],   "h0_entropy": s0["entropy"],
            "h1_count":   s1["count"], "h1_mean":    s1["mean"],
            "h1_std":     s1["std"],   "h1_max":     s1["max"],
            "h1_sum":     s1["sum"],   "h1_entropy": s1["entropy"],
        })
        print(f"  Año {year}  (n={n})")
        print(f"    H0 → count={s0['count']:3d}  mean={s0['mean']:8.3f}  "
              f"std={s0['std']:8.3f}  max={s0['max']:8.3f}  "
              f"sum={s0['sum']:9.3f}  entropy={s0['entropy']:.4f}")
        print(f"    H1 → count={s1['count']:3d}  mean={s1['mean']:8.3f}  "
              f"std={s1['std']:8.3f}  max={s1['max']:8.3f}  "
              f"sum={s1['sum']:9.3f}  entropy={s1['entropy']:.4f}")
        print(f"    v  = [{s0['count']}, {s0['mean']:.3f}, {s0['std']:.3f}, "
              f"{s0['max']:.3f}, {s0['sum']:.3f}, {s0['entropy']:.4f}, "
              f"{s1['count']}, {s1['mean']:.3f}, {s1['std']:.3f}, "
              f"{s1['max']:.3f}, {s1['sum']:.3f}, {s1['entropy']:.4f}]")
        print()

results = pd.DataFrame(records)
csv_path = os.path.join(OUT_DIR, "tda_statistics.csv")
results.to_csv(csv_path, index=False, encoding="utf-8")

# ── Tabla resumen por bloque ──────────────────────────────────────────────────

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)

for block_name in BLOCKS:
    bdata = results[results["bloque"] == block_name].copy()
    if bdata.empty:
        continue
    bdata = bdata.drop(columns=["bloque"]).set_index("año")
    print(f"\n{'='*60}")
    print(f"  TABLA RESUMEN — {block_name}")
    print(f"{'='*60}")
    print(bdata.to_string())
    print()

_log_file.close()
sys.stdout = sys.stdout.terminal

# ── Gráfico 1: Evolución temporal por bloque (6 métricas) ────────────────────

METRICS = [
    ("h0_count",   "Nº componentes H₀",      "#6366f1"),
    ("h1_count",   "Nº bucles H₁",           "#f59e0b"),
    ("h0_entropy", "Entropía H₀",            "#ef4444"),
    ("h1_entropy", "Entropía H₁",            "#10b981"),
    ("h1_max",     "Persistencia máxima H₁", "#8b5cf6"),
    ("h1_mean",    "Persistencia media H₁",  "#0ea5e9"),
]

for block_name, cols in BLOCKS.items():
    bdata = results[results["bloque"] == block_name].sort_values("año")
    if bdata.empty:
        continue

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Evolución temporal — {block_name}",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, (metric, label, color) in zip(axes, METRICS):
        ax.plot(bdata["año"], bdata[metric], marker="o", color=color,
                linewidth=2, markersize=5, zorder=3)
        ax.fill_between(bdata["año"], bdata[metric], alpha=0.10, color=color)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Año", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(bdata["año"].min() - 1, bdata["año"].max() + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if bdata[metric].max() > 0:
            idx_max = bdata[metric].idxmax()
            ax.annotate(
                str(int(bdata.loc[idx_max, "año"])),
                xy=(bdata.loc[idx_max, "año"], bdata.loc[idx_max, metric]),
                xytext=(6, 5), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
            )

    plt.tight_layout()
    safe = block_name.replace(" ", "_").replace(":", "").replace("+", "mas")
    out = os.path.join(OUT_DIR, f"evolucion_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado → {out}")

# ── Gráfico 2: Comparación entre bloques (4 métricas) ────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Comparación entre bloques — evolución temporal",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

comp_metrics = [
    ("h1_count",   "Nº bucles H₁"),
    ("h1_entropy", "Entropía H₁"),
    ("h0_count",   "Nº componentes H₀"),
    ("h0_entropy", "Entropía H₀"),
]

for ax, (metric, title) in zip(axes, comp_metrics):
    for block_name, color in BLOCK_COLORS.items():
        bdata = results[results["bloque"] == block_name].sort_values("año")
        if bdata.empty:
            continue
        short = block_name.split(":")[0]
        ax.plot(bdata["año"], bdata[metric], marker="o", label=short,
                color=color, linewidth=2, markersize=4)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Año", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
out = os.path.join(OUT_DIR, "comparacion_bloques.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gráfico guardado → {out}")

# ── Gráfico 3: Barcodes para años clave ──────────────────────────────────────

for block_name, cols in BLOCKS.items():
    bdata_avail = results[results["bloque"] == block_name]
    available   = sorted(bdata_avail["año"].unique())
    years_plot  = [y for y in KEY_YEARS if y in available]
    if not years_plot:
        continue

    n_years = len(years_plot)
    if n_years > 4:
        n_rows = 2
        n_cols = int(np.ceil(n_years / 2))
    else:
        n_rows = 1
        n_cols = n_years
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, 7 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f"Barcodes — {block_name}", fontsize=13, fontweight="bold")

    for ax, year in zip(axes, years_plot):
        dgm0, dgm1, n = run_tda(df[df[YEAR_COL] == year], cols)
        if dgm0 is None:
            ax.set_visible(False)
            continue
        plot_barcode(ax, dgm0, dgm1, year, n)

    for ax in axes[n_years:]:
        ax.set_visible(False)

    plt.tight_layout()
    safe = block_name.replace(" ", "_").replace(":", "").replace("+", "mas")
    out = os.path.join(OUT_DIR, f"barcodes_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado → {out}")

# ── Gráfico 4: Diagramas de persistencia para años clave ─────────────────────

for block_name, cols in BLOCKS.items():
    bdata_avail = results[results["bloque"] == block_name]
    available   = sorted(bdata_avail["año"].unique())
    years_plot  = [y for y in KEY_YEARS if y in available]
    if not years_plot:
        continue

    n_years = len(years_plot)
    if n_years > 4:
        n_rows = 2
        n_cols = int(np.ceil(n_years / 2))
    else:
        n_rows = 1
        n_cols = n_years
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, 5.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f"Diagramas de persistencia — {block_name}",
                 fontsize=13, fontweight="bold")

    for ax, year in zip(axes, years_plot):
        dgm0, dgm1, n = run_tda(df[df[YEAR_COL] == year], cols)
        if dgm0 is None:
            ax.set_visible(False)
            continue

        d0 = dgm0[np.isfinite(dgm0[:, 1])]
        if len(d0):
            ax.scatter(d0[:, 0], d0[:, 1], c="#6366f1", s=30,
                       alpha=0.7, label="H₀", zorder=3)
        if len(dgm1):
            ax.scatter(dgm1[:, 0], dgm1[:, 1], c="#f59e0b", s=45,
                       marker="^", alpha=0.85, label="H₁", zorder=4)

        all_vals = []
        if len(d0):
            all_vals.extend(d0[:, 1].tolist())
        if len(dgm1):
            all_vals.extend(dgm1[np.isfinite(dgm1[:, 1]), 1].tolist())
        lim = max(all_vals) * 1.05 if all_vals else 1.05

        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4)
        ax.set_xlim(-lim * 0.02, lim)
        ax.set_ylim(-lim * 0.02, lim)
        ax.set_title(f"{year}  (n={n})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Nacimiento", fontsize=8)
        ax.set_ylabel("Muerte", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)

    for ax in axes[n_years:]:
        ax.set_visible(False)

    plt.tight_layout()
    safe = block_name.replace(" ", "_").replace(":", "").replace("+", "mas")
    out = os.path.join(OUT_DIR, f"diagramas_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado → {out}")

# ── Resumen final ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60, flush=True)
print("ANÁLISIS COMPLETADO")
print(f"Outputs en: {OUT_DIR}")
print()
for block_name in BLOCKS:
    bdata = results[results["bloque"] == block_name]
    if bdata.empty:
        continue
    años = sorted(bdata["año"].unique())
    print(f"  {block_name}")
    print(f"    Años con datos: {años[0]}–{años[-1]}  ({len(años)} puntos)")
    idx_max_h1 = bdata["h1_count"].idxmax()
    print(f"    Año con más bucles H₁:    {int(bdata.loc[idx_max_h1,'año'])}  "
          f"({int(bdata.loc[idx_max_h1,'h1_count'])} bucles)")
    idx_max_ent = bdata["h1_entropy"].idxmax()
    print(f"    Año con mayor entropía H₁: {int(bdata.loc[idx_max_ent,'año'])}  "
          f"({bdata.loc[idx_max_ent,'h1_entropy']:.3f})")
    print()

print(f"Log guardado → {os.path.join(OUT_DIR, 'resultados.txt')}")
