# graficas_encuestas.py
import argparse, os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ utilidades de carga/limpieza ============

SPANISH_VERSION_COLS = {"versión", "version", "modo", "mode"}
ID_COLS = {"id"}
NAME_COLS = {"nombre", "nombre del jugador", "player", "jugador", "participant", "usuario"}

def _norm(s: str) -> str:
    return str(s).strip().lower().replace("\xa0"," ").replace("  "," ")

def comma_decimal_to_dot(df: pd.DataFrame, cols):
    """Convierte comas decimales a punto, preservando separadores."""
    pat = re.compile(r"(?P<int>\d),(?=\d)")
    for c in cols:
        df[c] = df[c].astype(str).map(lambda x: pat.sub(r"\g<int>.", x))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        try:
            import openpyxl  # asegura motor para xlsx
        except ImportError:
            print("Instala openpyxl: pip install openpyxl", file=sys.stderr)
            raise
        df = pd.read_excel(path)
    else:
        # CSV con separador automático
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [_norm(c) for c in df.columns]
    return df

def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    version_col = None
    for c in cols:
        if _norm(c) in SPANISH_VERSION_COLS:
            version_col = c; break
    if version_col is None:
        # fallback: tercera columna (como en tu ejemplo)
        version_col = cols[2]

    id_col   = next((c for c in cols if _norm(c) in ID_COLS), None)
    name_col = next((c for c in cols if _norm(c) in NAME_COLS), None)

    meta = {version_col}
    if id_col: meta.add(id_col)
    if name_col: meta.add(name_col)
    item_cols = [c for c in cols if c not in meta]
    return id_col, name_col, version_col, item_cols

# ============ gráficos ============

def plot_grouped_bars(item_means_by_mode, outpath):
    plt.figure(figsize=(12, 6))
    modes = list(item_means_by_mode.columns)
    x = np.arange(len(item_means_by_mode))
    width = 0.8 / max(3, len(modes))
    for i, m in enumerate(modes):
        plt.bar(x + i*width, item_means_by_mode[m].values, width, label=m)
    plt.xticks(x + width*(len(modes)-1)/2, [f"Q{j+1}" for j in range(len(item_means_by_mode))], rotation=60, ha="right")
    plt.ylim(0, 5.1)
    plt.ylabel("Media (1–5)")
    plt.title("Medias por ítem y modo")
    plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_boxplot_overall(df, version_col, outpath):
    modes = list(df[version_col].dropna().unique())
    data = [df[df[version_col]==m]["overall_mean"].dropna().values for m in modes]
    plt.figure(figsize=(6,5))
    plt.boxplot(data, labels=modes, showmeans=True)
    plt.ylabel("Media por jugador (1–5)")
    plt.title("Distribución por modo (boxplot de medias por jugador)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_heatmap(item_means_by_mode, outpath):
    plt.figure(figsize=(6, 10))
    mat = item_means_by_mode.values
    im = plt.imshow(mat, aspect='auto')
    plt.colorbar(im, label="Media (1–5)")
    plt.yticks(range(len(item_means_by_mode.index)), [f"Q{j+1}" for j in range(len(item_means_by_mode.index))])
    plt.xticks(range(len(item_means_by_mode.columns)), item_means_by_mode.columns, rotation=45, ha="right")
    plt.title("Heatmap de medias por ítem y modo")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_bar_global(mean_global_by_mode, outpath):
    plt.figure(figsize=(6,4))
    vals = mean_global_by_mode.values
    plt.bar(mean_global_by_mode.index.astype(str), vals)
    for i, v in enumerate(vals):
        plt.text(i, v+1, f"{v:.1f}", ha="center", va="bottom")
    plt.ylim(0, 105)
    plt.ylabel("Puntuación (0–100)")
    plt.title("Encuestas — Media global por modo (reescalado)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_violin(long_df, version_col, outpath):
    modes = list(long_df[version_col].dropna().unique())
    data = [long_df[long_df[version_col]==m]["Score"].dropna().values for m in modes]
    plt.figure(figsize=(7,5))
    plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks([i+1 for i in range(len(modes))], modes)
    plt.ylabel("Respuestas (1–5)")
    plt.title("Distribución de respuestas por modo (violin)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_radar(construct_means_by_mode: dict, labels: list, outpath: str):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(0,5)
    for mode, vals in construct_means_by_mode.items():
        vals_loop = vals + vals[:1]
        ax.plot(angles, vals_loop, label=mode)
        ax.fill(angles, vals_loop, alpha=0.1)
    ax.set_title("Radar por constructos")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

# ============ agrupación de constructos ============

def default_construct_mapping(item_cols):
    """
    Si no defines tus grupos reales, reparte los ítems en 4 bloques contiguos.
    Edita este mapeo para poner tus constructos (Flow, Inmersión, Disfrute, Motivación).
    """
    n = len(item_cols)
    k = 4
    sizes = [n//k + (1 if i < n % k else 0) for i in range(k)]
    blocks, start = [], 0
    for s in sizes:
        blocks.append(item_cols[start:start+s])
        start += s
    labels = [f"Bloque {i+1}" for i in range(k)]
    return {labels[i]: blocks[i] for i in range(k)}

# ============ flujo principal ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default=r"D:\TFM\NeuroDoomPy\sessions\01_features\Libro1.xlsx",
                        help="Ruta al .xlsx o .csv con la tabla de encuestas")
    parser.add_argument("--outdir", required=False, default=r"D:\TFM\NeuroDoomPy\sessions\01_features\figuresR",
                        help="Carpeta de salida para las imágenes")
    args = parser.parse_args()

    excel_path = args.input
    outdir     = args.outdir

    os.makedirs(outdir, exist_ok=True)

    df = read_table(excel_path)
    id_col, name_col, version_col, item_cols = detect_columns(df)

    # Limpia comas decimales -> punto y convierte a numérico
    df = comma_decimal_to_dot(df, item_cols)

    # Orden opcional de modos
    mode_order = ["Preconfigured", "HeuristicTree", "ML"]
    if version_col in df.columns:
        df[version_col] = df[version_col].astype("category")
        try:
            df[version_col] = df[version_col].cat.set_categories(mode_order, ordered=True)
        except Exception:
            pass

    # Media por fila (participante×modo)
    df["overall_mean"] = df[item_cols].mean(axis=1)

    # ----- 1) Barras agrupadas: media por ítem y modo
    item_means_by_mode = df.groupby(version_col)[item_cols].mean().T
    item_means_by_mode = item_means_by_mode[[c for c in mode_order if c in item_means_by_mode.columns]]
    plot_grouped_bars(item_means_by_mode, os.path.join(outdir, "encuestas_grouped_items.png"))

    # ----- 2) Boxplot: media por jugador y modo
    plot_boxplot_overall(df, version_col, os.path.join(outdir, "encuestas_boxplot_overall.png"))

    # ----- 3) Heatmap: items × modo (medias)
    plot_heatmap(item_means_by_mode, os.path.join(outdir, "encuestas_heatmap_items.png"))

    # ----- 4) Barras globales 0–100
    mean_global_by_mode = df.groupby(version_col)["overall_mean"].mean() * 20.0
    mean_global_by_mode = mean_global_by_mode[[m for m in mode_order if m in mean_global_by_mode.index]]
    plot_bar_global(mean_global_by_mode, os.path.join(outdir, "encuestas_bar_global.png"))

    # ----- 5) Violin: todas las respuestas crudas por modo
    id_vars = [c for c in [id_col, name_col, version_col] if c]
    long_df = df.melt(id_vars=id_vars, value_vars=item_cols, var_name="Item", value_name="Score")
    plot_violin(long_df, version_col, os.path.join(outdir, "encuestas_violin.png"))

    # ----- 6) Radar: por constructos (EDITA mapping si tienes grupos reales)
    mapping = default_construct_mapping(item_cols)
    construct_means_by_mode = {}
    modes = [m for m in mode_order if m in df[version_col].unique()]
    for m in modes:
        sub = df[df[version_col]==m]
        vals = [sub[cols].mean(axis=1).mean() for cols in mapping.values()]
        construct_means_by_mode[m] = vals
    plot_radar(construct_means_by_mode, list(mapping.keys()), os.path.join(outdir, "encuestas_radar.png"))

    # ----- CSV limpio opcional
    clean_csv = os.path.join(outdir, "encuestas_clean.csv")
    df.to_csv(clean_csv, index=False, encoding="utf-8")
    print("\nHe generado las figuras en:", os.path.abspath(outdir))
    for f in ["encuestas_grouped_items.png",
              "encuestas_boxplot_overall.png",
              "encuestas_heatmap_items.png",
              "encuestas_bar_global.png",
              "encuestas_violin.png",
              "encuestas_radar.png"]:
        print(" -", f)
    print("También he guardado el CSV normalizado:", clean_csv)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
