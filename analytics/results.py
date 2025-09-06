# resultados_6_1.py
# Genera tablas y figuras del apartado 6.1 (Resultados) a partir de features_horda.xlsx
# Uso:
#   python resultados_6_1.py --excel /ruta/a/features_horda.xlsx
# Requisitos: pandas, matplotlib, openpyxl

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import utils_io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# Utilidades
# ----------------------------
def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", str(s)) if unicodedata.category(ch) != "Mn")

def normalize_mode(x):
    if pd.isna(x):
        return np.nan
    t = _strip_accents(str(x).strip().lower())
    if "heur" in t:
        return "heuristic"
    if "pre" in t:
        return "preconfigured"
    if "ml" in t or "band" in t or "pred" in t:
        return "bandit"
    return t

def normalize_state(x):
    if pd.isna(x):
        return np.nan
    t = _strip_accents(str(x).strip().lower())
    mapping = {
        "flow": "Flow", "flujo": "Flow",
        "stress": "Stress", "estres": "Stress", "estresado": "Stress",
        "relax": "Relax", "relajacion": "Relax", "relajado": "Relax",
        "bored": "Bored", "aburrido": "Bored", "aburrimiento": "Bored"
    }
    return mapping.get(t, t.capitalize())

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main(excel_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Carga
    df_h = pd.read_excel(excel_path, sheet_name="Hordas")
    # (No necesitamos la hoja Encuestas para 6.1)
    if "player_id" not in df_h.columns:
        raise ValueError("Falta la columna 'player_id' en la hoja Hordas.")

    # 2) Normalizaciones y detección de columnas
    if "mode" not in df_h.columns:
        raise ValueError("Falta la columna 'mode' en la hoja Hordas.")
    df_h["mode"] = df_h["mode"].map(normalize_mode)

    state_col = pick_col(df_h, ["RealState", "real_state", "state"])
    if state_col:
        df_h["RealState_norm"] = df_h[state_col].map(normalize_state)
    else:
        df_h["RealState_norm"] = np.nan

    horde_idx_col = pick_col(df_h, ["session_horde_idx", "horde_idx", "horde_index", "round", "horde"])
    enemies_col   = pick_col(df_h, ["enemies_number", "n_enemies", "enemies", "enemy_count"])
    score_col     = pick_col(df_h, ["score", "points", "puntuacion"])
    att_col       = pick_col(df_h, ["avgAtt", "attention", "att", "meanAttention"])
    med_col       = pick_col(df_h, ["avgMed", "meditation", "med", "meanMeditation"])
    session_col   = pick_col(df_h, ["mode_block_id", "source_file", "session_id"])

    # 6.1.1 Descripción de la muestra
    n_players = df_h["player_id"].nunique()
    n_hordas_total = len(df_h)

    counts_player_mode = (
        df_h.groupby(["player_id", "mode"], dropna=False).size()
            .unstack(fill_value=0)
            .sort_index()
    )
    counts_player_mode["Total"] = counts_player_mode.sum(axis=1)
    counts_player_mode.to_csv(outdir / "tabla_hordas_por_jugador_y_modo.csv", index=True)

    avg_hordes_per_player_by_mode = (
        counts_player_mode.drop(columns=["Total"]).mean(axis=0).to_frame("avg_hordes_per_player").T
    )
    avg_hordes_per_player_by_mode.to_csv(outdir / "resumen_promedio_hordas_por_jugador_y_modo.csv", index=True)

    # 6.1.2 Distribución por modos
    mode_counts = df_h["mode"].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(7,4))
    mode_counts.plot(kind="bar")
    plt.title("Total de hordas por modo")
    plt.xlabel("Modo")
    plt.ylabel("Nº de hordas")
    savefig(outdir / "modos_barras.png")

    plt.figure(figsize=(5,5))
    mode_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.title("Proporciones de hordas por modo")
    savefig(outdir / "modos_pie.png")

    # 6.1.3 Evolución temporal en cada sesión
    # === Tabla por usuario y modo: puntuación máxima ===
    if score_col:
        # Filtrar registros válidos
        df_scores = df_h.dropna(subset=[score_col, "player_id", "mode"]).copy()

        # Tabla de máximos por jugador × modo
        score_max = df_scores.pivot_table(
            index="player_id",
            columns="mode",
            values=score_col,
            aggfunc="max"
        )

        # Reordenar columnas en el orden deseado si existen
        desired_order = [m for m in ["preconfigured", "heuristic", "bandit"] if m in score_max.columns]
        if desired_order:
            score_max = score_max.reindex(columns=desired_order)

        # Guardar CSV
        score_max.to_csv(outdir / "tabla_score_max_por_jugador_y_modo.csv", index=True)

        print("Guardada tabla de puntuación máxima por jugador × modo:",
            outdir / "tabla_score_max_por_jugador_y_modo.csv")




    # 6.1.4 Estados afectivos generales
    states_order = ["Flow", "Stress", "Relax", "Bored"]
    df_states = df_h.dropna(subset=["RealState_norm"])[["mode", "RealState_norm"]].copy()
    if not df_states.empty:
        prop = (
            df_states.groupby(["mode", "RealState_norm"]).size()
                     .groupby(level=0)
                     .apply(lambda s: s / s.sum())
                     .unstack()
                     .reindex(columns=[c for c in states_order if c in df_states["RealState_norm"].unique()])
                     .fillna(0.0)
                     .sort_index()
        )
        # --- 6.1.4 Barras apiladas robustas (Index o MultiIndex) ---
    prop.to_csv(outdir / "proporcion_estados_por_modo.csv", index=True)

    # Construir etiquetas seguras para el eje X
    labels = []
    for idx in prop.index:
        if isinstance(idx, tuple):
            labels.append(" / ".join(map(str, idx)))
        else:
            labels.append(str(idx))

    x = np.arange(len(labels))

    plt.figure(figsize=(max(7, 1.8*len(labels)), 4))
    bottom = np.zeros(len(labels))
    for col in prop.columns:
        plt.bar(x, prop[col].values, bottom=bottom, label=col)
        bottom += prop[col].values

    plt.title("Proporciones de estados afectivos por modo (RealState)")
    plt.xlabel("Modo")
    plt.ylabel("Proporción")
    plt.xticks(x, labels, rotation=0)
    plt.legend()
    savefig(outdir / "estados_apilados_por_modo.png")


    # 6.1.5 Variables EEG (Atención y Meditación)
    desc_list = []
    for var, name in [(att_col, "Attention"), (med_col, "Meditation")]:
        if var:
            g = df_h.groupby("mode")[var]
            desc = pd.DataFrame({
                "N": g.count(),
                "mean": g.mean(),
                "std": g.std(),
                "min": g.min(),
                "q25": g.quantile(0.25),
                "median": g.median(),
                "q75": g.quantile(0.75),
                "max": g.max()
            }).reset_index()
            desc["variable"] = name
            desc_list.append(desc)
    if desc_list:
        desc_all = pd.concat(desc_list, ignore_index=True)
        desc_all.to_csv(outdir / "descriptivos_attention_meditation_por_modo.csv", index=False)

    # === Boxplot conjunto de Attention y Meditation por modo (con leyenda correcta) ===
    if att_col and med_col:


        df_long = pd.DataFrame({
            "Attention": df_h[att_col],
            "Meditation": df_h[med_col],
            "mode": df_h["mode"]
        }).melt(id_vars="mode", value_vars=["Attention", "Meditation"],
                var_name="Variable", value_name="Valor").dropna(subset=["Valor", "mode"])

        modos = sorted(df_long["mode"].dropna().unique())
        x = np.arange(len(modos))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        # Datos por variable
        data_att = [df_long[(df_long["mode"] == m) & (df_long["Variable"] == "Attention")]["Valor"].values for m in modos]
        data_med = [df_long[(df_long["mode"] == m) & (df_long["Variable"] == "Meditation")]["Valor"].values for m in modos]

        # Posiciones
        pos_att = x - width/2
        pos_med = x + width/2

        # Dos boxplots separados
        bp_att = ax.boxplot(data_att, positions=pos_att, widths=width*0.9, patch_artist=True)
        bp_med = ax.boxplot(data_med, positions=pos_med, widths=width*0.9, patch_artist=True)

        # Estilo mínimo (colores distintos para reconocerlos en la leyenda)
        for box in bp_att["boxes"]:
            box.set_facecolor("tab:blue"); box.set_alpha(0.5)
        for box in bp_med["boxes"]:
            box.set_facecolor("tab:green"); box.set_alpha(0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(modos)
        ax.set_title("Attention vs Meditation por modo")
        ax.set_xlabel("Modo")
        ax.set_ylabel("Valor EEG")

        # Leyenda usando proxies
        legend_handles = [
            Patch(facecolor="tab:blue", alpha=0.5, label="Attention"),
            Patch(facecolor="tab:green", alpha=0.5, label="Meditation"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        savefig(outdir / "boxplot_attention_meditation.png")


    # 6.1.6 Resumen visual global — Transiciones
    group_cols = ["player_id"] + ([session_col] if session_col else [])
    order_col = horde_idx_col or pick_col(df_h, ["start_ts", "timestamp"])

    df_trans_rows = []
    if order_col is not None and "RealState_norm" in df_h.columns:
        for _, g in df_h.dropna(subset=["RealState_norm"]).sort_values(order_col).groupby(group_cols):
            vals = g["RealState_norm"].tolist()
            for i in range(len(vals) - 1):
                a, b = vals[i], vals[i + 1]
                if pd.isna(a) or pd.isna(b):
                    continue
                df_trans_rows.append((a, b))
    df_trans = pd.DataFrame(df_trans_rows, columns=["from", "to"])

    if not df_trans.empty:
        trans_mat = (
            df_trans.groupby(["from", "to"]).size()
                    .unstack(fill_value=0)
                    .reindex(index=[s for s in states_order if s in df_trans["from"].unique()],
                             columns=[s for s in states_order if s in df_trans["to"].unique()])
                    .fillna(0)
        )
        trans_mat.to_csv(outdir / "matriz_transiciones_estados.csv", index=True)

        # Heatmap de transiciones
        plt.figure(figsize=(6,5))
        plt.imshow(trans_mat.values, interpolation="nearest", aspect="auto", cmap="Blues")
        plt.title("Heatmap — Transiciones de estados (t → t+1)")
        plt.xlabel("Estado t+1")
        plt.ylabel("Estado t")
        plt.xticks(range(len(trans_mat.columns)), trans_mat.columns)
        plt.yticks(range(len(trans_mat.index)), trans_mat.index)
        cbar = plt.colorbar()
        cbar.set_label("Conteos")
        savefig(outdir / "transiciones_heatmap.png")

        # Alluvial/Sankey simple sin librerías extra
        left_totals = trans_mat.sum(axis=1)
        right_totals = trans_mat.sum(axis=0)
        total = left_totals.sum()
        if total > 0:
            left_heights = (left_totals / total).fillna(0.0)
            right_heights = (right_totals / total).fillna(0.0)

            def stacked_positions(values):
                starts, ends = {}, {}
                y = 0.0
                for key, h in values.items():
                    starts[key] = y
                    y2 = y + float(h)
                    ends[key] = y2
                    y = y2
                return starts, ends

            left_start, left_end = stacked_positions(left_heights.to_dict())
            right_start, right_end = stacked_positions(right_heights.to_dict())

            left_offsets = {k: left_start[k] for k in left_start}
            right_offsets = {k: right_start[k] for k in right_start}

            fig = plt.figure(figsize=(9,5))
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            # Bloques laterales (bordes)
            for side, starts, ends, xpos in [
                ("left", left_start, left_end, 0.0),
                ("right", right_start, right_end, 1.0),
            ]:
                for s in (left_totals.index if side == "left" else right_totals.index):
                    y0 = starts[s]; y1 = ends[s]
                    ax.add_patch(plt.Rectangle((xpos - (0.02 if side=="left" else 0.0), y0),
                                               0.02, y1 - y0, fill=False))
                    ax.text(xpos - (0.03 if side=="left" else -0.01), (y0 + y1)/2, s,
                            va="center", ha="right" if side=="left" else "left")

            # Flujos
            for s in trans_mat.index:
                for t in trans_mat.columns:
                    count = trans_mat.loc[s, t]
                    if count <= 0:
                        continue
                    frac = count / total
                    ly0 = left_offsets[s]
                    ly1 = ly0 + frac
                    left_offsets[s] = ly1

                    ry0 = right_offsets[t]
                    ry1 = ry0 + frac
                    right_offsets[t] = ry1

                    verts = [
                        (0.02, ly0),
                        (0.35, ly0),
                        (0.65, ry0),
                        (0.98, ry0),
                        (0.98, ry1),
                        (0.65, ry1),
                        (0.35, ly1),
                        (0.02, ly1),
                        (0.02, ly0),
                    ]
                    codes = [
                        MplPath.MOVETO,
                        MplPath.CURVE4,
                        MplPath.CURVE4,
                        MplPath.CURVE4,
                        MplPath.LINETO,
                        MplPath.CURVE4,
                        MplPath.CURVE4,
                        MplPath.CURVE4,
                        MplPath.CLOSEPOLY
                    ]
                    path = MplPath(verts, codes)
                    patch = PathPatch(path, alpha=0.5)  # sin especificar color
                    ax.add_patch(patch)

            ax.set_title("Transiciones de estados (t → t+1) — Diagrama tipo alluvial")
            savefig(outdir / "transiciones_alluvial.png")

    # Resumen textual mínimo
    resumen = pd.DataFrame([
        {"Métrica": "Jugadores únicos", "Valor": int(n_players)},
        {"Métrica": "Hordas totales registradas", "Valor": int(n_hordas_total)},
        {"Métrica": "Columna de enemigos usada", "Valor": enemies_col or "N/D"},
        {"Métrica": "Columna de score usada", "Valor": score_col or "N/D"},
        {"Métrica": "Variables EEG", "Valor": f"{att_col or 'N/D'} (Attention), {med_col or 'N/D'} (Meditation)"},
    ])
    resumen.to_csv(outdir / "resumen_6_1_metricas.csv", index=False)

    # Mensaje final en consola
    print("== RESUMEN 6.1 ==")
    print(f"Jugadores únicos: {n_players}")
    print(f"Hordas totales:   {n_hordas_total}")
    print(f"Salidas guardadas en: {outdir.resolve()}")

    # === Promedios de encuestas por versión (Preconfigured / HeuristicTree / ML) ===

    # ---- Configura AQUÍ la ruta de tus encuestas ----
    encuestas_path = Path(r"D:\TFM\NeuroDoomPy\sessions\00_masters\NeuroDoom_GameExperience.xlsx")

    def _strip_accents(s: str) -> str:
        return ''.join(ch for ch in unicodedata.normalize('NFD', str(s)) if unicodedata.category(ch) != 'Mn')

    def _pick_col(df, candidates):
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        # búsqueda flexible por empieza-con
        for c in df.columns:
            low = c.lower()
            if any(low.startswith(cc.lower()) for cc in candidates):
                return c
        return None

    def _normalize_version(x):
        if pd.isna(x):
            return np.nan
        t = _strip_accents(str(x)).strip().lower()
        if t.startswith("preconfig") or "preconfigur" in t:
            return "Preconfigured"
        if "heur" in t:
            return "HeuristicTree"
        if t in ("ml", "bandit", "pred", "predictivo"):
            return "ML"
        return str(x)

    # 1) Leer encuestas
    df_enc = pd.read_excel(encuestas_path, sheet_name=0)

    # 2) Localizar columnas clave
    version_col = _pick_col(df_enc, ["Versión", "Version"])
    if version_col is None:
        raise ValueError("No se encontró la columna 'Versión' en el archivo de encuestas.")

    id_like = {"id", "nombre del jugador", "nombre", "jugador", "player", "player_id"}
    meta_cols = [c for c in df_enc.columns if _strip_accents(c).strip().lower() in id_like]
    meta_cols = list(dict.fromkeys(meta_cols + [version_col]))  # únicos y preserva orden

    # 3) Normalizar versiones
    df_enc["version_norm"] = df_enc[version_col].map(_normalize_version)

    # 4) Codificar respuestas Likert -> 1..5
    agreement_map = {
        "Totalmente en desacuerdo": 1,
        "En desacuerdo": 2,
        "Ni de acuerdo ni en desacuerdo": 3,
        "De acuerdo": 4,
        "Totalmente de acuerdo": 5,
        "Para nada": 1,
        "Poco": 3,
        "Ni poco ni mucho": 4,
        "Bastante": 5,
        "Mucho": 6,
        "Muy poco": 2,
        "Muchísimo": 7
    }

    intensity_map = {
        "Nada": 1,
        "Poco": 2,
        "Ni poco ni mucho": 3,
        "Bastante": 4,
        "Mucho": 4,         # <- si prefieres Mucho=5, cambia a 5
        "Muchísimo": 5,
    }
    likert_map = {**agreement_map, **intensity_map}

    df_num = df_enc.replace(likert_map)

    # 5) Seleccionar columnas de respuesta (no meta) y quedarnos con las numéricas
    resp_cols = [c for c in df_enc.columns if c not in meta_cols]
    num_cols = [c for c in resp_cols if pd.api.types.is_numeric_dtype(df_num[c])]

    if not num_cols:
        raise ValueError("No se detectaron columnas numéricas tras codificar las respuestas Likert.")

    # 6) Promedios por versión (tabla: preguntas × versión)
    promedios = df_num.groupby("version_norm")[num_cols].mean().T
    promedios = promedios[ [c for c in ["Preconfigured", "HeuristicTree", "ML"] if c in promedios.columns] ]
    promedios_rounded = promedios.round(2)
    (promedios_rounded
    ).to_csv(outdir / "encuestas_promedio_por_version.csv", encoding="utf-8-sig")

    # 7) (Opcional) N válidos por pregunta y versión
    n_validos = df_num.groupby("version_norm")[num_cols].count().T
    n_validos = n_validos[ [c for c in ["Preconfigured", "HeuristicTree", "ML"] if c in n_validos.columns] ]
    n_validos.to_csv(outdir / "encuestas_N_validos_por_version.csv", encoding="utf-8-sig")

    # 8) (Opcional) media global por versión (promedio de todas las preguntas)
    media_global = promedios.mean(axis=0).to_frame("media_global").round(2)
    media_global.to_csv(outdir / "encuestas_media_global_por_version.csv", encoding="utf-8-sig")

    print("== Encuestas ==")



if __name__ == "__main__":
    excel_path = Path(r"D:\TFM\NeuroDoomPy\sessions\01_features\features_horda.xlsx")
    outdir     = Path(r"D:\TFM\NeuroDoomPy\sessions\01_features\figuresR")
    main(excel_path, outdir)