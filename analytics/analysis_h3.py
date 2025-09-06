# -*- coding: utf-8 -*-
"""
Análisis H3 NeuroDoom EEG-DDA
- Lee NeuroDoom_GameExperience.xlsx
- Mapea respuestas textuales a Likert (1–5 y 1–7)
- Reescala todo a 0–1
- Calcula índices: Flow (FSS 2,3,4), Inmersión (MiniPXI 6), Disfrute (MiniPXI 7), Motivación-BCI (BCI 2,4,6)
- Friedman + Wilcoxon pareado (Holm)
- Guarda resultados y figuras
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_XLSX = r"D:\TFM\NeuroDoomPy\sessions\00_masters\NeuroDoom_GameExperience.xlsx"
OUTDIR = Path("outputs_h3")
OUTDIR.mkdir(exist_ok=True)

# Columnas de identificación (ajusta si difieren en tu archivo)
ID_COL = "ID"
NAME_COL = "Nombre del Jugador"

# Columnas que indican qué modo jugó cada persona en "Versión 1/2/3".
# Sus celdas deben contener A/B/C (A=preconfigured, B=HeuristicTree, C=ML)
VERSION_FLAG_COLS = [r"^Versi[oó]n\s*1$", r"^Versi[oó]n\s*2$", r"^Versi[oó]n\s*3$"]

# Map de letra a etiqueta de modo
MODE_LETTER_TO_LABEL = {
    "A": "preconfigured",
    "B": "heuristic",
    "C": "ML"
}

# Ítems que usaremos por cuestionario (regex en castellano, insensible a mayúsculas)
# Sufijos ".2"/".3" o " 2"/" 3" se interpretan como versiones 2/3
ITEMS = {
    "FSS": {
        # (etiqueta_interna, patrón_regex)
        "FSS_2": r"Mis pensamientos/acciones flu[ií]an de manera suave y natural\.?$",
        "FSS_3": r"No me di cuenta de que pasaba el tiempo\.?$",
        "FSS_4": r"No tuve ninguna dificultad para concentrarme\.?$",
        # Escala de FSS: 1-7
        "scale_max": 7
    },
    "MiniPXI": {
        "MiniPXI_6": r"Me sent[ií] completamente concentrado en lo que hac[ií]a\.?$",
        "MiniPXI_7": r"Disfrut[eé] mucho jugando a este juego\.?$",
        # Escala 1-5
        "scale_max": 5
    },
    "BCI": {
        "BCI_2": r"Sent[ií] que el BCI aumentaba mi sensaci[oó]n de estar “dentro” del juego\.?$",
        "BCI_4": r"Jugar mientras se med[ií]a mi actividad cerebral result[oó] interesante para m[ií]\.?$",
        "BCI_6": r"Volver[ií]a a jugar usando este dispositivo en el futuro\.?$",
        # Escala 1-5
        "scale_max": 5
    }
}

# Construcciones/índices para H3
CONSTRUCTS = {
    "Flow": ["FSS_2", "FSS_3", "FSS_4"],  # FSS (1-7) → reescalado a 0-1
    "Inmersion": ["MiniPXI_6"],            # 1-5 → 0-1
    "Disfrute": ["MiniPXI_7"],             # 1-5 → 0-1
    "Motivacion_BCI": ["BCI_2", "BCI_4", "BCI_6"]  # 1-5 → 0-1
}

# Mapeos de texto → valor Likert
LIKERT_1_5 = {
    "totalmente en desacuerdo": 1,
    "en desacuerdo": 2,
    "ni de acuerdo ni en desacuerdo": 3,
    "de acuerdo": 4,
    "totalmente de acuerdo": 5,
}

# Escala FSS 1–7 típica con anclajes en castellano
LIKERT_1_7 = {
    "para nada": 1,
    "muy poco": 2,
    "poco": 3,
    "ni poco ni mucho": 4,
    "bastante": 5,
    "mucho": 6,
    "muchísimo": 7,
}

# También aceptamos ya-numéricos por si el Excel tiene números
# =========================

def read_excel(input_path):
    df = pd.read_excel(input_path)
    # Limpieza básica de cabeceras
    df.columns = [str(c).strip() for c in df.columns]
    return df

def compile_patterns():
    # Compilamos patrones de items y versiones
    item_patterns = {}
    for block, cfg in ITEMS.items():
        for k, patt in cfg.items():
            if k == "scale_max":
                continue
            item_patterns[k] = re.compile(patt, flags=re.IGNORECASE)
    version_cols = [re.compile(p) for p in VERSION_FLAG_COLS]
    return item_patterns, version_cols

def detect_version_flags(df, version_col_patterns):
    # Encuentra las columnas reales que casan con "Versión 1/2/3"
    found = []
    for patt in version_col_patterns:
        col = next((c for c in df.columns if patt.search(c)), None)
        if col is None:
            raise ValueError(f"No se encontró columna que case con patrón de versión: {patt.pattern}")
        found.append(col)
    return found  # [col_v1, col_v2, col_v3]

def normalize_text(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip().lower()
    # normaliza comillas y espacios
    s = s.replace('"', '\"').replace("“", "\"").replace("”", "\"")
    return s

def map_likert_value(raw, scale_max_guess=None):
    """Mapea texto → número. Si ya es numérico, lo devuelve.
    Usa heurística: si el texto está en diccionario 1–5 ó 1–7.
    """
    if pd.isna(raw):
        return np.nan
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    s = normalize_text(raw)

    # Primero probar diccionario 1-5
    if s in LIKERT_1_5:
        return float(LIKERT_1_5[s])
    # Luego 1-7
    if s in LIKERT_1_7:
        return float(LIKERT_1_7[s])

    # A veces los datos son "Muchísimo" con mayúscula/acentos:
    # ya normalizamos a lower, debería casar.

    # Si nada casa y parece un número en texto:
    try:
        return float(s.replace(",", "."))
    except:
        # No mapeable: devolver NaN para revisar
        return np.nan

def rescale_to_01(x, max_scale):
    if pd.isna(x):
        return np.nan
    return (x - 1.0) / (max_scale - 1.0)

def extract_item_columns(df, item_patterns):
    """Devuelve un dict: etiqueta_item -> {version: [column_names]}.
    Busca columnas cuyo 'texto base' case con el patrón y detecta versión por sufijo (.2/.3/' 2'/' 3').
    """
    by_item = {k: {1: [], 2: [], 3: []} for k in item_patterns.keys()}

    for col in df.columns:
        base = col
        version = 1
        # Detectar versión por sufijo
        m = re.search(r"(?:\.|\s)([23])$", base)
        if m:
            version = int(m.group(1))
            base = re.sub(r"(?:\.|\s)([23])$", "", base).strip()

        # ¿Esta columna es uno de nuestros ítems?
        for item_key, patt in item_patterns.items():
            if patt.search(base):
                by_item[item_key][version].append(col)

    return by_item

def build_long(df, id_col, name_col, version_flag_cols, item_columns_by_item):
    """Convierte a formato largo: una fila por jugador × versión × ítem.
    Usa las columnas 'Versión 1/2/3' para mapear 1/2/3 → A/B/C → modo.
    """
    required_id_cols = [id_col, name_col] + version_flag_cols
    for c in required_id_cols:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: {c}")

    # Mapeo por jugador: v1/v2/v3 → letter (A/B/C) → mode label
    player_maps = {}
    for _, row in df[required_id_cols].iterrows():
        pid = row[id_col]
        mapping = {}
        for i, vcol in enumerate(version_flag_cols, start=1):
            letter = str(row[vcol]).strip().upper()
            mapping[i] = MODE_LETTER_TO_LABEL.get(letter, f"unknown_{letter}")
        player_maps[pid] = mapping

    # Recopilar datos en largo
    records = []
    for _, row in df.iterrows():
        pid = row[id_col]
        pname = row[name_col]
        if pid not in player_maps:
            continue
        vmap = player_maps[pid]  # {1:'preconfigured', 2:'heuristic', 3:'ML'}

        for item_key, versions_dict in item_columns_by_item.items():
            for v in (1, 2, 3):
                cols = versions_dict[v]
                if not cols:
                    # Puede que el Excel tenga exactamente una columna por versión/ítem.
                    continue
                # Si hay varias columnas que casan, tomamos la primera no vacía (por seguridad)
                val = np.nan
                for c in cols:
                    if not pd.isna(row[c]):
                        val = row[c]
                        break
                records.append({
                    "player_id": pid,
                    "player_name": pname,
                    "version_idx": v,
                    "mode": vmap.get(v, f"unknown_v{v}"),
                    "item": item_key,
                    "raw": val
                })

    long_df = pd.DataFrame.from_records(records)
    return long_df

def add_numeric_and_rescales(long_df):
    # Determinar escala por item
    item_to_scale_max = {}
    for block, cfg in ITEMS.items():
        scale_max = cfg["scale_max"]
        for k in cfg.keys():
            if k == "scale_max":
                continue
            item_to_scale_max[k] = scale_max

    # Mapear a valores y reescalar 0-1
    long_df["value_num"] = long_df.apply(
        lambda r: map_likert_value(r["raw"], scale_max_guess=item_to_scale_max.get(r["item"], 5)),
        axis=1
    )
    long_df["scale_max"] = long_df["item"].map(item_to_scale_max)
    long_df["value_0_1"] = long_df.apply(
        lambda r: rescale_to_01(r["value_num"], r["scale_max"]) if pd.notna(r["value_num"]) else np.nan,
        axis=1
    )
    # También versión reescalada a 1–5 si la necesitas:
    long_df["value_1_5"] = long_df.apply(
        lambda r: (rescale_to_01(r["value_num"], r["scale_max"]) * 4 + 1) if pd.notna(r["value_num"]) else np.nan,
        axis=1
    )
    return long_df

def compute_constructs(long_df):
    """Devuelve wide a nivel jugador×modo con los índices por constructo (media de ítems reescalados 0–1)."""
    # Filtramos solo ítems que formen parte de los constructs
    items_needed = {it for lst in CONSTRUCTS.values() for it in lst}
    df = long_df[long_df["item"].isin(items_needed)].copy()

    # Media por jugador×modo×ítem (0–1)
    g = df.groupby(["player_id", "player_name", "mode", "item"], as_index=False)["value_0_1"].mean()

    # Para cada constructo, hacer la media de sus ítems
    rows = []
    for (pid, pname, mode), sub in g.groupby(["player_id", "player_name", "mode"]):
        rec = {"player_id": pid, "player_name": pname, "mode": mode}
        for cons_name, item_list in CONSTRUCTS.items():
            vals = sub.loc[sub["item"].isin(item_list), "value_0_1"].dropna()
            rec[cons_name] = np.nan if vals.empty else vals.mean()
        rows.append(rec)

    wide = pd.DataFrame(rows)
    return wide

def friedman_and_posthoc(wide_df, construct):
    """Friedman (3 condiciones intra-sujetos) y Wilcoxon pareado (Holm). 
    Devuelve dict con resultados y dataframe con pares.
    """
    # Pivot a players × mode
    pivot = wide_df.pivot_table(index="player_id", columns="mode", values=construct)
    modes = list(pivot.columns)
    if len(modes) != 3:
        return {"error": f"Se esperaban 3 modos, encontrados: {modes}"}, None

    # Asegurar filas completas
    pivot = pivot.dropna()
    if pivot.shape[0] < 2:
        return {"error": "No hay suficientes jugadores con datos completos para Friedman."}, None

    # Friedman
    stat, pval = friedmanchisquare(*[pivot[m].values for m in modes])
    res = {
        "construct": construct,
        "friedman_chi2": float(stat),
        "friedman_p": float(pval),
        "N": int(pivot.shape[0]),
        "modes": modes
    }

    # Post-hoc Wilcoxon con corrección Holm
    pairs = list(itertools.combinations(modes, 2))
    ph_rows = []
    pvals = []
    for a, b in pairs:
        x = pivot[a].values
        y = pivot[b].values
        # Wilcoxon pareado bidireccional
        wstat, pw = wilcoxon(x, y, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
        # tamaño de efecto r = Z / sqrt(N)
        # Z ~ aproximación de wstat; scipy no lo da directamente. Aproximación normal: 
        n = len(x)
        # Media y sd de W bajo H0: mu = n(n+1)/4 ; sigma = sqrt(n(n+1)(2n+1)/24)
        mu = n*(n+1)/4.0
        sigma = np.sqrt(n*(n+1)*(2*n+1)/24.0)
        z = (wstat - mu) / sigma if sigma > 0 else np.nan
        r = abs(z) / np.sqrt(n) if n > 0 else np.nan

        ph_rows.append({"pair": f"{a} vs {b}", "wilcoxon_W": float(wstat), "p_raw": float(pw), "N": int(n), "effect_size_r": float(r)})
        pvals.append(pw)

    # Corrección Holm
    order = np.argsort(pvals)  # de menor a mayor
    m = len(pvals)
    adj = [None]*m
    for rank, idx in enumerate(order):
        adj_p = pvals[idx] * (m - rank)
        adj[idx] = min(adj_p, 1.0)

    for i, row in enumerate(ph_rows):
        row["p_holm"] = float(adj[i])

    posthoc_df = pd.DataFrame(ph_rows)
    return res, posthoc_df

def plot_bar_with_error(wide_df, construct, outdir=OUTDIR):
    """Barplot de medias por modo con barras de error (SEM)."""
    agg = wide_df.groupby("mode")[construct].agg(["mean", "std", "count"]).dropna()
    if agg.empty:
        return
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    ax = agg["mean"].plot(kind="bar", yerr=agg["sem"], capsize=4)
    ax.set_ylabel(f"{construct} (0–1)")
    ax.set_title(f"{construct} por modo")
    plt.tight_layout()
    plt.savefig(outdir / f"bar_{construct}.png", dpi=150)
    plt.close()

def main():
    df = read_excel(INPUT_XLSX)
    item_patterns, version_col_patterns = compile_patterns()
    version_cols = detect_version_flags(df, version_col_patterns)

    # Extraer columnas por ítem/versión
    item_cols_by_item = extract_item_columns(df, item_patterns)

    # Construir largo con mapeo por jugador de v1/v2/v3 → A/B/C → modo
    long_df = build_long(df, ID_COL, NAME_COL, version_cols, item_cols_by_item)

    # Mapear a numérico y reescalar
    long_df = add_numeric_and_rescales(long_df)

    # Guardar tabla larga para auditoría
    long_df.to_csv(OUTDIR / "long_items_0_1.csv", index=False, encoding="utf-8-sig")

    # Índices por constructo (0–1)
    wide_df = compute_constructs(long_df)
    wide_df.to_csv(OUTDIR / "constructs_0_1.csv", index=False, encoding="utf-8-sig")

    # Análisis: Friedman + post-hoc por constructo
    summary_rows = []
    for cons in CONSTRUCTS.keys():
        res, ph = friedman_and_posthoc(wide_df, cons)
        if "error" in res:
            print(f"[{cons}] ERROR:", res["error"])
        else:
            summary_rows.append(res)
            if ph is not None:
                ph.to_csv(OUTDIR / f"posthoc_{cons}.csv", index=False, encoding="utf-8-sig")
        # Plot
        plot_bar_with_error(wide_df, cons, OUTDIR)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUTDIR / "friedman_summary.csv", index=False, encoding="utf-8-sig")

    # Impresión breve
    print("\n== RESUMEN FRIEDMAN ==")
    if summary_rows:
        print(pd.DataFrame(summary_rows))
    else:
        print("Sin resultados (comprueba columnas y patrones).")

    print("\nArchivos generados en:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
