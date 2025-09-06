"""
estadistics.py
--------------
Cruza unity_events + EEG + logs con estas reglas:
- El logs.csv tiene 14 columnas (cada fila = horda).
- Solo son válidas las FILAS desde la 84 en adelante (1-based) → raw.iloc[83:, :].
- Los logs se alinean por BLOQUES consecutivos de `mode` con desplazamiento: hordas 2..N-1.
- El modo 'GameOver' NO se cruza con logs (siempre NaN en columnas de log).
"""

import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import fnmatch

# ========================= CONFIG =========================
USER = "Noah_28"   # CAMBIAMOS EL USUARIO AQUÍ
name = "NGL"  # Nombre del usuario (para el nombre de la sesión)

BASE_DIR = r"D:\TFM\NeuroDoomPy\sessions"
SESSION_DIR = os.path.join(BASE_DIR, USER)
BASE_DIR_MASTERS = r"D:\TFM\NeuroDoomPy\sessions\00_masters"

UNITY_EVENTS_CSV = os.path.join(SESSION_DIR, "unity_events.csv")
DATA_CSV        = os.path.join(SESSION_DIR, "data.csv")
LOGS_CSV        = os.path.join(SESSION_DIR, "logs.csv")
OUT_CSV         = os.path.join(SESSION_DIR, "master_hordas.csv")
OUT_XLSX        = os.path.join(SESSION_DIR, "master_hordas_" + name + ".xlsx")
OUT_XLSX_MASTERS        = os.path.join(BASE_DIR_MASTERS, "master_hordas_" + name + ".xlsx")
# ==========================================================

# ========================= VISTAS/EXPORT =========================
# (1) Si rellenas INCLUDE_MASTER, solo exportará esas columnas en 'master'.
# (2) Si está vacío, exportará todas excepto las que estén en EXCLUDE_MASTER (acepta comodines).
INCLUDE_MASTER = []  # p.ej.: ["mode","horde","avgAtt","avgMed","fFlow","fStress","fBored","fRelax","score"]
EXCLUDE_MASTER = [
    "*_scaled",       # quita backups
    "frameCount",
    "domThisHorde",
    "kills"


]
# ================================================================



def _filter_columns(df: pd.DataFrame,
                    include_patterns=None,
                    exclude_patterns=None) -> pd.DataFrame:
    cols = list(df.columns)

    # Lista blanca (si la usas, manda sobre la negra)
    if include_patterns:
        keep = set()
        for pat in include_patterns:
            keep.update([c for c in cols if fnmatch.fnmatch(c, pat)])
        selected = [c for c in cols if c in keep]
    else:
        selected = cols[:]  # todas

    # Lista negra
    if exclude_patterns and not include_patterns:
        drop = set()
        for pat in exclude_patterns:
            drop.update([c for c in selected if fnmatch.fnmatch(c, pat)])
        selected = [c for c in selected if c not in drop]

    return df[selected]



def classify_state(att, med, att_hi=60, att_lo=40, med_hi=60, med_lo=40):
    if pd.isna(att) or pd.isna(med): return np.nan
    if att > att_hi and med > med_hi: return "flow"
    if att > att_hi and med < med_lo: return "stress"
    if att < att_lo and med > med_hi: return "relax"
    if att < att_lo and med < med_lo: return "bored"
    return "neutral"


def state_fractions(df):
    if df.empty: return 0.0, 0.0, 0.0, 0.0, np.nan
    vc = df["state"].value_counts(normalize=True).to_dict()
    f_flow   = float(vc.get("flow", 0.0))
    f_stress = float(vc.get("stress", 0.0))
    f_bored  = float(vc.get("bored", 0.0))
    f_relax  = float(vc.get("relax", 0.0))
    pool = {"flow": f_flow, "stress": f_stress, "bored": f_bored, "relax": f_relax}
    dom = max(pool, key=pool.get) if any(pool.values()) else np.nan
    return f_flow, f_stress, f_bored, f_relax, dom


def _as_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def load_unity_events(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Timestamp": "timestamp", "Time": "timestamp", "time": "timestamp",
        "Mode": "mode", "Horde": "horde", "Score": "score"
    })

    # Normaliza tipos / espacios
    if "mode" not in df.columns:
        raise ValueError("unity_events.csv: falta columna 'Mode/mode'")

    df["mode"] = df["mode"].astype(str).str.strip()

    if "timestamp" not in df.columns:
        raise ValueError("unity_events.csv: falta columna 'Timestamp/Time/time'")

    df["timestamp"] = _as_numeric(df["timestamp"])

    if "horde" in df.columns:
        df["horde"] = _as_numeric(df["horde"])
    else:
        df["horde"] = np.arange(1, len(df) + 1, dtype=float)

    if "score" in df.columns:
        df["score"] = _as_numeric(df["score"])
    else:
        df["score"] = np.nan

    # >>> CLAVE: conservar el ORDEN CRONOLÓGICO (no ordenar por mode)
    df = df.sort_values(["timestamp", "horde"], kind="mergesort").reset_index(drop=True)

    # Índice por modo (respetando orden original)
    df["session_horde_idx"] = df.groupby("mode", sort=False).cumcount() + 1

    # Bloques consecutivos de mode (run-length)
    mode_changes = (df["mode"] != df["mode"].shift(1)).astype(int)
    df["mode_block_id"] = mode_changes.cumsum()  # 1,2,3,... cada vez que cambia el modo

    return df


def load_eeg(path):
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["__time__"] = _as_numeric(df["time"])
    elif "timestamp" in df.columns:
        df["__time__"] = _as_numeric(df["timestamp"])
    else:
        raise ValueError("data.csv: no hay columna 'time' o 'timestamp'")

    if "attention" not in df.columns:
        raise ValueError("data.csv: falta columna 'attention'")
    if "meditation" not in df.columns:
        raise ValueError("data.csv: falta columna 'meditation'")

    df["attention"] = _as_numeric(df["attention"])
    df["meditation"] = _as_numeric(df["meditation"])
    return df


def _maybe_rescale_minus1_to_1_to_0_to_1(series: pd.Series) -> pd.Series:
    """Si detectamos valores fuera de [0,1] asumimos que está en [-1,1] y remapeamos a [0,1]."""
    if series.dropna().empty:
        return series
    mn, mx = series.min(skipna=True), series.max(skipna=True)
    if (mn < 0.0) or (mx > 1.0):
        return (series + 1.0) / 2.0
    return series


def build_master():
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

    # --- Eventos & EEG ---
    ue = load_unity_events(UNITY_EVENTS_CSV)
    eeg = load_eeg(DATA_CSV)

    # --- Ventanas por horda usando "desde el fin previo de este mode hasta el fin actual"
    rows = []
    prev_end_per_mode = {}
    eeg_time_min = eeg["__time__"].min()

    for _, r in ue.iterrows():
        mode = r["mode"]
        end_ts = float(r["timestamp"])
        start_ts = prev_end_per_mode.get(mode, eeg_time_min)
        prev_end_per_mode[mode] = end_ts

        # Ventana (start_ts, end_ts]
        chunk = eeg.loc[(eeg["__time__"] > start_ts) & (eeg["__time__"] <= end_ts),
                        ["attention", "meditation"]].copy()

        if not chunk.empty:
            chunk["state"] = [classify_state(a, m) for a, m in zip(chunk["attention"], chunk["meditation"])]
            f_flow, f_stress, f_bored, f_relax, dom = state_fractions(chunk)
            avg_att = float(chunk["attention"].mean())
            avg_med = float(chunk["meditation"].mean())
        else:
            f_flow = f_stress = f_bored = f_relax = 0.0
            dom = np.nan
            avg_att = avg_med = np.nan

        rows.append({
            "mode": mode,
            "mode_block_id": int(r["mode_block_id"]),
            "horde": r["horde"],
            "session_horde_idx": r["session_horde_idx"],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "avgAtt": avg_att,
            "avgMed": avg_med,
            "fFlow": f_flow,
            "fStress": f_stress,
            "fBored": f_bored,
            "fRelax": f_relax,
            "RealState": dom,
            "score": r.get("score", np.nan),
        })

    eeg_by_horde = pd.DataFrame(rows).reset_index(drop=True)

    # Indice previo de estado dominante (0..3) por modo y bloque
    state_to_idx = {"flow": 0, "stress": 1, "bored": 2, "relax": 3}
    def _map_state(s):
        if isinstance(s, str):
            return state_to_idx.get(s.lower(), np.nan)
        return np.nan

    eeg_by_horde["pIdx"] = (
        eeg_by_horde.groupby(["mode", "mode_block_id"], sort=False)["RealState"]
        .shift(1).map(_map_state)
    )

    # --- Logs: 14 columnas; quedarse con FILAS 83..fin (1-based) ---
    raw = pd.read_csv(LOGS_CSV, header=None)
    raw_valid = raw.iloc[83:, :].copy()  # ← desde la fila 84 (1-based)

    desired_cols = [
        "frameCount", "dificultad", "domThisHorde", "kills",
        "aAtt", "aMed", "prevEmotion",
        "fFlow_log", "fStress_log", "fBored_log", "fRelax_log",
        "predState", "RealState_log", "reward"
    ]

    # Normaliza a 14 columnas con nombres
    if raw_valid.shape[1] >= len(desired_cols):
        logs_pool = raw_valid.iloc[:, :len(desired_cols)].copy()
        logs_pool.columns = desired_cols
    else:
        logs_pool = raw_valid.copy()
        # renombra lo que exista
        for i in range(raw_valid.shape[1]):
            logs_pool.rename(columns={raw_valid.columns[i]: desired_cols[i]}, inplace=True)
        # crea columnas faltantes
        for i in range(raw_valid.shape[1], len(desired_cols)):
            logs_pool[desired_cols[i]] = np.nan
        logs_pool = logs_pool[desired_cols]

    # Convierte numéricas cuando proceda (no fuerzo prevEmotion/predState si son strings)
    num_cols = ["frameCount", "dificultad", "domThisHorde", "kills", "aAtt", "aMed",
                "fFlow_log", "fStress_log", "fBored_log", "fRelax_log",
                "RealState_log", "reward"]
    for c in num_cols:
        if c in logs_pool.columns:
            logs_pool[c] = _as_numeric(logs_pool[c])

    # --- Alinear por BLOQUES consecutivos de mode con desplazamiento (2..N-1), omitiendo GameOver ---
    aligned_chunks = []
    cursor = 0
    per_block_report = OrderedDict()

    # Recorremos en orden cronológico por bloques
    for (block_id, mode), g in (
        eeg_by_horde.groupby(["mode_block_id", "mode"], sort=False)
    ):
        n = len(g)  # nº de hordas en ESTE bloque consecutivo
        key = f"{mode}#b{block_id}"

        if str(mode).strip().lower() == "gameover":
            block = pd.DataFrame({c: [np.nan] * n for c in desired_cols})
            aligned_chunks.append(block)
            per_block_report[key] = {"hordas": n, "logs_consumidos": 0}
            continue

        k = max(0, n - 2)  # logs esperados (hordas 2..N-1 en este bloque)
        chunk = logs_pool.iloc[cursor:cursor + k].copy()
        cursor += k

        # padding: horda 1 y horda N → NaN
        pad_top = pd.DataFrame({c: [np.nan] for c in desired_cols})
        pad_bottom = pd.DataFrame({c: [np.nan] for c in desired_cols})
        block = pd.concat([pad_top, chunk, pad_bottom], ignore_index=True)

        # normaliza tamaño a n
        if len(block) < n:
            miss = n - len(block)
            block = pd.concat([block,
                               pd.DataFrame({c: [np.nan] * miss for c in desired_cols})],
                              ignore_index=True)
        elif len(block) > n:
            block = block.iloc[:n].copy()

        aligned_chunks.append(block.reset_index(drop=True))
        per_block_report[key] = {"hordas": n, "logs_consumidos": len(chunk)}

    logs_aligned = pd.concat(aligned_chunks, ignore_index=True)

    # --- Concatena por posición ---
    master = pd.concat(
        [eeg_by_horde.reset_index(drop=True), logs_aligned.reset_index(drop=True)],
        axis=1
    )

    # ==== DES-ESCALADO DE CAMPOS DE LOGS ====

    def _backup_and_assign(df, col, new_values, backup_suffix="_scaled"):
        """Guarda la columna original y escribe la desescalada."""
        if col in df.columns:
            #df[col + backup_suffix] = df[col]
            df[col] = new_values

    # diff → dificultad real
    if "dificultad" in master.columns:
        _backup_and_assign(master, "dificultad", 1.75 * master["dificultad"].astype(float) + 2.25)

    # pIdx → índice entero 0..4
    if "pIdx" in master.columns:
        master["pIdx_scaled"] = master["pIdx"]
        prev_idx = 2.0 * master["pIdx"].astype(float) + 2.0
        master["pIdx"] = np.clip(np.rint(prev_idx), 0, 4).astype("Int64")

    # aAtt, aMed → valores en 0..100
    for col in ["aAtt", "aMed"]:
        if col in master.columns:
            _backup_and_assign(master, col, 50.0 * master[col].astype(float) + 50.0)
    
    # aAtt, aMed → valores en 0..100
    if "prevEmotion" in master.columns:
        _backup_and_assign(master, "prevEmotion", 2.00 * master["prevEmotion"].astype(float) + 2.00)

    # --- Normalizar f* de [-1,1] → [0,1] SOLO si hace falta
    for col in ["fFlow_log", "fStress_log", "fBored_log", "fRelax_log"]:
        if col in master.columns:
            master[col] = _maybe_rescale_minus1_to_1_to_0_to_1(master[col])




    # --- Exporta ---
    master_full = master.copy()
    master_export = _filter_columns(master_full,
                                    include_patterns=INCLUDE_MASTER,
                                    exclude_patterns=EXCLUDE_MASTER)

    # CSVs: uno limpio y otro completo (por si luego necesitas todo)
    OUT_CSV_FULL = os.path.splitext(OUT_CSV)[0] + "_full.csv"
    master_export.to_csv(OUT_CSV, index=False)
    master_full.to_csv(OUT_CSV_FULL, index=False)

    try:
        with pd.ExcelWriter(OUT_XLSX_MASTERS, engine="xlsxwriter") as writer:
            # Hoja 'master' = filtrada (como tú mandes)
            master_export.to_excel(writer, index=False, sheet_name="master")
            # Hoja 'master_full' = TODO por si necesitas depurar
            master_full.to_excel(writer, index=False, sheet_name="master_full")

            # Hojas con los datos originales / intermedios sin tocar
            ue.to_excel(writer, index=False, sheet_name="unity_events")
            eeg[["__time__","attention","meditation"]].to_excel(writer, index=False, sheet_name="eeg_samples")
            logs_pool.to_excel(writer, index=False, sheet_name="logs_valid_rows_83on")
            logs_aligned.to_excel(writer, index=False, sheet_name="logs_aligned")
        x_out = OUT_XLSX_MASTERS
    except PermissionError:
        alt = os.path.splitext(OUT_XLSX_MASTERS)[0] + "_alt.xlsx"

    try:
        with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
            # Hoja 'master' = filtrada (como tú mandes)
            master_export.to_excel(writer, index=False, sheet_name="master")
            # Hoja 'master_full' = TODO por si necesitas depurar
            master_full.to_excel(writer, index=False, sheet_name="master_full")

            # Hojas con los datos originales / intermedios sin tocar
            ue.to_excel(writer, index=False, sheet_name="unity_events")
            eeg[["__time__","attention","meditation"]].to_excel(writer, index=False, sheet_name="eeg_samples")
            logs_pool.to_excel(writer, index=False, sheet_name="logs_valid_rows_83on")
            logs_aligned.to_excel(writer, index=False, sheet_name="logs_aligned")
        x_out = OUT_XLSX
    except PermissionError:
        alt = os.path.splitext(OUT_XLSX)[0] + "_alt.xlsx"
        with pd.ExcelWriter(alt, engine="xlsxwriter") as writer:
            master_export.to_excel(writer, index=False, sheet_name="master")
            master_full.to_excel(writer, index=False, sheet_name="master_full")
            ue.to_excel(writer, index=False, sheet_name="unity_events")
            eeg[["__time__","attention","meditation"]].to_excel(writer, index=False, sheet_name="eeg_samples")
            logs_pool.to_excel(writer, index=False, sheet_name="logs_valid_rows_83on")
            logs_aligned.to_excel(writer, index=False, sheet_name="logs_aligned")
        x_out = alt

    print(f"✅ Exportado:\n- {OUT_CSV} (filtrado)\n- {OUT_CSV_FULL} (completo)\n- {x_out}")


    # --- Reportes ---
    print(f"✅ Exportado:\n- {OUT_CSV}\n- {x_out}")
    print(f"Eventos totales: {len(eeg_by_horde)} | Filas log válidas (83..fin): {len(logs_pool)}")

    total_consumed = 0
    for key, rep in per_block_report.items():
        print(f"  · {key}: hordas={rep['hordas']} | logs_consumidos={rep['logs_consumidos']}")
        total_consumed += rep["logs_consumidos"]

    leftover = max(0, len(logs_pool) - total_consumed)
    print(f"Logs consumidos: {total_consumed} | Logs restantes sin consumir: {leftover}")
    if leftover != 0:
        print("⚠️ Aviso: hay desajuste entre hordas esperadas y filas de logs. "
              "Revisa cortes por bloque/mode o filas inválidas en logs.csv.")

    # Sanidad: mismas filas
    if len(master) != len(eeg_by_horde) or len(logs_aligned) != len(eeg_by_horde):
        print("❗ Inconsistencia de tamaños tras la alineación. Revisa la lógica de bloques.")

if __name__ == "__main__":
    build_master()
