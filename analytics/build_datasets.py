
import os
import re
import glob
import math
import pandas as pd
from typing import Dict, List, Tuple, Optional

# -------------------------
# CONFIG (edit if needed)
# -------------------------
STATE_MAP = {0: "flow", 1: "stress", 2: "bored", 3: "relax"}  # enum mapping for RealState_log
INPUT_PATTERNS = ["*.xlsx", "*.xls"]  # file patterns to include
ORDER_COLS = ["session_horde_idx", "horde"]  # order within a session
ID_FROM_FILENAME_REGEX = r"(\\d+)"  # extract an integer from filename as player_id (fallback to ordinal)

# Columns we expect (soft-checked; the script is defensive)
COL_MODE = "mode"
COL_BLOCK = "mode_block_id"        # per-player block (version instance)
COL_HORDE = "horde"
COL_PRED = "predState"
COL_REAL = "RealState_log"
COL_SCORE = "score"
COL_REWARD = "reward"

# Define las rutas como variables
INPUT_DIR = r"D:\TFM\NeuroDoomPy\sessions\00_masters"           # carpeta con todos los maestros (uno por jugador)
OUTPUT_DIR = r"D:\TFM\NeuroDoomPy\sessions\01_features"  # carpeta de salida


# -------------------------
# Helpers
# -------------------------
def extract_player_id_from_filename(stem: str, ordinal:int) -> int:
    """Try to extract a number from the filename; fallback to the ordinal index (starting at 1)."""
    m = re.search(ID_FROM_FILENAME_REGEX, stem)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    return ordinal

def load_all_masters(input_dir: str) -> pd.DataFrame:
    """Load and concatenate all per-player master files; add player_id from filename."""
    files: List[str] = []
    for pat in INPUT_PATTERNS:
        files.extend(glob.glob(os.path.join(input_dir, pat)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No input files found in {input_dir} matching {INPUT_PATTERNS}")
    
    frames = []
    for i, path in enumerate(files, start=1):
        try:
            df = pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"Error reading {path}: {e}")
        stem = os.path.splitext(os.path.basename(path))[0]
        player_id = extract_player_id_from_filename(stem, i)
        df = df.copy()
        df["player_id"] = player_id
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out

def normalize_states(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized columns for predicted/real states, mapping numeric codes via STATE_MAP."""
    d = df.copy()
    if COL_REAL in d.columns:
        def map_real(x):
            try:
                if pd.isna(x): return None
                xi = int(float(x))
                return STATE_MAP.get(xi, str(x))
            except:
                return str(x).strip().lower()
        d["_real_state"] = d[COL_REAL].apply(map_real)
        d["_is_flow"] = (d["_real_state"] == "flow").astype("float")
    else:
        d["_real_state"] = None
        d["_is_flow"] = math.nan
    if COL_PRED in d.columns:
        def map_pred(x):
            try:
                if pd.isna(x): return None
                xi = int(float(x))
                return STATE_MAP.get(xi, str(x))
            except:
                return str(x).strip().lower()
        d["_pred_state"] = d[COL_PRED].apply(map_pred)
    else:
        d["_pred_state"] = None
    return d

def max_consecutive_flow(series_states: pd.Series) -> int:
    """Compute maximum consecutive 'flow' runs in a sequence of state labels."""
    run = 0
    best = 0
    for val in series_states:
        if val == "flow":
            run += 1
            if run > best: best = run
        else:
            run = 0
    return best

def choose_last_score(df_session: pd.DataFrame) -> Optional[float]:
    """Pick session final score as the last non-null by ORDER_COLS (fallback to max)."""
    cols = [c for c in ORDER_COLS if c in df_session.columns]
    g = df_session.sort_values(cols) if cols else df_session
    s = g[COL_SCORE] if COL_SCORE in g.columns else pd.Series(dtype=float)
    s = s.dropna()
    if not s.empty:
        return float(s.iloc[-1])
    # fallback
    if COL_SCORE in df_session.columns and df_session[COL_SCORE].notna().any():
        return float(df_session[COL_SCORE].max())
    return None

# -------------------------
# Pipeline
# -------------------------
def build_features_horda(df_raw: pd.DataFrame) -> pd.DataFrame:
    d = normalize_states(df_raw)
    # Ensure key columns exist
    for c in [COL_MODE, COL_BLOCK, COL_HORDE]:
        if c not in d.columns:
            d[c] = None
    # Keep all original columns + normalized
    return d

def build_features_sesion(df_horda: pd.DataFrame) -> pd.DataFrame:
    d = df_horda.copy()
    keys = ["player_id", COL_MODE]
    for k in keys:
        if k not in d.columns:
            d[k] = None
    # Aggregate per (player_id, mode)
    groups = d.groupby(keys, dropna=False)
    rows = []
    for (pid, mode), g in groups:
        row = {"player_id": pid, "mode": mode}
        # Flow stats
        if "_is_flow" in g.columns and g["_is_flow"].notna().any():
            row["flow_share_session"] = float(g["_is_flow"].mean())
        else:
            row["flow_share_session"] = None
        if "_real_state" in g.columns:
            # Order within session for run computation
            cols = [c for c in ORDER_COLS if c in g.columns]
            gg = g.sort_values(cols) if cols else g
            row["max_consec_flow_hordes"] = int(max_consecutive_flow(gg["_real_state"].tolist()))
        else:
            row["max_consec_flow_hordes"] = None
        # Reward/Score
        if COL_REWARD in g.columns and g[COL_REWARD].notna().any():
            row["reward_mean"] = float(g[COL_REWARD].mean())
        else:
            row["reward_mean"] = None
        row["score_final"] = choose_last_score(g)
        # Hordas contadas
        row["n_hordas"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)

def run_pipeline(input_dir: str, output_dir: str) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    df_all = load_all_masters(input_dir)
    df_horda = build_features_horda(df_all)
    df_sesion = build_features_sesion(df_horda)

    path_horda = os.path.join(output_dir, "features_horda.xlsx")
    path_sesion = os.path.join(output_dir, "features_sesion.xlsx")

    # Guardar en Excel (necesita openpyxl instalado)
    df_horda.to_excel(path_horda, index=False, engine="openpyxl")
    df_sesion.to_excel(path_sesion, index=False, engine="openpyxl")

    return path_horda, path_sesion


if __name__ == "__main__":

    path_horda, path_sesion  = run_pipeline(INPUT_DIR, OUTPUT_DIR)
    print("Wrote:", path_horda)
    print("Wrote:", path_sesion)
