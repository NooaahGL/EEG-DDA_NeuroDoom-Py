import os
import pandas as pd


class CONFIG:
    # Carpetas principales
    INPUT_DIR = r"D:\TFM\NeuroDoomPy\sessions\00_masters"
    OUTPUT_DIR = r"D:\TFM\NeuroDoomPy\sessions\01_features"


    # Archivos derivados
    PATH_HORDA = os.path.join(OUTPUT_DIR, "features_horda.xlsx")
    PATH_SESION = os.path.join(OUTPUT_DIR, "features_sesion.xlsx")


    # CSV/Excel flags
    EXCEL_ENGINE = "openpyxl"


    # Columnas clave (ajusta si difiere)
    COL_MODE = "mode"
    COL_BLOCK = "mode_block_id"
    COL_HORDE = "horde"
    COL_PRED = "predState"
    COL_REAL = "RealState_log"
    COL_SCORE = "score"
    COL_REWARD = "reward"


    # Estados (enum de RealState_log)
    STATE_MAP = {0: "flow", 1: "stress", 2: "bored", 3: "relax"}




def read_excel_safe(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine=CONFIG.EXCEL_ENGINE)

def save_excel_safe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False, engine=CONFIG.EXCEL_ENGINE)

def check_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")