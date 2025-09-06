# analysis_h2.py
import os
import glob
import itertools
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from scipy.stats import norm




# =========================
# Utilidades internas
# =========================

CANONICAL_MODE = {
    "preconfigured": "preconfigured",
    "baseline": "preconfigured",
    "heuristic": "heuristic",
    "heuristictree": "heuristic",
    "tree": "heuristic",
    "ml": "ML",
    "bandit": "ML",
    "thompson": "ML",
    "bandit_thompson": "ML",
}


def _canon_mode(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    key = str(x).strip().lower()
    return CANONICAL_MODE.get(key, str(x).strip())

def _iqr(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

def _load_concat_excel(input_dir: str, pattern: str, sheet_name: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No se encontraron ficheros con patrón: {os.path.join(input_dir, pattern)}")
    frames = []
    for p in paths:
        try:
            df = pd.read_excel(p, sheet_name=sheet_name)
            df["__source_file"] = os.path.basename(p)
            frames.append(df)
        except Exception as e:
            warnings.warn(f"Error leyendo {p}: {e}")
    if not frames:
        raise RuntimeError("No se pudo leer ningún Excel válido.")
    return pd.concat(frames, ignore_index=True)

def _ensure_columns(df: pd.DataFrame, needed: List[str]):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas obligatorias: {missing}. "
                       f"Columnas disponibles: {list(df.columns)}")

def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    # Requisitos: columnas
    _ensure_columns(df, ["mode", "horde", "fFlow_log"])

    # Canonizar modo
    df = df.copy()
    df["mode"] = df["mode"].apply(_canon_mode)

    # Filtrar GameOver y horde=1 (calibración)
    df = df[df["mode"].str.lower() != "gameover"]
    df = df[df["horde"] != 1]

    # Mantener columnas clave si existen
    # player_id puede venir como 'player_id' o 'participant_id' o 'user_name'
    if "player_id" in df.columns:
        pid_col = "player_id"
    elif "participant_id" in df.columns:
        pid_col = "participant_id"
    elif "user_name" in df.columns:
        pid_col = "user_name"
    else:
        # Si no hay identificador, creamos uno por fichero (peor caso)
        pid_col = "player_id"
        df[pid_col] = df["__source_file"].str.replace(".xlsx", "", regex=False)

    # Asegurar tipos
    df["fFlow_log"] = pd.to_numeric(df["fFlow_log"], errors="coerce")
    df = df.dropna(subset=["mode", "fFlow_log"])

    # Normalizar a porcentaje [0,100] si viniera 0-1
    if df["fFlow_log"].dropna().between(0, 1).mean() > 0.9:
        df["fFlow_log"] = df["fFlow_log"] * 100.0

    # Selección final de columnas útiles
    keep_cols = [c for c in [pid_col, "mode", "horde", "fFlow_log", "__source_file"] if c in df.columns]
    df = df[keep_cols].rename(columns={pid_col: "player_id"}).reset_index(drop=True)

    # Filtrar a los 3 modos de interés solamente
    valid_modes = {"preconfigured", "heuristic", "ML"}
    df = df[df["mode"].isin(valid_modes)]
    return df

def _descriptives(df: pd.DataFrame) -> pd.DataFrame:
    by_mode = (
        df.groupby("mode")["fFlow_log"]
          .agg(N="count", mean=np.mean, std=np.std, median=np.median, IQR=_iqr,
               q25=lambda s: float(np.nanpercentile(s, 25)),
               q75=lambda s: float(np.nanpercentile(s, 75)))
          .reset_index()
          .sort_values("mode")
    )
    return by_mode

def _per_player_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    # Media por jugador y modo (agrega hordas)
    ppm = (
        df.groupby(["player_id", "mode"])["fFlow_log"]
          .agg(N_hordes="count", mean=np.mean, median=np.median)
          .reset_index()
    )
    return ppm

def _paired_players(df: pd.DataFrame, mode_a: str, mode_b: str) -> pd.DataFrame:
    # Jugadores que tienen ambos modos
    players_a = set(df[df["mode"] == mode_a]["player_id"].unique())
    players_b = set(df[df["mode"] == mode_b]["player_id"].unique())
    common = sorted(players_a & players_b)

    # Media por jugador y modo
    ppm = _per_player_descriptives(df)
    wide = (ppm.pivot(index="player_id", columns="mode", values="mean")
               .reindex(columns=["preconfigured", "heuristic", "ML"]))
    paired = wide.loc[common, [mode_a, mode_b]].dropna()
    paired = paired.rename(columns={mode_a: f"{mode_a}_mean", mode_b: f"{mode_b}_mean"})
    return paired.reset_index()

def _pairwise_test(df: pd.DataFrame, mode_a: str, mode_b: str) -> Dict[str, object]:
    """
    Estrategia:
      1) Intentar prueba PAREADA sobre medias por jugador cuando hay jugadores en común.
         - Si n_pares >= 5: Wilcoxon signed-rank (no paramétrico) y también t pareada como referencia.
         - Si 2 <= n_pares < 5: reportar t pareada + aviso por n pequeño.
      2) Si no hay jugadores en común, usar Mann–Whitney sobre observaciones por horda (no pareado).
    """
    # Pareado (medias por jugador)
    paired = _paired_players(df, mode_a, mode_b)
    n_pairs = int(paired.shape[0])

    result = {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "test_type": None,
        "n_pairs": n_pairs,
        "statistic": np.nan,
        "p_value": np.nan,
        "effect_size": np.nan,     # r de Wilcoxon o d de Cohen (según corresponda)
        "note": "",
    }

    if n_pairs >= 2:
        diff = paired[f"{mode_b}_mean"] - paired[f"{mode_a}_mean"]

        # Wilcoxon (no paramétrico, pareado)
        if n_pairs >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(paired[f"{mode_a}_mean"], paired[f"{mode_b}_mean"], zero_method="wilcox", alternative="two-sided")
                # Efecto r ≈ Z / sqrt(N); aproximamos Z desde p para dos colas
                # Nota: SciPy no da Z directamente; para N>=10 esto es mejor interpretado.
                result.update({
                    "test_type": "Wilcoxon paired",
                    "statistic": float(w_stat),
                    "p_value": float(w_p),
                    "effect_size": float(np.nan),  # dejamos vacío por robustez
                })
            except Exception as e:
                result["note"] += f"Wilcoxon no disponible ({e}). "

        # T pareada como referencia
        try:
            t_stat, t_p = stats.ttest_rel(paired[f"{mode_a}_mean"], paired[f"{mode_b}_mean"])
            # Cohen's d para medidas repetidas (d_av por Morris & DeShon, aprox simple):
            sd_diff = np.std(diff, ddof=1)
            d = np.nan
            if sd_diff > 0:
                d = np.nanmean(diff) / sd_diff
            result.setdefault("test_type", "t paired")  # si Wilcoxon falló
            # Si Wilcoxon ya estaba, añadimos nota
            if result["test_type"] != "t paired":
                result["note"] += "Incluye t pareada como referencia. "
            result.update({
                "statistic": float(t_stat) if np.isnan(result["statistic"]) else result["statistic"],
                "p_value": float(t_p) if np.isnan(result["p_value"]) else result["p_value"],
                "effect_size": float(d) if np.isnan(result["effect_size"]) else result["effect_size"],
            })
        except Exception as e:
            result["note"] += f"t pareada no disponible ({e}). "

        if n_pairs < 5:
            result["note"] += f"n_pairs={n_pairs}: potencia limitada; interpretar con cautela. "
        return result

    # No hay jugadores en común → no pareado sobre hordas
    a = df.loc[df["mode"] == mode_a, "fFlow_log"].dropna()
    b = df.loc[df["mode"] == mode_b, "fFlow_log"].dropna()

    if len(a) < 2 or len(b) < 2:
        result.update({
            "test_type": "insuficiente",
            "note": "Muestras demasiado pequeñas para contraste robusto."
        })
        return result

    try:
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Efecto r ≈ Z / sqrt(N). Aproximación desde U:
        # SciPy no expone Z; omitimos r por simplicidad robusta.
        result.update({
            "test_type": "Mann-Whitney U (unpaired hordes)",
            "statistic": float(u_stat),
            "p_value": float(u_p),
            "effect_size": float(np.nan),
            "note": "Sin emparejamiento por jugador.",
        })
    except Exception as e:
        result.update({
            "test_type": "error",
            "note": f"Error en Mann-Whitney: {e}"
        })
    return result

def _pairwise_matrix(df: pd.DataFrame) -> pd.DataFrame:
    modes = ["preconfigured", "heuristic", "ML"]
    results = []
    for a, b in itertools.combinations(modes, 2):
        results.append(_pairwise_test(df, a, b))
    return pd.DataFrame(results)

# =========================
# API pública del módulo
# =========================

def run_h2(input_dir: str,
           pattern: str = "master_hordas_*.xlsx",
           sheet_name: str = "master",
           export_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    Carga y filtra los datos y calcula:
      - descriptivos por modo (hordas como unidades)
      - medias por jugador y modo (agregación dentro de jugador)
      - tests por pares de modos, emparejando por jugador cuando sea posible
    """
    raw = _load_concat_excel(input_dir, pattern, sheet_name)
    df = _filter_df(raw)

    if df.empty:
        raise RuntimeError("Tras el filtrado no quedan datos para H2.")

    by_mode = _descriptives(df)
    per_player_mode = _per_player_descriptives(df)
    pairwise = _pairwise_matrix(df)

    out = {
        "by_mode": by_mode,
        "per_player_mode": per_player_mode,
        "pairwise_tests": pairwise,
        "n_players": pd.Series(df["player_id"].unique()).nunique(),
        "n_hordes": df.shape[0],
        "df_hordas_filtered": df,   # <--- AÑADIDO
    }

    # Export opcional
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        by_mode.to_csv(os.path.join(export_dir, "H2_by_mode.csv"), index=False)
        per_player_mode.to_csv(os.path.join(export_dir, "H2_per_player_mode.csv"), index=False)
        pairwise.to_csv(os.path.join(export_dir, "H2_pairwise_tests.csv"), index=False)

    return out

# analysis_h2.py (añadir al final)

def run_h2_mixed(df_hordas: pd.DataFrame, export_dir: str = None):
    """
    Modelo lineal mixto a nivel de horda:
        fFlow_log ~ C(mode) + (1 | player_id)
    Devuelve:
      - anova_like: LRT global del efecto 'mode' (full vs null, ML no REML)
      - pairwise: contrastes Wald entre modos (heuristic-preconfigured, ML-preconfigured, ML-heuristic)
      - model_full: objeto statsmodels del modelo completo
    """
    df = df_hordas.copy()
    # Categorías con baseline 'preconfigured'
    df["mode"] = pd.Categorical(df["mode"], categories=["preconfigured", "heuristic", "ML"], ordered=True)

    # Ajuste (ML para LRT)
    null_m = smf.mixedlm("fFlow_log ~ 1", df, groups=df["player_id"]).fit(reml=False)
    full_m = smf.mixedlm("fFlow_log ~ C(mode)", df, groups=df["player_id"]).fit(reml=False)

    # --- LRT global (ANOVA-like) ---
    lrt = 2 * (full_m.llf - null_m.llf)
    df_diff = full_m.df_modelwc - null_m.df_modelwc
    p_lrt = float(chi2.sf(lrt, df_diff))
    anova_like = {
        "effect": "mode",
        "statistic": float(lrt),
        "df": int(df_diff),
        "p_value": p_lrt
    }

    # --- Contrastes Wald por nombres de coeficientes (solo FE) ---
    # Esperados en fe_params: ['Intercept', 'C(mode)[T.heuristic]', 'C(mode)[T.ML]']
    fe_params = full_m.fe_params               # Series con nombres
    fe_names  = list(fe_params.index)
    cov_fe    = full_m.cov_params().loc[fe_names, fe_names]  # covarianza FE

    def wald_from_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Construye vector de pesos por nombre de parámetro y calcula estimación, SE, z y p (aprox normal).
        Cualquier nombre no presente se ignora (peso=0).
        """
        w = np.zeros(len(fe_names))
        for k, v in weights_dict.items():
            if k in fe_names:
                w[fe_names.index(k)] = v
        est = float(np.dot(w, fe_params.values))
        var = float(w @ cov_fe.values @ w.T)
        se  = float(np.sqrt(var)) if var > 0 else np.nan
        z   = est / se if (se is not None and se > 0) else np.nan
        p   = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        return {"estimate": est, "se": se, "z": z, "p_value": p}

    c1 = wald_from_weights({"C(mode)[T.heuristic]": 1.0})                         # heuristic - preconfigured
    c2 = wald_from_weights({"C(mode)[T.ML]": 1.0})                                # ML - preconfigured
    c3 = wald_from_weights({"C(mode)[T.ML]": 1.0, "C(mode)[T.heuristic]": -1.0})  # ML - heuristic

    pairwise = pd.DataFrame([
        {"contrast": "heuristic - preconfigured", **c1},
        {"contrast": "ML - preconfigured",        **c2},
        {"contrast": "ML - heuristic",            **c3},
    ])

    # Export opcional
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        with open(os.path.join(export_dir, "H2_mixed_full_summary.txt"), "w") as f:
            f.write(str(full_m.summary()))
        with open(os.path.join(export_dir, "H2_mixed_null_summary.txt"), "w") as f:
            f.write(str(null_m.summary()))
        pd.DataFrame([anova_like]).to_csv(os.path.join(export_dir, "H2_mixed_anova_like.csv"), index=False)
        pairwise.to_csv(os.path.join(export_dir, "H2_mixed_pairwise.csv"), index=False)

    return {
        "model_full": full_m,
        "anova_like": anova_like,
        "pairwise": pairwise
    }

