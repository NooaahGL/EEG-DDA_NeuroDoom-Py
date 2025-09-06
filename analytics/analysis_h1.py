# -*- coding: utf-8 -*-
import os, re, glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)

# ----------------- Configuración de etiquetas -----------------
LABEL_MAP_INT2STR = {0: "Flow", 1: "Stress", 2: "Bored", 3: "Relax"}
# Orden canónico de reporte (para tablas/figuras):
LABEL_ORDER_CANON = ["Flow", "Relax", "Bored", "Stress"]

# ----------------- Utilidades -----------------
def _read_master_sheet(xlsx_path: str, sheet_name: str = "master") -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    df["source_file"] = os.path.basename(xlsx_path)
    return df

def _infer_player_id_from_filename(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"_([A-Za-z0-9]+)$", stem)
    return m.group(1) if m else stem

def _ensure_player_id(df: pd.DataFrame) -> pd.DataFrame:
    if "player_id" not in df.columns:
        df["player_id"] = df["source_file"].apply(_infer_player_id_from_filename)
    return df

def _string_normalize(s: str) -> str:
    t = str(s).strip().lower()
    t = (t.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u"))
    synonyms = {
        "flow": "Flow",
        "relax": "Relax", "relajacion": "Relax", "relajación": "Relax",
        "bored": "Bored", "aburrimiento": "Bored", "aburrido": "Bored",
        "stress": "Stress", "estres": "Stress", "estrés": "Stress",
    }
    return synonyms.get(t, t.title())

def _to_label_series(col: pd.Series) -> pd.Series:
    """Mapea 0..3 o texto a {'Flow','Stress','Bored','Relax'}; valores inválidos -> NaN."""
    allowed = {"Flow","Stress","Bored","Relax"}
    if pd.api.types.is_numeric_dtype(col):
        colnum = pd.to_numeric(col, errors="coerce").astype("Int64")
        mapped = colnum.map(LABEL_MAP_INT2STR)
        return mapped.where(mapped.isin(allowed))
    else:
        def norm_map(x):
            t = str(x).strip().lower()
            t = (t.replace("á","a").replace("é","e").replace("í","i")
                   .replace("ó","o").replace("ú","u"))
            synonyms = {
                "flow":"Flow",
                "relax":"Relax","relajacion":"Relax","relajación":"Relax",
                "bored":"Bored","aburrimiento":"Bored","aburrido":"Bored",
                "stress":"Stress","estres":"Stress","estrés":"Stress",
            }
            lab = synonyms.get(t, None)
            return lab if lab in allowed else None
        mapped = col.apply(norm_map)
        return mapped.astype("string")


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Filtros pedidos
    if "mode" in df.columns:
        df = df[df["mode"].astype(str) != "GameOver"]
    # La columna puede llamarse 'horde' o 'horde_idx' o similar
    for hcol in ["horde", "horde_idx", "Horde", "HordeIdx"]:
        if hcol in df.columns:
            df = df[df[hcol] != 1]
            break
    # reward en blanco -> fuera
    if "reward" not in df.columns:
        raise ValueError("Falta la columna 'reward'.")
    # convierte '' a NaN y luego filtra
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df = df[~df["reward"].isna()]
    return df

# ----------------- Métricas y tests -----------------
def _acc(y_true, y_pred): return float(accuracy_score(y_true, y_pred))
def _macro_f1(y_true, y_pred): return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
def _bal_acc(y_true, y_pred): return float(balanced_accuracy_score(y_true, y_pred))

def _cluster_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_boot=2000, random_state=42) -> Tuple[float, Tuple[float,float]]:
    """Bootstrap por clusters (jugador): remuestrea jugadores completos."""
    rng = np.random.RandomState(random_state)
    uniq = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        samp_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp_groups])
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    return float(np.mean(vals)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def _perm_pvalue_blocked(y_true, y_pred, groups, metric_fn, n_perm=10000, random_state=42) -> float:
    """Permuta y_pred dentro de cada jugador. p-valor de cola derecha."""
    rng = np.random.RandomState(random_state)
    obs = metric_fn(y_true, y_pred)
    uniq = np.unique(groups)
    count = 1  # corrección +1
    for _ in range(n_perm):
        y_perm = y_pred.copy()
        for g in uniq:
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                y_perm[idx] = rng.permutation(y_perm[idx])
        if metric_fn(y_true, y_perm) >= obs:
            count += 1
    return count / (n_perm + 1)

def _per_class_metrics(y_true, y_pred, labels: List[str]) -> pd.DataFrame:
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, lab in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        TPR = R[i]
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        BA = 0.5 * (TPR + TNR)
        rows.append({"Emoción": lab, "Accuracy": TPR, "Macro-F1": F1[i], "Balanced Accuracy": BA})
    return pd.DataFrame(rows)

def _bootstrap_per_class_cluster(y_true, y_pred, groups, labels: List[str], n_boot=2000, random_state=42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    uniq = np.unique(groups)
    store = {lab: {"Accuracy": [], "Macro-F1": [], "Balanced Accuracy": []} for lab in labels}
    for _ in range(n_boot):
        samp_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp_groups])
        dfc = _per_class_metrics(y_true[idx], y_pred[idx], labels)
        for _, row in dfc.iterrows():
            lab = row["Emoción"]
            for m in ["Accuracy", "Macro-F1", "Balanced Accuracy"]:
                store[lab][m].append(row[m])
    rows = []
    for lab in labels:
        for m in ["Accuracy", "Macro-F1", "Balanced Accuracy"]:
            vals = np.array(store[lab][m])
            rows.append({"Emoción": lab, "Métrica": m,
                         "IC95%": (float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5)))})
    return pd.DataFrame(rows)

def _perm_per_class_blocked(y_true, y_pred, groups, labels: List[str], metric_name: str, n_perm=10000, random_state=42) -> Dict[str, float]:
    rng = np.random.RandomState(random_state)
    uniq = np.unique(groups)
    obs_df = _per_class_metrics(y_true, y_pred, labels).set_index("Emoción")
    obs_vals = {lab: obs_df.loc[lab, metric_name] for lab in labels}
    counts = {lab: 1 for lab in labels}
    for _ in range(n_perm):
        y_perm = y_pred.copy()
        for g in uniq:
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                y_perm[idx] = rng.permutation(y_perm[idx])
        per_df = _per_class_metrics(y_true, y_perm, labels).set_index("Emoción")
        for lab in labels:
            if per_df.loc[lab, metric_name] >= obs_vals[lab]:
                counts[lab] += 1
    tot = n_perm + 1
    return {lab: counts[lab] / tot for lab in labels}

# ----------------- Orquestador -----------------
def evaluate_from_excels(
    input_dir: str,
    pattern: str = "master_hordas_*.xlsx",
    sheet_name: str = "master",
    n_boot: int = 2000,
    n_perm: int = 10000,
    random_state: int = 42,
    save_prefix: Optional[str] = None,
) -> Dict:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No se encontraron ficheros con patrón {pattern} en {input_dir}")

    frames = []
    for p in paths:
        try:
            frames.append(_read_master_sheet(p, sheet_name=sheet_name))
        except Exception as e:
            print(f"[WARN] No pude leer {p}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df = _ensure_player_id(df)
    df = _filter_df(df)

    need = ["RealState", "predState", "reward", "player_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas necesarias: {miss}")

    # Mapear etiquetas
    y_true_series = _to_label_series(df["RealState"])
    y_pred_series = _to_label_series(df["predState"])
    reward_num = pd.to_numeric(df["reward"], errors="coerce")

    # Máscaras
    mask_r1 = reward_num == 1                   # acierto
    mask_rne = (reward_num != 1) & (~reward_num.isna())  # no-acierto válido

    # Filas válidas:
    # - Siempre mantenemos reward==1 (forzaremos igualdad)
    # - Para reward!=1, solo si predState es válido (no NaN)
    mask_valid_pred = mask_rne & (~y_pred_series.isna())
    mask_keep = mask_r1 | mask_valid_pred

    # Aplicamos selección
    y_true_series = y_true_series[mask_keep]
    y_pred_series = y_pred_series[mask_keep]
    reward_num    = reward_num[mask_keep]
    groups = _ensure_player_id(df.loc[mask_keep]).loc[mask_keep, "player_id"].astype(str).values

    # Forzar la regla: si reward==1, entonces y_pred = y_true
    y_true = y_true_series.astype(str).values
    y_pred = y_pred_series.astype("string").values
    force_n = int(mask_r1[mask_keep].sum())
    y_pred[mask_r1[mask_keep].values] = y_true[mask_r1[mask_keep].values]

    # Reporte informativo (opcional)
    print(f"[INFO] Filas usadas: {len(y_true)} | Forzadas por reward=1: {force_n} | "
        f"Descartadas por reward!=1 sin predState válido: {int((mask_rne & y_pred_series.isna()).sum())}")

    # Orden de reporte (solo las presentes)
    present = list(pd.unique(y_true))
    labels = [c for c in LABEL_ORDER_CANON if c in present] + [c for c in present if c not in LABEL_ORDER_CANON]

    # Chequeo de coherencia: reward = 1 implica acierto

    # --------- Métricas del modelo ----------
    acc = _acc(y_true, y_pred)
    mF1 = _macro_f1(y_true, y_pred)
    bAcc = _bal_acc(y_true, y_pred)

    acc_mean, acc_ci = _cluster_bootstrap_ci(y_true, y_pred, groups, _acc, n_boot, random_state)
    f1_mean,  f1_ci  = _cluster_bootstrap_ci(y_true, y_pred, groups, _macro_f1, n_boot, random_state)
    ba_mean,  ba_ci  = _cluster_bootstrap_ci(y_true, y_pred, groups, _bal_acc, n_boot, random_state)

    # --------- Baselines ----------
    # 1) Mayoritaria global
    maj = pd.Series(y_true).value_counts().idxmax()
    y_maj = np.array([maj] * len(y_true), dtype=object)
    # 2) Mayoritaria por jugador (más fuerte)
    y_maj_by_player = np.empty_like(y_true, dtype=object)
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        mj = pd.Series(y_true[gi]).value_counts().idxmax()
        y_maj_by_player[gi] = mj
    # 3) Azar uniforme
    rng = np.random.RandomState(random_state)
    y_uni = rng.choice(labels, size=len(y_true), replace=True)

    def all_metrics(yhat):
        return _acc(y_true, yhat), _macro_f1(y_true, yhat), _bal_acc(y_true, yhat)

    acc_maj_g, f1_maj_g, ba_maj_g = all_metrics(y_maj)
    acc_maj_p, f1_maj_p, ba_maj_p = all_metrics(y_maj_by_player)
    acc_uni,   f1_uni,   ba_uni   = all_metrics(y_uni)

    # --------- Permutacionales bloqueados ----------
    p_acc = _perm_pvalue_blocked(y_true, y_pred, groups, _acc, n_perm, random_state)
    p_f1  = _perm_pvalue_blocked(y_true, y_pred, groups, _macro_f1, n_perm, random_state)
    p_ba  = _perm_pvalue_blocked(y_true, y_pred, groups, _bal_acc, n_perm, random_state)

    # --------- Confusiones ----------
    cm_abs  = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_abs_df  = pd.DataFrame(cm_abs, index=[f"T:{l}" for l in labels], columns=[f"P:{l}" for l in labels])
    cm_norm_df = pd.DataFrame(cm_norm, index=[f"T:{l}" for l in labels], columns=[f"P:{l}" for l in labels])

    # --------- Tabla general ----------
    general = pd.DataFrame([
        {"Métrica": "Accuracy",
         "Valor": acc, "IC95%": acc_ci,
         "Baseline mayoritaria (global)": acc_maj_g,
         "Baseline mayoritaria (por jugador)": acc_maj_p,
         "Baseline azar uniforme": acc_uni,
         "p-valor permutacional": p_acc},
        {"Métrica": "Macro-F1",
         "Valor": mF1, "IC95%": f1_ci,
         "Baseline mayoritaria (global)": f1_maj_g,
         "Baseline mayoritaria (por jugador)": f1_maj_p,
         "Baseline azar uniforme": f1_uni,
         "p-valor permutacional": p_f1},
        {"Métrica": "Balanced Accuracy",
         "Valor": bAcc, "IC95%": ba_ci,
         "Baseline mayoritaria (global)": ba_maj_g,
         "Baseline mayoritaria (por jugador)": ba_maj_p,
         "Baseline azar uniforme": ba_uni,
         "p-valor permutacional": p_ba},
    ])

    # --------- Por emoción ----------
    per_now  = _per_class_metrics(y_true, y_pred, labels)               # puntuales
    per_boot = _bootstrap_per_class_cluster(y_true, y_pred, groups, labels, n_boot, random_state)
    p_acc_c  = _perm_per_class_blocked(y_true, y_pred, groups, labels, "Accuracy", n_perm, random_state)
    p_f1_c   = _perm_per_class_blocked(y_true, y_pred, groups, labels, "Macro-F1", n_perm, random_state)
    p_ba_c   = _perm_per_class_blocked(y_true, y_pred, groups, labels, "Balanced Accuracy", n_perm, random_state)

    # baselines por emoción (1-vs-rest)
    def per_cls_on(yhat):
        return _per_class_metrics(y_true, yhat, labels).set_index("Emoción")
    per_maj_g = per_cls_on(y_maj)
    per_uni   = per_cls_on(y_uni)

    rows = []
    for lab in labels:
        ci_acc = per_boot[(per_boot["Emoción"]==lab) & (per_boot["Métrica"]=="Accuracy")]["IC95%"].values[0]
        ci_f1  = per_boot[(per_boot["Emoción"]==lab) & (per_boot["Métrica"]=="Macro-F1")]["IC95%"].values[0]
        ci_ba  = per_boot[(per_boot["Emoción"]==lab) & (per_boot["Métrica"]=="Balanced Accuracy")]["IC95%"].values[0]
        vals   = per_now[per_now["Emoción"]==lab].iloc[0]
        rows += [
            {"Emoción": lab, "Métrica":"Accuracy", "Valor":vals["Accuracy"],
             "IC95%":ci_acc,
             "Baseline mayoritaria (global)": per_maj_g.loc[lab,"Accuracy"],
             "Baseline azar uniforme": per_uni.loc[lab,"Accuracy"],
             "p-valor permutacional": p_acc_c[lab]},
            {"Emoción": lab, "Métrica":"Macro-F1", "Valor":vals["Macro-F1"],
             "IC95%":ci_f1,
             "Baseline mayoritaria (global)": per_maj_g.loc[lab,"Macro-F1"],
             "Baseline azar uniforme": per_uni.loc[lab,"Macro-F1"],
             "p-valor permutacional": p_f1_c[lab]},
            {"Emoción": lab, "Métrica":"Balanced Accuracy", "Valor":vals["Balanced Accuracy"],
             "IC95%":ci_ba,
             "Baseline mayoritaria (global)": per_maj_g.loc[lab,"Balanced Accuracy"],
             "Baseline azar uniforme": per_uni.loc[lab,"Balanced Accuracy"],
             "p-valor permutacional": p_ba_c[lab]},
        ]
    per_class_table = pd.DataFrame(rows)

    if save_prefix:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        general.to_csv(save_prefix + "_tabla_general.csv", index=False)
        per_class_table.to_csv(save_prefix + "_tabla_por_emocion.csv", index=False)
        cm_abs_df.to_csv(save_prefix + "_cm_abs.csv")
        cm_norm_df.to_csv(save_prefix + "_cm_norm.csv")

    return {
        "files_used": paths,
        "labels_order": labels,
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics_general": general,
        "metrics_per_class": per_class_table,
        "cm_absolute": cm_abs_df,
        "cm_normalized": cm_norm_df,
    }

# ----------------- Ejemplo de uso -----------------
if __name__ == "__main__":
    INPUT_DIR = r"D:\TFM\NeuroDoomPy\sessions\00_masters"
    out_prefix = os.path.join(INPUT_DIR, "H1_EVAL_predState")

    R = evaluate_from_excels(
        INPUT_DIR,
        pattern="master_hordas_*.xlsx",
        sheet_name="master",
        n_boot=2000,
        n_perm=10000,
        random_state=42,
        save_prefix=out_prefix,
    )

    print("\n== Tabla general ==")
    print(R["metrics_general"])
    print("\n== Tabla por emoción ==")
    print(R["metrics_per_class"])
    print("\n== Matriz de confusión normalizada ==")
    print(R["cm_normalized"].round(3))
