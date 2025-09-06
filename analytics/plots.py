import os
import pandas as pd
import matplotlib.pyplot as plt
from utils_io import CONFIG
import numpy as np


plt.rcParams.update({
"figure.autolayout": True,
"axes.grid": True,
})




def ensure_dir(path):
    os.makedirs(path, exist_ok=True)




def confusion_matrix_plot(cm: pd.DataFrame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            ax.text(j, i, cm.values[i,j], ha='center', va='center')
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.colorbar(im)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)




def spaghetti_by_player(df_sesion: pd.DataFrame, value_col: str, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    modes = ["Preconfigured","HeuristicTree","ML"]
    fig, ax = plt.subplots(figsize=(7,5))
    for pid, g in df_sesion.groupby("player_id"):
        g = g.set_index("mode").reindex(modes)
        ax.plot(modes, g[value_col], marker='o', alpha=0.6, label=f"P{pid}")
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} por versión (líneas = jugador)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_matrix_plot_normalized(cm: pd.DataFrame, out_path: str, normalize='true'):
    """
    Versión normalizada de la matriz de confusión.
    normalize='true' -> porcentajes por fila (condicionados a la clase real)
    """
    ensure_dir(os.path.dirname(out_path))
    cm_values = cm.values.astype(float)
    if normalize == 'true':
        row_sums = cm_values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm_values / row_sums
        title = "Matriz de confusión (normalizada por clase real)"
    elif normalize == 'pred':
        col_sums = cm_values.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        cm_norm = cm_values / col_sums
        title = "Matriz de confusión (normalizada por predicción)"
    else:
        cm_norm = cm_values / cm_values.sum()
        title = "Matriz de confusión (normalizada global)"
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns))); ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right"); ax.set_yticklabels(cm.index)
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%", ha='center', va='center')
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Proporción")
    fig.savefig(out_path, dpi=150); plt.close(fig)


def metrics_ci_barplot(ci_dict: dict, out_path: str, title: str = "Rendimiento con IC95%"):
    """
    Barras con IC95% para Macro-F1 y Balanced Accuracy.
    Espera el dict devuelto en res['bootstrap'] de analysis_h1.run_h1*_...
    """
    ensure_dir(os.path.dirname(out_path))
    metrics = ["Macro-F1", "Balanced Acc."]
    means = [ci_dict["macro_f1_mean"], ci_dict["bacc_mean"]]
    ci = [ci_dict["macro_f1_ci95"], ci_dict["bacc_ci95"]]
    yerr = np.array([[m - c[0], c[1] - m] for m, c in zip(means, ci)]).T

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(metrics, means, yerr=yerr, capsize=6)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title(title)
    for i, m in enumerate(means):
        ax.text(i, m + 0.03, f"{m:.3f}", ha="center")
    fig.savefig(out_path, dpi=150); plt.close(fig)


def permutation_null_hist(perm_scores: np.ndarray, true_score: float, out_path: str,
                          title: str = "Test permutacional (macro-F1)"):
    """
    Histograma de la distribución nula permutada con línea del score real.
    Pasa el array de permutaciones y el true_score (p.ej., res['permutation']['true_score']).
    """
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(perm_scores, bins=30, alpha=0.7)
    ax.axvline(true_score, linestyle="--", linewidth=2)
    ax.set_xlabel("Score (macro-F1)"); ax.set_ylabel("Frecuencia"); ax.set_title(title)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def per_player_metrics_plot(pred_df: pd.DataFrame, out_path: str):
    """
    Muestra Macro-F1 por jugador (útil para apéndice del TFM).
    Requiere un dataframe con columnas: ['player_id','_y_true','_y_pred'].
    """
    ensure_dir(os.path.dirname(out_path))
    from sklearn.metrics import f1_score
    rows = []
    for pid, g in pred_df.groupby("player_id"):
        rows.append({
            "player_id": pid,
            "macro_f1": f1_score(g["_y_true"], g["_y_pred"], average="macro", zero_division=0)
        })
    m = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(m["player_id"].astype(str), m["macro_f1"])
    ax.set_ylim(0,1); ax.set_xlabel("Jugador"); ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 por jugador (LPO)")
    for i, v in enumerate(m["macro_f1"].values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.savefig(out_path, dpi=150); plt.close(fig)

# ======== H2: PLOTS DE FLOW POR MODO ========

def _mode_display_name(m: str) -> str:
    # Mapea a etiquetas limpias para la memoria
    mapping = {"preconfigured": "Preconfigured",
               "heuristic": "HeuristicTree",
               "ML": "ML"}
    return mapping.get(str(m), str(m))

def flow_boxplot_players(per_player_df: pd.DataFrame, out_path: str):
    """
    Boxplot del %Flow por modo usando la MEDIA por jugador (input: h2['per_player_mode']).
    Espera columnas: ['player_id','mode','mean'].
    """
    ensure_dir(os.path.dirname(out_path))
    df = per_player_df.copy()
    df["mode_disp"] = df["mode"].apply(_mode_display_name)

    modes = ["Preconfigured","HeuristicTree","ML"]
    data = [df.loc[df["mode_disp"]==m, "mean"].values for m in modes]

    fig, ax = plt.subplots(figsize=(6,4))
    bp = ax.boxplot(data, labels=modes, showmeans=True)
    ax.set_ylabel("% Flow (media por jugador)")
    ax.set_title("Distribución de %Flow por modo (agregado por jugador)")
    # puntos individuales superpuestos
    for i, m in enumerate(modes, start=1):
        y = df.loc[df["mode_disp"]==m, "mean"].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.plot(x, y, marker='o', linestyle='', alpha=0.7)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def flow_spaghetti_players(per_player_df: pd.DataFrame, out_path: str):
    """
    Spaghetti por jugador conectando su %Flow medio en cada modo.
    """
    ensure_dir(os.path.dirname(out_path))
    df = per_player_df.copy()
    df["mode_disp"] = df["mode"].apply(_mode_display_name)
    order = ["Preconfigured","HeuristicTree","ML"]
    pivot = df.pivot(index="player_id", columns="mode_disp", values="mean").reindex(columns=order)

    fig, ax = plt.subplots(figsize=(7,5))
    for pid, row in pivot.iterrows():
        ax.plot(order, row.values, marker='o', alpha=0.7, label=str(pid))
    ax.set_ylabel("% Flow (media por jugador)")
    ax.set_title("Evolución del %Flow por jugador entre modos")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def flow_differences_barplot(per_player_df: pd.DataFrame, out_path: str):
    """
    Barras de diferencias (media ± IC95%) por jugador entre modos:
    heuristic-preconfigured, ML-preconfigured, ML-heuristic.
    """
    ensure_dir(os.path.dirname(out_path))
    df = per_player_df.copy()
    df["mode_disp"] = df["mode"].apply(_mode_display_name)
    pivot = df.pivot(index="player_id", columns="mode_disp", values="mean")

    diffs = {
        "Heuristic - Preconfigured": pivot["HeuristicTree"] - pivot["Preconfigured"],
        "ML - Preconfigured":        pivot["ML"] - pivot["Preconfigured"],
        "ML - Heuristic":            pivot["ML"] - pivot["HeuristicTree"]
    }
    labels = list(diffs.keys())
    vals = [v.dropna() for v in diffs.values()]
    means = [float(v.mean()) for v in vals]
    ses = [float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0 for v in vals]
    cis = [1.96*s for s in ses]

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(len(labels)), means, yerr=cis, capsize=6)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Δ % Flow (media por jugador)")
    ax.set_title("Diferencias entre modos (media ± IC95%)")
    for i, m in enumerate(means):
        ax.text(i, m + (0.02 if m>=0 else -0.06)*max(1, np.nanmax(np.abs(means))), f"{m:.1f}", ha="center")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
