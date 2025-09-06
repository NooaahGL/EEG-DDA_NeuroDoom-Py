import pandas as pd
import numpy as np

# ==== RUTAS ====
UNITY_EVENTS = r"D:/TFM/NeuroDoomPy/sessions/Jorge_01/unity_events.csv"
EEG_CSV      = r"D:/TFM/NeuroDoomPy/sessions/Jorge_01/data.csv"

# ==== 1) CARGAR UNITY EVENTS (sesiones y hordas) ====
ev = pd.read_csv(UNITY_EVENTS, engine="python")
ev.columns = [c.strip().lower() for c in ev.columns]

# columnas esperadas: mode,horde,timestamp,(opcional) score
for col in ("mode","horde","timestamp"):
    if col not in ev.columns:
        raise ValueError(f"Falta la columna '{col}' en unity_events.csv")

if "score" not in ev.columns:
    ev["score"] = np.nan

# tipos y orden
ev["mode"] = ev["mode"].astype(str)
ev["horde"] = pd.to_numeric(ev["horde"], errors="coerce").astype("Int64")
ev["timestamp"] = pd.to_numeric(ev["timestamp"], errors="coerce")
ev["score"] = pd.to_numeric(ev["score"], errors="coerce")
ev = ev.sort_values("timestamp").reset_index(drop=True)

# marcar gameover y construir session_id
is_go = ev["mode"].str.lower().eq("gameover")
ev["session_id"] = is_go.shift(fill_value=False).cumsum().astype(int)

# (opcional) quita duplicados exactos por bug
ev = ev.drop_duplicates(subset=["session_id","horde","timestamp","mode"], keep="last")

# separa inicios de horda y gameovers
starts = ev[~is_go].copy().sort_values(["session_id","timestamp","horde"])
go_ts  = ev[is_go].groupby("session_id")["timestamp"].min().rename("gameover_ts")
go_sc  = ev[is_go].groupby("session_id")["score"].max().rename("gameover_score")

# ==== 2) HORDAS: calcular fin de cada horda y duración ====
hordes = (
    starts
    .groupby("session_id", group_keys=False)
    .apply(lambda df: df.assign(next_ts=df["timestamp"].shift(-1),
                                next_score=df["score"].shift(-1)))
    .reset_index(drop=True)
    .merge(go_ts, on="session_id", how="left")
    .merge(go_sc, on="session_id", how="left")
)

# fin de horda = siguiente inicio; si no hay, GameOver
hordes["end_ts"] = hordes["next_ts"].fillna(hordes["gameover_ts"])
hordes["horde_duration_s"] = (hordes["end_ts"] - hordes["timestamp"]).clip(lower=0)

# puntuación ganada por horda (si hay score)
hordes["end_score"]   = hordes["next_score"].fillna(hordes["gameover_score"])
hordes["score_gained"] = hordes["end_score"] - hordes["score"]

# ==== 3) CARGAR EEG ====
eeg = pd.read_csv(EEG_CSV, engine="python", usecols=lambda c: c != "raw_data")
eeg.columns = [c.strip() for c in eeg.columns]
# renombrar y limpiar
if "time" in eeg.columns:
    eeg = eeg.rename(columns={"time":"timestamp"})
eeg["timestamp"]  = pd.to_numeric(eeg["timestamp"], errors="coerce")
eeg["attention"]  = pd.to_numeric(eeg.get("attention"),  errors="coerce")
eeg["meditation"] = pd.to_numeric(eeg.get("meditation"), errors="coerce")

# filtra sólo datos de headset (no blinks)
if "data_type" in eeg.columns:
    eeg["data_type"] = eeg["data_type"].astype(str).str.strip().str.strip('"').str.lower()
    eeg = eeg[eeg["data_type"].eq("headset_data")]

eeg = eeg.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# ==== 4) PROMEDIOS EEG POR HORDA (intervalo [start, end)) ====
# Para evitar problemas con hordas de duración 0 (bins duplicados), calculamos por máscara fila a fila.
def avg_eeg_in_window(ts0, ts1, chunk):
    if pd.isna(ts0) or pd.isna(ts1):
        return pd.Series({"att_mean": np.nan, "med_mean": np.nan, "n_samples": 0})
    m = (chunk["timestamp"] >= ts0) & (chunk["timestamp"] < ts1)
    if not m.any():
        return pd.Series({"att_mean": np.nan, "med_mean": np.nan, "n_samples": 0})
    sub = chunk.loc[m]
    return pd.Series({
        "att_mean":  sub["attention"].mean(),
        "med_mean":  sub["meditation"].mean(),
        "n_samples": int(m.sum())
    })

# Pre-computar límites de sesión para recortar EEG por sesión (eficiente)
sess_bounds = (
    starts.groupby("session_id")["timestamp"].min().rename("session_start_ts")
    .to_frame().merge(go_ts, left_index=True, right_index=True, how="left")
    .reset_index()
)

# contenedores de resultados por horda
att_means, med_means, n_samps = [], [], []

for sid, df_h in hordes.groupby("session_id"):
    # EEG de la sesión (entre primer inicio y gameover)
    srow = sess_bounds.loc[sess_bounds["session_id"]==sid].iloc[0]
    ts_s, ts_e = srow["session_start_ts"], srow["gameover_ts"]
    eeg_sess = eeg[(eeg["timestamp"] >= ts_s) & (eeg["timestamp"] < ts_e)] if pd.notna(ts_e) else eeg[eeg["timestamp"] >= ts_s]

    # calcula para cada horda del sid
    tmp = df_h.apply(lambda r: avg_eeg_in_window(r["timestamp"], r["end_ts"], eeg_sess), axis=1)
    att_means.extend(tmp["att_mean"].tolist())
    med_means.extend(tmp["med_mean"].tolist())
    n_samps.extend(tmp["n_samples"].tolist())

hordes["att_mean"]  = att_means
hordes["med_mean"]  = med_means
hordes["n_eeg_samples"] = n_samps

horde_summary = hordes[[
    "session_id","mode","horde","timestamp","end_ts","horde_duration_s",
    "att_mean","med_mean","n_eeg_samples","score","end_score","score_gained"
]]

# ==== 5) RESUMEN POR SESIÓN ====
# tiempo total
sess = sess_bounds.copy()
sess["session_total_time_s"] = (sess["gameover_ts"] - sess["session_start_ts"]).clip(lower=0)

# nº de hordas y tiempo medio por horda (sobre las hordas con duración > 0 opcionalmente)
g = horde_summary.groupby("session_id", as_index=False)
session_means = g.agg(
    n_hordes=("horde","count"),
    mean_horde_duration_s=("horde_duration_s", "mean")
)

# promedios EEG por sesión (usando todas las muestras EEG dentro del intervalo de sesión)
def avg_eeg_session(row):
    ts0, ts1 = row["session_start_ts"], row["gameover_ts"]
    m = (eeg["timestamp"] >= ts0) & (eeg["timestamp"] < ts1)
    sub = eeg.loc[m]
    return pd.Series({
        "session_att_mean": sub["attention"].mean() if len(sub) else np.nan,
        "session_med_mean": sub["meditation"].mean() if len(sub) else np.nan,
        "session_eeg_samples": int(len(sub))
    })

sess_eeg = sess.apply(avg_eeg_session, axis=1)
session_summary = pd.concat([sess, session_means.set_index("session_id").reindex(sess["session_id"]).reset_index(drop=True), sess_eeg], axis=1)

# puntuación total por sesión (si existe score)
start_score = starts.groupby("session_id")["score"].first().rename("session_start_score")
session_summary = (
    session_summary
    .merge(start_score, on="session_id", how="left")
    .merge(go_sc.rename("gameover_score"), on="session_id", how="left")
)
session_summary["session_total_score"] = session_summary["gameover_score"] - session_summary["session_start_score"]

# ==== 6) OPCIONAL: MÉTRICAS GLOBALES (por curiosidad) ====
global_mean_horde_time = horde_summary["horde_duration_s"].mean()

# ==== 7) MOSTRAR RESULTADOS ====
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

print("\n=== HORDAS (duración y promedios EEG dentro de cada intervalo) ===")
print(horde_summary.head(20))

print("\n=== SESIONES (duración total y promedios EEG) ===")
print(session_summary[[
    "session_id","session_start_ts","gameover_ts","session_total_time_s",
    "n_hordes","mean_horde_duration_s",
    "session_att_mean","session_med_mean","session_eeg_samples",
    "session_start_score","gameover_score","session_total_score"
]])

# =========  8) GRÁFICAS: Atención y Meditación por horda, una figura por sesión  =========

import math
import matplotlib.pyplot as plt

def fmt_mmss(seconds):
    if pd.isna(seconds):
        return "N/A"
    seconds = float(seconds)
    m = int(seconds // 60)
    s = int(round(seconds - 60*m))
    return f"{m:02d}:{s:02d}"

# Ordena por seguridad
horde_summary = horde_summary.sort_values(["session_id","timestamp","horde"]).reset_index(drop=True)
session_summary = session_summary.sort_values("session_id").reset_index(drop=True)

for _, sess in session_summary.iterrows():
    sid = int(sess["session_id"])
    sub = horde_summary[horde_summary["session_id"] == sid].copy()

    if sub.empty:
        continue

    # Eje X = nº de horda (si hay NaN o 0, ordena por timestamp)
    if sub["horde"].isna().any():
        sub = sub.sort_values("timestamp")
        x = range(1, len(sub) + 1)
    else:
        x = sub["horde"]

    fig, ax = plt.subplots()
    ax.plot(x, sub["att_mean"], marker="o", label="Atención")
    ax.plot(x, sub["med_mean"], marker="s", label="Meditación")

    # Info complementaria en el título
    dur_txt = fmt_mmss(sess["session_total_time_s"])
    score_final = sess.get("gameover_score")
    score_txt = "N/A" if pd.isna(score_final) else f"{int(score_final)}"

    ax.set_title(f"Sesión {sid} — Duración: {dur_txt} — Score final: {score_txt}")
    ax.set_xlabel("Horda")
    ax.set_ylabel("EEG promedio")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()



