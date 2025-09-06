import os
from utils_io import CONFIG, read_excel_safe
import build_datasets
import analysis_h1, analysis_h2, analysis_h3
from plots import confusion_matrix_plot, confusion_matrix_plot_normalized
from plots import metrics_ci_barplot, permutation_null_hist, per_player_metrics_plot
from plots import ensure_dir
import pandas as pd


# Toggles (activa/desactiva pasos)
RUN_BUILD = False # Unificar maestros y generar features_*.xlsx
RUN_H1 = True # Clasificación por horda
RUN_H2 = False # Comparación versiones (flow)
RUN_H3 = False # Cuestionarios (activa cuando tengas survey.xlsx)
MAKE_PLOTS = True
MAKE_PLOTS2 = False # sólo H1 (usa tu plots.py si lo tienes)


FIG_DIR = os.path.join(CONFIG.OUTPUT_DIR, "figures")


def main():
    # 0) Build datasets
    if RUN_BUILD:
        h_path, s_path = build_datasets.run_pipeline(CONFIG.INPUT_DIR, CONFIG.OUTPUT_DIR)
        print("Datasets escritos:", h_path, s_path)
    else:
        h_path, s_path = CONFIG.PATH_HORDA, CONFIG.PATH_SESION

    # 1) H1
    # 1) H1 (reward-only)
    if RUN_H1:

        # --- H1 (LOPO multinomial logistic): usar datos de analysis_h1 y graficar ---
        lopo = analysis_h1.run_multiclass_lopo_dir(
            CONFIG.INPUT_DIR,
            pattern="master_hordas_*.xlsx",
            sheet_name="master",
            n_boot=2000,
            n_perm=10000,
            random_state=42
        )

        # Matrices como DataFrame
        cm_abs_df  = lopo["cm_absolute"].copy()
        cm_norm_df = lopo["cm_normalized"].copy()

        # Quitar prefijos "T:" / "P:" para que los ejes salgan limpios en tu plotter
        cm_abs_df.index    = [s.replace("T:", "") for s in cm_abs_df.index]
        cm_abs_df.columns  = [s.replace("P:", "") for s in cm_abs_df.columns]
        cm_norm_df.index   = [s.replace("T:", "") for s in cm_norm_df.index]
        cm_norm_df.columns = [s.replace("P:", "") for s in cm_norm_df.columns]

        if MAKE_PLOTS:
            ensure_dir(FIG_DIR)
            abs_png  = os.path.join(FIG_DIR, "H1_cm_absolute.png")
            norm_png = os.path.join(FIG_DIR, "H1_cm_normalized.png")

            # Tus funciones NO aceptan title ni labels → solo df y out_path
            confusion_matrix_plot(cm_abs_df, out_path=abs_png)
            confusion_matrix_plot_normalized(cm_norm_df, out_path=norm_png)

            # (opcional) exportar también a SVG para el TFM
            confusion_matrix_plot(cm_abs_df, out_path=os.path.join(FIG_DIR, "H1_cm_absolute.svg"))
            confusion_matrix_plot_normalized(cm_norm_df, out_path=os.path.join(FIG_DIR, "H1_cm_normalized.svg"))

            print("Figuras guardadas:")
            print(" -", abs_png)
            print(" -", norm_png)


    # 2) H2
    if RUN_H2:
        h2 = analysis_h2.run_h2(
            CONFIG.INPUT_DIR,
            pattern="master_hordas_*.xlsx",
            sheet_name="master",
            export_dir=os.path.join(CONFIG.OUTPUT_DIR, "tables")  # opcional
        )

        print("==== H2 resumen ====")
        print("Jugadores:", h2["n_players"], "| Hordas:", h2["n_hordes"])
        print("\n-- Descriptivos por modo (hordas) --")
        print(h2["by_mode"])
        print("\n-- Media por jugador y modo --")
        print(h2["per_player_mode"])
        print("\n-- Tests por pares --")
        print(h2["pairwise_tests"])

        # ----- Modelo mixto (hordas) -----
        mixed = analysis_h2.run_h2_mixed(
            h2["df_hordas_filtered"],
            export_dir=os.path.join(CONFIG.OUTPUT_DIR, "tables")
        )
        print("\n-- ANOVA (LRT) efecto 'mode' --")
        print(mixed["anova_like"])
        print("\n-- Contrastes (Wald) --")
        print(mixed["pairwise"])

        # ----- Figuras -----
        if MAKE_PLOTS:
            ensure_dir(FIG_DIR)
            from plots import flow_boxplot_players, flow_spaghetti_players, flow_differences_barplot

            flow_boxplot_players(
                h2["per_player_mode"],
                out_path=os.path.join(FIG_DIR, "H2_flow_boxplot_players.png")
            )
            flow_spaghetti_players(
                h2["per_player_mode"],
                out_path=os.path.join(FIG_DIR, "H2_flow_spaghetti_players.png")
            )
            flow_differences_barplot(
                h2["per_player_mode"],
                out_path=os.path.join(FIG_DIR, "H2_flow_differences_barplot.png")
            )

            # (opcional) duplicado en SVG para la memoria
            flow_boxplot_players(
                h2["per_player_mode"],
                out_path=os.path.join(FIG_DIR, "H2_flow_boxplot_players.svg")
            )
            flow_spaghetti_players(
                h2["per_player_mode"],
                out_path=os.path.join(FIG_DIR, "H2_flow_spaghetti_players.svg")
            )
            flow_differences_barplot(
                h2["per_player_mode"],
                out_path=os.path.join(FIG_DIR, "H2_flow_differences_barplot.svg")
            )

 

    # 3) H3
    if RUN_H3:
        h3 = analysis_h3.run_h3(s_path)
        for k,v in h3.items():
            print("====", k, "====")
        print(v)


if __name__ == "__main__":
    main()