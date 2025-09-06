#!/usr/bin/env python3
"""
train_ts.py — Actualiza un bandido lineal bayesiano (Thompson Sampling)
======================================================================
* Lee logs.csv (frame, x0..x11, pred, real, hit)
* Para cada brazo k acumula:  A_k = Σ x x^T ,  b_k = Σ r x
* Calcula Σ_k = A_k^{-1}   y   μ_k = Σ_k b_k
* Guarda **A, b, Sigma, mu** en bandit.json (para Unity y para continuar entrenamiento)

Compatible con el nuevo **vector de contexto de 12 features** (CONTEXT_DIM = 10).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Generator, Tuple, List

import numpy as np

# --------------------------- CONFIG --------------------------------
CONTEXT_DIM = 10            # <── debe coincidir con EmotionTracker
N_ARMS      = 4             # Flow, Stress, Bored, Relax
PRIOR_VAR   = 1.0           # σ² para el prior gaussiano
EPS         = 1e-8          # regularización

# ------------------------- UTILIDADES ------------------------------

def safe_inv(A: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Devuelve A⁻¹ con jitter adaptativo; usa pinv si todo falla."""
    I = np.eye(A.shape[0])
    jitter = eps
    while jitter < 1e-2:
        try:
            return np.linalg.inv(A + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10
    print("⚠️  A singular; pinv con jitter =", jitter)
    return np.linalg.pinv(A + jitter * I)


def fresh_prior(dim: int = CONTEXT_DIM) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    eye = np.eye(dim) * PRIOR_VAR
    return [eye.copy() for _ in range(N_ARMS)], [np.zeros(dim) for _ in range(N_ARMS)]


def load_or_init(model_path: Path, dim: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not model_path.exists():
        print(f"↪️  {model_path} no existe → prior {dim}×{dim}")
        return fresh_prior(dim)

    raw = json.loads(model_path.read_text())
    if "A" not in raw or "b" not in raw:
        print("⚠️  JSON sin A/b → prior nuevo")
        return fresh_prior(dim)

    A_ls = [np.asarray(A) for A in raw["A"]]
    b_ls = [np.asarray(b) for b in raw["b"]]
    try:
        assert len(A_ls) == N_ARMS and len(b_ls) == N_ARMS
        assert all(A.shape == (dim, dim) for A in A_ls)
        assert all(b.shape == (dim,)     for b in b_ls)
    except AssertionError:
        print("⚠️  Dimensión en JSON ≠ logs → prior nuevo")
        return fresh_prior(dim)
    return A_ls, b_ls


def iterate_logs(csv_path: Path, dim: int) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < dim + 4:
                continue
            x = np.asarray(row[1:1+dim], dtype=float)
            pred   = int(row[1+dim])
            real   = int(row[2+dim])
            reward = float(row[3+dim])    # ⬅️
            yield x, pred, reward

# ------------------------- ENTRENAMIENTO ---------------------------

def train(csv_file: str, json_file: str, clear: bool = False) -> None:
    csv_path   = Path(csv_file)
    model_path = Path(json_file)
    if not csv_path.exists():
        print("❌ Log no encontrado:", csv_path)
        return

    dim = CONTEXT_DIM  # forzamos 12; ignoramos infer_dim para coherencia

    A_ls, b_ls = load_or_init(model_path, dim)

    updates = 0
    for x, arm, reward in iterate_logs(csv_path, dim):
        if 0 <= arm < N_ARMS:
            A_ls[arm] += np.outer(x, x)
            b_ls[arm] += reward * x
            updates += 1
    if updates == 0:
        print("No hay filas nuevas → nada que entrenar")
        return

    Sigma_ls, Mu_ls = [], []
    for A, b in zip(A_ls, b_ls):
        Sigma = safe_inv(A)
        mu    = Sigma @ b
        Sigma_ls.append(Sigma.tolist())
        Mu_ls.append(mu.tolist())

    tmp = model_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "A": [A.tolist() for A in A_ls],
        "b": [b.tolist() for b in b_ls],
        "Sigma": Sigma_ls,
        "mu": Mu_ls
    }, indent=2))
    tmp.replace(model_path)
    print(f"✓ {updates} filas procesadas — modelo actualizado → {model_path}")

    if clear:
        csv_path.write_text("")
        print(f"✔️  {csv_path} vaciado")

# ---------------------------- CLI ----------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena LinTS con contexto de 12 features")
    ap.add_argument("csv_log", help="logs.csv de Unity")
    ap.add_argument("json_out", help="bandit.json a actualizar")
    ap.add_argument("--clear-logs", action="store_true", help="Vacía el CSV tras entrenar")
    args = ap.parse_args()

    train(args.csv_log, args.json_out, clear=args.clear_logs)
