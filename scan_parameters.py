from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from mypkg.hubbard import HubbardParams, build_basis, build_hamiltonian


def run_scan() -> None:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    u_values = np.linspace(0.0, 8.0, 17)
    v_values = np.linspace(0.0, 4.0, 17)

    basis = build_basis(n_sites=4, n_up=1, n_down=1)
    rows: list[tuple[float, float, float]] = []

    print(">>> Démarrage du balayage optimisé (eigsh) des paramètres U et V")
    for U in u_values:
        for V in v_values:
            params = HubbardParams(t=1.0, U=float(U), V=float(V), lx=2, ly=2, periodic=False)
            H = build_hamiltonian(basis=basis, params=params)
            
            # Extraction efficace de l'état fondamental pour matrices creuses
            evals, _ = eigsh(H, k=1, which="SA", tol=1e-10, maxiter=5000)
            rows.append((float(U), float(V), float(evals[0])))

    csv_path = out_dir / "uv_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["U", "V", "ground_energy"])
        writer.writerows(rows)

    print(f"[OK] Balayage terminé. Données écrites dans {csv_path}")


if __name__ == "__main__":
    run_scan()
