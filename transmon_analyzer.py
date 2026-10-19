from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


class TransmonFluxEngine:
    """Transmon en base de charge avec SQUID asymétrique et solveur tridiagonal."""

    def __init__(
        self,
        EC_GHz: float = 0.25,
        EJ_max_GHz: float = 20.0,
        ng: float = 0.0,
        charge_cutoff: int = 100,
        asymmetry: float = 0.0,
    ):
        self.EC_GHz = float(EC_GHz)
        self.EJ_max_GHz = float(EJ_max_GHz)
        self.ng = float(ng)
        self.charge_cutoff = int(charge_cutoff)
        self.asymmetry = float(asymmetry)

        self.n_values = np.arange(-self.charge_cutoff, self.charge_cutoff + 1, dtype=float)
        self.dimension = len(self.n_values)

    def josephson_energy(self, phi: float) -> float:
        d = self.asymmetry
        return self.EJ_max_GHz * np.sqrt(np.cos(np.pi * phi) ** 2 + d**2 * np.sin(np.pi * phi) ** 2)

    def solve(self, phi: float, n_levels: int = 4):
        if n_levels < 1:
            raise ValueError("n_levels doit être >= 1.")

        EJ = self.josephson_energy(phi)
        diagonal = 4.0 * self.EC_GHz * (self.n_values - self.ng) ** 2
        off_diagonal = np.full(self.dimension - 1, -0.5 * EJ, dtype=float)

        energies, states = eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, min(n_levels, self.dimension) - 1),
            check_finite=False,
        )
        return energies, states

    def charge_operator(self) -> np.ndarray:
        return np.diag(self.n_values.astype(np.complex128))

    def dipole_matrix(self, phi: float, n_levels: int = 4) -> np.ndarray:
        _, states = self.solve(phi, n_levels=n_levels)
        n_op = self.charge_operator()
        return states.conj().T @ n_op @ states


def robust_eigenstate_path(
    engine: TransmonFluxEngine,
    phi_values: np.ndarray,
    state_index: int = 0,
) -> np.ndarray:
    phi_values = np.asarray(phi_values, dtype=float)
    if phi_values.size == 0:
        raise ValueError("phi_values ne peut pas être vide.")

    states = []

    _, vecs0 = engine.solve(float(phi_values[0]), n_levels=max(state_index + 1, 4))
    prev = vecs0[:, state_index]
    norm = np.linalg.norm(prev)
    if norm <= 1e-14:
        raise ValueError("État initial de norme nulle.")
    prev = prev / norm
    states.append(prev)

    for phi in phi_values[1:]:
        _, vecs = engine.solve(float(phi), n_levels=max(state_index + 4, 4))
        overlaps = np.abs(vecs.conj().T @ prev)
        j = int(np.argmax(overlaps))
        state = vecs[:, j]
        norm = np.linalg.norm(state)
        if norm <= 1e-14:
            raise ValueError("État propre de norme nulle.")
        state = state / norm

        if np.vdot(prev, state).real < 0:
            state = -state

        states.append(state)
        prev = state

    return np.asarray(states, dtype=np.complex128)


def parallel_transport_gauge(
    states: np.ndarray,
    *,
    tolerance: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.complex128)
    norms = np.linalg.norm(states, axis=1)

    if np.any(norms <= tolerance):
        raise ValueError("Au moins un état possède une norme nulle.")

    gauged = states / norms[:, None]
    overlaps = []

    for index in range(1, len(gauged)):
        overlap = np.vdot(gauged[index - 1], gauged[index])
        magnitude = abs(overlap)

        if magnitude <= tolerance:
            raise ValueError(f"Recouvrement trop faible au point {index}.")

        overlaps.append(overlap)
        gauged[index] *= np.conj(overlap) / magnitude

    return gauged, np.asarray(overlaps, dtype=np.complex128)


def wilson_loop_u1(
    states: np.ndarray,
    *,
    tolerance: float = 1e-14,
) -> tuple[complex, float]:
    states = np.asarray(states, dtype=np.complex128)
    norms = np.linalg.norm(states, axis=1)

    if np.any(norms <= tolerance):
        raise ValueError("Présence d'un état de norme nulle dans le calcul de Wilson.")

    states = states / norms[:, None]
    wilson = 1.0 + 0.0j

    for k in range(len(states)):
        left = states[k]
        right = states[(k + 1) % len(states)]

        overlap = np.vdot(left, right)
        magnitude = abs(overlap)

        if magnitude <= tolerance:
            raise ValueError(f"Recouvrement insuffisant sur le lien {k}.")

        wilson *= overlap / magnitude

    phase = float(-np.angle(wilson))
    return wilson, phase


if __name__ == "__main__":
    engine = TransmonFluxEngine(
        EC_GHz=0.25,
        EJ_max_GHz=20.0,
        ng=0.0,
        charge_cutoff=100,
        asymmetry=0.0,
    )

    path = np.linspace(0.0, 1.0, 201, endpoint=True)
    states = robust_eigenstate_path(engine, path, state_index=0)
    gauged_states, local_links = parallel_transport_gauge(states)
    wilson, phase = wilson_loop_u1(states)

    local_overlaps = np.array([np.vdot(gauged_states[k], gauged_states[k + 1]) for k in range(len(gauged_states) - 1)])

    energies_path = np.array([engine.solve(float(phi), n_levels=4)[0] for phi in path])

    df_diag = pd.DataFrame(
        {
            "phi": path,
            "E0": energies_path[:, 0],
            "E1": energies_path[:, 1],
            "E2": energies_path[:, 2],
            "E3": energies_path[:, 3],
            "gap_E1_E0": energies_path[:, 1] - energies_path[:, 0],
            "overlap_real": np.r_[np.nan, np.real(local_links)],
            "overlap_imag": np.r_[np.nan, np.imag(local_links)],
            "overlap_abs": np.r_[np.nan, np.abs(local_links)],
            "gauged_link_real": np.r_[np.nan, np.real(local_overlaps)],
            "gauged_link_imag": np.r_[np.nan, np.imag(local_overlaps)],
            "gauged_link_abs": np.r_[np.nan, np.abs(local_overlaps)],
        }
    )

    dip_rows = []
    for phi in path:
        energies, states_sol = engine.solve(float(phi), n_levels=4)
        n_esys = states_sol.conj().T @ engine.charge_operator() @ states_sol

        row = {"phi": float(phi)}
        for i in range(4):
            row[f"E{i}"] = float(energies[i])

        row["n_01_abs"] = float(abs(n_esys[0, 1]))
        row["n_02_abs"] = float(abs(n_esys[0, 2]))
        row["n_03_abs"] = float(abs(n_esys[0, 3]))
        row["n_11_real"] = float(np.real(n_esys[1, 1]))
        dip_rows.append(row)

    df_dipole = pd.DataFrame(dip_rows)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    
    csv_path_diag = out_dir / "transmon_wilson_diagnostics.csv"
    csv_path_dipole = out_dir / "transmon_dipole_summary.csv"

    df_diag.to_csv(csv_path_diag, index=False)
    df_dipole.to_csv(csv_path_dipole, index=False)

    print("--- RÉSULTATS ANALYSE TRANSMON (COMPLET & DIPÔLE) ---")
    print(f"Phase géométrique (Wilson) : {phase:.12f} rad")
    print(f"Phase / pi : {phase / np.pi:.12f}")
    print(f"Module de Wilson |W| : {abs(wilson):.12f}")
    print("Partie imaginaire max (liens jaugés) :", np.max(np.abs(local_overlaps.imag)))
    print("Valeur réelle min (liens jaugés) :", np.min(local_overlaps.real))
    print(f"Diagnostics de Wilson écrits dans : {csv_path_diag}")
    print(f"Résumé du dipôle écrit dans : {csv_path_dipole}")
