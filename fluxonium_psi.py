#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM Ψ — Quantum-Geometric Configuration Toolkit
=======================================================

Purpose
-------
A single, robust research/experimental scaffold built around a fluxonium model.

PHYSICAL CORE
-------------
    Fluxonium -> Hamiltonian -> spectrum -> matrix elements
              -> coherence estimates -> geometric/information descriptors

FORMAL CONFIGURATION LAYER
---------------------------
    state -> fidelity -> metric -> configuration Ψ
          -> entanglement/information proxies
          -> sensoriel / langagier projections
          -> internal model -> metacognitive diagnostics

IMPORTANT EPISTEMIC STATUS
--------------------------
The fluxonium, spectrum, matrix elements, Berry phase and quantum-information
quantities are physical/mathematical objects within the stated model.

The SENSORIEL, LANGAGIER, MODELE INTERNE and METACOGNITION layers are NOT
claims that a superconducting circuit is conscious, sentient, or linguistic.
They are a formal interface that maps physical state descriptors into an
abstract configuration space. The bridge is deliberately explicit so that
metaphor is not confused with experimental evidence.

The "esoteric" element used here is the idea of a geometric phase / holonomy:
the state can acquire a phase associated with a path through parameter space.
This is legitimate quantum physics. Any interpretation of that holonomy as
"meaning", "experience", or "consciousness" remains speculative and is kept
outside the physical model.

Dependencies
------------
numpy, scipy, matplotlib
Optional: pandas is NOT required.

Usage
-----
    python fluxonium_psi.py
    python fluxonium_psi.py --no-plots
    python fluxonium_psi.py --output results
    python fluxonium_psi.py --EJ 5.0 --basis 100

Outputs
-------
    spectrum_fluxonium.png
    coherence_vs_EJ.png
    geometry_information.png
    configuration_pipeline.png
    fluxonium_results.csv
    fluxonium_summary.json
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import math
from typing import Iterable

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    EC: float = 1.0          # GHz
    EL: float = 0.5          # GHz
    EJ: float = 5.0          # GHz
    N_basis: int = 90
    N_levels: int = 6

    Q_diel: float = 5.0e5
    x_qp: float = 1.0e-8
    Delta: float = 50.0      # GHz
    T: float = 0.020         # K
    A_phi: float = 2.0e-6    # Phi0

    kB_h: float = 20.8366    # GHz/K


@dataclass(frozen=True)
class ScanParams:
    EJ_min: float = 1.5
    EJ_max: float = 12.0
    EJ_points: int = 35
    flux_points: int = 161


@dataclass
class PhysicalState:
    energies: np.ndarray
    evecs: np.ndarray
    phi_op: np.ndarray
    n_op: np.ndarray
    phi_zpf: float
    H: np.ndarray


@dataclass
class ConfigurationPsi:
    """
    Formal configuration vector.

    All fields are normalized descriptors in [0, 1] except where explicitly
    documented. They are NOT observables of consciousness.
    """
    spectral_coherence: float
    geometric_coherence: float
    fidelity_local: float
    metric_local: float
    entanglement_proxy: float
    sensoriel: float
    langagier: float
    internal_model: float
    metacognition: float
    geometric_phase: float


# ============================================================================
# 2. NUMERICAL UTILITIES
# ============================================================================

def safe_float(x: complex | float) -> float:
    return float(np.real_if_close(x))


def normalize_state(psi: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    norm = np.linalg.norm(psi)
    if norm < eps:
        raise ValueError("Impossible de normaliser un état de norme quasi nulle.")
    return psi / norm


def matrix_element(op: np.ndarray, evecs: np.ndarray, i: int, j: int) -> complex:
    return np.vdot(evecs[:, i], op @ evecs[:, j])


def unitary_function_of_hermitian(
    op: np.ndarray,
    fn,
) -> np.ndarray:
    """Calcule f(op) par diagonalisation spectrale."""
    vals, vecs = eigh(op)
    return (vecs * fn(vals)) @ vecs.conj().T


def clamp01(x: float) -> float:
    return float(np.clip(np.real(x), 0.0, 1.0))


# ============================================================================
# 3. FLUXONIUM PHYSICS
# ============================================================================

def harmonic_oscillator_basis(
    EC: float,
    EL: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    LC oscillator basis.

    phi_zpf = (2 EC / EL)^(1/4)
    phi = phi_zpf (a + a†)
    n   = -i/(2 phi_zpf) (a - a†)
    """
    if EC <= 0 or EL <= 0:
        raise ValueError("EC et EL doivent être strictement positifs.")
    if N < 8:
        raise ValueError("N_basis doit être >= 8.")

    phi_zpf = (2.0 * EC / EL) ** 0.25

    a = np.zeros((N, N), dtype=complex)
    for k in range(1, N):
        a[k - 1, k] = math.sqrt(k)

    adag = a.conj().T
    phi = phi_zpf * (a + adag)
    n_op = -1j / (2.0 * phi_zpf) * (a - adag)

    return phi, n_op, phi_zpf


def fluxonium_hamiltonian(
    EC: float,
    EL: float,
    EJ: float,
    phi_ext: float,
    phi: np.ndarray,
    n_op: np.ndarray,
) -> np.ndarray:
    """
    H = 4 EC n² + 1/2 EL phi² - EJ cos(phi - phi_ext).

    cos(phi - phi_ext) is evaluated as a matrix function because phi is a
    finite-dimensional Hermitian operator.
    """
    if EJ < 0:
        raise ValueError("EJ doit être >= 0.")

    H_C = 4.0 * EC * (n_op @ n_op)
    H_L = 0.5 * EL * (phi @ phi)

    cos_shifted = unitary_function_of_hermitian(
        phi,
        lambda x: np.cos(x - phi_ext),
    )

    H_J = -EJ * cos_shifted
    H = H_C + H_L + H_J

    # Remove tiny anti-Hermitian numerical roundoff.
    return 0.5 * (H + H.conj().T)


def solve_fluxonium(
    params: FluxoniumParams,
    phi_ext: float,
) -> PhysicalState:
    phi, n_op, phi_zpf = harmonic_oscillator_basis(
        params.EC, params.EL, params.N_basis
    )

    H = fluxonium_hamiltonian(
        params.EC,
        params.EL,
        params.EJ,
        phi_ext,
        phi,
        n_op,
    )

    evals, evecs = eigh(
        H,
        subset_by_index=[0, params.N_levels - 1],
        check_finite=True,
    )

    evals = evals - evals[0]

    return PhysicalState(
        energies=evals,
        evecs=evecs,
        phi_op=phi,
        n_op=n_op,
        phi_zpf=phi_zpf,
        H=H,
    )


def transition_frequencies(energies: np.ndarray) -> tuple[float, float]:
    if len(energies) < 3:
        raise ValueError("Au moins trois niveaux sont nécessaires.")
    return float(energies[1] - energies[0]), float(energies[2] - energies[1])


def anharmonicity(energies: np.ndarray) -> tuple[float, float]:
    f01, f12 = transition_frequencies(energies)
    alpha = f12 - f01
    alpha_rel = alpha / f01 if abs(f01) > 1e-12 else np.nan
    return float(alpha), float(alpha_rel)


# ============================================================================
# 4. COHERENCE / DÉCOHÉRENCE
# ============================================================================

def T1_dielectric(
    f01: float,
    n01: complex,
    params: FluxoniumParams,
) -> float:
    """Phenomenological dielectric-loss estimate, returned in microseconds."""
    if f01 <= 1e-12:
        return 1e9

    x = f01 / (params.kB_h * params.T)
    coth = 1.0 / np.tanh(0.5 * x) if x > 1e-8 else 2.0 / x

    gamma_GHz = (
        16.0 * params.EC * abs(n01) ** 2 / params.Q_diel * coth
    )
    return 1.0 / gamma_GHz * 1e-3 if gamma_GHz > 0 else 1e9


def T1_qp_junction(
    f01: float,
    sin01: complex,
    params: FluxoniumParams,
) -> float:
    """Phenomenological quasiparticle estimate for the junction channel."""
    if f01 <= 1e-12:
        return 1e9

    factor = math.sqrt(max(0.0, 2.0 * params.Delta / f01))
    boltz = math.exp(-min(700.0, f01 / (params.kB_h * params.T)))

    gamma_GHz = (
        (8.0 * params.EJ / math.pi)
        * factor
        * params.x_qp
        * abs(sin01) ** 2
        * (1.0 + boltz)
    )

    return 1.0 / gamma_GHz * 1e-3 if gamma_GHz > 0 else 1e9


def T1_qp_array(
    f01: float,
    phi01: complex,
    params: FluxoniumParams,
) -> float:
    """Phenomenological quasiparticle estimate for the array channel."""
    if f01 <= 1e-12:
        return 1e9

    factor = math.sqrt(max(0.0, 2.0 * params.Delta / f01))
    boltz = math.exp(-min(700.0, f01 / (params.kB_h * params.T)))

    gamma_GHz = (
        (8.0 * params.EL / math.pi)
        * factor
        * params.x_qp
        * abs(phi01 / 2.0) ** 2
        * (1.0 + boltz)
    )

    return 1.0 / gamma_GHz * 1e-3 if gamma_GHz > 0 else 1e9


def T_phi_flux(
    d_omega_d_phi: float,
    A_phi: float,
) -> float:
    """First-order phenomenological 1/f flux-noise estimate in microseconds."""
    A_rad = 2.0 * math.pi * A_phi
    gamma_GHz = abs(d_omega_d_phi) * math.sqrt(2.0 * math.log(2.0)) * A_rad

    if gamma_GHz < 1e-12:
        return 1e6

    return 1.0 / gamma_GHz * 1e-3


def coherence_observables(
    params: FluxoniumParams,
    phi_ext: float = math.pi,
    d_phi: float = 1e-4,
) -> dict[str, float]:
    state = solve_fluxonium(params, phi_ext)
    E = state.energies
    f01, f12 = transition_frequencies(E)
    alpha, alpha_rel = anharmonicity(E)

    n01 = matrix_element(state.n_op, state.evecs, 0, 1)
    phi01 = matrix_element(state.phi_op, state.evecs, 0, 1)

    sin_half = unitary_function_of_hermitian(
        state.phi_op,
        lambda x: np.sin(x / 2.0),
    )
    sin01 = matrix_element(sin_half, state.evecs, 0, 1)

    state_p = solve_fluxonium(params, phi_ext + d_phi)
    state_m = solve_fluxonium(params, phi_ext - d_phi)

    d_omega_d_phi = (
        (state_p.energies[1] - state_m.energies[1])
        / (2.0 * d_phi)
    )

    T1d = T1_dielectric(f01, n01, params)
    T1j = T1_qp_junction(f01, sin01, params)
    T1a = T1_qp_array(f01, phi01, params)

    gamma1 = 1.0 / T1d + 1.0 / T1j + 1.0 / T1a
    T1_total = 1.0 / gamma1

    Tphi = T_phi_flux(d_omega_d_phi, params.A_phi)

    gamma2 = 0.5 / T1_total + 1.0 / Tphi
    T2_star = 1.0 / gamma2

    return {
        "f01_GHz": f01,
        "f12_GHz": f12,
        "alpha_GHz": alpha,
        "alpha_relative": alpha_rel,
        "T1_diel_us": T1d,
        "T1_qp_JJ_us": T1j,
        "T1_qp_array_us": T1a,
        "T1_total_us": T1_total,
        "T_phi_us": Tphi,
        "T2_star_us": T2_star,
        "n01": abs(n01),
        "phi01": abs(phi01),
        "sin_half_01": abs(sin01),
        "domega_dphi_GHz_per_rad": float(d_omega_d_phi),
    }


# ============================================================================
# 5. GÉOMÉTRIE QUANTIQUE : FIDÉLITÉ, MÉTRIQUE, HOLONOMIE
# ============================================================================

def ground_state(
    params: FluxoniumParams,
    phi_ext: float,
) -> np.ndarray:
    return solve_fluxonium(params, phi_ext).evecs[:, 0]


def state_fidelity(psi: np.ndarray, chi: np.ndarray) -> float:
    """
    F = |<psi|chi>|² for pure normalized states.
    """
    psi = normalize_state(psi)
    chi = normalize_state(chi)
    return clamp01(abs(np.vdot(psi, chi)) ** 2)


def local_fubini_study_metric(
    params: FluxoniumParams,
    phi_ext: float,
    d_phi: float = 1e-3,
) -> tuple[float, float]:
    """
    Discrete pure-state metric:
        g ≈ (1 - F(phi, phi+dphi)) / dphi²

    This is a local information-geometric descriptor. The numerical prefactor
    depends on convention; we retain the convention explicitly here.
    """
    psi0 = ground_state(params, phi_ext)
    psip = ground_state(params, phi_ext + d_phi)

    F = state_fidelity(psi0, psip)
    g = max(0.0, (1.0 - F) / (d_phi ** 2))

    return float(g), float(F)


def berry_phase_loop(
    params: FluxoniumParams,
    phi_path: np.ndarray,
) -> float:
    """
    Discrete Berry/Pancharatnam phase along a closed parameter path.

    For a 1D flux parameter, a truly closed loop requires the path itself to
    return to its initial point. The phase is therefore a mathematical
    holonomy diagnostic of the chosen path, not a claim of an additional
    physical force.
    """
    if len(phi_path) < 3:
        raise ValueError("La boucle doit contenir au moins 3 points.")

    states = [ground_state(params, float(p)) for p in phi_path]

    phase = 0.0
    for a, b in zip(states[:-1], states[1:]):
        overlap = np.vdot(a, b)
        if abs(overlap) < 1e-14:
            return float("nan")
        phase += np.angle(overlap)

    # Close the loop.
    overlap = np.vdot(states[-1], states[0])
    if abs(overlap) < 1e-14:
        return float("nan")

    phase += np.angle(overlap)
    return float(np.angle(np.exp(1j * phase)))


def geometric_descriptors(
    params: FluxoniumParams,
    phi_ext: float = math.pi,
) -> dict[str, float]:
    g, F = local_fubini_study_metric(params, phi_ext)

    # A closed excursion around the flux coordinate.
    path = np.linspace(phi_ext - 0.5 * math.pi, phi_ext + 0.5 * math.pi, 31)
    path = np.concatenate([path, [path[0]]])

    gamma = berry_phase_loop(params, path)

    return {
        "metric_local": g,
        "fidelity_local": F,
        "geometric_phase_rad": gamma,
        "geometric_phase_abs_normalized": (
            clamp01(abs(gamma) / math.pi) if np.isfinite(gamma) else np.nan
        ),
    }


# ============================================================================
# 6. INFORMATION / "ENTRELACS" : EXPLICITLY A PROXY
# ============================================================================

def reduced_density_matrix_from_two_qubit_state(
    psi: np.ndarray,
) -> np.ndarray:
    """
    Exact partial trace for a 2x2 pure state.

    This is provided as a small information-theoretic laboratory. It is NOT
    the same thing as proving that a single fluxonium contains a bipartite
    conscious state.
    """
    psi = normalize_state(psi)

    if psi.size != 4:
        raise ValueError("Le laboratoire 2-qubits exige un état de dimension 4.")

    C = psi.reshape(2, 2)
    rho_A = C @ C.conj().T
    return 0.5 * (rho_A + rho_A.conj().T)


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(np.real(vals), 0.0, 1.0)
    vals = vals[vals > 1e-14]

    if len(vals) == 0:
        return 0.0

    return float(-np.sum(vals * np.log(vals)) / np.log(base))


def entanglement_proxy_from_transition_geometry(
    params: FluxoniumParams,
    phi_ext: float,
) -> float:
    """
    Proxy, not a physical entanglement measure of the fluxonium.

    We build a two-component normalized state from two low-energy transition
    matrix elements. This deliberately marks the abstraction boundary.
    """
    state = solve_fluxonium(params, phi_ext)

    n01 = matrix_element(state.n_op, state.evecs, 0, 1)
    phi01 = matrix_element(state.phi_op, state.evecs, 0, 1)

    z = np.array([n01, phi01], dtype=complex)
    z = normalize_state(z)

    # Two-mode vector embedded into a 2x2 amplitude matrix.
    C = np.array([[z[0], 0.0], [0.0, z[1]]], dtype=complex)
    psi4 = normalize_state(C.reshape(-1))

    rho_A = reduced_density_matrix_from_two_qubit_state(psi4)
    S = von_neumann_entropy(rho_A, base=2.0)

    # S=1 is maximal for two qubits, S=0 separable.
    return clamp01(S)


# ============================================================================
# 7. CONFIGURATION Ψ : SENSORIEL -> LANGAGIER -> MODÈLE INTERNE -> META
# ============================================================================

def sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-x))


def build_configuration_psi(
    physical: dict[str, float],
    geometry: dict[str, float],
    entanglement_proxy: float,
) -> ConfigurationPsi:
    """
    Formal information architecture.

    The coefficients are intentionally transparent. They are not fitted to
    biological consciousness and must not be interpreted as empirical laws.
    """
    # Physical coherence score.
    t1 = max(physical["T1_total_us"], 1e-12)
    t2 = max(physical["T2_star_us"], 1e-12)
    coherence = clamp01(
        0.5 * sigmoid(math.log10(t1 + 1e-12) - 1.0)
        + 0.5 * sigmoid(math.log10(t2 + 1e-12) - 1.0)
    )

    # Geometry: high fidelity + bounded metric gives a stable local regime.
    F = clamp01(geometry["fidelity_local"])
    g = max(0.0, geometry["metric_local"])
    geometric_coherence = clamp01(F * math.exp(-0.1 * min(g, 50.0)))

    # "Sensoriel": coupling/information intake proxy.
    sensoriel = clamp01(
        0.45 * coherence
        + 0.30 * geometric_coherence
        + 0.25 * entanglement_proxy
    )

    # "Langagier": symbolic projection proxy from spectral structure.
    anh = abs(physical["alpha_relative"])
    spectral_selectivity = clamp01(anh / (1.0 + anh))
    langagier = clamp01(
        0.55 * spectral_selectivity
        + 0.25 * geometric_coherence
        + 0.20 * entanglement_proxy
    )

    # Internal model = recurrent integration of physical + geometry channels.
    internal_model = clamp01(
        0.35 * sensoriel
        + 0.35 * langagier
        + 0.30 * geometric_coherence
    )

    # Metacognition proxy = consistency between state, geometry and model.
    metacognition = clamp01(
        0.40 * internal_model
        + 0.30 * coherence
        + 0.20 * F
        + 0.10 * entanglement_proxy
    )

    gamma = geometry["geometric_phase_abs_normalized"]
    if not np.isfinite(gamma):
        gamma = 0.0

    return ConfigurationPsi(
        spectral_coherence=coherence,
        geometric_coherence=geometric_coherence,
        fidelity_local=F,
        metric_local=g,
        entanglement_proxy=entanglement_proxy,
        sensoriel=sensoriel,
        langagier=langagier,
        internal_model=internal_model,
        metacognition=metacognition,
        geometric_phase=clamp01(gamma),
    )


# ============================================================================
# 8. SCANS
# ============================================================================

def scan_flux(
    params: FluxoniumParams,
    scan: ScanParams,
) -> dict[str, np.ndarray]:
    phi_ext = np.linspace(0.0, 2.0 * math.pi, scan.flux_points)
    flux = phi_ext / (2.0 * math.pi)

    energies = np.zeros((len(phi_ext), params.N_levels))
    for i, p in enumerate(phi_ext):
        energies[i] = solve_fluxonium(params, float(p)).energies

    return {
        "phi_ext": phi_ext,
        "flux": flux,
        "energies": energies,
    }


def scan_EJ(
    base: FluxoniumParams,
    scan: ScanParams,
    phi_ext: float = math.pi,
) -> list[dict[str, float]]:
    rows = []

    for EJ in np.linspace(scan.EJ_min, scan.EJ_max, scan.EJ_points):
        p = FluxoniumParams(**{**asdict(base), "EJ": float(EJ)})

        physical = coherence_observables(p, phi_ext)
        geometry = geometric_descriptors(p, phi_ext)
        ent = entanglement_proxy_from_transition_geometry(p, phi_ext)
        psi = build_configuration_psi(physical, geometry, ent)

        row = {
            "EJ_GHz": float(EJ),
            **physical,
            **geometry,
            **asdict(psi),
        }
        rows.append(row)

    return rows


# ============================================================================
# 9. GRAPHIQUES
# ============================================================================

def save_spectrum_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(1, data["energies"].shape[1]):
        ax.plot(
            data["flux"],
            data["energies"][:, k],
            lw=1.5,
            label=f"niveau {k}",
        )

    ax.axvline(0.5, ls="--", alpha=0.5)
    ax.set_xlabel(r"$\Phi_{\rm ext}/\Phi_0$")
    ax.set_ylabel("Énergie relative (GHz)")
    ax.set_title("Fluxonium — spectre")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "spectrum_fluxonium.png", dpi=170)
    plt.close(fig)


def save_coherence_plot(
    rows: list[dict[str, float]],
    output: Path,
) -> None:
    EJ = [r["EJ_GHz"] for r in rows]
    T1 = [r["T1_total_us"] for r in rows]
    T2 = [r["T2_star_us"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(EJ, T1, lw=2, label=r"$T_1$")
    ax.semilogy(EJ, T2, lw=2, ls="--", label=r"$T_2^*$")
    ax.set_xlabel(r"$E_J$ (GHz)")
    ax.set_ylabel("Temps (µs)")
    ax.set_title("Cohérence au sweet spot")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "coherence_vs_EJ.png", dpi=170)
    plt.close(fig)


def save_geometry_information_plot(
    rows: list[dict[str, float]],
    output: Path,
) -> None:
    EJ = [r["EJ_GHz"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(EJ, [r["fidelity_local"] for r in rows],
            lw=2, label="Fidélité locale")
    ax.plot(EJ, [r["geometric_coherence"] for r in rows],
            lw=2, label="Cohérence géométrique")
    ax.plot(EJ, [r["entanglement_proxy"] for r in rows],
            lw=2, label="Entrelacs — proxy")
    ax.plot(EJ, [r["metacognition"] for r in rows],
            lw=2, ls="--", label="Métacognition — proxy formel")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$E_J$ (GHz)")
    ax.set_ylabel("Indice normalisé")
    ax.set_title("Géométrie → information → configuration Ψ")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "geometry_information.png", dpi=170)
    plt.close(fig)


def save_pipeline_plot(
    psi: ConfigurationPsi,
    output: Path,
) -> None:
    labels = [
        "spectral\ncoherence",
        "geometric\ncoherence",
        "sensoriel",
        "langagier",
        "modèle\ninterne",
        "métacognition",
    ]
    values = [
        psi.spectral_coherence,
        psi.geometric_coherence,
        psi.sensoriel,
        psi.langagier,
        psi.internal_model,
        psi.metacognition,
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    ax.plot(x, values, marker="o", lw=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Indice normalisé")
    ax.set_title(
        "Configuration Ψ — chaîne formelle\n"
        "état → géométrie → entrelacs → modèle interne"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "configuration_pipeline.png", dpi=170)
    plt.close(fig)


# ============================================================================
# 10. EXPORTS
# ============================================================================

def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return

    keys = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_summary(
    params: FluxoniumParams,
    rows: list[dict[str, float]],
    output: Path,
) -> None:
    best = max(rows, key=lambda r: r["metacognition"])

    summary = {
        "epistemic_note": (
            "Les grandeurs fluxonium et geometriques sont modelisees "
            "physiquement. Les couches sensoriel/langagier/modele_interne/"
            "metacognition sont des proxies formels et non des preuves de "
            "conscience."
        ),
        "parameters": asdict(params),
        "rows": len(rows),
        "best_configuration_by_formal_proxy": best,
    }

    (output / "fluxonium_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================================
# 11. RAPPORT TERMINAL
# ============================================================================

def print_report(
    params: FluxoniumParams,
    physical: dict[str, float],
    geometry: dict[str, float],
    psi: ConfigurationPsi,
) -> None:
    print("=" * 82)
    print("FLUXONIUM Ψ — QUANTUM-GEOMETRIC CONFIGURATION TOOLKIT")
    print("=" * 82)
    print(
        f"EC={params.EC:.3f} GHz | EL={params.EL:.3f} GHz | "
        f"EJ={params.EJ:.3f} GHz | N={params.N_basis}"
    )
    print("-" * 82)

    print("PHYSIQUE")
    print(f"  f01                    : {physical['f01_GHz']:.6f} GHz")
    print(f"  f12                    : {physical['f12_GHz']:.6f} GHz")
    print(f"  anharmonicité          : {physical['alpha_GHz']:.6f} GHz")
    print(f"  T1 total               : {physical['T1_total_us']:.4g} µs")
    print(f"  T2*                    : {physical['T2_star_us']:.4g} µs")
    print(f"  dω/dφ                  : {physical['domega_dphi_GHz_per_rad']:.6g}")

    print("\nGÉOMÉTRIE QUANTIQUE")
    print(f"  fidélité locale        : {geometry['fidelity_local']:.8f}")
    print(f"  métrique locale        : {geometry['metric_local']:.6g}")
    print(f"  phase géométrique      : {geometry['geometric_phase_rad']:.6g} rad")

    print("\nCONFIGURATION Ψ")
    print(f"  cohérence spectrale    : {psi.spectral_coherence:.6f}")
    print(f"  cohérence géométrique  : {psi.geometric_coherence:.6f}")
    print(f"  entrelacs — proxy      : {psi.entanglement_proxy:.6f}")
    print(f"  sensoriel — proxy      : {psi.sensoriel:.6f}")
    print(f"  langagier — proxy      : {psi.langagier:.6f}")
    print(f"  modèle interne — proxy : {psi.internal_model:.6f}")
    print(f"  métacognition — proxy  : {psi.metacognition:.6f}")

    print("\nSTATUT ÉPISTÉMIQUE")
    print("  La phase géométrique est un objet de physique quantique.")
    print("  Les couches sensoriel/langagier/métacognition sont des")
    print("  transformations formelles, pas des observables de conscience.")
    print("=" * 82)


# ============================================================================
# 12. MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fluxonium Ψ — toolkit physique + géométrie + configuration."
    )
    parser.add_argument("--EC", type=float, default=1.0)
    parser.add_argument("--EL", type=float, default=0.5)
    parser.add_argument("--EJ", type=float, default=5.0)
    parser.add_argument("--basis", type=int, default=90)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("fluxonium_psi_results"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.levels < 3:
        raise SystemExit("--levels doit être >= 3.")

    params = FluxoniumParams(
        EC=args.EC,
        EL=args.EL,
        EJ=args.EJ,
        N_basis=args.basis,
        N_levels=args.levels,
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    # ---- Point central : sweet spot Phi_ext = pi.
    phi_ext = math.pi

    physical = coherence_observables(params, phi_ext)
    geometry = geometric_descriptors(params, phi_ext)
    entanglement_proxy = entanglement_proxy_from_transition_geometry(
        params, phi_ext
    )

    psi = build_configuration_psi(
        physical,
        geometry,
        entanglement_proxy,
    )

    print_report(params, physical, geometry, psi)

    # ---- Scan EJ.
    scan = ScanParams()
    rows = scan_EJ(params, scan, phi_ext=phi_ext)

    # ---- Figures.
    if not args.no_plots:
        flux_data = scan_flux(params, scan)
        save_spectrum_plot(flux_data, output)
        save_coherence_plot(rows, output)
        save_geometry_information_plot(rows, output)
        save_pipeline_plot(psi, output)

    # ---- Data.
    write_csv(rows, output / "fluxonium_results.csv")
    write_summary(params, rows, output)

    print("\nSorties :")
    print(f"  {output / 'fluxonium_results.csv'}")
    print(f"  {output / 'fluxonium_summary.json'}")

    if not args.no_plots:
        print(f"  {output / 'spectrum_fluxonium.png'}")
        print(f"  {output / 'coherence_vs_EJ.png'}")
        print(f"  {output / 'geometry_information.png'}")
        print(f"  {output / 'configuration_pipeline.png'}")

    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()
