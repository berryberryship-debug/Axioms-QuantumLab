#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRANSMON — BANC DE CARACTÉRISATION V3.2-DEV
===========================================
"""

import numpy as np
import json
import sys
from datetime import datetime

from src.test_transmon_driven_v2 import build_system_operators, rk4_step_td

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value

def validate_characterization_inputs(d, psi_target, target_subspace, coh_idx, omega_d_ghz):
    if not isinstance(d, (int, np.integer)) or isinstance(d, bool) or d < 2:
        raise TypeError("d doit être un entier >= 2.")
    if not isinstance(coh_idx, (int, np.integer)) or isinstance(coh_idx, bool):
        raise TypeError("coh_idx doit être un entier.")
        
    if (
        isinstance(omega_d_ghz, (bool, np.bool_))
        or not isinstance(omega_d_ghz, (int, float, np.integer, np.floating))
    ):
        raise TypeError("omega_d_ghz doit être numérique.")

    if not np.isfinite(omega_d_ghz) or omega_d_ghz <= 0:
        raise ValueError("omega_d_ghz doit être fini et strictement positif.")
        
    if coh_idx >= d or coh_idx < 1:
        raise ValueError(f"Indice de cohérence {coh_idx} incompatible avec d={d}.")
    
    psi_target = np.asarray(psi_target, dtype=complex)
    
    if not isinstance(target_subspace, (set, list, tuple)):
        raise TypeError("target_subspace doit être un ensemble ou une séquence d'indices.")
        
    target_subspace_set = set(target_subspace)

    if psi_target.shape != (d,):
        raise ValueError(f"Dimension cible incompatible : attendu {(d,)}, reçu {psi_target.shape}.")
    if not target_subspace_set:
        raise ValueError("Le sous-espace cible est vide.")
        
    if any(
        not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
        for i in target_subspace_set
    ):
        raise TypeError("Les indices du sous-espace doivent être des entiers non booléens.")
        
    if any(i < 0 or i >= d for i in target_subspace_set):
        raise ValueError("Indice du sous-espace hors dimension.")

    norm = np.linalg.norm(psi_target)
    if not np.isclose(norm, 1.0, atol=1e-12):
        raise ValueError(f"Le vecteur cible doit être normalisé (norme = {norm}).")

    outside = [i for i in range(d) if i not in target_subspace_set]
    if outside and np.linalg.norm(psi_target[outside]) > 1e-12:
        raise ValueError("Le vecteur cible doit appartenir intégralement au sous-espace accepté.")

    return psi_target, target_subspace_set

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def circular_statistics(angles_list):
    valid_angles = [a for a in angles_list if not np.isnan(a)]
    if not valid_angles:
        return float("nan"), 0.0
    sin_sum = np.sum(np.sin(valid_angles))
    cos_sum = np.sum(np.cos(valid_angles))
    return float(np.arctan2(sin_sum, cos_sum)), float(np.sqrt(sin_sum**2 + cos_sum**2) / len(valid_angles))

def compute_fidelities(rho, psi_target, target_subspace):
    f_noncond = float(np.real(np.vdot(psi_target, rho @ psi_target)))
    indices = sorted(target_subspace)
    p_survie = float(np.real(np.trace(rho[np.ix_(indices, indices)])))
    f_cond = f_noncond / p_survie if p_survie > 1e-12 else float("nan")
    return f_noncond, float(f_cond), p_survie

def run_v3_2_characterization(regime_name, omega_d_ghz, psi_target, target_subspace, coh_idx, d=4):
    psi_target, target_subspace = validate_characterization_inputs(d, psi_target, target_subspace, coh_idx, omega_d_ghz)

    omega_d = omega_d_ghz * 2.0 * np.pi
    DT = 0.001
    T_FINAL = 24.0
    SAMPLE_STRIDE = 10
    SIGMA = 3.0
    T_CENTER = 12.0
    PHASE = 0.0
    AMP = np.pi / (SIGMA * np.sqrt(2.0 * np.pi))
    
    TOL_TRACE = 1e-10
    TOL_HERM = 1e-10
    TOL_POS = 1e-12

    H0, X, c_ops = build_system_operators(d)
    energies = np.real(np.diagonal(H0))
    freq_ref_coherence = energies[coh_idx] - energies[0]

    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    rho = np.outer(psi0, psi0.conjugate())
    steps = int(round(T_FINAL / DT))
    
    max_trace_drift = 0.0
    max_hermiticity_err = 0.0
    min_eigenvalue_obs = np.inf
    max_stepwise_leakage = 0.0
    t_max_leakage = 0.0
    p3_max_stepwise = 0.0

    times, p1_hist, p2_hist, d_phases = [], [], [], []

    for step in range(1, steps + 1):
        t_start = (step - 1) * DT
        t_end = t_start + DT
        
        rho = rk4_step_td(rho, t_start, DT, H0, X, c_ops, omega_d, 
                          include_dissipators=True, 
                          amp=AMP, t_center=T_CENTER, sigma=SIGMA, phase=PHASE)
        
        if not np.all(np.isfinite(rho)):
            raise ValueError(f"Instabilité numérique critique au pas {step}: rho non fini.")
            
        max_trace_drift = max(max_trace_drift, np.abs(np.trace(rho) - 1.0))
        herm_err = np.linalg.norm(rho - rho.conjugate().T, ord="fro")
        max_hermiticity_err = max(max_hermiticity_err, herm_err)
        
        eigvals = np.linalg.eigvalsh((rho + rho.conjugate().T) / 2.0)
        min_eigenvalue_obs = min(min_eigenvalue_obs, np.min(eigvals))

        diag_real = np.real(np.diagonal(rho))
        
        if d > 3:
            p3_max_stepwise = max(p3_max_stepwise, float(diag_real[3]))
            
        l_val = np.sum([diag_real[n] for n in range(d) if n not in target_subspace])
        if l_val > max_stepwise_leakage:
            max_stepwise_leakage = float(l_val)
            t_max_leakage = t_end

        if step % SAMPLE_STRIDE == 0 or step == steps:
            times.append(t_end)
            p1_hist.append(diag_real[1])
            p2_hist.append(diag_real[2] if d > 2 else 0.0)
            
            coh_target = rho[0, coh_idx]
            if np.abs(coh_target) > 1e-3:
                coh_rot = coh_target * np.exp(-1j * freq_ref_coherence * t_end)
                d_phases.append(wrap_angle(np.angle(coh_rot) - 0.0))
            else:
                d_phases.append(np.nan)

    f_noncond, f_cond, p_surv = compute_fidelities(rho, psi_target, target_subspace)
    mean_phase, r_conc = circular_statistics(d_phases)
    
    integrity_values = np.array([max_trace_drift, max_hermiticity_err, min_eigenvalue_obs], dtype=float)
    mandatory_values = np.array([f_noncond, p_surv, max_stepwise_leakage, np.max(p1_hist), r_conc], dtype=float)
    
    all_mandatory_finite = bool(np.all(np.isfinite(integrity_values)) and np.all(np.isfinite(mandatory_values)))
    
    integrity_pass = (
        all_mandatory_finite and
        (max_trace_drift <= TOL_TRACE) and
        (max_hermiticity_err <= TOL_HERM) and
        (min_eigenvalue_obs >= -TOL_POS)
    )

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "regime_name": regime_name,
            "dimension_d": d,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__
        },
        "layer_4_reproducibility": {
            "drive_freq_ghz": omega_d_ghz,
            "pulse_amp": AMP,
            "pulse_sigma": SIGMA,
            "pulse_t_center": T_CENTER,
            "pulse_phase": PHASE,
            "dt_ns": DT,
            "target_subspace": sorted(list(target_subspace)),
            "coherence_idx": coh_idx,
            "coherence_reference_freq_rad_ns": float(freq_ref_coherence)
        },
        "layer_2_integrity": {
            "max_trace_drift": float(max_trace_drift),
            "max_hermiticity_error": float(max_hermiticity_err),
            "min_eigenvalue_obs": float(min_eigenvalue_obs),
            "all_mandatory_finite": all_mandatory_finite,
            "status_overall": "PASS" if integrity_pass else "FAIL",
            "tolerances": {"trace": TOL_TRACE, "hermiticity": TOL_HERM, "positivity": TOL_POS}
        },
        "layer_1_dynamics": {
            "P1_max_sampled": float(np.max(p1_hist)),
            "P2_max_sampled": float(np.max(p2_hist)),
            "P3_max_stepwise": p3_max_stepwise,
            "Leakage_max_stepwise": max_stepwise_leakage,
            "t_Leakage_max_stepwise": t_max_leakage,
            "Leakage_final": float(l_val)
        },
        "layer_3_operation": {
            "fidelity_state_unconditional": f_noncond,
            "fidelity_state_conditional": f_cond,
            "survival_probability_final": p_surv,
            "coherence_circular_mean_phase": mean_phase,
            "coherence_phase_concentration_R": r_conc,
            "coherence_valid_phase_samples": sum(~np.isnan(d_phases)),
            "gate_phase_error": "NON DEFINI",
            "status": {
                "conditional_fidelity_defined": bool(np.isfinite(f_cond)),
                "coherence_phase_defined": bool(np.isfinite(mean_phase))
            }
        }
    }

    print(json.dumps(json_safe(report), indent=4, ensure_ascii=False, allow_nan=False))
    return report

def main():
    psi_target_1 = np.zeros(4, dtype=complex)
    psi_target_1[1] = 1.0
    run_v3_2_characterization("G_01 (1-photon)", 5.0, psi_target_1, target_subspace={0, 1}, coh_idx=1)

    psi_target_2 = np.zeros(4, dtype=complex)
    psi_target_2[2] = 1.0
    run_v3_2_characterization("G_02 (2-photons)", 4.85, psi_target_2, target_subspace={0, 2}, coh_idx=2)

if __name__ == "__main__":
    main()
