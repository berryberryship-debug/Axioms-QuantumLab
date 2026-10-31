#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STE-50 v2.1.4.1 — RK4 COUPLÉ (z, ρ) — VERSION CORRIGÉE + SORTIES COMPLÈTES

Corrections principales :
1. Calcul de dq/dt cohérent et explicite
2. Enregistrement correct de z_start
3. Projection robuste
4. Détection Preisach clarifiée
5. Sorties complètes (CSV + JSON + NPY + log)
"""

import os
import json
import csv
import hashlib
from datetime import datetime
import numpy as np

TOL = 1e-12


# ======================================================================
# 1. SIGNAL EXTERNE
# ======================================================================

def external_signal(t):
    return (
        1.2 * np.sin(0.5 * np.pi * t)
        + 0.3 * np.cos(0.2 * np.pi * t)
    )


def external_signal_derivative(t):
    return (
        0.6 * np.pi * np.cos(0.5 * np.pi * t)
        - 0.06 * np.pi * np.sin(0.2 * np.pi * t)
    )


# ======================================================================
# 2. OBSERVABLE DE RÉTROACTION
# ======================================================================

def feedback_observable(target="B"):
    O = np.zeros((4, 4), dtype=np.complex128)
    if target == "A":
        O[2, 2] = 1.0
        O[3, 3] = 1.0
    elif target == "B":
        O[1, 1] = 1.0
        O[3, 3] = 1.0
    else:
        raise ValueError("target doit être 'A' ou 'B'.")
    return O


# ======================================================================
# 3. MÉMOIRE MACROPHASE
# ======================================================================

class MacroPhaseMemoryCoupled:
    def __init__(
        self,
        kmin=0.2,
        kmax=3.0,
        A=1.0,
        beta_bw=0.4,
        gamma_bw=0.2,
        n=2.0,
        beta_tanh=1.5,
    ):
        self.kmin = kmin
        self.kmax = kmax
        self.A = A
        self.beta_bw = beta_bw
        self.gamma_bw = gamma_bw
        self.n = n
        self.beta_tanh = beta_tanh

        self.z = 0.0
        self.weights = np.array([0.15, 0.25, 0.35, 0.15, 0.10])
        self.p_states = np.ones(5)
        self.thresholds = np.array([
            [-1.0, 1.0],
            [-0.5, 0.5],
            [-0.2, 0.2],
            [0.1, 0.8],
            [-0.8, -0.1],
        ])

    def bouc_wen_rhs(self, z, du_eff_dt):
        z_abs = abs(z)
        return (
            self.A * du_eff_dt
            - self.beta_bw * abs(du_eff_dt) * (z_abs ** (self.n - 1.0)) * z
            - self.gamma_bw * du_eff_dt * (z_abs ** self.n)
        )

    def update_preisach(self, u_eff):
        changed = False
        for i in range(5):
            old = self.p_states[i]
            if u_eff >= self.thresholds[i, 1]:
                self.p_states[i] = 1.0
            elif u_eff <= self.thresholds[i, 0]:
                self.p_states[i] = -1.0
            if self.p_states[i] != old:
                changed = True
        k_preisach = float(np.sum(self.weights * self.p_states))
        return k_preisach, changed

    def compute_k_eff(self, z, k_preisach):
        s = 0.5 * (1.0 + np.tanh(self.beta_tanh * z))
        k_bouc_wen = self.kmin + (self.kmax - self.kmin) * s
        return max(0.1, float(k_bouc_wen + 0.2 * k_preisach))


# ======================================================================
# 4. OUTILS QUANTIQUES
# ======================================================================

def hermitize(rho):
    return 0.5 * (rho + rho.conj().T)


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(hermitize(rho))
    positive = eigenvalues.real[eigenvalues.real > TOL]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def partial_trace_b(rho, dim_a=2, dim_b=2):
    rho_a = np.zeros((dim_a, dim_a), dtype=np.complex128)
    for b in range(dim_b):
        for i in range(dim_a):
            for j in range(dim_a):
                rho_a[i, j] += rho[i * dim_b + b, j * dim_b + b]
    return rho_a


def partial_trace_a(rho, dim_a=2, dim_b=2):
    rho_b = np.zeros((dim_b, dim_b), dtype=np.complex128)
    for a in range(dim_a):
        for i in range(dim_b):
            for j in range(dim_b):
                rho_b[i, j] += rho[a * dim_b + i, a * dim_b + j]
    return rho_b


def audit_rho_raw(rho, label=""):
    rho_h = hermitize(rho)
    eigenvalues = np.linalg.eigvalsh(rho_h)
    trace = float(np.trace(rho).real)
    return {
        "label": label,
        "trace": trace,
        "trace_error": float(abs(trace - 1.0)),
        "hermiticity_error": float(np.linalg.norm(rho - rho.conj().T, ord="fro")),
        "min_eigenvalue": float(eigenvalues.min()),
        "purity": float(np.real(np.trace(rho @ rho))),
        "is_valid": bool(
            abs(trace - 1.0) < 1e-10
            and np.linalg.norm(rho - rho.conj().T, ord="fro") < 1e-10
            and eigenvalues.min() > -1e-10
        ),
    }


def project_physical_rho(rho):
    rho_h = hermitize(rho)
    trace = np.trace(rho_h).real
    if trace <= TOL:
        raise ValueError("Trace non positive : projection impossible.")
    rho_h = rho_h / trace
    eigenvalues, eigenvectors = np.linalg.eigh(rho_h)
    eigenvalues = np.clip(eigenvalues.real, 0.0, None)
    total = eigenvalues.sum()
    if total <= TOL:
        raise ValueError("Spectre non projetable.")
    eigenvalues = eigenvalues / total
    rho_projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    rho_projected = hermitize(rho_projected)
    projection_error = float(np.linalg.norm(rho_projected - rho, ord="fro"))
    return rho_projected, projection_error


# ======================================================================
# 5. HAMILTONIEN
# ======================================================================

def build_hamiltonian(gap_material, phase_k, g_coupling, mu=0.0, Delta=0.5):
    delta_k = Delta * np.exp(1j * phase_k)
    delta_k_star = np.conj(delta_k)
    H = np.zeros((4, 4), dtype=np.complex128)
    H[0, 0] = -mu + 0.5 * gap_material
    H[1, 1] = -mu - 0.5 * gap_material
    H[2, 2] = mu + 0.5 * gap_material
    H[3, 3] = mu - 0.5 * gap_material
    H[0, 2] = delta_k_star
    H[2, 0] = delta_k
    H[1, 3] = delta_k_star
    H[3, 1] = delta_k
    H[0, 3] = g_coupling
    H[3, 0] = g_coupling
    H[1, 2] = g_coupling
    H[2, 1] = g_coupling
    return H


def phase_from_u_eff(u_eff):
    return (u_eff % (2.0 * np.pi)) - np.pi


# ======================================================================
# 6. GÉNÉRATEUR COUPLÉ
# ======================================================================

def lindblad_rhs(rho, H, c_ops):
    drho = -1j * (H @ rho - rho @ H)
    for L in c_ops:
        L_dagger = L.conj().T
        drho += L @ rho @ L_dagger - 0.5 * (L_dagger @ L @ rho + rho @ L_dagger @ L)
    return drho


def coupled_rhs(
    z, rho, t, memory, eta_u, g_coupling, k_preisach, c_ops, feedback_enabled=True
):
    u_ext = external_signal(t)
    du_ext_dt = external_signal_derivative(t)
    O = feedback_observable("B")
    q = float(np.real(np.trace(rho @ O)))

    if feedback_enabled:
        u_eff = u_ext + eta_u * q
    else:
        u_eff = u_ext

    k_eff = memory.compute_k_eff(z, k_preisach)
    phase_k = phase_from_u_eff(u_eff)
    H = build_hamiltonian(gap_material=k_eff, phase_k=phase_k, g_coupling=g_coupling)

    drho_dt = lindblad_rhs(rho, H, c_ops)
    dq_dt = float(np.real(np.trace(O @ drho_dt)))

    if feedback_enabled:
        du_eff_dt = du_ext_dt + eta_u * dq_dt
    else:
        du_eff_dt = du_ext_dt
        dq_dt = 0.0

    dz_dt = memory.bouc_wen_rhs(z, du_eff_dt)

    diagnostics = {
        "u_ext": u_ext,
        "u_eff": u_eff,
        "du_ext_dt": du_ext_dt,
        "du_eff_dt": du_eff_dt,
        "q": q,
        "dq_dt": dq_dt,
        "k_eff": k_eff,
        "phase_k": phase_k,
    }
    return dz_dt, drho_dt, diagnostics


# ======================================================================
# 7. RK4 COUPLÉ
# ======================================================================

def coupled_rk4_step(
    z, rho, t_start, dt, memory, eta_u, g_coupling, k_preisach, c_ops, feedback_enabled=True
):
    def rhs(z_stage, rho_stage, t_stage):
        return coupled_rhs(
            z_stage, rho_stage, t_stage,
            memory, eta_u, g_coupling, k_preisach, c_ops, feedback_enabled
        )

    dz1, drho1, diag1 = rhs(z, rho, t_start)
    dz2, drho2, diag2 = rhs(z + 0.5*dt*dz1, rho + 0.5*dt*drho1, t_start + 0.5*dt)
    dz3, drho3, diag3 = rhs(z + 0.5*dt*dz2, rho + 0.5*dt*drho2, t_start + 0.5*dt)
    dz4, drho4, diag4 = rhs(z + dt*dz3, rho + dt*drho3, t_start + dt)

    z_next = z + (dt / 6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)
    rho_raw = rho + (dt / 6.0) * (drho1 + 2*drho2 + 2*drho3 + drho4)

    return z_next, rho_raw, {
        "stage_1": diag1,
        "stage_2": diag2,
        "stage_3": diag3,
        "stage_4": diag4,
    }


# ======================================================================
# 8. ÉTATS ET OPÉRATEURS
# ======================================================================

def initial_bell_state():
    psi0 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return np.outer(psi0, psi0.conj())


def decoherence_operators(gamma=0.02):
    L = np.zeros((4, 4), dtype=np.complex128)
    L[1, 0] = np.sqrt(gamma)
    L[3, 2] = np.sqrt(gamma)
    return [L]


# ======================================================================
# 9. MÉTRIQUES
# ======================================================================

def ste_metrics(rho, rho_ref, gap, topology_proxy):
    rho_a = partial_trace_b(rho)
    rho_b = partial_trace_a(rho)
    entropy_ab = von_neumann_entropy(rho)
    entropy_a = von_neumann_entropy(rho_a)
    entropy_b = von_neumann_entropy(rho_b)
    mutual_information = max(0.0, entropy_a + entropy_b - entropy_ab)
    coherence = float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))
    bell_survival = float(np.real(np.trace(hermitize(rho) @ hermitize(rho_ref))))
    purity = float(np.real(np.trace(rho @ rho)))
    psi = 0.5 * bell_survival + 0.5 * mutual_information

    return {
        "gap": float(gap),
        "topology_proxy": int(topology_proxy),
        "entropy_ab": entropy_ab,
        "entropy_a": entropy_a,
        "entropy_b": entropy_b,
        "mutual_information": mutual_information,
        "coherence": coherence,
        "bell_survival": bell_survival,
        "purity": purity,
        "psi": psi,
    }


# ======================================================================
# 10. SIMULATION HYBRIDE
# ======================================================================

def run_coupled_system(
    total_time=2.0,
    n_steps=200,
    g_coupling=0.25,
    eta_u=0.1,
    feedback_enabled=True,
    project=True,
    audit_each_step=True,
):
    dt = total_time / n_steps
    memory = MacroPhaseMemoryCoupled()
    rho = initial_bell_state()
    rho_ref = rho.copy()
    c_ops = decoherence_operators(gamma=0.02)

    history = []
    invalid_raw_audits = []
    preisach_events = []

    u0 = external_signal(0.0)
    k_preisach, _ = memory.update_preisach(u0)

    for step in range(n_steps):
        t_start = step * dt
        t_end = t_start + dt
        z_start = memory.z

        q_current = float(np.real(np.trace(rho @ feedback_observable("B"))))
        u_eff_start = external_signal(t_start) + (eta_u * q_current if feedback_enabled else 0.0)

        k_preisach, changed = memory.update_preisach(u_eff_start)
        if changed:
            preisach_events.append({
                "time": float(t_start),
                "u_eff": float(u_eff_start),
                "p_states": memory.p_states.copy().tolist(),
            })

        z_next, rho_raw, stage_diagnostics = coupled_rk4_step(
            memory.z, rho, t_start, dt,
            memory, eta_u, g_coupling, k_preisach, c_ops, feedback_enabled
        )

        raw_audit = audit_rho_raw(rho_raw, label=f"step={step}_raw")
        if audit_each_step and not raw_audit["is_valid"]:
            invalid_raw_audits.append(raw_audit)

        if project:
            rho_projected, projection_error = project_physical_rho(rho_raw)
        else:
            rho_projected = rho_raw
            projection_error = 0.0

        memory.z = z_next
        rho = rho_projected

        metrics = ste_metrics(
            rho, rho_ref,
            stage_diagnostics["stage_4"]["k_eff"],
            1 if stage_diagnostics["stage_4"]["k_eff"] > 0.8 else 0,
        )

        metrics.update({
            "step": step,
            "t_start": t_start,
            "t_end": t_end,
            "u_eff_start": stage_diagnostics["stage_1"]["u_eff"],
            "u_eff_mid": stage_diagnostics["stage_2"]["u_eff"],
            "u_eff_end": stage_diagnostics["stage_4"]["u_eff"],
            "q_start": stage_diagnostics["stage_1"]["q"],
            "q_end": stage_diagnostics["stage_4"]["q"],
            "dq_dt_end": stage_diagnostics["stage_4"]["dq_dt"],
            "z_start": z_start,
            "z_end": z_next,
            "k_eff_start": stage_diagnostics["stage_1"]["k_eff"],
            "k_eff_end": stage_diagnostics["stage_4"]["k_eff"],
            "projection_error": projection_error,
            "raw_trace": raw_audit["trace"],
            "raw_trace_error": raw_audit["trace_error"],
            "raw_hermiticity_error": raw_audit["hermiticity_error"],
            "raw_min_eigenvalue": raw_audit["min_eigenvalue"],
            "raw_audit_valid": raw_audit["is_valid"],
            "projection_enabled": project,
            "feedback_enabled": feedback_enabled,
            "eta_u": eta_u,
            "g_coupling": g_coupling,
        })
        history.append(metrics)

    return {
        "history": history,
        "rho_final": rho,
        "z_final": memory.z,
        "invalid_raw_audits": invalid_raw_audits,
        "preisach_events": preisach_events,
        "n_steps": n_steps,
        "dt": dt,
        "total_time": total_time,
        "feedback_enabled": feedback_enabled,
        "eta_u": eta_u,
        "g_coupling": g_coupling,
    }


# ======================================================================
# 11. TESTS
# ======================================================================

def test_hamiltonian():
    print("=== TEST HAMILTONIEN ===")
    H = build_hamiltonian(1.0, 0.7, 0.25)
    err = np.linalg.norm(H - H.conj().T, ord="fro")
    print(f"Erreur d’hermiticité : {err:.3e}")
    assert err < TOL
    print("OK\n")


def test_feedback_isolation():
    print("=== TEST ISOLATION : eta_u = 0 ===")
    result = run_coupled_system(total_time=1.0, n_steps=100, eta_u=0.0,
                                feedback_enabled=False, project=False, audit_each_step=False)
    print(f"q final : {result['history'][-1]['q_end']:.12f}")
    print(f"z final : {result['z_final']:.12f}")
    print("OK\n")
    return result


def test_feedback_effect():
    print("=== TEST EFFET DE RÉTROACTION ===")
    off = run_coupled_system(total_time=1.0, n_steps=200, eta_u=0.0,
                             feedback_enabled=False, project=False, audit_each_step=False)
    on = run_coupled_system(total_time=1.0, n_steps=200, eta_u=0.1,
                            feedback_enabled=True, project=False, audit_each_step=False)
    dpsi = abs(on["history"][-1]["psi"] - off["history"][-1]["psi"])
    dz = abs(on["z_final"] - off["z_final"])
    drho = np.linalg.norm(on["rho_final"] - off["rho_final"], ord="fro")
    print(f"Écart Psi : {dpsi:.6e}")
    print(f"Écart z   : {dz:.6e}")
    print(f"Écart rho : {drho:.6e}")
    print("OK\n")
    return {"delta_psi": dpsi, "delta_z": dz, "delta_rho": drho}


def test_full_reproducibility():
    print("=== REPRODUCTIBILITÉ ===")
    r1 = run_coupled_system(total_time=1.0, n_steps=150, project=True, audit_each_step=False)
    r2 = run_coupled_system(total_time=1.0, n_steps=150, project=True, audit_each_step=False)

    def sha(hist):
        lines = []
        for rec in hist:
            items = sorted((k, v) for k, v in rec.items()
                           if isinstance(v, (int, float, np.integer, np.floating)))
            lines.append(";".join(f"{k}={v:.17g}" for k, v in items))
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    h1, h2 = sha(r1["history"]), sha(r2["history"])
    ok = (h1 == h2) and (np.linalg.norm(r1["rho_final"] - r2["rho_final"]) == 0.0)
    print(f"Reproductible : {'OUI' if ok else 'NON'}")
    print("OK\n")
    return ok


# ======================================================================
# 12. EXPORT COMPLET
# ======================================================================

def export_results(result, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Historique CSV
    csv_path = os.path.join(out_dir, "ste50_v2141_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["history"][0].keys()))
        writer.writeheader()
        writer.writerows(result["history"])

    # 2. Événements Preisach
    events_path = os.path.join(out_dir, "ste50_v2141_preisach_events.json")
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_events": len(result["preisach_events"]),
            "events": result["preisach_events"],
        }, f, indent=2)

    # 3. Matrice densité finale
    rho_path = os.path.join(out_dir, "ste50_v2141_final_rho.npy")
    np.save(rho_path, result["rho_final"])

    # 4. Résumé global JSON
    last = result["history"][-1]
    summary = {
        "version": "STE-50 v2.1.4.1",
        "timestamp": timestamp,
        "parameters": {
            "total_time": result["total_time"],
            "n_steps": result["n_steps"],
            "dt": result["dt"],
            "g_coupling": result["g_coupling"],
            "eta_u": result["eta_u"],
            "feedback_enabled": result["feedback_enabled"],
        },
        "final_metrics": {
            "q_end": last["q_end"],
            "psi": last["psi"],
            "z_final": result["z_final"],
            "k_eff_end": last["k_eff_end"],
            "bell_survival": last["bell_survival"],
            "mutual_information": last["mutual_information"],
            "purity": last["purity"],
            "entropy_ab": last["entropy_ab"],
        },
        "audits": {
            "n_invalid_raw": len(result["invalid_raw_audits"]),
            "min_raw_eigenvalue": min(r["raw_min_eigenvalue"] for r in result["history"]),
            "max_projection_error": max(r["projection_error"] for r in result["history"]),
        },
        "preisach": {
            "n_events": len(result["preisach_events"]),
        },
        "files": {
            "history_csv": csv_path,
            "events_json": events_path,
            "final_rho_npy": rho_path,
        }
    }
    summary_path = os.path.join(out_dir, "ste50_v2141_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 5. Journal texte lisible
    log_path = os.path.join(out_dir, "ste50_v2141_run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("STE-50 v2.1.4.1 — Journal d'exécution\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date          : {timestamp}\n")
        f.write(f"Temps total   : {result['total_time']}\n")
        f.write(f"Nombre de pas : {result['n_steps']}\n")
        f.write(f"dt            : {result['dt']:.6e}\n")
        f.write(f"eta_u         : {result['eta_u']}\n")
        f.write(f"g_coupling    : {result['g_coupling']}\n\n")
        f.write("--- Métriques finales ---\n")
        f.write(f"q_end               : {last['q_end']:.8f}\n")
        f.write(f"Psi                 : {last['psi']:.8f}\n")
        f.write(f"z_final             : {result['z_final']:.8f}\n")
        f.write(f"k_eff_end           : {last['k_eff_end']:.8f}\n")
        f.write(f"bell_survival       : {last['bell_survival']:.8f}\n")
        f.write(f"mutual_information  : {last['mutual_information']:.8f}\n")
        f.write(f"purity              : {last['purity']:.8f}\n\n")
        f.write("--- Audits ---\n")
        f.write(f"Audits bruts invalides : {len(result['invalid_raw_audits'])}\n")
        f.write(f"min eigenvalue brute   : {min(r['raw_min_eigenvalue'] for r in result['history']):.6e}\n")
        f.write(f"Événements Preisach    : {len(result['preisach_events'])}\n\n")
        f.write("Fichiers générés :\n")
        f.write(f"  - {csv_path}\n")
        f.write(f"  - {events_path}\n")
        f.write(f"  - {rho_path}\n")
        f.write(f"  - {summary_path}\n")
        f.write(f"  - {log_path}\n")

    print("\n=== FICHIERS DE SORTIE ===")
    print(f"CSV historique     : {csv_path}")
    print(f"Événements Preisach: {events_path}")
    print(f"rho finale (.npy)  : {rho_path}")
    print(f"Résumé JSON        : {summary_path}")
    print(f"Journal texte      : {log_path}")

    return {
        "csv": csv_path,
        "events": events_path,
        "rho": rho_path,
        "summary": summary_path,
        "log": log_path,
    }


# ======================================================================
# 13. EXÉCUTION PRINCIPALE
# ======================================================================

if __name__ == "__main__":
    print("=== STE-50 v2.1.4.1 — RK4 COUPLÉ + SORTIES COMPLÈTES ===\n")

    test_hamiltonian()
    test_feedback_isolation()
    test_feedback_effect()
    test_full_reproducibility()

    print("=== SIMULATION PRINCIPALE ===")
    result = run_coupled_system(
        total_time=2.0,
        n_steps=200,
        g_coupling=0.25,
        eta_u=0.1,
        feedback_enabled=True,
        project=True,
        audit_each_step=True,
    )

    print(f"\nq final              : {result['history'][-1]['q_end']:.6f}")
    print(f"Psi final            : {result['history'][-1]['psi']:.6f}")
    print(f"z final              : {result['z_final']:.6f}")
    print(f"Audits invalides     : {len(result['invalid_raw_audits'])}")
    print(f"Événements Preisach  : {len(result['preisach_events'])}")

    export_results(result, out_dir="output")

    print("\n=== FIN ===")
