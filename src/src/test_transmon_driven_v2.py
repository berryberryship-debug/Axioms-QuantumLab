# Fichier : src/test_transmon_driven_v2.py

import numpy as np
from typing import Tuple, List, Callable, Union

def build_system_operators(d: int = 5, E_j: float = 20.0, E_c: float = 0.3, gamma_rel: float = 0.01, gamma_phi: float = 0.005) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Construit les opérateurs du système transmon pour la dynamique de Lindblad :
    - H0 : Hamiltonien de Duffing diagonal avec anharmonicité.
    - X : Opérateur de charge / couplage.
    - c_ops : Liste des opérateurs de collapse (relaxation et déphasage pur).
    """
    omega_p = np.sqrt(8.0 * E_j * E_c) - E_c
    
    # Hamiltonien diagonal (Duffing / transmon)
    H0 = np.zeros((d, d), dtype=complex)
    for n in range(d):
        E_n = n * omega_p - 0.5 * E_c * n * (n - 1)
        H0[n, n] = E_n

    # Opérateur de charge X (tridiagonal, couplage de voisinage)
    X = np.zeros((d, d), dtype=complex)
    for n in range(d - 1):
        val = np.sqrt(n + 1)
        X[n, n+1] = val
        X[n+1, n] = val

    # Opérateurs de collapse de Lindblad
    c_ops = []
    # Opérateur de relaxation (transition de |n> vers |n-1>)
    L_rel = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        L_rel[n-1, n] = np.sqrt(n * gamma_rel)
    c_ops.append(L_rel)

    # Opérateur de déphasage pur
    L_phi = np.zeros((d, d), dtype=complex)
    for n in range(d):
        L_phi[n, n] = n * gamma_phi
    c_ops.append(L_phi)

    return H0, X, c_ops

def rk4_step_lindblad(rho: np.ndarray, H_operator: Union[np.ndarray, Callable[[float], np.ndarray]], c_ops: List[np.ndarray], dt: float, t: float = 0.0) -> np.ndarray:
    """
    Réalise un pas d'intégration Runge-Kutta d'ordre 4 (RK4) pour l'équation de Lindblad :
    drho/dt = -i [H, rho] + sum_k (L_k rho L_k^dagger - 0.5 {L_k^dagger L_k, rho})
    """
    H = H_operator(t) if callable(H_operator) else H_operator

    def lindblad_rhs(r: np.ndarray) -> np.ndarray:
        # Partie unitaire : -i [H, r]
        comm = -1j * (np.dot(H, r) - np.dot(r, H))
        
        # Partie dissipative de Lindblad
        dissipator = np.zeros_like(r, dtype=complex)
        for L in c_ops:
            L_dag = L.conj().T
            L_dag_L = np.dot(L_dag, L)
            term1 = np.dot(L, np.dot(r, L_dag))
            term2 = 0.5 * (np.dot(L_dag_L, r) + np.dot(r, L_dag_L))
            dissipator += term1 - term2
            
        return comm + dissipator

    k1 = lindblad_rhs(rho)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2)
    k4 = lindblad_rhs(rho + dt * k3)

    rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Restauration de la trace (normalisation de la matrice densité)
    rho_next = (rho_next + rho_next.conj().T) / 2.0
    rho_next /= np.trace(rho_next)
    
    return rho_next
