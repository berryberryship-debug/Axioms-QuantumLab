# Fichier : src/test_transmon_driven_v2.py

import numpy as np
from typing import Dict, Union, Callable

def build_system_operators(n_levels: int = 5, E_j: float = 20.0, E_c: float = 0.3) -> Dict[str, np.ndarray]:
    """
    Construit l'hamiltonien initial H_0 et les opérateurs de base pour le transmon.
    Modélisation dans la base des états de charge / oscillateur avec anharmonicité.
    """
    omega_p = np.sqrt(8.0 * E_j * E_c) - E_c
    
    # Hamiltonien de base diagonal avec correction d'anharmonicité
    H_0 = np.zeros((n_levels, n_levels), dtype=complex)
    for n in range(n_levels):
        E_n = n * omega_p - 0.5 * E_c * n * (n - 1)
        H_0[n, n] = E_n
        
    # Couplages hors-diagonaux (termes d'interaction et de transition)
    for n in range(n_levels - 1):
        coupling = 0.1 * np.sqrt(n + 1)
        H_0[n, n+1] = coupling
        H_0[n+1, n] = coupling

    return {
        "H_0": H_0,
        "omega_p": omega_p
    }

def rk4_step_td(state: np.ndarray, H_operator: Union[np.ndarray, Callable[[float], np.ndarray]], dt: float, t: float = 0.0) -> np.ndarray:
    """
    Réalise un pas d'intégration Runge-Kutta d'ordre 4 (RK4) pour l'équation de Schrödinger.
    Prend en charge un opérateur hamiltonien statique ou dépendant du temps H(t).
    """
    H = H_operator(t) if callable(H_operator) else H_operator
    
    def derivative(s: np.ndarray) -> np.ndarray:
        return -1j * np.dot(H, s)

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state / np.linalg.norm(next_state)
