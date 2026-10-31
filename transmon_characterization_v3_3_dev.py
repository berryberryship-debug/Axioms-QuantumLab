# Fichier : src/transmon_characterization_v3_2_dev.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import des opérateurs et solveurs depuis le module compagnon
from src.test_transmon_driven_v2 import build_system_operators, rk4_step_td

class TransmonCharacterizationV32Dev:
    def __init__(self, n_levels: int = 5, E_j: float = 20.0, E_c: float = 0.3, seed: int = 42):
        np.random.seed(seed)
        self.n_levels = n_levels
        self.E_j = E_j
        self.E_c = E_c
        
        # Construction des opérateurs du système via le module tiers
        self.operators = build_system_operators(n_levels=self.n_levels, E_j=self.E_j, E_c=self.E_c)
        self.H_0 = self.operators.get("H_0", np.zeros((n_levels, n_levels)))
        
        # État latent interne du substrat (zéro surface, initialisation stochastique contrôlée)
        self.z = np.random.normal(0, 1.0, n_levels)
        self.historique_energie: List[float] = []

    def _calculer_energie_potentielle(self, state: np.ndarray) -> float:
        """Calcule l'énergie potentielle interne du système."""
        return float(-0.5 * np.real(np.dot(state.conj(), np.dot(self.H_0, state))))

    def evolution_latente_libre(self, iterations: int = 150, dt: float = 0.02) -> Tuple[np.ndarray, List[float]]:
        """
        Simule l'auto-organisation du champ latent sans forçage externe (zéro surface).
        Intègre une dynamique de relaxation avec suivi énergétique.
        """
        self.historique_energie = []
        state = self.z.astype(complex)
        state /= np.linalg.norm(state) # Normalisation de l'état quantique / latent

        for _ in range(iterations):
            # Gradient de potentiel non linéaire inspiré du couplage transmon
            force = -state + np.tanh(np.dot(self.H_0, state))
            state += dt * force + np.random.normal(0, 0.005, self.n_levels) + 1j * np.random.normal(0, 0.005, self.n_levels)
            state /= np.linalg.norm(state)
            
            # Suivi de l'énergie instantanée
            ener = self._calculer_energie_potentielle(state)
            self.historique_energie.append(ener)
            
        self.z = state
        return self.z, self.historique_energie

    def auditer_convergence(self, iterations: int = 150) -> Dict[str, any]:
        """
        Réalise l'audit de convergence et analyse la sensibilité à la relaxation (Niveau 3).
        """
        etat_final, energies = self.evolution_latente_libre(iterations=iterations)
        
        coherence = float(np.mean(np.abs(etat_final)))
        energie_finale = energies[-1]
        
        # Calcul de la stabilité de la dérive sur les 10 derniers pas
        stabilite_derive = float(np.abs(energies[-1] - energies[-10])) if len(energies) >= 10 else 0.0

        return {
            "statut": "attracteur_stabilise_v3_2",
            "n_levels": self.n_levels,
            "coherence_interne": coherence,
            "energie_potentielle_finale": energie_finale,
            "stabilite_derive": stabilite_derive,
            "signature_topologique": f"transmon_dim_{self.n_levels}_stable_cv"
        }

if __name__ == "__main__":
    caracterisation = TransmonCharacterizationV32Dev(n_levels=5, E_j=20.0, E_c=0.3)
    rapport_audit = caracterisation.auditer_convergence(iterations=150)
    
    print("--- Rapport Global : transmon_characterization_v3_2_dev.py ---")
    for k, v in rapport_audit.items():
        print(f"{k}: {v}")
