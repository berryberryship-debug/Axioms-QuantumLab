#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Spectroscopie Transmon & Validation des Risques d'Ionisation
"""
import sys
from pathlib import Path

racine_labo = Path(__file__).resolve().parent.parent
sys.path.append(str(racine_labo))

from core.transmon import TransmonQubit
import numpy as np

def executer_spectroscopie():
    print("=== SPECTROSCOPIE TRANSMON & IONISATION (Dumas et al., 2024) ===")
    qubit = TransmonQubit(E_J=18.0, E_C=0.22)
    
    ng_plage = np.linspace(0.0, 0.5, 6)
    anharmonicite = qubit.calculer_anharmonicite()
    print(f"[*] Anharmonicité mesurée : {anharmonicite:.4f} GHz")
    
    print("\n[Balayage n_g] Niveaux d'énergie (GHz) :")
    for ng in ng_plage:
        evals, _ = qubit.spectre(n_g=ng, num_niveaux=3)
        print(f"  n_g={ng:.2f} | E0={evals[0]:.3f}, E1={evals[1]:.3f}, E2={evals[2]:.3f}")

    puissance_test = 0.90
    risque = qubit.risque_ionisation_lecture(puissance_test)
    print(f"\n[*] Test d'ionisation pour une puissance de lecture {puissance_test} :")
    print(f"    Probabilité d'ionisation induite : {risque * 100:.2f}%")

if __name__ == "__main__":
    executer_spectroscopie()
