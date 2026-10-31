
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDIT DE STABILITÉ — DYNAMIQUE LATENTE TRANSMON V3.2-DEV

Attention :
Cet audit porte sur la stabilité de la dynamique latente
stochastique, et non sur la convergence du solveur Lindblad.
"""

import json
import numpy as np

from src.transmon_characterization_v3_2_dev import (
    TransmonCharacterizationV32Dev
)


def executer_audit_global(
    d: int = 5,
    steps: int = 150
) -> dict:

    print("=== AUDIT DE STABILITÉ — TRANSMON V3.2-DEV ===")

    banc = TransmonCharacterizationV32Dev(
        n_levels=d,
        E_j=20.0,
        E_c=0.3,
        seed=42
    )

    rapport = banc.auditer_convergence(
        iterations=steps
    )

    print("\n--- Résultats ---")
    print(json.dumps(
        rapport,
        indent=4,
        ensure_ascii=False,
        allow_nan=False
    ))

    derive = rapport["stabilite_derive"]

    if not np.isfinite(derive):
        raise ValueError("Dérive non finie.")

    if derive >= 0.05:
        raise AssertionError(
            "Stabilité non satisfaite : dérive >= 0.05."
        )

    print("\n[PASS] Seuil de dérive respecté.")

    return rapport


if __name__ == "__main__":
    executer_audit_global()
