#!/usr/bin/env python3
"""
AXIOMS BASE LABS - STE-60
Démo d'exécution du Hamiltonien Hubbard 2x2.
"""

from mypkg.hubbard import HubbardParams, build_basis, build_hamiltonian


def main():
    params = HubbardParams(t=1.0, U=4.0, V=1.0, lx=2, ly=2, periodic=False)
    basis = build_basis(n_sites=4, n_up=1, n_down=1)
    H = build_hamiltonian(basis=basis, params=params)
    print(f"[OK] Hamiltonien Hubbard construit avec succès. Dimensions : {H.shape}")

if __name__ == "__main__":
    main()
