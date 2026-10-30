"""
V100 — Dispersion des Magnons 1D (Ondes de spin)
Test d'un modèle de Heisenberg ferromagnétique 1D (Approximation linéaire).
Audit physique : Prise en compte de la constante de Planck réduite (hbar) 
pour le calcul de la vitesse physique en m/s.
"""
import matplotlib
matplotlib.use('Agg') # Pour Termux / serveur sans affichage
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMÈTRES PHYSIQUES ET CONSTANTES
# ==========================================
J = 1.0       # Constante de couplage d'échange (meV)
S = 0.5       # Spin (1/2 pour un électron)
a = 1.0       # Distance inter-atomique (Angström)
N = 200       # Nombre de points

# Constante de Planck réduite en meV·s
HBAR_MEV_S = 6.582119569e-13 
# Conversion Angström -> mètres
A_TO_M = 1e-10

# ==========================================
# 2. CALCUL DE LA DISPERSION
# ==========================================
q = np.linspace(-np.pi/a, np.pi/a, N)

# Relation de dispersion : E(q) = 2JS(1 - cos(qa)) [en meV]
E_q = 2 * J * S * (1 - np.cos(q * a))

# Pente énergétique : dE/dq = 2JSa sin(qa) [en meV·Å]
dE_dq = 2 * J * S * a * np.sin(q * a)

# Vitesse physique : v = (1/hbar) * dE/dq
v_g_physique = (dE_dq / HBAR_MEV_S) * A_TO_M

# ==========================================
# 3. AFFICHAGE DES RÉSULTATS
# ==========================================
print("========================================")
print("V100 — DISPERSION DES MAGNONS (1D)")
print("========================================")
print(f"Couplage J = {J} meV, Spin S = {S}")
print(f"Énergie max (bord de zone) : {E_q.max():.4f} meV")
print(f"Pente énergétique max : {np.max(np.abs(dE_dq)):.4f} meV·Å")
print(f"Vitesse physique max : {np.max(np.abs(v_g_physique)):.2f} m/s")
print("Note : À q=0 (Gamma), E=0 et v=0 (Dispersion quadratique)")
print("========================================")

# ==========================================
# 4. VISUALISATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panneau 1 : Dispersion E(q)
ax1.plot(q, E_q, 'b-', lw=2)
ax1.set_title("Dispersion des Magnons (1D)")
ax1.set_xlabel("Vecteur d'onde q")
ax1.set_ylabel(r"Énergie \(\hbar\omega(q)\) (meV)")
ax1.set_xticks([-np.pi/a, 0, np.pi/a])
ax1.set_xticklabels([r'\(-\pi/a\)', '0 (Γ)', r'\(\pi/a\)'])
ax1.grid(True, alpha=0.3)

# Panneau 2 : Vitesse de groupe physique
ax2.plot(q, v_g_physique, 'r-', lw=2)
ax2.set_title("Vitesse de groupe (Ondes de spin)")
ax2.set_xlabel("Vecteur d'onde q")
ax2.set_ylabel(r"Vitesse \(v_g(q)\) (m/s)")
ax2.set_xticks([-np.pi/a, 0, np.pi/a])
ax2.set_xticklabels([r'\(-\pi/a\)', '0 (Γ)', r'\(\pi/a\)'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("v100_magnon_dispersion.png", dpi=150)
print("Figure sauvegardée : v100_magnon_dispersion.png")
