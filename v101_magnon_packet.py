"""
V101 — Propagation Dynamique d'un Paquet d'Ondes de Spin (Magnons 1D)
Chaîne discrète périodique avec intégration spectrale exacte (FFT).
Audit : Contrôle explicite de la localisation (coupure périodique) et de l'énergie.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMÈTRES PHYSIQUES ET RÉSEAU
# ==========================================
J = 1.0          # Constante d'échange (meV)
S = 0.5          # Spin
a = 1.0          # Distance inter-atomique (Å)
N_sites = 1024   # Nombre de sites de la chaîne
x = np.arange(N_sites) * a

HBAR_MEV_S = 6.582119569e-13  # meV·s
A_TO_M = 1e-10                # 1 Å = 1e-10 m

# Vecteurs d'onde discrets (première zone de Brillouin périodique)
k_modes = 2 * np.pi * np.fft.fftfreq(N_sites, d=a)

# Moteur spectral : formulation numériquement stable
omega_k = (4 * J * S / HBAR_MEV_S) * np.sin(k_modes * a / 2.0)**2  # en rad/s

def v_groupe_theorique(q):
    """Vitesse de groupe théorique en Å/s"""
    return (2 * J * S * a / HBAR_MEV_S) * np.sin(q * a)

def courbure_theorique(q):
    """Courbure de dispersion d2w/dq2 en Å²/s"""
    return (2 * J * S * (a**2) / HBAR_MEV_S) * np.cos(q * a)

# ==========================================
# 2. GÉNÉRATEUR DU PAQUET D'ONDES INITIAL
# ==========================================
q0 = np.pi / (4.0 * a)         # Vecteur d'onde central
sigma_q = 0.04 * (np.pi / a)   # Largeur spectrale (écart-type)
x0 = N_sites * a / 4.0         # Positionnement initial à 1/4 de boîte

# Correction de l'audit : sigma_x0 = 1 / (2 * sigma_q)
sigma_x0 = 1.0 / (2.0 * sigma_q)
psi_0 = np.exp(-((x - x0)**2) / (4.0 * (sigma_x0**2))) * np.exp(1j * q0 * x)

# Normalisation
psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2))

# ==========================================
# 3. PROPAGATEUR ET OBSERVABLES
# ==========================================
def propager(psi_init, t):
    psi_k_0 = np.fft.fft(psi_init)
    psi_k_t = psi_k_0 * np.exp(-1j * omega_k * t)
    return np.fft.ifft(psi_k_t)

def observables_spatiales(psi):
    densite = np.abs(psi)**2
    norme = np.sum(densite)
    x_moyen = np.sum(x * densite) / norme
    x2_moyen = np.sum((x**2) * densite) / norme
    variance = x2_moyen - (x_moyen**2)
    return norme, x_moyen, variance

def energie_moyenne(psi):
    psi_k = np.fft.fft(psi)
    poids = np.abs(psi_k)**2
    poids /= np.sum(poids)
    energie_k = 4.0 * J * S * np.sin(k_modes * a / 2.0)**2
    return float(np.sum(poids * energie_k))

# ==========================================
# 4. VITESSE MOYENNE DU PAQUET
# ==========================================
psi_k_0 = np.fft.fft(psi_0)
poids_k = np.abs(psi_k_0)**2
poids_k /= np.sum(poids_k)
v_paquet_theorique = float(np.sum(poids_k * v_groupe_theorique(k_modes)))

# ==========================================
# 5. BANC DE VALIDATION NUMÉRIQUE
# ==========================================
t_sim = 1.5e-12
temps = np.linspace(0, t_sim, 50)

normes, centroides, variances, energies = [], [], [], []
probabilite_bords_max = 0.0
marge = 30 # Marge de sécurité aux bords (en sites)

for t in temps:
    psi_t = propager(psi_0, t)
    n_t, cx_t, var_t = observables_spatiales(psi_t)
    normes.append(n_t)
    centroides.append(cx_t)
    variances.append(var_t)
    energies.append(energie_moyenne(psi_t))
    
    # Contrôle de la périodicité (Audit : Paquet loin des bords)
    densite_t = np.abs(psi_t)**2
    prob_bords = np.sum(densite_t[:marge]) + np.sum(densite_t[-marge:])
    probabilite_bords_max = max(probabilite_bords_max, prob_bords)

# --- Tests de validation ---
ecart_norme_max = np.max(np.abs(np.array(normes) - 1.0))
assert ecart_norme_max < 1e-12, f"Échec norme : {ecart_norme_max}"

assert probabilite_bords_max < 1e-10, (
    f"Paquet trop proche de la coupure périodique : {probabilite_bords_max}"
)

pente_num, _ = np.polyfit(temps, centroides, 1)
erreur_vitesse = np.abs(pente_num - v_paquet_theorique) / v_paquet_theorique
assert erreur_vitesse < 0.01, f"Échec vitesse : {erreur_vitesse}"

omega_seconde = courbure_theorique(q0)
variance_theo_finale = variances[0] + ((omega_seconde * t_sim)**2) / (4.0 * variances[0])
erreur_etalement = np.abs(variances[-1] - variance_theo_finale) / variance_theo_finale
assert erreur_etalement < 0.05, f"Échec étalement : {erreur_etalement}"

ecart_energie = np.max(np.abs(np.asarray(energies) - energies[0]))
assert ecart_energie < 1e-10, f"Échec énergie : {ecart_energie}"

# ==========================================
# 6. RÉSULTATS ET FIGURE
# ==========================================
print("==================================================")
print("V101 — AUDIT DYNAMIQUE DU PAQUET DE MAGNONS")
print("==================================================")
print(f"Vitesse mode central    : {v_groupe_theorique(q0) * A_TO_M:.2f} m/s")
print(f"Vitesse moyenne paquet  : {v_paquet_theorique * A_TO_M:.2f} m/s")
print(f"Vitesse mesurée (reg.)  : {pente_num * A_TO_M:.2f} m/s")
print(f"Écart relatif vitesse   : {erreur_vitesse:.4%}")
print(f"Écart relatif étalement : {erreur_etalement:.4%} (Validité de l'approx. GVD)")
print(f"Écart max énergie       : {ecart_energie:.2e} meV")
print(f"Probabilité aux bords   : {probabilite_bords_max:.2e} (Condition de centroïde)")
print("✓ Validation réussie : conditions aux limites et invariants respectés.")
print("==================================================")

# Tracé de la propagation
fig, ax = plt.subplots(figsize=(10, 5))
for idx, t in enumerate([0, 25, 49]):
    psi_t = propager(psi_0, temps[t])
    ax.plot(x, np.abs(psi_t)**2, label=f"t = {temps[t]*1e12:.1f} ps")

ax.set_xlim(x0 - 100, x0 + 300)
ax.set_xlabel("Position x (Å)")
ax.set_ylabel(r"Densité de probabilité \(|\psi(x,t)|^2\)")
ax.set_title("Propagation et étalement du paquet de magnons")
ax.legend()
plt.tight_layout()
plt.savefig("v101_magnon_packet.png", dpi=150)
print("Figure sauvegardée : v101_magnon_packet.png")
