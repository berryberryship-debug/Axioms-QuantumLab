import numpy as np

def simuler_lindblad_decoherence(chi, kappa=0.01, gamma_phi=0.005):
    print(f"🌀 Simulation Lindblad avec chi = {chi*1000:.2f} MHz")
    print(f"  • Taux de fuite cavité (kappa) : {kappa} GHz")
    print(f"  • Taux de déphasage qubit (gamma_phi) : {gamma_phi} GHz")
    
    t2_star = 1.0 / gamma_phi if gamma_phi > 0 else float('inf')
    visibilite = abs(chi) / np.sqrt(chi**2 + kappa**2)
    
    print(f"  • Temps de cohérence T2* estimé : {t2_star * 1000:.1f} ns")
    print(f"  • Visibilité de lecture dispersive : {visibilite:.4f}")
    
    return visibilite

if __name__ == "__main__":
    simuler_lindblad_decoherence(chi=-0.0028)
