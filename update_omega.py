import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, eye, kron, bmat
import csv

def gksl_liouvillian(H, ops, rates):
    H_m = csc_matrix(H)
    I = eye(H_m.shape[0], format="csc")
    L = -1j * (kron(I, H_m) - kron(H_m.T, I))
    for A, r in zip(ops, rates):
        A = csc_matrix(A)
        AdA = A.getH() @ A
        L += r * (kron(A.conjugate(), A) - 0.5 * kron(I, AdA) - 0.5 * kron(AdA.T, I))
    return csc_matrix(L)

def hamiltonian_sta_5d(t, t_max, omega_max):
    t_mid = t_max / 2.0
    sigma = t_max / 6.0
    g = omega_max * np.exp(-((t - t_mid)**2) / (2 * sigma**2))
    dg_dt = g * (-(t - t_mid) / (sigma**2))
    H_0 = np.zeros((5, 5), dtype=complex)
    for i in range(4):
        H_0[i, i+1] = g
        H_0[i+1, i] = g
    H_cd = np.zeros((5, 5), dtype=complex)
    for i in range(4):
        H_cd[i, i+1] = 1j * dg_dt / (2 * omega_max)
        H_cd[i+1, i] = -1j * dg_dt / (2 * omega_max)
    return H_0 + H_cd

if __name__ == "__main__":
    d = 5
    t_max = 15.0
    t_steps = 100
    t_c = np.linspace(0.0, t_max, t_steps)
    A_bruit = np.diag([1.0, 0.7, 0.2, -0.4, -1.5]).astype(complex)
    rho_0 = np.zeros((d, d), dtype=complex)
    rho_0[0, 0] = 1.0
    y0 = np.concatenate([rho_0.flatten(order="F").real, rho_0.flatten(order="F").imag])
    def rhs(t, y):
        L = gksl_liouvillian(hamiltonian_sta_5d(t, t_max, 4.2), [A_bruit], [0.015])
        return bmat([[L.real, -L.imag], [L.imag, L.real]], format="csc") @ y
    sol = solve_ivp(rhs, (0.0, t_max), y0, method="Radau", t_eval=t_c, rtol=1e-6, atol=1e-8)
    print("\n[+] CALCUL ET EXPORT DES DONNEES STA EN COURS...")
    with open("purete_sta.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Temps", "Purete", "Entropie_Lineaire", "Perte_Information_Pct"])
        for idx, t in enumerate(sol.t):
            yf = sol.y[:25, idx] + 1j * sol.y[25:, idx]
            rho_t = yf.reshape((d, d), order="F")
            purity = float(np.real(np.trace(rho_t @ rho_t)))
            entropy = 1.0 - purity
            writer.writerow([f"{t:.4f}", f"{purity:.6f}", f"{entropy:.6f}", f"{entropy*100:.2f}"])
    print("[+] Fichier CSV genere.")
