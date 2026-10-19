import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, eye, kron, bmat

def gksl_liouvillian(H, ops, rates):
    H_m = csc_matrix(H)
    I = eye(H_m.shape[0], format="csc")
    L = -1j * (kron(I, H_m) - kron(H_m.T, I))
    for A, r in zip(ops, rates):
        A = csc_matrix(A)
        AdA = A.getH() @ A
        L += r * (kron(A.conjugate(), A) - 0.5 * kron(I, AdA) - 0.5 * kron(AdA.T, I))
    return csc_matrix(L)

def triphasic_curves(t, phase_idx):
    horizon = max(3.0 - t - [0.0, 0.3, 0.6][phase_idx], 1e-5)
    return float(np.exp(min(1.0 / horizon, 40.0)))

def hamiltonian_system(t, t_max, y_vec):
    d = 5
    rho = y_vec[:25].reshape((d, d), order="F") + 1j * y_vec[25:].reshape((d, d), order="F")
    C1 = triphasic_curves(t, 0)
    V_higgs = np.diag([-0.6 * np.abs(rho[i,i]) + 0.25 * (rho[i,i].real**2) for i in range(d)])
    phase_bohm = np.angle(np.sum(np.diag(rho, k=1)) + 1e-12)
    g = 4.2 * np.exp(-((t - (t_max/2.0))**2) / (2 * (t_max/6.0)**2))
    dg_dt = g * (-(t - (t_max/2.0)) / ((t_max/6.0)**2))
    H = np.zeros((d, d), dtype=complex)
    for i in range(4):
        H[i, i+1] = g * C1
        H[i+1, i] = g * C1
    for i in range(4):
        H[i, i+1] += 1j * dg_dt / 8.4
        H[i+1, i] += -1j * dg_dt / 8.4
    return (H + V_higgs) * np.exp(1j * phase_bohm / d)

if __name__ == "__main__":
    d, t_max = 5, 2.92
    t_c = np.linspace(0.0, t_max, 80)
    A_bruit = np.diag([1.0, 0.7, 0.2, -0.4, -1.5]).astype(complex)
    rho_0 = np.zeros((d, d), dtype=complex)
    rho_0[0, 0] = 0.5
    rho_0[1, 1] = 0.5
    rho_0[0, 1] = 0.4
    rho_0[1, 0] = 0.4
    y0 = np.concatenate([rho_0.flatten(order="F").real, rho_0.flatten(order="F").imag])
    def rhs(t, y):
        C_sum = triphasic_curves(t, 0) + triphasic_curves(t, 1)
        L = gksl_liouvillian(hamiltonian_system(t, t_max, y), [A_bruit], [0.05 / (1.0 + 0.002 * C_sum)])
        return bmat([[L.real, -L.imag], [L.imag, L.real]], format="csc") @ y
    sol = solve_ivp(rhs, (0.0, t_max), y0, method="Radau", t_eval=t_c, rtol=1e-6, atol=1e-8)
    print("CARTOGRAPHIE BOHM")
    for idx, t in enumerate(sol.t):
        yf = sol.y[:25, idx] + 1j * sol.y[25:, idx]
        rho_t = yf.reshape((d, d), order="F")
        phase_bohm = np.angle(np.sum(np.diag(rho_t, k=1)) + 1e-12)
        if idx % 10 == 0:
            print(f"Temps {t:.2f} | Phase {phase_bohm:+.3f} rad")
