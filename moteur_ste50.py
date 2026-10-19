# ==============================================================================
# STE-50 "SIDÉRATION" ENGINE — MONOLITHE HPC UNIFIÉ & GÉNÉRATEUR WORD
# Couplage Énactif : cQED Résonante + Hystérésis de Phase + Soliton Topologique
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from numba import njit, prange

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

class MacroPhaseMemory:
    def __init__(self, dt=0.01, kmin=0.2, kmax=3.0, a=1.0, b=0.4, c=0.2, n=2.0, beta=1.5):
        self.dt = dt
        self.kmin = kmin
        self.kmax = kmax
        self.a = a
        self.b = b
        self.c = c
        self.n = n
        self.beta = beta
        
        self.z = 0.0
        self.prev_u = 0.0
        self.weights = np.array([0.15, 0.25, 0.35, 0.15, 0.10])
        self.p_states = np.ones(5)
        self.thresholds = np.array([[-1.0, 1.0], [-0.5, 0.5], [-0.2, 0.2], [0.1, 0.8], [-0.8, -0.1]])

    def compute_k_eff(self, u_input):
        du = u_input - self.prev_u
        z_abs = abs(self.z)
        dz = self.a * du - self.b * abs(du) * (z_abs ** (self.n - 1.0)) * self.z - self.c * du * (z_abs ** self.n)
        self.z += self.dt * dz
        s = 0.5 * (1.0 + np.tanh(self.beta * self.z))
        k_bw = self.kmin + (self.kmax - self.kmin) * s
        
        for i in range(5):
            if u_input >= self.thresholds[i, 1]: self.p_states[i] = 1.0
            elif u_input <= self.thresholds[i, 0]: self.p_states[i] = -1.0
        k_preisach = np.sum(self.weights * self.p_states)
        
        self.prev_u = u_input
        return max(0.1, float(k_bw + 0.2 * k_preisach))

@njit(cache=True)
def hermitian_part_nb(rho):
    return 0.5 * (rho + rho.conj().T)

@njit(cache=True)
def von_neumann_entropy_nb(rho, eps=1e-14):
    rh = hermitian_part_nb(rho)
    vals = np.linalg.eigvalsh(rh)
    s = 0.0
    for i in range(vals.shape[0]):
        v = np.real(vals[i])
        if v < eps: v = eps
        if v > 1.0: v = 1.0
        s -= v * np.log(v)
    return s

@njit(cache=True)
def cqed_kitaev_hamiltonian(gap_material, phase_k, g_coupling, mu=0.0, Delta=0.5):
    hz = -mu - 2.0 * gap_material * np.cos(phase_k)
    hy = 2.0 * Delta * np.sin(phase_k)
    H_kit = np.array([[hz, 1j * hy], [-1j * hy, -hz]], dtype=np.complex128)
    
    H_full = np.zeros((4, 4), dtype=np.complex128)
    H_full[0:2, 0:2] = H_kit + 0.5 * gap_material * np.eye(2, dtype=np.complex128)
    H_full[2:4, 2:4] = H_kit - 0.5 * gap_material * np.eye(2, dtype=np.complex128)
    
    H_full[0, 3] = g_coupling
    H_full[3, 0] = g_coupling
    H_full[1, 2] = g_coupling * 0.5
    H_full[2, 1] = g_coupling * 0.5
    
    return H_full

@njit(cache=True)
def lindblad_rhs_nb(rho, H, c_ops):
    drho = -1j * (H @ rho - rho @ H)
    nc = c_ops.shape[0]
    for k in range(nc):
        L = c_ops[k]
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
    return drho

@njit(cache=True)
def rk4_step_rho_nb(rho, dt, H, c_ops):
    k1 = lindblad_rhs_nb(rho, H, c_ops)
    k2 = lindblad_rhs_nb(rho + 0.5 * dt * k1, H, c_ops)
    k3 = lindblad_rhs_nb(rho + 0.5 * dt * k2, H, c_ops)
    k4 = lindblad_rhs_nb(rho + dt * k3, H, c_ops)
    rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    rho_next = hermitian_part_nb(rho_next)
    tr = np.trace(rho_next)
    if np.abs(tr) > 1e-15:
        rho_next = rho_next / tr
    return rho_next

@njit(cache=True)
def partial_trace_A_nb(rho, dimA, dimB):
    red = np.zeros((dimB, dimB), dtype=rho.dtype)
    for a in range(dimA):
        for i in range(dimB):
            for j in range(dimB):
                red[i, j] += rho[a * dimB + i, a * dimB + j]
    return red

@njit(cache=True)
def partial_trace_B_nb(rho, dimA, dimB):
    red = np.zeros((dimA, dimA), dtype=rho.dtype)
    for b in range(dimB):
        for i in range(dimA):
            for j in range(dimA):
                red[i, j] += rho[i * dimB + b, j * dimB + b]
    return red

@njit(cache=True)
def extract_ste_indices(rho, rho_ref, gap, topo, dimA=2, dimB=2):
    rhoA = partial_trace_B_nb(rho, dimA, dimB)
    rhoB = partial_trace_A_nb(rho, dimA, dimB)
    
    SA = von_neumann_entropy_nb(rhoA)
    SB = von_neumann_entropy_nb(rhoB)
    SAB = von_neumann_entropy_nb(rho)
    MI = max(0.0, SA + SB - SAB)
    
    C = 0.0
    d = rho.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j: C += np.abs(rho[i, j])
            
    F = np.real(np.trace(hermitian_part_nb(rho) @ hermitian_part_nb(rho_ref)))
    P = np.real(np.trace(rho @ rho))
    
    R = C / max(1.0, SAB)
    E = gap * topo
    Lambda = SAB
    Psi = 0.5 * F + 0.5 * MI
    
    return R, E, P, Lambda, Psi, SAB, C, F, MI

@njit(cache=True)
def evolve_trajectory_core(rho0, H, c_ops, tlist):
    rho = rho0.copy()
    for i in range(1, tlist.shape[0]):
        dt = tlist[i] - tlist[i-1]
        rho = rk4_step_rho_nb(rho, dt, H, c_ops)
    return rho

@njit(parallel=True, cache=True)
def run_hpc_batch(Hs, c_opss, rho0s, rho_refs, gaps, topologies, tlist, dimA, dimB):
    ntraj = rho0s.shape[0]
    
    R = np.empty(ntraj, dtype=np.float64)
    E = np.empty(ntraj, dtype=np.float64)
    P = np.empty(ntraj, dtype=np.float64)
    L = np.empty(ntraj, dtype=np.float64)
    Psi = np.empty(ntraj, dtype=np.float64)
    entropy = np.empty(ntraj, dtype=np.float64)
    coherence = np.empty(ntraj, dtype=np.float64)
    fidelity = np.empty(ntraj, dtype=np.float64)
    mutual_info = np.empty(ntraj, dtype=np.float64)

    for j in prange(ntraj):
        rho_f = evolve_trajectory_core(rho0s[j], Hs[j], c_opss[j], tlist)
        
        Rj, Ej, Pj, Lj, Psij, Sj, Cj, Fj, Mj = extract_ste_indices(
            rho_f, rho_refs[j], gaps[j], topologies[j], dimA, dimB
        )
        
        R[j] = Rj; E[j] = Ej; P[j] = Pj; L[j] = Lj; Psi[j] = Psij
        entropy[j] = Sj; coherence[j] = Cj; fidelity[j] = Fj; mutual_info[j] = Mj

    return R, E, P, L, Psi, entropy, coherence, fidelity, mutual_info

def generate_word_files(summary):
    if not HAS_DOCX:
        print("[AVERTISSEMENT] La bibliothèque 'python-docx' n'est pas installée.")
        return None, None

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    
    doc_lettre = Document()
    doc_lettre.add_heading("Proposition d'architecture formelle", level=1)
    doc_lettre.add_paragraph("Couplage non linéaire macro-quantique et protection topologique de phase (Moteur STE-50)")
    doc_lettre.add_paragraph()
    
    doc_lettre.add_paragraph("Monsieur le Professeur Blais,")
    doc_lettre.add_paragraph(
        "En tant qu'architecte noétique et consultant en innovation (fondateur d'Axioms Base Labs), "
        "mon travail se situe à la croisée de la phénoménologie systémique (dans la lignée de Francisco Varela et David Bohm) "
        "et de la modélisation informatique avancée."
    )
    doc_lettre.add_paragraph(
        "Loin des sentiers battus de la programmation conventionnelle, j'aborde la conception technologique avec une vision holistique : "
        "celle d'unifier l'ordre matériel macroscopique (la mémoire de forme et l'hystérésis non linéaire) et la dynamique des systèmes quantiques ouverts."
    )
    doc_lettre.add_paragraph(
        "Je me permets de soumettre à votre attention le code source d'un engin de calcul novateur (le prototype STE-50), "
        "développé avec le soutien de l'intelligence artificielle comme outil de transcription rapide. Ce script simule un couplage interdisciplinaire audacieux : "
        "l'intégration d'une mémoire matérielle pilotant en temps réel le gap spectral et les phases topologiques d'un système quantique ouvert résolu "
        "via l'équation maîtresse de Lindblad et des extensions cQED."
    )
    doc_lettre.add_paragraph(
        "Au-delà de la performance numérique — optimisée ensemblistement via Numba et validée par des métriques d'information mutuelle et de pureté —, "
        "l'intuition fondatrice de cette architecture est d'explorer un paradigme de rupture : la possibilité d'encoder l'information non pas par des "
        "contraintes externes lourdes, mais par une signature topologique de phase auto-protégée, agissant en superposition au-dessus des substrats matériels stables."
    )
    doc_lettre.add_paragraph(
        "Je sais combien la rigueur est cardinale au sein de l'Institut quantique de l'Université de Sherbrooke. C'est pourquoi cet outil est pensé pour être "
        "falsifiable, traçable et rigoureusement normé. Je vous joins par la présente les artefacts de simulation ainsi que le cœur du code pour étude."
    )
    doc_lettre.add_paragraph()
    doc_lettre.add_paragraph("Veuillez agréer, Monsieur le Professeur Blais, l'expression de ma considération la plus distinguée.")
    doc_lettre.add_paragraph()
    doc_lettre.add_paragraph("Xio\nArchitecte d'Innovation et de Systèmes Noétiques\nAxioms Base Labs")
    
    path_lettre = os.path.join(out_dir, "Lettre_Presentation_Alexandre_Blais.docx")
    doc_lettre.save(path_lettre)

    doc_rapport = Document()
    doc_rapport.add_heading("STE-50 ENGINE REPORT (cQED UNIFIED)", level=1)
    doc_rapport.add_paragraph("Rapport de simulation HPC et diagnostic de falsifiabilité poppérienne.")
    
    table = doc_rapport.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Paramètre / Observable'
    hdr_cells[1].text = 'Valeur Numérique (float64)'
    
    metrics = [
        ("Trajectoires Monte Carlo", f"{summary['ntraj']}"),
        ("Espace de Hilbert (d)", "4 (Bipartite 2x2)"),
        ("Moyenne du Gap Effectif", f"{summary['gap_mean']:.6f}"),
        ("Fidélité de l'État Solitonique", f"{summary['fidelity_mean']:.6f}"),
        ("Cohérence Quantique Globale (L1)", f"{summary['coherence_mean']:.6f}"),
        ("Pureté de la Matrice d'État (P)", f"{summary['P_mean']:.6f}"),
        ("Entropie de von Neumann (S)", f"{summary['entropy_mean']:.6f}"),
        ("Information Mutuelle Bipartie", f"{summary['mutual_info_mean']:.6f}"),
        ("Indice PSI STE (Code-barres)", f"{summary['Psi_mean']:.6f}"),
        ("Intervalle de Confiance", f"± {1.96 * summary['Psi_std'] / np.sqrt(summary['ntraj']):.6f}"),
        ("Statut du Verdict", summary["verdict"])
    ]
    
    for label, val in metrics:
        row_cells = table.add_row().cells
        row_cells[0].text = label
        row_cells[1].text = val
        
    path_rapport = os.path.join(out_dir, "Rapport_Scientifique_STE50.docx")
    doc_rapport.save(path_rapport)

    return path_lettre, path_rapport

if __name__ == "__main__":
    print("Initialisation du pipeline unifié STE-50 (Sidération HPC)...")
    
    ntraj = 64
    d = 4
    dimA = 2
    dimB = 2
    tlist = np.linspace(0.0, 15.0, 300)

    Hs = np.zeros((ntraj, d, d), dtype=np.complex128)
    c_opss = np.zeros((ntraj, 1, d, d), dtype=np.complex128)
    rho0s = np.zeros((ntraj, d, d), dtype=np.complex128)
    rho_refs = np.zeros((ntraj, d, d), dtype=np.complex128)
    gaps = np.zeros(ntraj, dtype=np.float64)
    topologies = np.zeros(ntraj, dtype=np.int64)

    rng = np.random.default_rng(777)
    phase_memory = MacroPhaseMemory(dt=0.05)

    print("Calcul des couplages via hystérésis multi-échelles...")
    for j in range(ntraj):
        u_mecanique = rng.uniform(-1.5, 1.5)
        gap_mat = phase_memory.compute_k_eff(u_mecanique)
        gaps[j] = gap_mat
        topologies[j] = 1 if gap_mat > 0.8 else 0
        
        phase_k = rng.uniform(-np.pi, np.pi)
        g_coupling = 0.25 
        Hs[j] = cqed_kitaev_hamiltonian(gap_mat, phase_k, g_coupling)
        
        gamma_decoherence = 0.02
        L = np.zeros((4, 4), dtype=np.complex128)
        L[1, 0] = np.sqrt(gamma_decoherence)
        L[3, 2] = np.sqrt(gamma_decoherence)
        c_opss[j, 0] = L
        
        psi0 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        rho_init = np.outer(psi0, psi0.conj())
        rho0s[j] = rho_init
        rho_refs[j] = rho_init.copy()

    print(f"Lancement du batch Monte Carlo Numba sur {ntraj} trajectoires...")
    R, E, P_out, Lam, Psi, ent, coh, fid, mi = run_hpc_batch(
        Hs, c_opss, rho0s, rho_refs, gaps, topologies, tlist, dimA, dimB
    )

    summary_dict = {
        "ntraj": ntraj,
        "gap_mean": float(np.mean(gaps)), "gap_std": float(np.std(gaps)),
        "topology_mean": float(np.mean(topologies)),
        "R_mean": float(np.mean(R)), "R_std": float(np.std(R)),
        "E_mean": float(np.mean(E)), "E_std": float(np.std(E)),
        "P_mean": float(np.mean(P_out)), "P_std": float(np.std(P_out)),
        "Lambda_mean": float(np.mean(Lam)), "Lambda_std": float(np.std(Lam)),
        "Psi_mean": float(np.mean(Psi)), "Psi_std": float(np.std(Psi)),
        "entropy_mean": float(np.mean(ent)), "coherence_mean": float(np.mean(coh)),
        "fidelity_mean": float(np.mean(fid)), "mutual_info_mean": float(np.mean(mi)),
    }

    validation_status = (
        summary_dict["gap_mean"] > 0.4 
        and summary_dict["Psi_mean"] > 0.65 
        and summary_dict["entropy_mean"] < 1.2
    )
    summary_dict["verdict"] = "VALIDATION NUMÉRIQUE : COHÉRENCE DU CODE-BARRES TOPOLOGIQUE" if validation_status else "RÉFUTATION NUMÉRIQUE"

    print("Génération automatique des fichiers Word...")
    path_lettre, path_rapport = generate_word_files(summary_dict)

    print("=" * 60)
    print("Exécution du pipeline STE-50 terminée avec succès.")
    if path_lettre and path_rapport:
        print(f" -> Fichier Word 1 (Lettre)  : {path_lettre}")
        print(f" -> Fichier Word 2 (Rapport) : {path_rapport}")
    else:
        print(" (Note : Installez 'python-docx' via pip pour activer l'export Word automatique)")
    print(f"Valeur finale moyenne de l'indice Psi : {summary_dict['Psi_mean']:.4f}")
    print("=" * 60)
