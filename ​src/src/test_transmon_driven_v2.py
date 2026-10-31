#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRANSMON — NOYAU DYNAMIQUE (V2)
================================
Noyau mathématique et physique (Hamiltonien de Duffing, 
dissipateurs de Lindblad, et intégrateur RK4 non autonome).
"""

import numpy as np

# Constantes physiques (rad/ns et ns^-1)
OMEGA = 5.0 * 2.0 * np.pi
ALPHA = 0.3 * 2.0 * np.pi
GAMMA_10 = 0.002
GAMMA_21 = 0.004

# Paramètres par défaut de l'impulsion
SIGMA = 3.0
T_CENTER = 12.0
PHASE = 0.0
AMP = np.pi / (SIGMA * np.sqrt(2.0 * np.pi))

def build_system_operators(d=4):
    energies = [n * OMEGA - 0.5 * n * (n - 1) * ALPHA for n in range(d)]
    H0 = np.diag(energies).astype(complex)

    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    
    X = a + a.conjugate().T

    c_ops = []
    if d > 1:
        c10 = np.zeros((d, d), dtype=complex)
        c10[0, 1] = np.sqrt(GAMMA_10)
        c_ops.append(c10)
    if d > 2:
        c21 = np.zeros((d, d), dtype=complex)
        c21[1, 2] = np.sqrt(GAMMA_21)
        c_ops.append(c21)

    return H0, X, c_ops

def gaussian_envelope(t, amp, t_center, sigma):
    return amp * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)

def drive_signal(t, omega_d, amp, t_center, sigma, phase):
    return gaussian_envelope(t, amp, t_center, sigma) * np.cos(omega_d * t + phase)

def lindblad_rhs_td(t, rho, H0, X, c_ops, omega_d, include_dissipators, amp, t_center, sigma, phase):
    Ht = H0 + drive_signal(t, omega_d, amp, t_center, sigma, phase) * X
    commutator = Ht @ rho - rho @ Ht
    result = -1j * commutator

    if include_dissipators:
        for C in c_ops:
            Cd = C.conjugate().T
            CdC = Cd @ C
            result += C @ rho @ Cd - 0.5 * (CdC @ rho + rho @ CdC)

    return result

def rk4_step_td(rho, t, dt, H0, X, c_ops, omega_d, include_dissipators=True, 
                amp=AMP, t_center=T_CENTER, sigma=SIGMA, phase=PHASE):
    k1 = lindblad_rhs_td(t, rho, H0, X, c_ops, omega_d, include_dissipators, amp, t_center, sigma, phase)
    k2 = lindblad_rhs_td(t + dt / 2.0, rho + dt * k1 / 2.0, H0, X, c_ops, omega_d, include_dissipators, amp, t_center, sigma, phase)
    k3 = lindblad_rhs_td(t + dt / 2.0, rho + dt * k2 / 2.0, H0, X, c_ops, omega_d, include_dissipators, amp, t_center, sigma, phase)
    k4 = lindblad_rhs_td(t + dt, rho + dt * k3, H0, X, c_ops, omega_d, include_dissipators, amp, t_center, sigma, phase)
    
    return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
