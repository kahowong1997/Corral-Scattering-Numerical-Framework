"""
MSH Quasi-1D Module: Hybrid Propagators & T-Matrix Renormalization
Author: Ka Ho Wong
Date: Jan 2026

This module implements the Quasi-1D (strip) workflow for MSH systems. It utilizes 
the analytical kernel to construct the hybrid Green's Function G(kx, Y) and 
applies a T-matrix scattering formalism to compute renormalized topological bands.
"""

import numba
import numpy as np
from MSH_Analytical_Kernel import (
    coefficients, denominator, get_adjugate, solve_poles, txs0, tzs0
)

@numba.njit(cache=True)
def GF_hybrid(kx, Y, params):
    """
    Constructs the hybrid Green's Function in (kx, Y) via the Cauchy Residue Theorem.
    
    Note: The analytical residue summation is derived strictly for the Y > 0 
    half-plane. To compute the Green's Function for Y < 0, we invoke the 
    Hermitian relation G(kx, -Y) = [G(kx*, Y)]^dagger.
    """
    t, mu, alpha, Delta = params.t, params.mu, params.alpha, params.Delta
    ny = abs(int(np.round(Y * 2 / np.sqrt(3))))
    
    # Mapping Y < 0 to the analytical domain Y > 0 via Hermitian symmetry
    if Y < 0:
        kx = kx.conjugate()  
        
    # --- Block 1: Chiral q ---
    c4, c3, c2, c1, c0 = coefficients(kx, t, mu, alpha, Delta) 
    z_in = solve_poles(c4, c3, c2, c1, c0)
    
    Res1 = (1.0/denominator(z_in[0], c4, c3, c2, c1)) * get_adjugate(z_in[0], kx, t, mu, alpha, Delta, False)
    Res2 = (1.0/denominator(z_in[1], c4, c3, c2, c1)) * get_adjugate(z_in[1], kx, t, mu, alpha, Delta, False)
    G_chiral_q = -Res1 * (z_in[0]**ny) - Res2 * (z_in[1]**ny)
    
    # --- Block 2: Chiral dagger (Delta -> -Delta) ---
    c4_d, c3_d, c2_d, c1_d, c0_d = coefficients(kx, t, mu, alpha, -Delta)
    z_in_d = solve_poles(c4_d, c3_d, c2_d, c1_d, c0_d)
    
    Res1_d = (1.0/denominator(z_in_d[0], c4_d, c3_d, c2_d, c1_d)) * get_adjugate(z_in_d[0], kx, t, mu, alpha, Delta, True)
    Res2_d = (1.0/denominator(z_in_d[1], c4_d, c3_d, c2_d, c1_d)) * get_adjugate(z_in_d[1], kx, t, mu, alpha, Delta, True)
    G_chiral_q_dagger = -Res1_d * (z_in_d[0]**ny) - Res2_d * (z_in_d[1]**ny)
    
    # --- Final Construction ---
    G_block = np.zeros((4, 4), dtype=np.complex128)
    G_block[0:2, 2:4] = G_chiral_q_dagger
    G_block[2:4, 0:2] = G_chiral_q
    
    U = (1.0/np.sqrt(2.0)) * (np.eye(4, dtype=np.complex128) + 1j * txs0)
    U_adj = np.ascontiguousarray(U.conj().T)
    G_mid = np.ascontiguousarray(U @ G_block)
