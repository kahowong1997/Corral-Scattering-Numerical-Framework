import numpy as np
from collections import namedtuple
from MSH_2D_Library import (
    build_GF_library, build_GF_matrix_final, build_corral_basis, build_T_matrix
    )

tzs0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=np.complex128)

# 1. Setup Parameters
Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta'])

def verify_T_matrix():
    params = Params(t=1.0, mu=-6.5, alpha=0.21, Delta=0.38)
    V_strength = 5.0  
    
    # 1. Generate a small 3-site triangular cluster for the test
    pos = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2]
    ])
    
    print("Step 1: Generating Green's Function Substrate...")
    library_data = build_GF_library(params, steps=256)
    G0 = build_GF_matrix_final(pos, pos, library_data)
    dim = G0.shape[0]

    # 2. Construct the Scattering Potential Matrix V
    # V is onsite: -V_strength * tau_z
    V_mat = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(len(pos)):
        V_mat[4*i:4*i+4, 4*i:4*i+4] = -V_strength * tzs0

    print(f"Step 2: Computing T-matrix via Chiral Basis Decomposition (V={V_strength})...")
    basis_data = build_corral_basis(G0)
    T = build_T_matrix(V_strength, basis_data)

    # 3. Verify the Dyson Identity: T = V + V @ G0 @ T
    # This is the fundamental definition of the T-matrix.
    # We check if T - (V @ G0 @ T) - V = 0
    
    # Mathematical check: T must satisfy (I - V*G0)T = V
    I = np.eye(dim, dtype=np.complex128)
    dyson_lhs = (I - V_mat @ G0) @ T
    
    max_diff = np.max(np.abs(dyson_lhs - V_mat))
    
    print("\n" + "="*45)
    print("T-MATRIX DYSON IDENTITY CHECK")
    print("="*45)
    print(f"Matrix Dimension:    {dim}x{dim}")
    print(f"Max Identity Error:  {max_diff:.2e}")
    print("="*45)
    
    assert max_diff < 1e-12, f"T-matrix does not satisfy Dyson Equation! Error: {max_diff}"
    print("✅ VERIFICATION SUCCESSFUL: Chiral decomposition is mathematically exact.")

if __name__ == "__main__":
    verify_T_matrix()
