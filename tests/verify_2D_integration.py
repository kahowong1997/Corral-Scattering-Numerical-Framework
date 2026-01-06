import numpy as np
from MSH_2D_Library import build_GF_library, apply_GF_library_vectorized
from MSH_Quasi_1D import GF_hybrid  # Your ANALYTICAL, verified function
from collections import namedtuple

Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta'])

def test_2D_integration_consistency():
    params = Params(t=1.0, mu=-3.5, alpha=0.2, Delta=0.3)
    
    # 1. Setup spatial coordinates for testing
    X_test = -6.5
    Y_test = 7*np.sqrt(3)/2
    
    # 2. Method A: 2D Library Vectorized Integration (Analytical Summation)
    steps = 1024
    print(f"Building Library and Computing 2D G(X={X_test}, Y={Y_test})...")
    library_data = build_GF_library(params, steps=steps)
    G_2D_library = apply_GF_library_vectorized(np.array([X_test]), np.array([Y_test]), library_data)[0]
    
    # 3. Method B: Numerical Integration of the Analytical GF_hybrid
    # We integrate: (1/2pi) * \int GF_hybrid(kx) * exp(i * kx * X) dkx
    k_vec = np.linspace(0, 2*np.pi, steps)
    dk = k_vec[1] - k_vec[0]
    G_sum = np.zeros((4, 4), dtype=np.complex128)
    
    print("Integrating analytical GF_hybrid over kx...")
    for i in range(steps):
        kx = k_vec[i]
        weight = 0.5 if (i == 0 or i == steps - 1) else 1.0
        
        # Call your verified analytical 1D function
        G_kx = GF_hybrid(kx, Y_test, params)
        
        # Apply the longitudinal phase shift
        G_sum += weight * G_kx * np.exp(1j * kx * X_test)
        
    G_reference = G_sum * (dk / (2 * np.pi))
    
    # 4. Consistency Check
    diff = np.max(np.abs(G_2D_library - G_reference))
    print(f"\n2D Integration Consistency Check:")
    print(f"Max Absolute Difference: {diff:.2e}")
    
    # Accuracy should be limited only by the integration step size (approx 10^-12)
    assert diff < 1e-10, f"2D Integration logic inconsistent with 1D analytical! Error: {diff}"
    print("✅ 2D Integration Verified: Vectorized residues match Integrated Analytical GF.")

if __name__ == "__main__":
    test_2D_integration_consistency()
