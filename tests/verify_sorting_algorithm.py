import numpy as np
import time
from collections import namedtuple
from MSH_2D_Library import build_GF_library, build_GF_matrix_final, apply_GF_library_vectorized

Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta'])

def test_2D_performance_benchmark():
    params = Params(t=1.0, mu=-3.5, alpha=0.2, Delta=0.3)
    library_data = build_GF_library(params, steps=512)
    
    N_sites = 20
    
    # --- Ensure Repeated Differences ---
    while True:
        pos = np.zeros((N_sites, 2))
        for i in range(N_sites):
            # We use a small range (-4 to 4) to force "collisions" (repeated vectors)
            n1, n2 = np.random.randint(-4, 4), np.random.randint(-4, 4)
            pos[i] = [n1 + 0.5*n2, (np.sqrt(3)/2)*n2]
        
        # Calculate all possible differences
        diffs = []
        for i in range(N_sites):
            for j in range(N_sites):
                dx = round(pos[i,0] - pos[j,0], 8)
                dy = round(pos[i,1] - pos[j,1], 8)
                diffs.append((dx, dy))
        
        # Check if unique differences < total pairs
        unique_diffs = len(set(diffs))
        if unique_diffs < (N_sites**2):
            print(f"Lattice generated: {unique_diffs} unique vectors for {N_sites**2} pairs.")
            break
        else:
            print("No repeated vectors found. Re-generating lattice...")

    # --- Method A: Optimized Folding ---
    start_opt = time.time()
    G_optimized = build_GF_matrix_final(pos, pos, library_data)
    t_optimized = time.time() - start_opt
    
    # --- Method B: Brute-Force Assembly ---
    start_brute = time.time()
    G_brute = np.zeros((N_sites * 4, N_sites * 4), dtype=np.complex128)
    for i in range(N_sites):
        for j in range(N_sites):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            G_brute[4*i:4*i+4, 4*j:4*j+4] = apply_GF_library_vectorized(
                np.array([dx]), np.array([dy]), library_data
            )[0]
    t_brute = time.time() - start_brute

    # --- Results ---
    max_diff = np.max(np.abs(G_optimized - G_brute))
    speedup = t_brute / t_optimized if t_optimized > 0 else 0
    
    print(f"Brute-Force: {t_brute:.4f}s | Optimized: {t_optimized:.4f}s")
    print(f"Speedup: {speedup:.2f}x | Error: {max_diff:.2e}")
    
    assert max_diff < 1e-14

if __name__ == "__main__":
    test_2D_performance_benchmark()
