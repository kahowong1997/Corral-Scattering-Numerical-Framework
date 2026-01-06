# Corral-Scattering-Numerical-Model
Numerical framework for computing Green's Functions under renormalization from scattering potentials. Optimized for Magnet-Superconductor Hybrid (MSH) systems using Python (Numba).

# Project Overview
This framework provides a suite of high-performance tools designed to obtain the topological Hamiltonian for MSH systems. The core of the project focuses on the Renormalization of the zero energy Green's Functions due to corral scattering potentials, allowing for the precise identification of topological phases and Majorana signatures in real space.

The framework is built on analytical foundations, using Cauchy's residue theorem to bypass the computational bottlenecks of standard numerical integration.

# Framework Modules
## 1. Bare Green's Function Engine ('MSH_Bare_GF.py')
This is the fundamental kernel of the framework. It computes the bare **Hybrid Propagator** $G(k_x, Y)$ using a theoretically-informed analytical approach.

### **Methodology & Features**
* **Mixed-Representation Construction:** Directly computes the $(k_x, Y)$ Green's Function, preserving momentum-space resolution in the longitudinal direction while resolving the transverse spatial coordinate.
* **Cauchy Residue Theorem:** Analytically evaluates the $k_y \to Y$ Fourier transform, bypassing the need for computationally intensive numerical integration over the Brillouin Zone.
* **Analytical domain resolution:** The summation is derived strictly for $Y > 0$. The engine resolves the $Y < 0$ sector via the Hermitian relation $G(k_x, -Y) = [G(k_x^*, Y)]^\dagger$.
* **Pole Analysis:** Utilizes a companion matrix solver to identify characteristic evanescent modes in the complex $z$-plane, ensuring numerical stability for long-range spatial correlations.

### **Mathematical Definition**
The engine evaluates the Fourier transform from momentum space to the hybrid $(k_x, Y)$ representation:

$$G(k_x, Y) = \frac{\sqrt{3}}{4\pi} \int_{-\frac{2\pi}{\sqrt{3}}}^{\frac{2\pi}{\sqrt{3}}} G(k_x, k_y) e^{ik_y Y} dk_y$$

By identifying the poles $z_i$ of the characteristic polynomial $P(z)$ where $z = e^{ik_y a_y}$, the integral is computed via the residue sum:

$$G(k_x, Y) = \sum_{z_i < 1} \text{Res}\left[-\frac{\text{Adj}(H(z))}{P(z)} z^{n_y-1}, z_i\right]$$
