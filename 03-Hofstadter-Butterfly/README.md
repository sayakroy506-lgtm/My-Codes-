# 03 · Hofstadter Butterfly (Harper Model)

Spectrum of a 2D square-lattice tight-binding electron in a perpendicular magnetic field. We plot the fractal Hofstadter butterfly versus flux, and the three magnetic bands at flux $\Phi/\Phi_0=1/3$.

## Model

**Hamiltonian** (Landau gauge, hopping $t=1$, flux per plaquette $\alpha=\Phi/\Phi_0$):

$$H=-t\sum_{\langle ij\rangle}e^{iA_{ij}}\,c_i^\dagger c_j+\text{h.c.}$$

**Harper equation** (after Fourier transforming along $y$, $k_y$ is conserved):

$$E\,\psi_m=\psi_{m+1}+\psi_{m-1}+2\cos(2\pi\alpha m+k_y)\,\psi_m$$

**Rational flux** $\alpha=p/q$ (with $p,q$ coprime): the potential repeats every $q$ sites, so the magnetic unit cell has $q$ sites and the spectrum splits into $q$ bands. The Bloch condition on the $q\times q$ matrix is

$$\psi_{m+q}=e^{iqk_x}\psi_m,\qquad k_x,k_y\in\left[0,\tfrac{2\pi}{q}\right)$$

**Butterfly:** plot all eigenvalues of $H(\alpha=p/q,k_x,k_y)$ for every coprime $p/q$ up to $q_{\max}$.

**Flux $\alpha=1/3$:** three bands with edges (in units of $t$)

$$[-(1+\sqrt3),\,-2],\quad[-(\sqrt3-1),\,\sqrt3-1],\quad[2,\,1+\sqrt3]$$

and the total spectrum always lies in $E\in[-4,4]$.

---

## Code, block by block

### 1. Imports and PRB-style plot settings
Same serif, inward-tick style and colours as the rest of the repo.

```python
import numpy as np
from math import gcd
import matplotlib.pyplot as plt

# ---------------- PRB-style plotting ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "savefig.dpi": 300,
})

BLUE, RED, GREEN = "#1f4e9c", "#c0392b", "#2a7f62"
```

### 2. Parameters
Largest denominator $q$ used for the flux $\alpha=p/q$.

```python
QMAX = 80
```

### 3. Harper Hamiltonian
Builds the $q\times q$ matrix: cosine potential on the diagonal, hopping 1 between neighbours, and a Bloch phase $e^{\pm iqk_x}$ on the wrap-around bond.

```python
def harper_matrix(alpha, kx, ky, q):
    n = np.arange(q)
    H = np.diag(2 * np.cos(ky + 2 * np.pi * alpha * n)).astype(complex)

    for i in range(q - 1):
        H[i, i + 1] = 1
        H[i + 1, i] = 1

    # Bloch boundary condition: phase e^{i q kx} on the wrap-around bond
    H[q - 1, 0] += np.exp(1j * q * kx)
    H[0, q - 1] += np.exp(-1j * q * kx)

    return H
```

### 4. Figure 1: Hofstadter butterfly
Loops over every coprime $p/q$ with $q\le q_{\max}$, samples a few $(k_x,k_y)$ points in the magnetic Brillouin zone (enough to trace the band edges), and scatters all eigenvalues against $\alpha$.

```python
def butterfly():
    xs, ys = [], []

    for q in range(1, QMAX + 1):
        for p in range(0, q + 1):
            if gcd(p, q) != 1:
                continue
            alpha = p / q

            # sample the magnetic Brillouin zone (period 2*pi/q in kx, ky)
            for kx in np.linspace(0, 2 * np.pi / q, 2):
                for ky in np.linspace(0, 2 * np.pi / q, 4):
                    E = np.linalg.eigvalsh(harper_matrix(alpha, kx, ky, q))
                    xs.extend([alpha] * q)
                    ys.extend(E)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(xs, ys, s=0.15, color=BLUE, linewidths=0, rasterized=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel(r"Magnetic flux $\alpha=\Phi/\Phi_0$")
    ax.set_ylabel(r"Energy $E/t$")
    ax.set_title("Hofstadter butterfly")

    fig.tight_layout()
    fig.savefig("images/01_hofstadter_butterfly.png", bbox_inches="tight")
    plt.show()
```

<p align="center"><img src="images/01_hofstadter_butterfly.png" width="50%"></p>

### 5. Figure 2: Bands at $\alpha=1/3$
Diagonalizes the $3\times3$ matrix over $k_y$ for many $k_x$. Shaded regions are the full band width (all $k_x$), and solid lines are the $k_x=0$ cut. The three bands are separated by two gaps.

```python
def fixed_flux():
    alpha, q = 1 / 3, 3

    ky = np.linspace(0, 2 * np.pi / q, 300)
    kx_values = np.linspace(0, 2 * np.pi / q, 40)

    # energies[kx, ky, band]
    energies = np.array([
        [np.linalg.eigvalsh(harper_matrix(alpha, kx, k, q)) for k in ky]
        for kx in kx_values
    ])

    colors = [BLUE, GREEN, RED]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    for b in range(q):
        lo = energies[:, :, b].min(axis=0)
        hi = energies[:, :, b].max(axis=0)
        ax.fill_between(ky, lo, hi, color=colors[b], alpha=0.25, lw=0)
        ax.plot(ky, energies[0, :, b], lw=1.4, color=colors[b])

    ax.set_xlim(0, 2 * np.pi / q)
    ax.set_xticks([0, np.pi / q, 2 * np.pi / q])
    ax.set_xticklabels([r"$0$", r"$\pi/3$", r"$2\pi/3$"])
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"Energy $E/t$")
    ax.set_title(r"Spectrum at $\Phi/\Phi_0=1/3$")

    fig.tight_layout()
    fig.savefig("images/02_fixed_flux_bands.png", bbox_inches="tight")
    plt.show()
```

<p align="center"><img src="images/02_fixed_flux_bands.png" width="45%"></p>

### 6. Main

```python
if __name__ == "__main__":
    butterfly()
    fixed_flux()
```

---

## Run

```bash
mkdir -p images
python 03_hofstadter.py
```

Takes about 10 s. Figures are saved to `images/` at 300 dpi.
