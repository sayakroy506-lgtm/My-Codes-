# 04 · Haldane Model: Chern Insulator on the Honeycomb Lattice

Haldane's model of graphene with a staggered sublattice mass $M$ and complex next-nearest-neighbour hopping $t_2e^{\pm i\phi}$ that breaks time-reversal symmetry. We compute the Chern number of the lower band over the full $(\phi,M)$ plane and plot the Berry curvature.

## Model

**Bloch Hamiltonian** (hopping $t_1$ between nearest neighbours, $t_2$ between next-nearest neighbours):

$$H(\mathbf k)=d_0\,\mathbb 1+\mathbf d(\mathbf k)\cdot\boldsymbol\sigma=\begin{pmatrix}d_0+d_z & t_1 f(\mathbf k)\\ t_1 f^*(\mathbf k) & d_0-d_z\end{pmatrix}$$

**Nearest-neighbour term** ($\boldsymbol\delta_j$ are the three bond vectors; phases are measured from $\boldsymbol\delta_1$ so that $H(\mathbf k+\mathbf G)=H(\mathbf k)$):

$$f(\mathbf k)=\sum_{j=1}^{3}e^{i\mathbf k\cdot(\boldsymbol\delta_j-\boldsymbol\delta_1)}$$

**Next-nearest-neighbour (Haldane) terms** ($\mathbf b_i$ are the three Bravais vectors, $\mathbf b_1+\mathbf b_2+\mathbf b_3=0$):

$$d_z=M-2t_2\sin\phi\sum_{i}\sin(\mathbf k\cdot\mathbf b_i),\qquad d_0=2t_2\cos\phi\sum_{i}\cos(\mathbf k\cdot\mathbf b_i)$$

**Bands:** $E_\pm=d_0\pm|\mathbf d|$.

**Berry curvature and Chern number** of the lower band $|u_-\rangle$:

$$\Omega(\mathbf k)=\partial_{k_x}A_y-\partial_{k_y}A_x,\quad \mathbf A=i\langle u_-|\nabla_{\mathbf k}u_-\rangle,\qquad C=\frac1{2\pi}\int_{\text{BZ}}\Omega\,d^2k$$

**Phase boundary:** the gap closes at $K$ or $K'$ when $M=\pm3\sqrt3\,t_2\sin\phi$. Inside,

$$|M|<3\sqrt3\,t_2|\sin\phi|\ \Rightarrow\ C=\operatorname{sgn}(\sin\phi)\ (\text{in our convention}),\qquad\text{otherwise } C=0.$$

For $t_2=0.15\,t_1$ the largest boundary value is $3\sqrt3\,t_2\approx0.78\,t_1$.

---

## Code, block by block

### 1. Imports and PRB-style plot settings
Same serif, inward-tick style and blue/red palette as the rest of the repo.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

BLUE, RED = "#1f4e9c", "#c0392b"
```

### 2. Parameters and lattice vectors
Hoppings, the three bond vectors $\boldsymbol\delta_j$, the Bravais vectors $\mathbf b_i$, and the reciprocal lattice vectors $\mathbf G_{1,2}$ that span the Brillouin zone.

```python
# ---------------- Parameters ----------------
T1 = 1.0       # nearest-neighbour hopping
T2 = 0.15      # next-nearest-neighbour hopping

# Nearest-neighbour vectors
d1 = np.array([0.0, -1.0])
d2 = np.array([np.sqrt(3) / 2, 0.5])
d3 = np.array([-np.sqrt(3) / 2, 0.5])

# Next-nearest-neighbour vectors (= Bravais lattice vectors)
b1 = d2 - d3
b2 = d3 - d1
b3 = d1 - d2

# Reciprocal lattice vectors (a1 = b1, a2 = b2)
G1 = np.array([2 * np.pi / np.sqrt(3), 2 * np.pi / 3])
G2 = np.array([0.0, 4 * np.pi / 3])
```

### 3. Haldane Hamiltonian
Builds $H(\mathbf k)$ from $d_0$, $d_z$ and $f(\mathbf k)$. It works on whole $k$-grids at once and is exactly periodic in $\mathbf k$.

```python
def hamiltonian(kx, ky, M, phi):
    kx, ky = np.asarray(kx, float), np.asarray(ky, float)

    # Nearest-neighbour term (phases measured from d1 -> H(k+G) = H(k))
    nn = sum(
        np.exp(1j * (kx * (d[0] - d1[0]) + ky * (d[1] - d1[1])))
        for d in (d1, d2, d3)
    )

    # Haldane next-nearest-neighbour terms
    sine = sum(np.sin(kx * b[0] + ky * b[1]) for b in (b1, b2, b3))
    cosine = sum(np.cos(kx * b[0] + ky * b[1]) for b in (b1, b2, b3))

    dz = M - 2 * T2 * np.sin(phi) * sine
    d0 = 2 * T2 * np.cos(phi) * cosine

    H = np.zeros(kx.shape + (2, 2), dtype=complex)
    H[..., 0, 0] = d0 + dz
    H[..., 1, 1] = d0 - dz
    H[..., 0, 1] = T1 * nn
    H[..., 1, 0] = np.conj(T1 * nn)
    return H
```

### 4. Eigenvector and link variables
`lower_band` returns the lower-band eigenvector. `link` is the normalised overlap between neighbouring $k$-points, and `plaquette_flux` combines four links into the Berry flux through one small square (Fukui–Hatsugai–Suzuki method). It does not depend on the arbitrary phase of the eigenvectors.

```python
def lower_band(kx, ky, M, phi):
    _, U = np.linalg.eigh(hamiltonian(kx, ky, M, phi))
    return U[..., :, 0]


def link(a, b):
    z = np.sum(np.conj(a) * b, axis=-1)
    return z / np.abs(z)


def plaquette_flux(p00, p10, p11, p01):
    """Berry flux through one plaquette (Fukui-Hatsugai-Suzuki)."""
    return np.angle(
        link(p00, p10) * link(p10, p11)
        * np.conj(link(p01, p11)) * np.conj(link(p00, p01))
    )
```

### 5. Chern number
Divides the **true** Brillouin zone (the parallelogram $\mathbf k=u\mathbf G_1+v\mathbf G_2$) into a grid and sums the plaquette fluxes: $C=\frac1{2\pi}\sum F$. The result is an exact integer.

> **Note:** looping over the square $[-\pi,\pi]^2$ instead is wrong here. Its area ($4\pi^2$) is 2.6 times the Brillouin-zone area ($8\pi^2/3\sqrt3$), so it overcounts the curvature and gives $|C|\approx2.8$ instead of 1.

```python
def chern_number(M, phi, N=30):
    u = np.arange(N) / N
    U, V = np.meshgrid(u, u, indexing="ij")
    KX = U * G1[0] + V * G2[0]
    KY = U * G1[1] + V * G2[1]

    psi = lower_band(KX, KY, M, phi)
    psi_x = np.roll(psi, -1, axis=0)
    psi_y = np.roll(psi, -1, axis=1)
    psi_xy = np.roll(psi_x, -1, axis=1)

    return plaquette_flux(psi, psi_x, psi_xy, psi_y).sum() / (2 * np.pi)
```

### 6. Berry curvature map
Same plaquette flux on a Cartesian grid, divided by the plaquette area to get $\Omega(\mathbf k)$. `bz_hexagon` returns the hexagon outline of the first Brillouin zone.

```python
def berry_curvature_map(M, phi, kmax=3.2, N=500):
    k = np.linspace(-kmax, kmax, N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    psi = lower_band(KX, KY, M, phi)

    dk = k[1] - k[0]
    flux = plaquette_flux(psi[:-1, :-1], psi[1:, :-1], psi[1:, 1:], psi[:-1, 1:])
    return k, flux / dk**2


def bz_hexagon():
    """Vertices of the first Brillouin zone (a regular hexagon)."""
    R = 4 * np.pi / (3 * np.sqrt(3))            # distance Gamma -> K
    ang = np.arange(7) * np.pi / 3
    return R * np.cos(ang), R * np.sin(ang)
```

### 7. Figure 1: Phase diagram
Computes $C$ at every $(\phi,M)$ point and overlays the analytic boundary $M=\pm3\sqrt3\,t_2\sin\phi$ (dashed). The Chern number is $+1$ for $\phi>0$, $-1$ for $\phi<0$, and 0 outside the lobes.

```python
def phase_diagram():
    Ms = np.linspace(-1.0, 1.0, 121)
    phis = np.linspace(-np.pi, np.pi, 161)

    C = np.zeros((len(Ms), len(phis)))
    for i, M in enumerate(Ms):
        for j, phi in enumerate(phis):
            C[i, j] = chern_number(M, phi, 30)
    C = np.rint(C)

    cmap = ListedColormap(["#9db8e0", "#ffffff", "#e8a79d"])   # C = -1, 0, +1

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.pcolormesh(phis / np.pi, Ms, C, cmap=cmap, vmin=-1.5, vmax=1.5,
                  shading="nearest", rasterized=True)

    # analytic boundary: M = +-3*sqrt(3)*t2*sin(phi)
    p = np.linspace(-np.pi, np.pi, 400)
    for s in (+1, -1):
        ax.plot(p / np.pi, s * 3 * np.sqrt(3) * T2 * np.sin(p),
                "k--", lw=0.8)

    for x, y, lab in [(0.5, 0.0, "$C=+1$"), (-0.5, 0.0, "$C=-1$"),
                      (0.0, 0.62, "$C=0$"), (0.0, -0.62, "$C=0$")]:
        ax.text(x, y, lab, ha="center", va="center", fontsize=9)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"Haldane phase $\phi$")
    ax.set_ylabel(r"Sublattice mass $M/t_1$")
    ax.set_title("Haldane phase diagram")

    fig.tight_layout()
    fig.savefig("images/01_haldane_phase_diagram.png", bbox_inches="tight")
    plt.show()
```

<p align="center"><img src="images/01_haldane_phase_diagram.png" width="50%"></p>

### 8. Figure 2: Berry curvature
Lower-band curvature at $M=0.2\,t_1$, $\phi=\pi/2$ (topological phase). It is concentrated at the $K$ valley with the smaller gap, and integrating over the hexagon gives $2\pi C=2\pi$.

```python
def berry_plot():
    M, phi = 0.2, np.pi / 2
    k, omega = berry_curvature_map(M, phi)

    vmax = np.percentile(np.abs(omega), 99.5)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(omega.T, origin="lower", extent=[k[0], k[-1], k[0], k[-1]],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")

    hx, hy = bz_hexagon()
    ax.plot(hx, hy, "k-", lw=0.8)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(r"Berry curvature, $M/t_1=0.2,\ \phi=\pi/2$")

    cbar = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85)
    cbar.set_label(r"$\Omega(k)$")
    cbar.ax.tick_params(direction="in")

    fig.tight_layout()
    fig.savefig("images/02_berry_curvature.png", bbox_inches="tight")
    plt.show()
```

<p align="center"><img src="images/02_berry_curvature.png" width="50%"></p>

### 9. Main

```python
if __name__ == "__main__":
    phase_diagram()
    berry_plot()
```

---

## Run

```bash
mkdir -p images
python 04_haldane.py
```

Takes about 25 s. Figures are saved to `images/` at 300 dpi.
