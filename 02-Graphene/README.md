# 02 · Graphene: Nearest-Neighbour Tight-Binding Model

Nearest-neighbour tight-binding description of the $\pi$ bands of graphene: band structure along $\Gamma\!-\!K\!-\!M\!-\!\Gamma$, the Dirac cone near $K$, and the density of states.

## Model

**Hamiltonian** (two sublattices A, B; hopping $t=-2.7$ eV; $a$ is the lattice constant):

$$H(\mathbf k)=\begin{pmatrix}0 & t\,f(\mathbf k)\\ t\,f^*(\mathbf k) & 0\end{pmatrix}$$

**Structure factor** (sum over the three nearest-neighbour bonds):

$$f(\mathbf k)=e^{ik_ya/\sqrt3}+2\,e^{-ik_ya/(2\sqrt3)}\cos\!\left(\frac{k_xa}{2}\right)$$

**Energy bands:**

$$E_\pm(\mathbf k)=\pm|t|\,|f(\mathbf k)|$$

**High-symmetry points:** $\Gamma=(0,0)$, $K=\left(\tfrac{4\pi}{3a},0\right)$, $M=\left(\tfrac{\pi}{a},\tfrac{\pi}{\sqrt3 a}\right)$.
Band edges: $E(\Gamma)=\pm3|t|=\pm8.1$ eV, $E(M)=\pm|t|=\pm2.7$ eV, $E(K)=0$.

**Dirac cone** (expanding around $K$, $\mathbf k=K+\mathbf q$):

$$E_\pm(\mathbf q)\approx\pm\hbar v_F|\mathbf q|,\qquad \hbar v_F=\frac{\sqrt3}{2}|t|a$$

**Density of states** (per unit cell, per spin; linear near $E=0$):

$$g(E)=\frac{2\sqrt3}{3\pi}\frac{|E|}{t^2}\quad(|E|\ll|t|)$$

with van Hove singularities at $E=\pm|t|$ (the $M$ points).

---

## Code, block by block

### 1. Imports and PRB-style plot settings
Same serif, inward-tick style and blue/red palette as the rest of the repo, so all figures match.

```python
import numpy as np
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

BLUE, RED = "#1f4e9c", "#c0392b"
```

### 2. Parameters
Hopping energy in eV and lattice constant. Wavevectors are in units of $1/a$.

```python
T = -2.7      # nearest-neighbour hopping (eV)
A = 1.0       # lattice constant (k in units of 1/A)
```

### 3. Structure factor $f(\mathbf k)$
Encodes the three nearest-neighbour bonds linking sublattice A to B.

```python
def f(kx, ky):
    return (
        np.exp(1j * ky * A / np.sqrt(3))
        + 2 * np.exp(-1j * ky * A / (2 * np.sqrt(3))) * np.cos(kx * A / 2)
    )
```

### 4. Energy bands
Returns the two bands $E_\mp=\mp|t||f|$ (valence, conduction). It works on arrays too, which is what lets the plots below avoid loops.

```python
def bands(kx, ky):
    energy = abs(T) * abs(f(kx, ky))
    return -energy, energy
```

### 5. Path helpers
`path_segment` gives a straight line between two $k$-points. `high_symmetry_path` chains $\Gamma\to K\to M\to\Gamma$ and returns the tick positions and labels for the plot.

```python
def path_segment(point_a, point_b, n=250):
    return np.linspace(point_a, point_b, n, endpoint=False)


def high_symmetry_path():
    Gamma = np.array([0.0, 0.0])
    K = np.array([4 * np.pi / 3, 0.0])
    M = np.array([np.pi, np.pi / np.sqrt(3)])

    part1 = path_segment(Gamma, K)
    part2 = path_segment(K, M)
    part3 = path_segment(M, Gamma)

    path = np.vstack([part1, part2, part3, Gamma[None, :]])

    ticks = [0, len(part1), len(part1) + len(part2), len(path) - 1]
    labels = [r"$\Gamma$", "K", "M", r"$\Gamma$"]

    return path, ticks, labels
```

### 6. Figure 1: Band structure
Evaluates $E_\pm$ along the path. The bands touch at $K$ (zero gap), and the width is $6|t|=16.2$ eV from bottom to top.

```python
def plot_bands():
    path, ticks, labels = high_symmetry_path()
    energies = np.array([bands(k[0], k[1]) for k in path])

    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    ax.plot(energies[:, 0], lw=1.6, color=BLUE)
    ax.plot(energies[:, 1], lw=1.6, color=RED)
    ax.axhline(0, ls="--", lw=0.7, color="gray")

    for tick in ticks:
        ax.axvline(tick, ls="-", lw=0.5, color="gray")

    ax.set_xlim(0, len(path) - 1)
    ax.set_xticks(ticks, labels)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Graphene band structure")

    fig.tight_layout()
    fig.savefig("images/01_graphene_bands.png", bbox_inches="tight")
    plt.show()
```
### 7. Figure 2: Dirac cone near $K$
Evaluates both bands on a grid of $\mathbf q$ around $K$. The surfaces meet at a point and are linear in $|\mathbf q|$, which gives massless Dirac fermions. Valence band in blue, conduction band in red.

```python
def plot_dirac_cone():
    K_point = np.array([4 * np.pi / 3, 0.0])

    q = np.linspace(-0.8, 0.8, 180)
    QX, QY = np.meshgrid(q, q)

    E_minus, E_plus = bands(K_point[0] + QX, K_point[1] + QY)

    fig = plt.figure(figsize=(3.6, 3.3))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(QX, QY, E_plus, cmap="Reds", linewidth=0,
                    antialiased=True, alpha=0.95)
    ax.plot_surface(QX, QY, E_minus, cmap="Blues_r", linewidth=0,
                    antialiased=True, alpha=0.95)

    ax.set_xlabel(r"$q_x\,(1/a)$", labelpad=1)
    ax.set_ylabel(r"$q_y\,(1/a)$", labelpad=1)
    # z-label drawn manually (3D labels can be clipped by the figure edge)
    ax.text2D(1.13, 0.5, "Energy (eV)", transform=ax.transAxes,
              rotation=90, va="center", ha="center")
    ax.set_title("Dirac cone near K", pad=0)
    ax.view_init(elev=18, azim=-60)
    ax.tick_params(pad=0)
    ax.minorticks_off()
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])

    fig.subplots_adjust(left=0.0, right=0.80, bottom=0.05, top=0.98)
    fig.savefig("images/02_dirac_cone.png", bbox_inches="tight", pad_inches=0.1)
    plt.show()
```

### 8. Figure 3: Density of states
Samples the true Brillouin-zone unit cell, $\mathbf k=u\mathbf b_1+v\mathbf b_2$ with $u,v\in[0,1)$, histograms all energies, and normalizes to 2 states per cell. The dashed red line is the analytic Dirac result.

> **Note:** sampling a square $[-\pi,\pi]^2$ instead misses the $K$ points (they sit at $k_x=4\pi/3>\pi$), which wrongly empties the DOS near $E=0$. Sampling the actual unit cell avoids this.

```python
def plot_dos(n=1200):
    # Sample the true Brillouin-zone unit cell: k = u*b1 + v*b2
    b1 = np.array([2 * np.pi / A, -2 * np.pi / (np.sqrt(3) * A)])
    b2 = np.array([0.0, 4 * np.pi / (np.sqrt(3) * A)])

    u = np.linspace(0, 1, n, endpoint=False)
    U, V = np.meshgrid(u, u)
    KX = U * b1[0] + V * b2[0]
    KY = U * b1[1] + V * b2[1]

    e_minus, e_plus = bands(KX, KY)
    energies = np.concatenate([e_minus.ravel(), e_plus.ravel()])

    hist, edges = np.histogram(energies, bins=300, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dos = 2 * hist          # states / eV / unit cell / spin (2 bands per cell)

    # Low-energy Dirac result: g(E) = 2*sqrt(3)/(3*pi) * |E| / t^2
    E_lin = np.linspace(-1.2, 1.2, 200)
    dos_lin = 2 * np.sqrt(3) / (3 * np.pi) * np.abs(E_lin) / T**2

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(centers, dos, lw=1.4, color=BLUE, label="Tight-binding")
    ax.plot(E_lin, dos_lin, ls="--", lw=1.0, color=RED, label="Dirac (linear)")

    ax.set_xlim(-9, 9)
    ax.set_ylim(0, 0.46)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (states/eV/cell/spin)")
    ax.set_title("Graphene density of states")
    ax.legend(loc="upper center", ncol=2, fontsize=8, columnspacing=1.0, handlelength=1.6)

    fig.tight_layout()
    fig.savefig("images/03_graphene_dos.png", bbox_inches="tight")
    plt.show()
```

### 9. Main

```python
if __name__ == "__main__":
    plot_bands()
    plot_dirac_cone()
    plot_dos()
```

---

## Run

```bash
mkdir -p images
python 02_graphene_tb.py
```

Figures are saved to `images/` at 300 dpi.
