# 01 · SSH + Aubry–André–Harper (AAH) Model

Open-boundary Su–Schrieffer–Heeger (SSH) chain with a quasiperiodic on-site (AAH) modulation. We look at the SSH bands, how the spectrum and eigenstate localization evolve with the modulation strength $\lambda$, and a few representative wavefunctions.

## Model

**SSH hopping** (alternating bonds, $t_1 = 0.65$, $t_2 = 1.0$):

$$H_{\text{SSH}} = \sum_n \left( t_1\, c^\dagger_{2n}c_{2n+1} + t_2\, c^\dagger_{2n+1}c_{2n+2} + \text{h.c.} \right)$$

**AAH on-site modulation** ($\beta = (\sqrt5-1)/2$, $\phi = 0$):

$$V_n = \lambda \cos(2\pi \beta n + \phi)$$

**Total Hamiltonian:** $H = H_{\text{SSH}} + \sum_n V_n\, c^\dagger_n c_n$

**Bloch Hamiltonian** (no modulation, $\lambda = 0$):

$$H(k)=\begin{pmatrix}0 & z(k)\\ z^*(k) & 0\end{pmatrix},\qquad z(k)=t_1+t_2e^{-ik},\qquad E_\pm(k)=\pm\sqrt{t_1^2+t_2^2+2t_1t_2\cos k}$$

**Inverse Participation Ratio** (localization measure):

$$\text{IPR}=\sum_n |\psi_n|^4$$

IPR $\sim 1/N$ for an extended state and $\to 1$ for a state localized on a single site.

---

## Code, block by block

### 1. Imports and PRB-style plot settings
Sets serif fonts, inward ticks and a two-colour palette so all figures share one publication-style look.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

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
Chain length, SSH hoppings (weak bond first, so the chain is in the topological phase), the golden-ratio AAH frequency, and the number of $\lambda$ points in the sweep.

```python
N = 80
T1 = 0.65
T2 = 1.00
BETA = (np.sqrt(5) - 1) / 2
PHI = 0.0
N_LAMBDA = 90
```

### 3. SSH Bloch Hamiltonian
Builds the $2\times2$ $H(k)$ above; used only for the clean band structure.

```python
def ssh_bloch(k, t1=T1, t2=T2):
    z = t1 + t2 * np.exp(-1j * k)
    return np.array([[0, z], [np.conjugate(z), 0]], dtype=complex)
```

### 4. AAH on-site potential
Returns $V_n = \lambda\cos(2\pi\beta n+\phi)$. Irrational $\beta$ makes the potential quasiperiodic.

```python
def aah_onsite(n, lam, beta=BETA, phi=PHI):
    return lam * np.cos(2 * np.pi * beta * n + phi)
```

### 5. Open-boundary Hamiltonian
Real-space $N\times N$ matrix: alternating hoppings off the diagonal, AAH potential on the diagonal. Open ends are needed to see edge states.

```python
def open_hamiltonian(lam, n_sites=N):
    H = np.zeros((n_sites, n_sites), dtype=float)

    # Alternating SSH hopping
    for i in range(n_sites - 1):
        hopping = T1 if i % 2 == 0 else T2
        H[i, i + 1] = hopping
        H[i + 1, i] = hopping

    # AAH onsite modulation
    for i in range(n_sites):
        H[i, i] += aah_onsite(i, lam)

    return H
```

### 6. Inverse Participation Ratio
Quantifies how localized an eigenstate is.

```python
def ipr(vec):
    probability = np.abs(vec) ** 2
    return np.sum(probability ** 2)
```

### 7. Figure 1: SSH band structure
Diagonalizes $H(k)$ over the Brillouin zone. The gap of size $2|t_2-t_1|$ at $k=\pm\pi$ is the reference for what follows.

```python
def plot_ssh_bands():
    k = np.linspace(-np.pi, np.pi, 600)
    energies = np.array([np.linalg.eigvalsh(ssh_bloch(ki)) for ki in k])

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(k, energies[:, 0], lw=1.6, color=BLUE)
    ax.plot(k, energies[:, 1], lw=1.6, color=RED)
    ax.axhline(0, ls="--", lw=0.7, color="gray")

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E/t$")
    ax.set_title("SSH band structure")

    fig.tight_layout()
    fig.savefig("images/01_ssh_bands.png", bbox_inches="tight")
    plt.show()
```
### 8. Figure 2: Spectrum and localization vs $\lambda$
For each $\lambda$, diagonalize the open chain and plot all eigenvalues, coloured by IPR. Dark = extended, bright = localized.

```python
def spectrum_and_ipr():
    lambdas = np.linspace(0, 2.5, N_LAMBDA)
    spectrum = np.zeros((N_LAMBDA, N))
    iprs = np.zeros_like(spectrum)

    for j, lam in enumerate(lambdas):
        E, V = eigh(open_hamiltonian(lam))
        spectrum[j] = E
        for i in range(N):
            iprs[j, i] = ipr(V[:, i])

    x = np.repeat(lambdas, N)
    y = spectrum.ravel()
    c = iprs.ravel()

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sc = ax.scatter(x, y, c=c, s=1.5, cmap="viridis", rasterized=True)

    ax.set_xlabel(r"AAH modulation $\lambda/t$")
    ax.set_ylabel(r"Energy $E/t$")
    ax.set_title("Spectrum and localization")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("IPR")
    cbar.ax.tick_params(direction="in")

    fig.tight_layout()
    fig.savefig("images/02_aah_spectrum_ipr.png", bbox_inches="tight")
    plt.show()
```
### 9. Figure 3: Wavefunction profiles
Picks the eigenstate closest to $E=0$ at three modulation strengths and plots $|\psi_n|^2$ along the chain: edge-localized at small $\lambda$, then moving into the bulk-localized regime as $\lambda$ grows.

```python
def plot_state_profiles():
    cases = [0.0, 1.0, 2.2]
    labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, len(cases), figsize=(7.0, 2.2))

    for ax, lam, lab in zip(axes, cases, labels):
        E, V = eigh(open_hamiltonian(lam))
        idx = np.argmin(np.abs(E))          # state closest to zero energy
        probability = np.abs(V[:, idx]) ** 2

        ax.plot(np.arange(N), probability, lw=1.4, color=BLUE)
        ax.fill_between(np.arange(N), probability, alpha=0.25, color=BLUE)
        ax.set_xlim(0, N - 1)
        ax.set_ylim(bottom=0)
        ax.set_title(rf"{lab}  $\lambda/t={lam:.1f}$")
        ax.set_xlabel("Site $n$")

    axes[0].set_ylabel(r"$|\psi_n|^2$")

    fig.tight_layout()
    fig.savefig("images/03_state_profiles.png", bbox_inches="tight")
    plt.show()
```
### 10. Main
Runs all three plots.

```python
if __name__ == "__main__":
    plot_ssh_bands()
    spectrum_and_ipr()
    plot_state_profiles()
```

---

## Run

```bash
mkdir -p images
python 01_ssh_aah.py
```

Figures are saved to `images/` at 300 dpi.
