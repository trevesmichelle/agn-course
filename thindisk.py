"""
Thin-disk (Shakura-Sunyaev, Newtonian) accretion-disk SED calculator.

In the context of this homework, a ThinDisk object is a black hole + its accretion disk. 
To use: construct it with the physical parameters (mass, accretion rate, spin) and then ask it for 
derived physical quantities: the ISCO, the radiative efficiency, the radial temperature profile, and the emergent SED.

AGN course, HW1's "Preliminary assumptions":

  spin <-> ISCO :  a = (x^1/2 / 3) [4 - (3x - 2)^1/2],   x = r_ISCO / r_g
  efficiency    :  eps_r = 1 - (1 - 2/(3x))^1/2
  ring temp.    :  T(r) = [ 3 G M Mdot / (8 sigma pi r^3) * f(r) ]^1/4
                   f(r) = 1 - (r_in / r)^1/2,   r_in = r_ISCO
  outer edge    :  r_out = r_SG = 2000 r_g

Units are CGS throughout.
"""
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
G       = 6.67430e-8        # cm^3 g^-1 s^-2
c       = 2.99792458e10     # cm s^-1
h       = 6.62607015e-27    # erg s
k_B     = 1.380649e-16      # erg K^-1
sigma   = 5.670374419e-5    # erg cm^-2 s^-1 K^-4  (Stefan-Boltzmann)
M_sun   = 1.98892e33        # g
yr      = 3.155760e7        # s  (Julian year)
m_p     = 1.67262192e-24    # g
sigma_T = 6.6524587e-25     # cm^2  (Thomson cross-section)

# convenient composite unit: 1 Msun/yr expressed in g/s
MDOT_SUN_PER_YR = M_sun / yr


class ThinDisk:
    """
    A Newtonian thin accretion disk around a (possibly spinning) black hole.

    Parameters
    ----------
    M_BH_Msun : float
        Black-hole mass in solar masses.
    Mdot_Msun_yr : float
        Accretion rate in solar masses per year.
    a : float, optional
        Dimensionless spin parameter, in [-1, 1]. a>0 prograde,
        a<0 retrograde, a=0 Schwarzschild. Default 0.
    r_out_rg : float, optional
        Outer disk radius in units of r_g. Default 2000 (the self-gravity
        radius adopted by the problem set).

    Notes
    -----
    Spin-dependent quantities (x = r_ISCO/r_g, the efficiency, r_g and the
    inner radius r_in) are computed once at construction and cached as
    attributes, since they never change for a given object.
    """

    def __init__(self, M_BH_Msun, Mdot_Msun_yr, a=0.0, r_out_rg=2000.0):
        # --- store inputs, converting to CGS ---
        self.M_BH_Msun   = M_BH_Msun
        self.Mdot_Msun_yr = Mdot_Msun_yr
        self.M    = M_BH_Msun * M_sun            # g
        self.Mdot = Mdot_Msun_yr * MDOT_SUN_PER_YR  # g/s
        self.a    = a
        self.r_out_rg = r_out_rg

        # --- cache spin-dependent geometry (computed once) ---
        self.r_g  = G * self.M / c**2            # gravitational radius (cm)
        self.x    = self._x_isco()               # r_ISCO / r_g  (dimensionless)
        self.eps  = self.efficiency()            # radiative efficiency
        self.r_in  = self.x * self.r_g           # inner edge = r_ISCO (cm)
        self.r_out = self.r_out_rg * self.r_g    # outer edge (cm)

    # ------------------------------------------------------------------
    # Spin / ISCO / efficiency
    # ------------------------------------------------------------------
    def _x_isco(self):
        """
        Return x = r_ISCO / r_g by inverting the spin-ISCO relation

            a = (x^1/2 / 3) [4 - (3x - 2)^1/2].

        The relation is monotonic in x over [1, 9] (x=1 at a=+1, x=6 at
        a=0, x=9 at a=-1), so a bracketed root find is robust and needs no
        GR algebra. We solve  g(x) = (x^1/2/3)[4-(3x-2)^1/2] - a = 0.
        """
        def residual(x):
            return (np.sqrt(x) / 3.0) * (4.0 - np.sqrt(3.0 * x - 2.0)) - self.a
        return brentq(residual, 1.0, 9.0, xtol=1e-12, rtol=1e-14)

    def efficiency(self):
        """
        Radiative (binding-energy) efficiency at the ISCO:

            eps_r = 1 - (1 - 2/(3x))^1/2 .
        """
        return 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * self.x))

    # ------------------------------------------------------------------
    # Disk structure
    # ------------------------------------------------------------------
    def T(self, r):
        """
        Effective (blackbody) temperature of the disk at radius r (cm).

            T(r) = [ 3 G M Mdot / (8 sigma pi r^3) * f(r) ]^1/4
            f(r) = 1 - (r_in / r)^1/2

        f is clipped at zero so that r <= r_in returns T = 0 (no emission
        inside the inner edge) rather than a NaN. Accepts scalar or array r.
        """
        r = np.asarray(r, dtype=float)
        f = 1.0 - np.sqrt(self.r_in / r)
        f = np.clip(f, 0.0, None)
        return (3.0 * G * self.M * self.Mdot
                / (8.0 * sigma * np.pi * r**3) * f) ** 0.25

    def planck_Bnu(self, nu, T):
        """
        Planck function B_nu(T) in erg s^-1 cm^-2 Hz^-1 sr^-1.

        Accepts scalars or broadcastable arrays. The Wien tail is guarded
        against overflow: where h*nu/kT is large the exponential is huge and
        B_nu underflows to 0, which we set explicitly.
        """
        nu = np.asarray(nu, dtype=float)
        T  = np.asarray(T,  dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            xq = h * nu / (k_B * T)
            small = xq < 700.0                  # expm1 overflows beyond ~700
            xq_safe = np.where(small, xq, 1.0)  # dummy where masked out
            Bnu = 2.0 * h * nu**3 / c**2 / np.expm1(xq_safe)
        return np.where(small, Bnu, 0.0)

    # ------------------------------------------------------------------
    # SED
    # ------------------------------------------------------------------
    def disk_sed(self, n_rings=4000, n_nu=600, nu_min=1e12, nu_max=1e18):
        """
        Compute the emergent disk SED by summing blackbody rings.

        The disk is divided into n_rings annuli between r_in and r_out, each
        radiating as a blackbody at its local T(r). The monochromatic
        luminosity is

            L_nu = 4 pi^2 * integral_{r_in}^{r_out} B_nu(T(r)) r dr ,

        where the factor accounts for both disk faces: a ring of (single-face)
        area 2 pi r dr emits pi B_nu per unit area into the outward
        hemisphere, and the disk radiates from two faces, giving
        2 * pi * 2 pi r dr = 4 pi^2 r dr.

        Parameters
        ----------
        n_rings : int
            Number of radial annuli (log-spaced; the hot inner region needs
            fine sampling).
        n_nu : int
            Number of frequency points (log-spaced).
        nu_min, nu_max : float
            Frequency range in Hz.

        Returns
        -------
        nu : ndarray, shape (n_nu,)
            Frequency grid (Hz).
        L_nu : ndarray, shape (n_nu,)
            Monochromatic luminosity (erg s^-1 Hz^-1).
        """
        # log-spaced radial grid; integrate on annulus midpoints
        r_edges = np.logspace(np.log10(self.r_in), np.log10(self.r_out),
                              n_rings + 1)
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        dr    = np.diff(r_edges)

        T_mid = self.T(r_mid)                       # (n_rings,)
        nu = np.logspace(np.log10(nu_min), np.log10(nu_max), n_nu)  # (n_nu,)

        # B_nu on the (frequency, ring) grid, then integrate over rings
        NU, TT = np.meshgrid(nu, T_mid, indexing="ij")   # (n_nu, n_rings)
        Bnu = self.planck_Bnu(NU, TT)
        L_nu = 4.0 * np.pi**2 * np.sum(Bnu * (r_mid * dr)[None, :], axis=1)

        return nu, L_nu

    def L_bol_sed(self, **sed_kwargs):
        """
        Bolometric luminosity from integrating the numerical SED: int L_nu dnu.
        Extra keyword args are forwarded to disk_sed (e.g. to widen the grid).
        """
        nu, L_nu = self.disk_sed(**sed_kwargs)
        return np.trapezoid(L_nu, nu)

    def L_bol_estimate(self):
        """Analytic estimate of the bolometric luminosity: eps_r * Mdot * c^2."""
        return self.eps * self.Mdot * c**2

    # ------------------------------------------------------------------
    # Eddington helpers
    # ------------------------------------------------------------------
    def L_Edd(self):
        """Eddington luminosity (erg/s), electron scattering on hydrogen."""
        return 4.0 * np.pi * G * self.M * m_p * c / sigma_T

    def eddington_ratio(self):
        """L_bol_estimate / L_Edd."""
        return self.L_bol_estimate() / self.L_Edd()

    def __repr__(self):
        return (f"ThinDisk(M={self.M_BH_Msun:.3g} Msun, "
                f"Mdot={self.Mdot_Msun_yr:.3g} Msun/yr, a={self.a:.3g})")


def Mdot_for_Eddington(M_BH_Msun, a=0.0):
    """
    Accretion rate (Msun/yr) for which L_bol_estimate = L_Edd, at fixed M, a.

    Free helper (not a method) because it answers "what Mdot do I need?",
    i.e. it constructs the disk rather than living on an existing one.
    """
    probe = ThinDisk(M_BH_Msun, 1.0, a=a)          # Mdot is a dummy here
    Mdot_cgs = probe.L_Edd() / (probe.eps * c**2)
    return Mdot_cgs / MDOT_SUN_PER_YR


if __name__ == "__main__":
    # sanity check: known ISCO / efficiency values
    for a in (-1.0, 0.0, 0.98):
        d = ThinDisk(1e8, 1.0, a=a)
        print(f"a={a:+.2f}  x=r_isco/rg={d.x:6.3f}  eps={d.eps:.4f}")