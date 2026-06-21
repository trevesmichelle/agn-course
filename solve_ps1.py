"""
AGN Problem Set 1 - solver for Questions 3-6 (thin-disk SEDs).

Each question function prints its physics quantities and saves a labelled
figure. The physical reasoning is kept in the lab notebook, not here - this
file is the numerics.

Usage
-----
Run all four questions (default when no flags are given):
    python solve_ps1.py

Run a single question:
    python solve_ps1.py --q3

Run a subset (flags combine):
    python solve_ps1.py --q4 --q6

Available flags: --q3  --q4  --q5  --q6  (and --all, equivalent to no flags).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from thindisk import ThinDisk, Mdot_for_Eddington, c

# Fiducial black hole (Question 3): the reference for all comparisons.
FIDUCIAL = dict(M_BH_Msun=1e8, Mdot_Msun_yr=1.0, a=0.0)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def wavelength_to_rgb(lambda_angstrom):
    """
    Map a wavelength to an approximate sRGB colour for the peak markers.

    Visible band (3800-7500 A): Dan Bruton's classic approximate
    wavelength->RGB algorithm, with intensity roll-off near the eye's limits.

    Outside the visible band the colour is clamped to a single edge hue (no
    darkening): far-UV -> bright violet, near-IR -> dark red. The visible band
    gets Dan Bruton's classic approximate wavelength->RGB rainbow. This is a
    visual convention, not a claim that the eye sees out-of-band wavelengths.
    Gradation *within* an out-of-band region is intentionally not encoded.

    Parameters
    ----------
    lambda_angstrom : float
        Peak wavelength in angstrom.

    Returns
    -------
    (r, g, b) : tuple of floats in [0, 1].
    """
    w = lambda_angstrom / 10.0          # angstrom -> nm (Bruton works in nm)

    # --- out-of-band: plain clamp to the edge hue (no darkening) ---
    # Within this problem set the peaks span far-UV to near-IR, so a single
    # bright violet for "UV" and a single dark red for "IR" is unambiguous.
    # (Gradation *within* a band is intentionally not encoded here; if a future
    # question has several peaks in the same band, distinguish them by the
    # curve's identity colour / line style, not by the physical wavelength hue.)
    if w < 380.0:
        return (0.55, 0.0, 1.0)          # bright violet (far-UV)
    if w > 750.0:
        return (0.55, 0.0, 0.0)          # dark red (near-IR)

    # --- visible band: Bruton's piecewise hue ---
    if w < 440:
        r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif w < 580:
        r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    # intensity fall-off at the very ends of the visible range
    if w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif w > 700:
        factor = 0.3 + 0.7 * (750 - w) / (750 - 700)
    else:
        factor = 1.0
    return (r * factor, g * factor, b * factor)


def plot_seds(disks, labels, title, filename,
              compare_with_fiducial=False, mark_peaks=True,
              vis_line=False, annotate=True,
              color_scheme="grey_wavelength"):
    """
    Plot one or more disk SEDs as log L_nu vs log nu.

    Two independent colour roles:
      - the curve + legend entry are coloured sequentially by BH mass
        (identity only: just to tell curves apart);
      - the peak marker and its vertical line are coloured by the *actual
        wavelength* of that peak via wavelength_to_rgb (physically meaningful:
        violet=UV peak, green=optical, red=IR).

    Parameters
    ----------
    disks : list[ThinDisk]
        Disks whose SEDs to plot, in legend order.
    labels : list[str]
        Legend label for each disk (same length as `disks`).
    title : str
        Axes title.
    filename : str
        Output PNG path.
    compare_with_fiducial : bool
        If True, also overplot the fiducial (Q3) SED as a dashed grey
        reference curve. Skipped automatically for any disk that already
        *is* the fiducial, to avoid drawing it twice.
    mark_peaks : bool
        If True, drop a dashed vertical line at each curve's L_nu peak
        (coloured by the peak wavelength).
    vis_line : bool
        If True, draw a vertical marker at 5000 A (used by Q6b).
    annotate : bool
        If True, label each curve's peak with its nu_peak and T_max.
    color_scheme : str
        Selects the identity/marker colour logic per question:
          "grey_wavelength" (default) - curves are a greyscale ramp by list
            position; peak marker, vertical line and box edge are coloured by
            the peak *wavelength* (violet=UV, green=optical, red=IR). Best when
            peaks fall in different bands (e.g. Q4: mass shifts UV->optical->IR).
          "distinct_hues" - each curve gets a distinct qualitative hue; the peak
            marker, line and box edge use that same curve hue (wavelength colour
            is dropped). Best when all peaks share a band, so wavelength colour
            would be uninformative (e.g. Q5: all peaks are UV).
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    n = len(disks)

    # --- curve identity colours, chosen by scheme ---
    if color_scheme == "distinct_hues":
        # qualitative palette: clearly different hues, ordered by list position
        palette = ["#1f77b4", "#17bec8", "#ff7f0e", "#d62728", "#9467bd"]
        colors = [palette[i % len(palette)] for i in range(n)]
        use_wavelength = False
    else:  # "grey_wavelength"
        if n > 1:
            greys = np.linspace(0.65, 0.10, n)     # light -> dark
        else:
            greys = np.array([0.30])
        colors = [str(g) for g in greys]           # matplotlib grey: "0.0".."1.0"
        use_wavelength = True

    # Inner helper: mark a curve's peak and annotate it (nu_peak, T_max).
    # `identity_color` is this curve's colour. When use_wavelength is True the
    # marker/line/box-edge are coloured by the peak wavelength (meaningful);
    # otherwise they use identity_color so everything for one curve is one hue.
    annot_slots = [0]   # mutable counter so each box stacks below the previous

    def mark_and_annotate(d, nu, Lnu, identity_color):
        ipk = np.argmax(Lnu)
        nu_pk = nu[ipk]
        logL_pk = np.log10(Lnu[ipk])
        lam_pk_A = (c / nu_pk) * 1e8
        mark_color = wavelength_to_rgb(lam_pk_A) if use_wavelength else identity_color
        if mark_peaks:
            ax.axvline(np.log10(nu_pk), ls="--", lw=1.6, color=mark_color, alpha=0.9)
            ax.plot(np.log10(nu_pk), logL_pk, marker="v", color=mark_color,
                    ms=9, mec="black", mew=0.7, zorder=5)
        if annotate:
            r_max = (49.0 / 36.0) * d.r_in        # T_max at analytic peak radius
            T_max = float(d.T(r_max))
            y_anchor = 0.97 - 0.115 * annot_slots[0]
            annot_slots[0] += 1
            ax.annotate(
                rf"$\nu_{{\rm pk}}={nu_pk:.2e}$ Hz,  "
                rf"$T_{{\rm max}}={T_max:.2e}$ K",
                xy=(0.02, y_anchor), xycoords="axes fraction",
                fontsize=7.5, color="0.15", va="top",   # dark text: always legible
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=mark_color, alpha=0.9, lw=1.4))   # box edge = marker colour

    # optional fiducial reference underneath everything (fully marked too)
    if compare_with_fiducial:
        fid = ThinDisk(**FIDUCIAL)
        is_fid = lambda d: (d.M_BH_Msun == fid.M_BH_Msun and
                            d.Mdot_Msun_yr == fid.Mdot_Msun_yr and
                            d.a == fid.a)
        if not any(is_fid(d) for d in disks):
            nu_f, Lnu_f = fid.disk_sed()
            ax.plot(np.log10(nu_f), np.log10(Lnu_f),
                    ls="--", lw=1.6, color="grey", alpha=0.85,
                    label="fiducial (Q3)")
            mark_and_annotate(fid, nu_f, Lnu_f, identity_color="grey")

    top = -np.inf
    for i, (d, lab, col) in enumerate(zip(disks, labels, colors)):
        nu, Lnu = d.disk_sed()
        with np.errstate(divide="ignore"):
            logLnu = np.log10(Lnu)        # -inf in far tails is fine; clipped by ylim
        ax.plot(np.log10(nu), logLnu, lw=2.2, color=col, label=lab)
        top = max(top, np.nanmax(logLnu[np.isfinite(logLnu)]))
        mark_and_annotate(d, nu, Lnu, identity_color=col)

    if vis_line:
        nu_5000 = c / 5000e-8
        ax.axvline(np.log10(nu_5000), ls="-.", lw=1.4, color="black",
                   alpha=0.7, label=r"5000 $\AA$")

    ax.set_xlabel(r"$\log_{10}(\nu\,/\,\mathrm{Hz})$")
    ax.set_ylabel(r"$\log_{10}(L_\nu\,/\,\mathrm{erg\,s^{-1}\,Hz^{-1}})$")
    ax.set_title(title)
    ax.set_ylim(top - 6, top + 0.8)        # a little extra headroom for labels
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.show()
    print(f"   saved {filename}")


# ---------------------------------------------------------------------------
# Question 3 - fiducial SED
# ---------------------------------------------------------------------------
def question_3():
    print("=== Question 3: fiducial SED "
          "(M=1e8 Msun, Mdot=1 Msun/yr, a=0) ===")
    d = ThinDisk(**FIDUCIAL)
    nu, Lnu = d.disk_sed()
    ipk = np.argmax(Lnu)
    nu_pk, lam_pk = nu[ipk], (c / nu[ipk]) * 1e8

    # T_max: sample the profile on a fine grid between r_in and r_out
    r = np.logspace(np.log10(d.r_in), np.log10(d.r_out), 5000)
    T_max = d.T(r).max()

    L_sed = d.L_bol_sed()
    L_est = d.L_bol_estimate()

    print(f"   r_ISCO/r_g        = {d.x:.3f}")
    print(f"   efficiency eps_r  = {d.eps:.4f}")
    print(f"   T_max             = {T_max:.3e} K")
    print(f"   L_nu peak freq    = {nu_pk:.3e} Hz  (lambda = {lam_pk:.1f} A)")
    print(f"   L_bol (SED int)   = {L_sed:.4e} erg/s")
    print(f"   L_bol (eps Mdot c^2) = {L_est:.4e} erg/s")
    print(f"   ratio SED/estimate   = {L_sed / L_est:.3f}")

    plot_seds([d], ["fiducial (Q3)"],
              r"Q3: fiducial thin-disk SED",
              "q3_sed.png", compare_with_fiducial=False)


# ---------------------------------------------------------------------------
# Question 4 - effect of black-hole mass
# ---------------------------------------------------------------------------
def question_4():
    print("=== Question 4: effect of BH mass (a=0, Mdot=1 Msun/yr) ===")
    d9  = ThinDisk(1e9, 1.0, a=0.0)
    d10 = ThinDisk(1e10, 1.0, a=0.0)
    disks = [d9, d10]
    labels = [r"$10^9\,M_\odot$", r"$10^{10}\,M_\odot$"]

    for d, lab in zip(disks, labels):
        nu, Lnu = d.disk_sed()
        ipk = np.argmax(Lnu)
        r = np.logspace(np.log10(d.r_in), np.log10(d.r_out), 5000)
        print(f"   {lab:16s}  nu_peak={nu[ipk]:.3e} Hz  "
              f"lam={(c/nu[ipk])*1e8:9.1f} A  T_max={d.T(r).max():.2e} K")

    # part (d): nu_peak vs M slope (include fiducial as the 1e8 anchor)
    fid = ThinDisk(**FIDUCIAL)
    masses, peaks = [], []
    for d in (fid, d9, d10):
        nu, Lnu = d.disk_sed()
        masses.append(d.M_BH_Msun)
        peaks.append(nu[np.argmax(Lnu)])
    slope = np.polyfit(np.log10(masses), np.log10(peaks), 1)[0]
    print(f"   d log(nu_peak)/d log(M) = {slope:.3f}  (expect -0.5)")

    # part (c): 1e6 Msun at Eddington
    mdot_edd = Mdot_for_Eddington(1e6, a=0.0)
    d6 = ThinDisk(1e6, mdot_edd, a=0.0)
    nu, Lnu = d6.disk_sed()
    ipk = np.argmax(Lnu)
    print(f"   [4c] 1e6 Msun at Eddington: Mdot={mdot_edd:.4f} Msun/yr, "
          f"nu_peak={nu[ipk]:.3e} Hz, edd_ratio={d6.eddington_ratio():.3f}")

    plot_seds(disks, labels,
              r"Q4: effect of BH mass ($a=0,\ \dot M=1\,M_\odot$/yr)",
              "q4_mass.png", compare_with_fiducial=True)


# ---------------------------------------------------------------------------
# Question 5 - effect of accretion rate
# ---------------------------------------------------------------------------
def question_5():
    print("=== Question 5: effect of accretion rate (M=1e8 Msun, a=0) ===")
    rates = [0.1, 4.0, 10.0]
    disks = [ThinDisk(1e8, mdot, a=0.0) for mdot in rates]
    labels = [rf"$\dot M={m:g}\,M_\odot$/yr" for m in rates]

    mdot_edd = Mdot_for_Eddington(1e8, a=0.0)
    print(f"   Eddington rate (1e8, a=0) = {mdot_edd:.3f} Msun/yr "
          f"-> Mdot=10 is super-Eddington")
    for d, lab in zip(disks, labels):
        nu, Lnu = d.disk_sed()
        ipk = np.argmax(Lnu)
        r = np.logspace(np.log10(d.r_in), np.log10(d.r_out), 5000)
        print(f"   {lab:20s}  nu_peak={nu[ipk]:.3e} Hz  "
              f"T_max={d.T(r).max():.2e} K  edd_ratio={d.eddington_ratio():.3f}")

    plot_seds(disks, labels,
              r"Q5: effect of $\dot M$ ($M=10^8\,M_\odot,\ a=0$)",
              "q5_mdot.png", compare_with_fiducial=True,
              color_scheme="distinct_hues")


# ---------------------------------------------------------------------------
# Question 6 - effect of spin
# ---------------------------------------------------------------------------
def question_6():
    print("=== Question 6: effect of spin (M=1e8 Msun, Mdot=1 Msun/yr) ===")
    spins = [0.98, -1.0]
    disks = [ThinDisk(1e8, 1.0, a=a) for a in spins]
    labels = [rf"$a={a:g}$" for a in spins]

    for d, lab in zip(disks, labels):
        nu, Lnu = d.disk_sed()
        ipk = np.argmax(Lnu)
        r = np.logspace(np.log10(d.r_in), np.log10(d.r_out), 5000)
        print(f"   {lab:10s}  r_isco/rg={d.x:5.3f}  eps={d.eps:.4f}  "
              f"T_max={d.T(r).max():.2e} K  nu_peak={nu[ipk]:.3e} Hz")

    # part (b): compare L_nu in the visual (5000 A) across spins + fiducial
    nu_5000 = c / 5000e-8
    print("   L_nu at 5000 A (optical):")
    for d, lab in zip([ThinDisk(**FIDUCIAL)] + disks,
                      ["a=0 (fiducial)"] + labels):
        nu, Lnu = d.disk_sed()
        logLnu_vis = np.interp(np.log10(nu_5000), np.log10(nu), np.log10(Lnu))
        print(f"      {lab:16s}  log L_nu = {logLnu_vis:.3f}")

    plot_seds(disks, labels,
              r"Q6: effect of spin ($M=10^8\,M_\odot,\ \dot M=1\,M_\odot$/yr)",
              "q6_spin.png", compare_with_fiducial=True, vis_line=True,
              color_scheme="distinct_hues")


# ---------------------------------------------------------------------------
# Main block - toggle which questions to run
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main block - choose which questions to run via command-line flags
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Solve AGN PS1 questions 3-6 (thin-disk SEDs).")
    p.add_argument("--q3", action="store_true", help="run Question 3")
    p.add_argument("--q4", action="store_true", help="run Question 4")
    p.add_argument("--q5", action="store_true", help="run Question 5")
    p.add_argument("--q6", action="store_true", help="run Question 6")
    p.add_argument("--all", action="store_true",
                   help="run all questions (same as giving no flags)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If no specific question flag is set, run everything.
    run_all = args.all or not (args.q3 or args.q4 or args.q5 or args.q6)

    if run_all or args.q3:
        question_3()
    if run_all or args.q4:
        question_4()
    if run_all or args.q5:
        question_5()
    if run_all or args.q6:
        question_6()