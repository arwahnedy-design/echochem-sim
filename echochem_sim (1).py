"""
EchoChem - Low-Clinker Cement from Local Wastes (Egypt)
CO2 balance + Supply Ceiling (Monte Carlo) + Pozzolanic kinetics + Net emissions
+ CBAM exposure + Sensitivity analysis + self-tests.

Python 3.9+  |  pip install numpy matplotlib   (streamlit only for the dashboard)

  python echochem_sim.py --selftest                 # checks the maths (run this first)
  python echochem_sim.py --run --n 20000 --out results [--clay 3]
  streamlit run echochem_sim.py                      # interactive dashboard

[PUB] = from a published source (see the Sources slide).
[ASSUMPTION] = our assumption: cite it or change it before presenting.
Results are a MODEL calibrated on published data, not laboratory measurements.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import os

import numpy as np



def _bisect(f, lo, hi, iters=200):
    """Simple bisection root finder (f(lo) and f(hi) must have opposite signs)."""
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spearman(x, y) -> float:
    """Spearman rank correlation (no ties expected for continuous draws)."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ============================ 1) PARAMETERS ============================
@dc.dataclass
class Params:
    # ---- [PUB] ----
    cement_mt: float = 50.0            # Mt cement/yr, Egypt 2023 (Maguire/Reuters 2025)
    clinker_factor: float = 0.89       # clinker/cement (EBRD, Cementis)
    ef_clinker: float = 1.107          # t CO2 per t clinker (EBRD)
    national_co2_mt: float = 279.0     # Mt CO2 energy+industry 2023 (Maguire 2025)
    exports_mt: float = 9.7            # Mt cement+clinker exports 2024 (Maguire 2025)
    straw_collected_mt: float = 2.7    # Mt straw collected, autumn 2025 (Daily News Egypt)
    marble_waste_mt: tuple = (0.448, 0.8)  # Mt/yr Shaq El-Thoban, published estimate range
    ash_max_share: float = 0.10        # optimum straw-ash share (Springer, Mater. Struct. 2025)
    lc3_limestone_share: float = 0.15  # limestone share in LC3-50 (LC3 project)
    clay_max_share: float = 0.30       # calcined-clay share in LC3-50 (LC3 project)
    target_clinker_factor: float = 0.80  # reference target (Cementis/EBRD)
    # ---- [ASSUMPTION] ----
    ash_yield: tuple = (0.12, 0.17, 0.22)         # ash/straw mass (min, mode, max)
    straw_to_cement: tuple = (0.05, 0.20, 0.50)   # share of straw reaching cement
    marble_to_cement: tuple = (0.10, 0.40, 0.80)  # share of marble waste reaching cement
    ratio_ash: float = 1.0       # t clinker replaced per t ash (simplification, <1 in reality)
    ratio_marble: float = 1.0    # t clinker replaced per t marble powder
    ratio_clay: float = 1.0      # t clinker replaced per t calcined clay
    clay_mt: float = 0.0         # Mt calcined clay available (scenario input)
    clay_ef: float = 0.20        # t CO2 per t calcined clay
    ash_ef: float = 0.05         # t CO2e per t ash (fossil share + handling)
    marble_ef: float = 0.01      # t CO2 per t marble powder (grinding)
    transport_km: float = 80.0   # average haul distance
    truck_ef: float = 0.0001     # t CO2 per t-km
    carbon_price_eur: float = 70.0  # EUR/t CO2, illustrative
    burn_temp_c: float = 600.0   # straw burning temperature


# ============================ 2) CO2 BALANCE ============================
def co2_per_t_cement(clinker_factor, p: Params):
    """t CO2 per t cement = clinker factor x 1.107 (EBRD: 0.89 x 1.107 = 0.985)."""
    return clinker_factor * p.ef_clinker


def reactivity(temp_c: float) -> float:
    """Amorphous-silica quality multiplier. [ASSUMPTION] shape: 1.0 up to 700 C
    (published window 450-650 C, amorphous to ~700 C), falling to 0.3 at 1000 C."""
    if temp_c <= 700:
        return 1.0
    return float(np.clip(1.0 - (temp_c - 700) / 300.0 * 0.7, 0.3, 1.0))


def scenario(p: Params, ash_mt, marble_mt, clay_mt) -> dict:
    """Net CO2 effect of ash/marble/clay masses (floats or numpy arrays, Mt/yr)."""
    ash = np.minimum(ash_mt, p.ash_max_share * p.cement_mt)
    marble = np.minimum(marble_mt, p.lc3_limestone_share * p.cement_mt)
    clay = np.minimum(clay_mt, p.clay_max_share * p.cement_mt)
    clinker_now = p.cement_mt * p.clinker_factor
    saved = np.minimum(ash * reactivity(p.burn_temp_c) * p.ratio_ash
                       + marble * p.ratio_marble + clay * p.ratio_clay, clinker_now)
    gross = saved * p.ef_clinker
    transport = (ash + marble + clay) * p.transport_km * p.truck_ef
    penalty = ash * p.ash_ef + marble * p.marble_ef + clay * p.clay_ef + transport
    net = gross - penalty
    new_cf = (clinker_now - saved) / p.cement_mt
    return dict(ash_used=ash, marble_used=marble, clay_used=clay, saved_clinker_mt=saved,
                gross_mt=gross, penalty_mt=penalty, net_mt=net, new_clinker_factor=new_cf,
                co2_per_t=co2_per_t_cement(new_cf, p), share_national_pct=100 * net / p.national_co2_mt)


# ============================ 3) SUPPLY CEILING ============================
def _tri(rng, t, n):
    lo, mode, hi = t
    hi = max(hi, lo + 1e-9)
    return rng.triangular(lo, min(max(mode, lo), hi), hi, n)


def _draw(p: Params, rng, n: int) -> dict:
    return dict(marble_waste=rng.uniform(*p.marble_waste_mt, n),
                marble_share=_tri(rng, p.marble_to_cement, n),
                straw_share=_tri(rng, p.straw_to_cement, n),
                ash_yield=_tri(rng, p.ash_yield, n))


def monte_carlo(p: Params, n: int = 20000, seed: int = 42) -> dict:
    d = _draw(p, np.random.default_rng(seed), n)
    ash = p.straw_collected_mt * d["straw_share"] * d["ash_yield"]
    marble = d["marble_waste"] * d["marble_share"]
    out = scenario(p, ash, marble, np.full(n, p.clay_mt))
    gap = max((p.clinker_factor - p.target_clinker_factor) * p.cement_mt, 1e-9)
    out["gap_mt"] = gap
    out["gap_closed_pct"] = 100 * out["saved_clinker_mt"] / gap
    out["inputs"] = d
    return out


def sensitivity(p: Params, mc: dict) -> dict:
    """Spearman rho between each uncertain input and the net CO2 saving (same draws as mc)."""
    net = mc["net_mt"]
    return {"marble waste (Mt)": spearman(mc["inputs"]["marble_waste"], net),
            "marble -> cement share": spearman(mc["inputs"]["marble_share"], net),
            "straw -> cement share": spearman(mc["inputs"]["straw_share"], net),
            "ash yield": spearman(mc["inputs"]["ash_yield"], net)}


def summarize(x) -> str:
    return (f"mean {np.mean(x):.2f} | P5 {np.percentile(x, 5):.2f} | "
            f"P50 {np.percentile(x, 50):.2f} | P95 {np.percentile(x, 95):.2f}")


# ============================ 4) POZZOLANIC KINETICS ============================
def lime_fixation(t_days, alpha_max, k, n=0.5):
    """Avrami-type: alpha(t) = alpha_max * (1 - exp(-(k t)^n)), n fixed at 0.5."""
    return alpha_max * (1.0 - np.exp(-(k * np.asarray(t_days, float)) ** n))


def fit_kinetics(points=((7, 0.82), (28, 0.87)), n=0.5):
    """Exact solution through two published lime-fixation points (82% @7d, 87% @28d).
    ILLUSTRATIVE: two points cannot validate a model; add more data when you find it."""
    (t1, a1), (t2, a2) = points
    g = lambda k: (1 - np.exp(-(k * t1) ** n)) / (1 - np.exp(-(k * t2) ** n)) - a1 / a2  # noqa: E731
    k = _bisect(g, 1e-6, 1e3)
    alpha_max = a1 / (1 - np.exp(-(k * t1) ** n))
    return float(alpha_max), float(k)


# ============================ 5) CBAM (indicative) ============================
def cbam_value(p: Params, net_mt: float) -> dict:
    """Value if the average saving per t cement also applied to exports (Mt x EUR/t = M EUR)."""
    per_t = net_mt / p.cement_mt
    exported = per_t * p.exports_mt
    return dict(saving_per_t=per_t, exported_mt_co2=exported, value_meur=exported * p.carbon_price_eur)


# ============================ 6) SELF-TEST ============================
def selftest() -> None:
    p = Params()
    assert abs(co2_per_t_cement(0.89, p) - 0.98523) < 1e-4, "baseline must be 0.985"
    assert abs(44 / 56 - 0.7857) < 1e-3, "CaCO3 -> CaO + CO2 stoichiometry"
    z = scenario(p, 0.0, 0.0, 0.0)
    assert abs(z["net_mt"]) < 1e-12 and abs(z["new_clinker_factor"] - 0.89) < 1e-12
    r = scenario(p, 0.0, 4.5, 0.0)  # 4.5 Mt marble (cap 7.5): 89% -> 80%
    assert abs(r["new_clinker_factor"] - 0.80) < 1e-9
    assert abs(r["gross_mt"] - 4.5 * 1.107) < 1e-9
    assert scenario(p, 100.0, 0.0, 0.0)["ash_used"] == p.ash_max_share * p.cement_mt, "ash cap"
    am, k = fit_kinetics()
    assert abs(lime_fixation(7, am, k) - 0.82) < 1e-9 and abs(lime_fixation(28, am, k) - 0.87) < 1e-9
    a, b = monte_carlo(p, 4000, 1), monte_carlo(p, 4000, 1)
    assert np.array_equal(a["net_mt"], b["net_mt"]), "seeded run must be reproducible"
    assert (a["net_mt"] >= 0).all() and (a["new_clinker_factor"] <= 0.89 + 1e-12).all()
    assert abs(cbam_value(p, 1.0)["value_meur"] - (1.0 / 50 * 9.7 * 70)) < 1e-9
    print("All self-tests passed.")


# ============================ 7) REPORT + PLOTS ============================
def plots(p: Params, mc: dict, sens: dict, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    cf = np.linspace(0.5, 0.89, 40)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(cf * 100, co2_per_t_cement(cf, p), lw=2)
    ax[0].axvline(89, ls="--", c="gray")
    ax[0].set(xlabel="Clinker factor (%)", ylabel="t CO2 per t cement", title="CO2 balance")
    ax[1].hist(mc["net_mt"], bins=50, color="#3fa796")
    ax[1].axvline(np.mean(mc["net_mt"]), c="k", ls="--")
    ax[1].set(xlabel="Net CO2 saving (Mt/yr)", title="Supply ceiling: local wastes only")
    ax[2].barh(list(sens), list(sens.values()), color="#e8a33d")
    ax[2].set(xlabel="Spearman rho", title="Sensitivity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "echochem_overview.png"), dpi=150)
    plt.close(fig)
    am, k = fit_kinetics()
    t = np.linspace(0, 90, 200)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(t, 100 * lime_fixation(t, am, k), lw=2)
    ax2.scatter([7, 28], [82, 87], c="r", zorder=3, label="published points")
    ax2.set(xlabel="Days", ylabel="Lime fixed (%)", title="Pozzolanic kinetics (illustrative)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "echochem_kinetics.png"), dpi=150)
    plt.close(fig2)


def report(p: Params, n: int, outdir: str | None = None) -> str:
    mc = monte_carlo(p, n)
    sens = sensitivity(p, mc)
    am, k = fit_kinetics()
    ref = (p.clinker_factor - p.target_clinker_factor) * p.cement_mt
    cb = cbam_value(p, float(np.mean(mc["net_mt"])))
    lines = [
        "=== EchoChem simulation ===",
        f"Baseline: {co2_per_t_cement(p.clinker_factor, p):.3f} t CO2 / t cement (clinker factor {p.clinker_factor:.0%})",
        f"Reference 89% -> 80%: needs {ref:.2f} Mt substitute, gross saving ~ {ref * p.ef_clinker:.2f} Mt CO2/yr",
        f"Local wastes, NET CO2 saving (Mt/yr): {summarize(mc['net_mt'])}",
        f"Share of national energy+industry CO2 (%): {summarize(mc['share_national_pct'])}",
        f"Share of the 80% gap closed (%): {summarize(mc['gap_closed_pct'])}",
        f"Clinker factor reached (%): {summarize(mc['new_clinker_factor'] * 100)}",
        f"Pozzolanic fit: alpha_max={am:.3f}, k={k:.3f}/day (illustrative, 2 points)",
        f"Ash reactivity at {p.burn_temp_c:.0f} C: {reactivity(p.burn_temp_c):.2f}",
        f"CBAM (indicative): {cb['saving_per_t']:.4f} t CO2/t cement -> ~{cb['value_meur']:.1f} M EUR/yr at {p.carbon_price_eur:.0f} EUR/t",
        "Sensitivity (Spearman rho): " + ", ".join(f"{a}: {b:+.2f}" for a, b in sens.items()),
        "Marble powder is CaCO3 (not burned); straw-ash CO2 is biogenic; ASSUMPTION values need citations.",
    ]
    if outdir:
        plots(p, mc, sens, outdir)
        lines.append(f"Plots saved to: {outdir}/")
    return "\n".join(lines)


# ============================ 8) STREAMLIT DASHBOARD ============================
def dashboard() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="EchoChem", layout="wide")
    st.markdown("<style>.block-container{direction:rtl;text-align:right}</style>", unsafe_allow_html=True)
    st.title("EchoChem — أسمنت أقل كربونًا من المخلفات")
    st.caption("نموذج محاكاة مبني على بيانات منشورة وليس قياسات معملية. القيم الموسومة ASSUMPTION افتراضات تحتاج مصدرًا.")
    p = Params()
    with st.sidebar:
        st.header("المدخلات · Inputs")
        p.clinker_factor = st.slider("نسبة الكلينكر الحالية", 0.81, 0.95, p.clinker_factor, 0.01)
        p.burn_temp_c = float(st.slider("حرارة حرق القش (°C)", 400, 1000, 600, 10))
        p.clay_mt = st.slider("طين مكلسن متاح (Mt)", 0.0, 10.0, 0.0, 0.5)
        p.transport_km = float(st.slider("مسافة النقل (km)", 10, 300, 80, 10))
        p.carbon_price_eur = float(st.slider("سعر الكربون (EUR/t)", 20, 150, 70, 5))
        lo, hi = st.slider("نسبة القش الواصل للأسمنت", 0.0, 1.0, (0.05, 0.5), 0.05)
        p.straw_to_cement = (lo, (lo + hi) / 2, hi)
        n = st.select_slider("عدد محاولات Monte Carlo", [2000, 5000, 20000, 50000], 5000)
    mc = monte_carlo(p, n)
    sens = sensitivity(p, mc)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CO₂ لكل طن أسمنت", f"{co2_per_t_cement(p.clinker_factor, p):.3f} t")
    c2.metric("متوسط التوفير الصافي", f"{mc['net_mt'].mean():.2f} Mt/yr")
    c3.metric("من انبعاثات مصر (طاقة+صناعة)", f"{mc['share_national_pct'].mean():.2f}%")
    c4.metric("من فجوة 80% المغلقة", f"{mc['gap_closed_pct'].mean():.0f}%")
    tabs = st.tabs(["CO₂ Balance", "Supply Ceiling", "Pozzolanic Kinetics", "CBAM", "Sensitivity", "Assumptions"])
    with tabs[0]:
        cf = np.linspace(0.5, 0.95, 40)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(cf * 100, co2_per_t_cement(cf, p), lw=2)
        ax.set(xlabel="Clinker factor (%)", ylabel="t CO2 / t cement")
        st.pyplot(fig); plt.close(fig)
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(mc["net_mt"], bins=50, color="#3fa796")
        ax.set(xlabel="Net CO2 saving (Mt/yr)")
        st.pyplot(fig); plt.close(fig)
        st.write(f"P5 = {np.percentile(mc['net_mt'], 5):.2f} | P95 = {np.percentile(mc['net_mt'], 95):.2f} Mt/yr")
    with tabs[2]:
        am, k = fit_kinetics()
        t = np.linspace(0, 90, 200)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(t, 100 * lime_fixation(t, am, k))
        ax.scatter([7, 28], [82, 87], c="r")
        ax.set(xlabel="Days", ylabel="Lime fixed (%)")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"جودة الرماد عند {p.burn_temp_c:.0f}°م = {reactivity(p.burn_temp_c):.2f} (تؤثر على الكلينكر المستبدل).")
    with tabs[3]:
        cb = cbam_value(p, float(mc["net_mt"].mean()))
        st.metric("قيمة تقديرية سنوية", f"{cb['value_meur']:.1f} M EUR")
        st.caption("إرشادي: يفترض نفس متوسط التوفير على الصادرات.")
    with tabs[4]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(sens), list(sens.values()), color="#e8a33d")
        ax.set(xlabel="Spearman rho")
        st.pyplot(fig); plt.close(fig)
    with tabs[5]:
        st.dataframe({k: [str(v)] for k, v in dc.asdict(p).items()})


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _safe_outdir(preferred: str) -> str:
    """Return a writable results folder (script folder first, then current folder)."""
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for root in (base, os.getcwd()):
        path = os.path.join(root, preferred)
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue
    return preferred


def _cli() -> None:
    import sys
    import traceback
    interactive = len(sys.argv) == 1          # e.g. pressing Run in Pydroid
    ap = argparse.ArgumentParser(description="EchoChem simulation")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--clay", type=float, default=0.0, help="Mt calcined clay available")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    try:
        if interactive or a.selftest:
            selftest()
        if interactive or a.run or not a.selftest:
            out = None
            if not a.no_plots:
                out = _safe_outdir(a.out)
            try:
                print(report(Params(clay_mt=a.clay), a.n, out))
            except Exception:                  # plotting problem: still print the numbers
                print("WARNING: plots failed, printing numbers only.")
                traceback.print_exc()
                print(report(Params(clay_mt=a.clay), a.n, None))
    except Exception:
        print("ERROR - send this text to get help:")
        traceback.print_exc()
    finally:
        if interactive:
            try:
                input("\nDone. Press Enter to close...")
            except Exception:
                pass


if _in_streamlit():
    dashboard()
elif __name__ == "__main__":
    _cli()
