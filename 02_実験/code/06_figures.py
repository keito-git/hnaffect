"""Result figures (matplotlib, Times New Roman). Canvas widths match IEEE two-column print
sizes (column 3.5 in, text 7.16 in) so that fonts print at their nominal size.

Fig. 2: (a) share of AI stories, (b) monthly M1 fear in both groups, (c) fear gap with ITS fit
Fig. 3: standardized level change beta_2 / SD_pre with 95% CI for every emotion x instrument
Fig. 4: (a) placebo distribution for M1 fear, (b) sensitivity to the pre-period start, (c) placebo false-positive rate by HAC lag
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "02_実験" / "results" / "its_seed2026"
OUT = [ROOT / "05_図表", ROOT / "06_paper" / "jp" / "figs", ROOT / "06_paper" / "en" / "figs"]
T0, HAC_LAG = 95, 4
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})
C_AI, C_CTL, C_FIT = "#C8651B", "#2F6FAE", "#222222"
EMOS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
INS = [("m1", "M1 DistilRoBERTa", "o"), ("m2", "M2 GoEmotions", "s"), ("m3", "M3 NRC", "^")]


def save(fig, name: str) -> None:
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(ROOT / "05_図表" / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def year_ticks(ax) -> None:
    ticks = list(range(0, 138, 24))
    ax.set_xticks(ticks, [str(2015 + t // 12) for t in ticks])
    ax.axvline(T0, color="0.4", lw=0.7, ls="--")


def fit_line(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    post = (t >= T0).astype(float)
    X = sm.add_constant(np.column_stack([t, post, (t - T0) * post]))
    return sm.OLS(y, X).fit().fittedvalues


def fig2(emotion: str = "fear") -> None:
    st = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet")
    st = st[st.num_comments >= 10].copy()
    st["t"] = (st.month.str[:4].astype(int) - 2015) * 12 + st.month.str[5:7].astype(int) - 1
    st = st[(st.t >= 0) & (st.t <= 137)]
    share = st.groupby("t").ai_broad.mean() * 100
    mm = pd.read_csv(RES / "monthly.csv", header=[0, 1], index_col=0)
    t = mm.index.values.astype(float)
    ai, ctl = mm[(f"m1_{emotion}", "ai")].values * 100, mm[(f"m1_{emotion}", "control")].values * 100
    gap = ai - ctl

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.55), constrained_layout=True)
    ax = axes[0]
    ax.plot(share.index, share.values, color=C_AI, lw=0.9)
    ax.set_ylabel("AI stories (%)")
    ax.set_title("(a) Share of AI stories", fontsize=8)
    year_ticks(ax)
    ax = axes[1]
    ax.plot(t, ai, color=C_AI, lw=0.7, label="AI")
    ax.plot(t, ctl, color=C_CTL, lw=0.7, label="Control")
    ax.set_ylabel(f"Mean {emotion} prob. (%)")
    ax.set_title(f"(b) M1 {emotion} by group", fontsize=8)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    year_ticks(ax)
    ax = axes[2]
    ax.plot(t, gap, color="0.55", lw=0.6, marker="o", ms=1.2)
    f = fit_line(t, gap)
    ax.plot(t[t < T0], f[t < T0], color=C_FIT, lw=1.1)
    ax.plot(t[t >= T0], f[t >= T0], color=C_FIT, lw=1.1)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel("AI $-$ control (pp)")
    ax.set_title(f"(c) {emotion.capitalize()} gap and ITS fit", fontsize=8)
    year_ticks(ax)
    save(fig, "fig2_trends")


def fig3() -> None:
    its = pd.read_csv(RES / "its_main.csv")
    its = its[its.model == "comparative"]
    fig, ax = plt.subplots(figsize=(3.5, 2.0), constrained_layout=True)
    y0 = np.arange(len(EMOS))[::-1]
    for j, (tag, lab, mk) in enumerate(INS):
        d = its[its.instrument == tag].set_index("emotion")
        emos = [e for e in EMOS if e in d.index]
        yy = np.array([y0[EMOS.index(e)] for e in emos]) + (1 - j) * 0.22
        scale = (d.loc[emos, "std_b2"] / d.loc[emos, "b2_level"]).values  # 1 / pre-period SD
        b = d.loc[emos, "std_b2"].values
        lo, hi = d.loc[emos, "ci_b2_lo"].values * scale, d.loc[emos, "ci_b2_hi"].values * scale
        ax.errorbar(b, yy, xerr=[b - lo, hi - b], fmt=mk, ms=3, lw=0.8, capsize=0,
                    color=["#222222", "#C8651B", "#2F6FAE"][j], label=lab)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_yticks(y0, EMOS)
    ax.set_xlabel(r"Standardized level change $\beta_2$ / SD$_{\mathrm{pre}}$")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig3_coefficients")


def fig4() -> None:
    pl = pd.read_csv(RES / "placebo.csv")
    its = pd.read_csv(RES / "its_main.csv")
    sc = pd.read_csv(RES / "prestart_curve.csv")
    true_b2 = its[(its.model == "comparative") & (its.instrument == "m1") & (its.emotion == "fear")].b2_level.iloc[0]
    d = pl[(pl.instrument == "m1") & (pl.emotion == "fear")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.55), constrained_layout=True)
    ax = axes[0]
    ax.hist(d.b2_level * 100, bins=15, color="0.75", edgecolor="white", lw=0.4, label="Placebo dates")
    ax.axvline(true_b2 * 100, color=C_AI, lw=1.4, label="ChatGPT release")
    ax.set_xlabel(r"$\beta_2$ for M1 fear (pp)")
    ax.set_ylabel("Count")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_title("(a) Placebo distribution", fontsize=8)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ax = axes[1]
    for j, (tag, lab, mk) in enumerate(INS):
        g = sc[(sc.instrument == tag) & (sc.emotion == "fear")].sort_values("start_year")
        ax.errorbar(g.start_year + (j - 1) * 0.12, g.b2_level * 100,
                    yerr=[(g.b2_level - g.ci_b2_lo) * 100, (g.ci_b2_hi - g.b2_level) * 100],
                    fmt=mk + "-", ms=2.5, lw=0.7, capsize=0,
                    color=["#222222", "#C8651B", "#2F6FAE"][j], label=lab.split()[0])
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Start of pre-period (January)")
    ax.set_ylabel(r"$\beta_2$ for fear (pp)")
    ax.set_title("(b) Sensitivity to pre-period", fontsize=8)
    ax.set_xticks(range(2015, 2022), [f"{y % 100:02d}" for y in range(2015, 2022)])
    ax = axes[2]
    hs = pd.read_csv(RES / "hac_sensitivity.csv")
    fp = hs.groupby("lag").placebo_share_p05.mean() * 100
    ax.bar([str(l) for l in fp.index], fp.values, color="0.6", width=0.6)
    ax.axhline(5, color=C_AI, lw=1.0, ls="--", label="Nominal 5%")
    ax.set_xlabel("HAC lag (months)")
    ax.set_ylabel(r"Placebos with $p<0.05$ (%)")
    ax.set_title("(c) Calibration of HAC tests", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    save(fig, "fig4_robustness")


def fig_combined() -> None:
    """Single full-width figure used in the 8-page version: (a) AI share, (b) fear gap with
    ITS fit, (c) placebo distribution for M1 fear, (d) placebo false-positive rate by HAC lag."""
    st = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet")
    st = st[st.num_comments >= 10].copy()
    st["t"] = (st.month.str[:4].astype(int) - 2015) * 12 + st.month.str[5:7].astype(int) - 1
    st = st[(st.t >= 0) & (st.t <= 137)]
    share = st.groupby("t").ai_broad.mean() * 100
    mm = pd.read_csv(RES / "monthly.csv", header=[0, 1], index_col=0)
    t = mm.index.values.astype(float)
    gap = (mm[("m1_fear", "ai")].values - mm[("m1_fear", "control")].values) * 100
    pl = pd.read_csv(RES / "placebo.csv")
    its_m = pd.read_csv(RES / "its_main.csv")
    true_b2 = its_m[(its_m.model == "comparative") & (its_m.instrument == "m1") & (its_m.emotion == "fear")].b2_level.iloc[0]
    hs = pd.read_csv(RES / "hac_sensitivity.csv")

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.55), constrained_layout=True)
    ax = axes[0]
    ax.plot(share.index, share.values, color=C_AI, lw=0.9)
    ax.set_ylabel("AI stories (%)")
    ax.set_title("(a) Share of AI stories", fontsize=8)
    year_ticks(ax)
    ax.set_xticks([0, 48, 96], ["2015", "2019", "2023"])
    ax = axes[1]
    ax.plot(t, gap, color="0.55", lw=0.5, marker="o", ms=1.0)
    f = fit_line(t, gap)
    ax.plot(t[t < T0], f[t < T0], color=C_FIT, lw=1.1)
    ax.plot(t[t >= T0], f[t >= T0], color=C_FIT, lw=1.1)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel("AI $-$ control (pp)")
    ax.set_title("(b) M1 fear gap, ITS fit", fontsize=8)
    year_ticks(ax)
    ax.set_xticks([0, 48, 96], ["2015", "2019", "2023"])
    ax = axes[2]
    d = pl[(pl.instrument == "m1") & (pl.emotion == "fear")]
    ax.hist(d.b2_level * 100, bins=12, color="0.75", edgecolor="white", lw=0.4)
    ax.axvline(true_b2 * 100, color=C_AI, lw=1.4)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(r"Placebo $\beta_2$ (pp)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Placebo, M1 fear", fontsize=8)
    ax = axes[3]
    fp = hs.groupby("lag").placebo_share_p05.mean() * 100
    ax.bar([str(l) for l in fp.index], fp.values, color="0.6", width=0.6)
    ax.axhline(5, color=C_AI, lw=1.0, ls="--")
    ax.set_xlabel("HAC lag (months)")
    ax.set_ylabel(r"Placebo $p<0.05$ (%)")
    ax.set_title("(d) Calibration", fontsize=8)
    save(fig, "fig2_combined")


def fig_ladder() -> None:
    """Full-width figure for the external-review version: (a) AI share, (b) M1 fear gap with ITS
    fit (primary sample), (c) distribution of fear beta_2 over bootstrap replicates with the ten
    independent samplings marked, (d) simulated false-positive rate by method and drift."""
    st = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet")
    st = st[st.num_comments >= 10].copy()
    st["t"] = (st.month.str[:4].astype(int) - 2015) * 12 + st.month.str[5:7].astype(int) - 1
    st = st[(st.t >= 0) & (st.t <= 137)]
    share = st.groupby("t").ai_broad.mean() * 100
    mm = pd.read_csv(RES / "monthly.csv", header=[0, 1], index_col=0)
    t = mm.index.values.astype(float)
    gap = (mm[("m1_fear", "ai")].values - mm[("m1_fear", "control")].values) * 100
    bs = pd.read_csv(ROOT / "02_実験" / "results" / "bootstrap.csv")
    seeds = []
    for d in sorted((ROOT / "02_実験" / "results").glob("its_seed*")):
        tt = pd.read_csv(d / "its_main.csv")
        r = tt[(tt.model == "comparative") & (tt.instrument == "m1") & (tt.emotion == "fear")].iloc[0]
        seeds.append((r.b2_level * 100, r.p_b2_holm < 0.05))
    sim = pd.read_csv(ROOT / "02_実験" / "results" / "simulation.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(share.index, share.values, color=C_AI, lw=0.9)
    ax.set_ylabel("AI stories (%)"); ax.set_title("(a) Share of AI stories", fontsize=8)
    year_ticks(ax); ax.set_xticks([0, 48, 96], ["2015", "2019", "2023"])
    ax = axes[1]
    ax.plot(t, gap, color="0.55", lw=0.5, marker="o", ms=1.0)
    f = fit_line(t, gap)
    ax.plot(t[t < T0], f[t < T0], color=C_FIT, lw=1.1); ax.plot(t[t >= T0], f[t >= T0], color=C_FIT, lw=1.1)
    ax.axhline(0, color="0.7", lw=0.5); ax.set_ylabel("AI $-$ control (pp)")
    ax.set_title("(b) M1 fear gap, sample 1", fontsize=8)
    year_ticks(ax); ax.set_xticks([0, 48, 96], ["2015", "2019", "2023"])
    ax = axes[2]
    ax.hist(bs.b2_fear * 100, bins=25, color="0.78", edgecolor="white", lw=0.3)
    ymax = ax.get_ylim()[1]
    for b2, sig in seeds:
        ax.plot([b2, b2], [0, ymax * 0.25], color=C_AI if sig else C_CTL, lw=1.0)
    ax.axvline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel(r"$\beta_2$ for M1 fear (pp)"); ax.set_ylabel("Bootstrap count")
    ax.set_title("(c) Sampling variability", fontsize=8)
    ax = axes[3]
    labs = {"single": "Single", "cits": "CITS", "cits_plac": "CITS+Placebo", "local24": "Local", "rcs": "Spline"}
    f0 = sim[sim.effect_sd == 0]
    x = np.arange(len(labs)); wdt = 0.26
    for j, (sc, col) in enumerate([("no_drift", "0.8"), ("half_drift", "0.55"), ("fitted", "0.3")]):
        v = [f0[(f0.scenario == sc) & (f0.method == m)].reject_rate.iloc[0] * 100 for m in labs]
        ax.bar(x + (j - 1) * wdt, v, wdt, color=col, label={"no_drift": "drift 0", "half_drift": "drift 0.5x", "fitted": "drift 1.0x"}[sc])
    ax.axhline(5, color=C_AI, lw=0.9, ls="--")
    ax.set_xticks(x, list(labs.values()), rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("False positives (%)"); ax.set_title("(d) Simulation, no effect", fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=3, handlelength=0.8, columnspacing=0.6,
              handletextpad=0.3, bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig2_ladder")


def fig_split() -> None:
    """Split the ladder figure so that each experiment subsection refers
    to its own figure. fig2_primary: (a) AI share, (b) M1 fear gap with ITS fit (column width);
    fig3_sampling: bootstrap distribution of fear beta_2 with the ten samplings marked."""
    st = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet")
    st = st[st.num_comments >= 10].copy()
    st["t"] = (st.month.str[:4].astype(int) - 2015) * 12 + st.month.str[5:7].astype(int) - 1
    st = st[(st.t >= 0) & (st.t <= 137)]
    share = st.groupby("t").ai_broad.mean() * 100
    mm = pd.read_csv(RES / "monthly.csv", header=[0, 1], index_col=0)
    t = mm.index.values.astype(float)
    gap = (mm[("m1_fear", "ai")].values - mm[("m1_fear", "control")].values) * 100
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.45), constrained_layout=True)
    ax = axes[0]
    ax.plot(share.index, share.values, color=C_AI, lw=0.9)
    ax.set_ylabel("AI stories (%)"); ax.set_title("(a) Share of AI stories", fontsize=8)
    year_ticks(ax); ax.set_xticks([0, 48, 96], ["2015", "2019", "2023"])
    ax = axes[1]
    ax.plot(t, gap, color="0.55", lw=0.5, marker="o", ms=1.0)
    f = fit_line(t, gap)
    ax.plot(t[t < T0], f[t < T0], color=C_FIT, lw=1.1); ax.plot(t[t >= T0], f[t >= T0], color=C_FIT, lw=1.1)
    ax.axhline(0, color="0.7", lw=0.5); ax.set_ylabel("AI $-$ control (pp)")
    ax.set_title("(b) M1 fear gap, sample 1", fontsize=8)
    year_ticks(ax); ax.set_xticks([0, 48, 96], ["2015", "2019", "2023"])
    save(fig, "fig2_primary")

    bs = pd.read_csv(ROOT / "02_実験" / "results" / "bootstrap.csv")
    seeds = []
    for d in sorted((ROOT / "02_実験" / "results").glob("its_seed*")):
        tt = pd.read_csv(d / "its_main.csv")
        r = tt[(tt.model == "comparative") & (tt.instrument == "m1") & (tt.emotion == "fear")].iloc[0]
        seeds.append((r.b2_level * 100, r.p_b2_holm < 0.05))
    fig, ax = plt.subplots(figsize=(3.5, 1.15), constrained_layout=True)
    ax.hist(bs.b2_fear * 100, bins=30, color="0.78", edgecolor="white", lw=0.3, label="Bootstrap replicates")
    ymax = ax.get_ylim()[1]
    done = set()
    for b2, sig in seeds:
        lab = ("Sampling, Holm $p<0.05$" if sig else "Sampling, Holm $p\\geq0.05$") if sig not in done else None
        done.add(sig)
        ax.plot([b2, b2], [0, ymax * 0.3], color=C_AI if sig else C_CTL, lw=1.1, label=lab)
    ax.axvline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel(r"$\beta_2$ for M1 fear (pp)"); ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "fig3_sampling")


if __name__ == "__main__":
    import sys
    if "split" in sys.argv:
        fig_split()
        sys.exit()
    if "ladder" in sys.argv:
        fig_ladder()
        sys.exit()
    fig_combined()
    fig2()
    fig3()
    fig4()
    print("figures saved")
