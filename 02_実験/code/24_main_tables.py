"""Generate the result tables of the English main text (no hand-typed numbers).

Output: 06_paper/en/tables/tab_emotions.tex, tab_instruments.tex, tab_spec.tex, tab_sim.tex
All values are read from 02_実験/results; probabilities in percentage points (pp).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "02_実験" / "results"
OUT = ROOT / "06_paper" / "en" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
EMOS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SEEDS = sorted(int(p.name.replace("its_seed", "")) for p in RES.glob("its_seed*"))


def pp(x: float, d: int = 2) -> str:
    s = f"{100 * x:+.{d}f}"
    return s.replace("-", "$-$")


def pv(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def star(p: float) -> str:
    return "$^{*}$" if p < 0.05 else ""


def tab_emotions() -> None:
    t = pd.read_csv(RES / "its_seed2026" / "its_main.csv")
    t = t[t.instrument == "m1"].set_index(["model", "emotion"])
    rows = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
            " & \\multicolumn{2}{c}{Single series} & \\multicolumn{4}{c}{Comparative} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}",
            "Emotion & $\\beta_2$ & $\\beta_3$ & $\\beta_2$ & Holm $p$ & $\\beta_3$ & Holm $p$ \\\\", "\\midrule"]
    for e in EMOS:
        s, c = t.loc[("single_ai", e)], t.loc[("comparative", e)]
        rows.append(f"{e.capitalize()} & {pp(s.b2_level)}{star(s.p_b2)} & {pp(s.b3_slope, 3)}{star(s.p_b3)} & "
                    f"{pp(c.b2_level)} & {pv(c.p_b2_holm)} & {pp(c.b3_slope, 3)} & {pv(c.p_b3_holm)} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_emotions.tex").write_text("\n".join(rows) + "\n")


def tab_instruments() -> None:
    ve = pd.read_csv(RES / "llm_annot" / "validation_emotions.csv")
    t0 = pd.read_csv(RES / "its_seed2026" / "its_main.csv")
    t0 = t0[(t0.model == "comparative") & (t0.emotion == "fear")].set_index("instrument")
    seed = {k: [] for k in ["m1", "m2", "m3"]}
    for sd in SEEDS:
        t = pd.read_csv(RES / f"its_seed{sd}" / "its_main.csv")
        t = t[(t.model == "comparative") & (t.emotion == "fear")].set_index("instrument")
        for k in seed:
            seed[k].append((t.loc[k, "b2_level"], t.loc[k, "p_b2"]))
    n = len(SEEDS)
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "Instrument & $\\rho$ & AUC & Std.\\ $\\beta_2$ & Positive & $p<0.05$ \\\\", "\\midrule"]
    for k, lab in [("m1", "M1"), ("m2", "M2"), ("m3", "M3")]:
        r = ve[(ve.instrument == k) & (ve.emotion == "fear")].iloc[0]
        b = np.array(seed[k])
        rows.append(f"{lab} & {r.spearman:.2f} & {r.auc:.2f} & {t0.loc[k, 'std_b2']:+.2f} & "
                    f"{int((b[:, 0] > 0).sum())}/{n} & {int((b[:, 1] < 0.05).sum())}/{n} \\\\".replace("+-", "-").replace(" -", " $-$"))
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_instruments.tex").write_text("\n".join(rows) + "\n")


def tab_spec() -> None:
    sp = pd.read_csv(RES / "spec_sensitivity.csv")
    sp = sp[(sp["sample"].astype(str) == "2026") & (sp.instrument == "m1")]
    g = sp.set_index(["weighting", "spec"])
    lab = [("comment", "linear", "Linear pre-trend (primary)"), ("comment", "month_FE", "Month fixed effects"),
           ("comment", "fourier_K2", "Fourier terms, $K=2$"), ("comment", "quadratic", "Quadratic pre-trend"),
           ("comment", "arima100", "ARIMA(1,0,0) errors"), ("comment", "rcs4_pre", "Spline pre-trend"),
           ("comment", "local_pm12", "Local window $\\pm$12 months"), ("comment", "local_pm24", "Local window $\\pm$24 months"),
           ("comment", "local_pm36", "Local window $\\pm$36 months"), ("story", "linear", "Story-weighted means")]
    rows = ["\\begin{tabular}{llrr}", "\\toprule", "Step & Variant & $\\beta_2$ (pp) & $p$ \\\\", "\\midrule"]
    for w, s, name in lab:
        r = g.loc[(w, s)]
        rows.append(f"6 & {name} & {pp(r.b2)} & {pv(r.p)} \\\\")
    rows.append("\\midrule")
    mc = pd.read_csv(RES / "matched_control.csv").set_index("instrument")
    rows.append(f"7 & Topic-matched control & {pp(mc.loc['m1', 'b2'])} & {pv(mc.loc['m1', 'p_b2'])} \\\\")
    sc = pd.read_csv(RES / "synthetic_control.csv"); a = sc[sc.unit == "AI"].iloc[0]
    rows.append(f"7 & Synthetic control & {pp(a.post_mean_gap)} & rank {int(a.rank_ratio)}/{len(sc)} \\\\")
    rows.append("\\midrule")
    ts = pd.read_csv(RES / "target_subset_seed2026.csv").set_index("subset")
    rows.append(f"-- & AI-targeted comments & {pp(ts.loc['nli_ai_target', 'b2'])} & {pv(ts.loc['nli_ai_target', 'p_b2'])} \\\\")
    rows.append(f"-- & AI-mentioning comments & {pp(ts.loc['mentions_ai', 'b2'])} & {pv(ts.loc['mentions_ai', 'p_b2'])} \\\\")
    pm = pd.read_csv(RES / "mention_pooled.csv").set_index("measure").loc["m1_fear"]
    rows.append(f"-- & AI-mentioning, pooled, spline & {pp(pm.b2_rcs)} & {pv(pm.p_rcs)} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_spec.tex").write_text("\n".join(rows) + "\n")


def tab_sim() -> None:
    sim = pd.read_csv(RES / "simulation.csv")
    meth = [("single", "Single series"), ("cits", "Comparative"), ("cits_plac", "Comp.\\ + placebo"),
            ("local24", "Local $\\pm$24"), ("rcs", "Spline pre-trend")]
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            " & \\multicolumn{3}{c}{False positives, $\\delta=0$} & \\multicolumn{2}{c}{Power, drift 1.0$\\times$} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
            "Method & 0 & 0.5$\\times$ & 1.0$\\times$ & $\\delta=0.5$ & $\\delta=1.0$ \\\\", "\\midrule"]
    for m, name in meth:
        v = lambda sc, d: sim[(sim.scenario == sc) & (sim.method == m) & (sim.effect_sd == d)].reject_rate.iloc[0] * 100
        rows.append(f"{name} & {v('no_drift', 0):.1f} & {v('half_drift', 0):.1f} & {v('fitted', 0):.1f} & "
                    f"{v('fitted', 0.5):.1f} & {v('fitted', 1.0):.1f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_sim.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    tab_emotions(); tab_instruments(); tab_spec(); tab_sim()
    for f in sorted(OUT.glob("*.tex")):
        print(f"== {f.name}\n{f.read_text()}")
