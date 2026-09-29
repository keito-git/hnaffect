"""Generate LaTeX macros for every number reported in the paper (no hand-typed numbers).

Writes 06_paper/{jp,en}/numbers.tex, which both the Japanese and English versions \\input.
Probabilities are reported in percentage points (pp); slopes in pp per month.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "02_実験" / "results"
MAIN_SEED = 2026
T0 = 95
INS = {"m1": "Mone", "m2": "Mtwo", "m3": "Mthree", "m1m": "Mmask"}
EMO = {"anger": "Anger", "disgust": "Disgust", "fear": "Fear", "joy": "Joy",
       "neutral": "Neutral", "sadness": "Sadness", "surprise": "Surprise"}


def fmt_p(p: float) -> str:
    """Includes the relation sign so that the text reads "$p$~\\pLev..." in both cases."""
    if p < 0.001:
        return "$<$ 0.001"
    return f"= {p:.3f}"


def fmt_pp(x: float, d: int = 2) -> str:
    s = f"{100 * x:+.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def main() -> None:
    out: dict[str, str] = {}
    d = RES / f"its_seed{MAIN_SEED}"
    meta = json.load(open(d / "meta.json"))
    stories = pd.read_parquet(ROOT / "02_実験" / "data" / "stories_all.parquet")
    stories["t"] = (stories["month"].str[:4].astype(int) - 2015) * 12 + stories["month"].str[5:7].astype(int) - 1
    stories = stories[(stories.t >= 0) & (stories.t <= 137)]
    elig = stories[stories.num_comments >= 10]
    out["NumMonths"] = "138"
    out["NumTzero"] = str(T0)
    out["NumStoriesAll"] = f"{len(stories):,}"
    out["NumStoriesElig"] = f"{len(elig):,}"
    out["NumAIStoriesElig"] = f"{int(elig.ai_broad.sum()):,}"
    out["NumComments"] = f"{meta['n_comments']:,}"
    out["NumStoriesSampled"] = f"{meta['n_stories']:,}"
    out["NumCommentsAI"] = f"{meta['by_group']['ai']:,}"
    out["NumCommentsCtl"] = f"{meta['by_group']['control']:,}"
    share = elig.groupby("t").ai_broad.mean()
    out["ShareAIPre"] = f"{100 * share[share.index < T0].mean():.1f}"
    out["ShareAIPost"] = f"{100 * share[share.index >= T0].mean():.1f}"
    out["ShareRatio"] = f"{share[share.index >= T0].mean() / share[share.index < T0].mean():.1f}"
    out["ShareAIMax"] = f"{100 * share.max():.1f}"
    stable_share = elig.groupby("t").ai_stable.mean()
    out["ShareStablePost"] = f"{100 * stable_share[stable_share.index >= T0].mean():.1f}"
    cnt = pd.read_csv(d / "counts.csv")
    out["MinStoriesAIMonth"] = str(int(cnt[cnt.group == "ai"].n_stories.min()))

    tc = json.load(open(RES / "title_check" / "title_check.json"))
    out["TitlePrecision"] = f"{tc['precision']:.2f}"
    out["TitleFN"] = str(tc["fn"])
    out["TitleContam"] = f"{100 * tc['control_contamination']:.1f}"
    for per in ["pre", "post"]:
        P = per.capitalize()
        out[f"TitleCtl{P}N"] = str(tc[f"ctl_{per}_n"])
        out[f"TitleCtl{P}FN"] = str(tc[f"ctl_{per}_fn"])

    its = pd.read_csv(d / "its_main.csv")
    for _, r in its.iterrows():
        if r.emotion not in EMO:
            continue
        prefix = {"comparative": "C", "single_ai": "S", "single_control": "K"}[r.model]
        key = prefix + INS[r.instrument] + EMO[r.emotion]
        out[f"std{key}"] = f"{r.std_b2:+.2f}".replace("-", "$-$")
        out[f"bLev{key}"] = fmt_pp(r.b2_level)
        out[f"bLevCI{key}"] = f"[{fmt_pp(r.ci_b2_lo)}, {fmt_pp(r.ci_b2_hi)}]"
        out[f"pvLev{key}"] = fmt_p(r.p_b2).replace("= ", "").replace("$<$ ", "$<$")
        out[f"pLev{key}"] = fmt_p(r.p_b2)
        out[f"bSlo{key}"] = fmt_pp(r.b3_slope, 3)
        out[f"pSlo{key}"] = fmt_p(r.p_b3)
        out[f"rel{key}"] = f"{100 * r.rel_b2:+.1f}".replace("-", "$-$")
        out[f"pre{key}"] = f"{100 * r.pre_mean_ai:.2f}"
        if not np.isnan(r.get("p_b2_holm", np.nan)):
            out[f"hvLev{key}"] = fmt_p(r.p_b2_holm).replace("= ", "").replace("$<$ ", "$<$")
            out[f"hLev{key}"] = fmt_p(r.p_b2_holm)
            out[f"hSlo{key}"] = fmt_p(r.p_b3_holm)

    cm = its[its.model == "comparative"].set_index(["instrument", "emotion"])
    out["FearStdRatio"] = f"{cm.loc[('m1', 'fear'), 'std_b2'] / cm.loc[('m2', 'fear'), 'std_b2']:.1f}"

    rob = pd.read_csv(d / "its_robust.csv")
    for _, r in rob[rob.model == "comparative"].iterrows():
        tag = "Stable" if r["sample"] == "stable_keywords" else "Mask"
        out[f"bLev{tag}{EMO[r.emotion]}"] = fmt_pp(r.b2_level)
        out[f"pLev{tag}{EMO[r.emotion]}"] = fmt_p(r.p_b2)
        out[f"bSlo{tag}{EMO[r.emotion]}"] = fmt_pp(r.b3_slope, 3)
        out[f"pSlo{tag}{EMO[r.emotion]}"] = fmt_p(r.p_b3)

    cl = pd.read_csv(d / "comment_level.csv")
    for _, r in cl.iterrows():
        out[f"bLevCL{EMO[r.emotion]}"] = fmt_pp(r.b2_level)
        out[f"pLevCL{EMO[r.emotion]}"] = fmt_p(r.p_b2)
        out[f"bSloCL{EMO[r.emotion]}"] = fmt_pp(r.b3_slope, 3)
        out[f"pSloCL{EMO[r.emotion]}"] = fmt_p(r.p_b3)

    pl = pd.read_csv(d / "placebo_summary.csv")
    for _, r in pl.iterrows():
        out[f"plEmp{INS[r.instrument]}{EMO[r.emotion]}"] = f"{r.emp_p:.3f}"
        out[f"plSig{INS[r.instrument]}{EMO[r.emotion]}"] = f"{100 * r.share_placebo_p05:.1f}"
        out[f"plSame{INS[r.instrument]}{EMO[r.emotion]}"] = f"{r.emp_p_same_sign:.3f}"
        out[f"plSparse{INS[r.instrument]}{EMO[r.emotion]}"] = f"{r.emp_p_sparse:.3f}"
        out[f"plNexc{INS[r.instrument]}{EMO[r.emotion]}"] = str(int(r.n_exceed))
        out[f"plOpp{INS[r.instrument]}{EMO[r.emotion]}"] = str(int(r.exceed_opposite_sign))
        if r.n_exceed:
            ym = lambda k: f"{2015 + k // 12}/{k % 12 + 1:02d}"
            out[f"plFrom{INS[r.instrument]}{EMO[r.emotion]}"] = ym(int(r.exceed_t0_min))
            out[f"plTo{INS[r.instrument]}{EMO[r.emotion]}"] = ym(int(r.exceed_t0_max))
    out["NumPlacebo"] = str(int(pl.n_placebo.iloc[0]))
    out["NumPlaceboSparse"] = str(int(pl.n_placebo_sparse.iloc[0]))
    fr = pl[(pl.instrument == "m1") & (pl.emotion == "fear")].iloc[0]
    out["plSparseExcMoneFear"] = str(int(round(fr.emp_p_sparse * fr.n_placebo_sparse)))
    out["plSigAllMone"] = f"{100 * pl[pl.instrument == 'm1'].share_placebo_p05.mean():.1f}"

    # post-hoc: trend-free local contrast and pre-period start curve
    lc = pd.read_csv(d / "local_contrast.csv")
    for _, r in lc.iterrows():
        k = INS[r.instrument] + EMO[r.emotion]
        out[f"loc{k}"] = fmt_pp(r["diff"])
        out[f"pLoc{k}"] = fmt_p(r.p_welch)
        out[f"pvLoc{k}"] = fmt_p(r.p_welch).replace("= ", "").replace("$<$ ", "$<$")
    sc = pd.read_csv(d / "prestart_curve.csv")
    for (tag, e), g in sc.groupby(["instrument", "emotion"]):
        k = INS[tag] + EMO[e]
        out[f"scMin{k}"] = fmt_pp(g.b2_level.min())
        out[f"scMax{k}"] = fmt_pp(g.b2_level.max())
        out[f"scNsig{k}"] = str(int((g.p_b2 < 0.05).sum()))
        out[f"scNpos{k}"] = str(int((g.b2_level > 0).sum()))
    out["NumStarts"] = str(sc.start_year.nunique())

    hs = pd.read_csv(d / "hac_sensitivity.csv")
    for lag, g in hs.groupby("lag"):
        L = {2: "Two", 4: "Four", 8: "Eight", 12: "Twelve"}[int(lag)]
        out[f"hacFP{L}"] = f"{100 * g.placebo_share_p05.mean():.1f}"
        out[f"hacPFear{L}"] = fmt_p(g[g.emotion == "fear"].p_b2.iloc[0])
        out[f"hacFPFear{L}"] = f"{100 * g[g.emotion == 'fear'].placebo_share_p05.iloc[0]:.1f}"
        out[f"hacSEFear{L}"] = f"{100 * g[g.emotion == 'fear'].se_b2.iloc[0]:.3f}"

    agg_path = d / "m2_aggregation.csv"
    if agg_path.exists():
        ag = pd.read_csv(agg_path)
        for _, r in ag.iterrows():
            A = {"max": "Max", "mean": "Mean", "noisyor": "Nor"}[r.aggregation]
            out[f"bAgg{A}{EMO[r.emotion]}"] = fmt_pp(r.b2_level)
            out[f"stdAgg{A}{EMO[r.emotion]}"] = f"{r.std_b2:+.2f}".replace("-", "$-$")
            out[f"pAgg{A}{EMO[r.emotion]}"] = fmt_p(r.p_b2)

    # seed replication (M1 comparative level change)
    seeds = sorted(int(p.name.replace("its_seed", "")) for p in RES.glob("its_seed*"))
    holm_f, emp_f = [], []
    for sd in seeds:
        t = pd.read_csv(RES / f"its_seed{sd}" / "its_main.csv")
        r = t[(t.instrument == "m1") & (t.model == "comparative") & (t.emotion == "fear")].iloc[0]
        holm_f.append(r.p_b2_holm)
        pls = pd.read_csv(RES / f"its_seed{sd}" / "placebo_summary.csv")
        emp_f.append(pls[(pls.instrument == "m1") & (pls.emotion == "fear")].emp_p.iloc[0])
    other = [sd for sd in seeds if sd != MAIN_SEED]
    oh = [h for sd, h in zip(seeds, holm_f) if sd != MAIN_SEED]
    oe = [e for sd, e in zip(seeds, emp_f) if sd != MAIN_SEED]
    out["NumSeedsOther"] = str(len(other))
    out["seedOtherHolmMinFear"] = f"{min(oh):.2f}"
    out["seedOtherHolmMaxFear"] = f"{max(oh):.2f}"
    out["seedOtherEmpMinFear"] = f"{min(oe):.2f}"
    out["seedOtherEmpMaxFear"] = f"{max(oe):.2f}"
    ob = []
    for sd in other:
        t = pd.read_csv(RES / f"its_seed{sd}" / "its_main.csv")
        ob.append(t[(t.instrument == "m1") & (t.model == "comparative") & (t.emotion == "fear")].b2_level.iloc[0])
    out["seedOtherMinFear"] = fmt_pp(min(ob))
    out["seedOtherMaxFear"] = fmt_pp(max(ob))
    n_sig_any = 0
    for sd in other:
        t = pd.read_csv(RES / f"its_seed{sd}" / "its_main.csv")
        f = t[(t.instrument == "m1") & (t.model == "comparative")]
        n_sig_any += int(((f.p_b2_holm < 0.05) | (f.p_b3_holm < 0.05)).any())
    # pooled (post-hoc)
    pdir = RES / "its_pooled"
    if pdir.exists():
        pm = pd.read_csv(pdir / "its_main.csv")
        pm = pm[pm.model == "comparative"].set_index(["instrument", "emotion"])
        f = pm.loc[("m1", "fear")]
        out["poolFear"] = fmt_pp(f.b2_level)
        out["poolCIFear"] = f"[{fmt_pp(f.ci_b2_lo)}, {fmt_pp(f.ci_b2_hi)}]"
        out["poolPFear"] = fmt_p(f.p_b2)
        out["poolHolmFear"] = fmt_p(f.p_b2_holm)
        out["poolStdFear"] = f"{f.std_b2:+.2f}"
        out["poolStdMtwoFear"] = f"{pm.loc[('m2', 'fear')].std_b2:+.2f}".replace("-", "$-$")
        out["poolStdMthreeFear"] = f"{pm.loc[('m3', 'fear')].std_b2:+.2f}".replace("-", "$-$")
        pp = pd.read_csv(pdir / "placebo_summary.csv")
        out["poolEmpFear"] = f"{pp[(pp.instrument == 'm1') & (pp.emotion == 'fear')].emp_p.iloc[0]:.3f}"
        pmeta = json.load(open(pdir / "meta.json"))
        out["poolComments"] = f"{pmeta['n_comments_unique']:,}"
        sl = pm.loc["m1"]
        out["poolSloSigEmos"] = ", ".join(e for e in sl.index if sl.loc[e, "p_b3_holm"] < 0.05)
    out["NumSeeds"] = str(len(seeds))
    for e, E in EMO.items():
        vals = []
        for s in seeds:
            t = pd.read_csv(RES / f"its_seed{s}" / "its_main.csv")
            row = t[(t.instrument == "m1") & (t.model == "comparative") & (t.emotion == e)].iloc[0]
            vals.append(row.b2_level)
        out[f"seedMin{E}"] = fmt_pp(min(vals))
        out[f"seedMax{E}"] = fmt_pp(max(vals))

    # ---- external-review additions (each block runs only if its result file exists) ----
    L2A = {"linear": "Lin", "month_FE": "Mfe", "fourier_K2": "Four", "quadratic": "Quad", "rcs4_pre": "Rcs",
           "arima100": "Arima", "local_pm12": "LocA", "local_pm24": "LocB", "local_pm36": "LocC"}
    # seeds: counts over all available samplings
    fear_rows = []
    for sd in seeds:
        t = pd.read_csv(RES / f"its_seed{sd}" / "its_main.csv"); c = t[t.model == "comparative"]
        f1 = c[(c.instrument == "m1") & (c.emotion == "fear")].iloc[0]
        f2 = c[(c.instrument == "m2") & (c.emotion == "fear")].iloc[0]
        f3 = c[(c.instrument == "m3") & (c.emotion == "fear")].iloc[0]
        fear_rows.append((f1.b2_level, f1.p_b2, f1.p_b2_holm, f2.b2_level, f3.b2_level))
    fr = np.array(fear_rows)
    out["seedN"] = str(len(fr))
    out["seedPos"] = str(int((fr[:, 0] > 0).sum()))
    out["seedNomSig"] = str(int((fr[:, 1] < 0.05).sum()))
    out["seedHolmSig"] = str(int((fr[:, 2] < 0.05).sum()))
    out["seedAgree"] = str(int(((fr[:, 0] > 0) & (fr[:, 3] > 0) & (fr[:, 4] > 0)).sum()))
    out["seedMedFear"] = fmt_pp(float(np.median(fr[:, 0])))
    bj = RES / "bootstrap_summary.json"
    if bj.exists():
        b = json.load(open(bj))
        out["bsB"] = f"{b['B']:,}"
        if "n_comments_pool" in b:
            out["bsPoolComments"] = f"{b['n_comments_pool']:,}"
        out["bsMean"] = fmt_pp(b["fear_b2_mean_pp"] / 100)
        out["bsLo"] = fmt_pp(b["fear_b2_q025_pp"] / 100); out["bsHi"] = fmt_pp(b["fear_b2_q975_pp"] / 100)
        for k, K in [("share_fear_b2_pos", "Pos"), ("share_fear_nominal_p05", "Nom"), ("share_fear_holm_p05", "Holm"),
                     ("share_sign_agree_3", "Agree"), ("share_surprise_b3_neg", "SurNeg"), ("share_anger_b3_pos", "AngPos"),
                     ("share_any_holm_p05", "Any")]:
            out[f"bs{K}"] = f"{100 * b[k]:.1f}"
    ss = RES / "spec_sensitivity.csv"
    if ss.exists():
        sp = pd.read_csv(ss)
        for (smp, w), g in sp[sp.instrument == "m1"].groupby(["sample", "weighting"]):
            S = "A" if str(smp) == "2026" else "P"
            W = "" if w == "comment" else "Story"
            for _, r in g.iterrows():
                out[f"spec{L2A[r.spec]}{W}{S}"] = fmt_pp(r.b2)
                out[f"specp{L2A[r.spec]}{W}{S}"] = fmt_p(r.p)
    mc = RES / "matched_control.csv"
    if mc.exists():
        m = pd.read_csv(mc).set_index("instrument")
        out["mtFear"] = fmt_pp(m.loc["m1", "b2"]); out["mtP"] = fmt_p(m.loc["m1", "p_b2"])
        out["mtExceed"] = f"{m.loc['m1', 'placebo_exceed']:.3f}"
        out["mtStdTwo"] = f"{m.loc['m2', 'std_b2']:+.2f}".replace("-", "$-$")
        out["mtStdThree"] = f"{m.loc['m3', 'std_b2']:+.2f}".replace("-", "$-$")
        pr = pd.read_csv(RES / "matched_pairs_seed2026.csv")
        out["mtSim"] = f"{pr.similarity.mean():.2f}"; out["mtPairs"] = f"{len(pr):,}"
    scf = RES / "synthetic_control.csv"
    if scf.exists():
        sc = pd.read_csv(scf); a = sc[sc.unit == "AI"].iloc[0]
        out["scRank"] = str(int(a.rank_ratio)); out["scUnits"] = str(len(sc))
        out["scGap"] = fmt_pp(a.post_mean_gap); out["scMonths"] = str(int(a.n_months))
        out["scP"] = f"{a.rank_ratio / len(sc):.3f}"
    ptf = RES / "pseudo_treated.csv"
    if ptf.exists():
        pt = pd.read_csv(ptf)
        out["ptTopics"] = str(pt.topic.nunique())
        out["ptShareBtwo"] = f"{100 * (pt.p_b2 < 0.05).mean():.1f}"
        out["ptShareBthree"] = f"{100 * (pt.p_b3 < 0.05).mean():.1f}"
        out["ptShareFear"] = f"{100 * (pt[pt.emotion == 'fear'].p_b2 < 0.05).mean():.1f}"
    smf = RES / "simulation.csv"
    if smf.exists() and "scenario" in pd.read_csv(smf).columns:
        sm_ = pd.read_csv(smf)
        M = {"single": "single", "cits": "cits", "cits_plac": "plac", "local24": "local", "rcs": "rcs"}
        SC = {"no_drift": "None", "half_drift": "Half", "fitted": "Fitted"}
        for _, r in sm_.iterrows():
            key = ("simFP" if r.effect_sd == 0 else f"simPow{ {0.5: 'H', 1.0: 'O', 1.5: 'T'}[r.effect_sd] }".replace(" ", "")) + M[r.method] + SC[r.scenario]
            out[key] = f"{100 * r.reject_rate:.1f}"
    # (stratified Claude-coded title check superseded by the LLM-reference check in validation.json)
    vj = RES / "llm_annot" / "validation.json"
    if vj.exists():
        v = json.load(open(vj))
        ve = pd.read_csv(RES / "llm_annot" / "validation_emotions.csv")
        kap = pd.read_csv(ROOT / "04_人手注釈キット" / "comments_annotation_sheet.csv")
        out["valNComments"] = f"{len(kap):,}"
        out["valNTitles"] = str(len(pd.read_csv(ROOT / "04_人手注釈キット" / "titles_annotation_sheet.csv")))
        out["valKappaFear"] = f"{v['agreement']['fear']:.2f}"
        out["valKappaMin"] = f"{min(v['agreement'][e] for e in EMO if e in v['agreement']):.2f}"
        out["valKappaMax"] = f"{max(v['agreement'][e] for e in EMO if e in v['agreement']):.2f}"
        out["valKappaTarget"] = f"{v['agreement']['target']:.2f}"
        out["valKappaTitle"] = f"{v['agreement']['title']:.2f}"
        for tag in ["m1", "m2", "m3"]:
            r = ve[(ve.instrument == tag) & (ve.emotion == "fear")].iloc[0]
            out[f"valRhoFear{INS[tag]}"] = f"{r.spearman:.2f}"
            if "auc" in r and not pd.isna(r.auc):
                out[f"valAucFear{INS[tag]}"] = f"{r.auc:.2f}"
            out[f"valDidAgree{INS[tag]}"] = str(int(ve[ve.instrument == tag].did_sign_agree.sum()))
            out[f"valDidN{INS[tag]}"] = str(int((ve.instrument == tag).sum()))
        out["tgtSharePre"] = f"{100 * v['ai_target_share_pre']:.0f}"
        out["tgtAgreeN"] = str(int(v["ai_target_n_pre"] + v["ai_target_n_post"]))
        out["tgtSharePost"] = f"{100 * v['ai_target_share_post']:.0f}"
        ia = v["instrument_agreement"]["fear"]
        out["instRhoOneTwo"], out["instRhoOneThree"], out["instRhoTwoThree"] = (f"{ia[k]:.2f}" for k in ["m1_m2", "m1_m3", "m2_m3"])
        tt = v["titles"]
        out["llmTitleRecall"] = f"{100 * tt['recall_overall']:.0f}"
        out["llmTitleRecallEarly"] = f"{100 * tt['2015-2018']['recall']:.0f}"
        out["llmTitleRecallLate"] = f"{100 * tt['2023-2026']['recall']:.0f}"
        # precision/contamination pooled over strata (unweighted, agreed titles)
        out["llmTitlePrec"] = f"{100 * np.mean([tt[s]['precision'] for s in ['2015-2018', '2019-2022', '2023-2026']]):.0f}"
        out["llmTitleCont"] = f"{100 * np.mean([tt[s]['contamination'] for s in ['2015-2018', '2019-2022', '2023-2026']]):.1f}"
    tsf = RES / "target_subset.csv"
    if tsf.exists():
        ts_ = pd.read_csv(tsf).set_index("subset")
        for nm, K in [("nli_ai_target", "Nli"), ("mentions_ai", "Men")]:
            out[f"tgt{K}Fear"] = fmt_pp(ts_.loc[nm, "b2"]); out[f"tgt{K}P"] = fmt_p(ts_.loc[nm, "p_b2"])
            out[f"tgt{K}Exc"] = f"{ts_.loc[nm, 'placebo_exceed']:.3f}"
            out[f"tgt{K}N"] = f"{int(ts_.loc[nm, 'n_ai_comments']):,}"
        tj = json.load(open(RES / "target_subset.json"))
        for k in ["nli_precision", "nli_recall"]:
            if k in tj:
                out["tgt" + ("NliPrec" if "prec" in k else "NliRec")] = f"{100 * tj[k]:.0f}"

    mr = RES / "mention_robustness.csv"
    if mr.exists():
        m = pd.read_csv(mr)
        for meas, K in [("m1_fear", "One"), ("m1m_fear", "Mask"), ("m2_fear", "Two"), ("m3_fear", "Three")]:
            g = m[m.measure == meas]
            out[f"menPos{K}"] = str(int((g.b2 > 0).sum())); out[f"menSig{K}"] = str(int((g.p_b2 < 0.05).sum()))
            out[f"menMin{K}"] = fmt_pp(g.b2.min()); out[f"menMax{K}"] = fmt_pp(g.b2.max())
        out["menN"] = str(m.seed.nunique())
        pm = pd.read_csv(RES / "mention_pooled.csv").set_index("measure")
        out["menPoolFear"] = fmt_pp(pm.loc["m1_fear", "b2"]); out["menPoolP"] = fmt_p(pm.loc["m1_fear", "p_b2"])
        out["menPoolExc"] = f"{pm.loc['m1_fear', 'placebo_exceed']:.3f}"
        out["menPoolRcs"] = fmt_pp(pm.loc["m1_fear", "b2_rcs"]); out["menPoolRcsP"] = fmt_p(pm.loc["m1_fear", "p_rcs"])
        out["menPoolN"] = f"{int(pm.loc['m1_fear', 'n_ai_mention_comments']):,}"
    nli_rows = []
    for sd in (2026, 2027, 2028):
        f = RES / f"target_subset_seed{sd}.csv"
        if f.exists():
            d = pd.read_csv(f).set_index("subset")
            if "nli_ai_target" in d.index:
                nli_rows.append((sd, d.loc["nli_ai_target", "b2"], d.loc["nli_ai_target", "p_b2"], d.loc["nli_ai_target", "placebo_exceed"]))
    if nli_rows:
        out["nliSeedsN"] = str(len(nli_rows))
        out["nliMin"] = fmt_pp(min(r[1] for r in nli_rows)); out["nliMax"] = fmt_pp(max(r[1] for r in nli_rows))
        out["nliSig"] = str(sum(r[2] < 0.05 for r in nli_rows)); out["nliCal"] = str(sum(r[3] < 0.05 for r in nli_rows))

    bad = [k for k in out if not k.isalpha()]
    assert not bad, f"LaTeX macro names must be letters only: {bad[:5]}"
    # per-sampling table (seeds + pooled), incl. post-hoc slope placebo calibration
    sp = pd.read_csv(RES / "slope_placebo.csv") if (RES / "slope_placebo.csv").exists() else None
    tags = {"2026": "Sa", "2027": "Sb", "2028": "Sc", "pooled": "Sp"}
    for smp, T in tags.items():
        dd = RES / (f"its_seed{smp}" if smp != "pooled" else "its_pooled")
        t = pd.read_csv(dd / "its_main.csv")
        c = t[t.model == "comparative"].set_index(["instrument", "emotion"])
        f = c.loc[("m1", "fear")]
        out[f"tab{T}FearB"] = fmt_pp(f.b2_level)
        out[f"tab{T}FearH"] = fmt_p(f.p_b2_holm).replace("= ", "").replace("$<$ ", "$<$")
        pls = pd.read_csv(dd / "placebo_summary.csv")
        out[f"tab{T}FearE"] = f"{pls[(pls.instrument == 'm1') & (pls.emotion == 'fear')].emp_p.iloc[0]:.3f}"
        out[f"tab{T}FearStdTwo"] = f"{c.loc[('m2', 'fear')].std_b2:+.2f}".replace("-", "$-$")
        out[f"tab{T}FearStdThree"] = f"{c.loc[('m3', 'fear')].std_b2:+.2f}".replace("-", "$-$")
        for e, E in [("surprise", "Sur"), ("anger", "Ang")]:
            r = c.loc[("m1", e)]
            out[f"tab{T}{E}B"] = fmt_pp(r.b3_slope, 3)
            out[f"tab{T}{E}H"] = fmt_p(r.p_b3_holm).replace("= ", "").replace("$<$ ", "$<$")
            if sp is not None:
                q = sp[(sp["sample"].astype(str) == smp) & (sp.instrument == "m1") & (sp.emotion == e)].iloc[0]
                out[f"tab{T}{E}E"] = f"{q.emp_p_b3:.3f}"
    if sp is not None:
        out["slopeFPMin"] = f"{100 * sp[sp.instrument == 'm1'].groupby('sample').placebo_share_p05_b3.mean().min():.1f}"
        out["slopeFPMax"] = f"{100 * sp[sp.instrument == 'm1'].groupby('sample').placebo_share_p05_b3.mean().max():.1f}"

    lines = ["% AUTO-GENERATED by 02_実験/code/05_make_numbers.py -- do not edit by hand"]
    for k, v in out.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    for sub in ["en"]:  # only the English manuscript is generated
        path = ROOT / "06_paper" / sub / "numbers.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
    json.dump(out, open(RES / "numbers.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(out)} macros -> {path}")


if __name__ == "__main__":
    main()
