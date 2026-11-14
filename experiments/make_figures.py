"""Figures and LaTeX tables from results/*.csv -> manuscript/figures, manuscript/tables."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "manuscript", "figures")
TAB = os.path.join(ROOT, "manuscript", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# fixed categorical order (validated palette); colour follows the method everywhere
STYLE = {
    "SPARE (ILP+greedy)": ("#2a78d6", "o", "-"),
    "SPARE-T (A1)": ("#eb6834", "s", "-"),
    "SPARE-J (A2)": ("#2a78d6", "^", "--"),
    "JustFreq": ("#1baf7a", "D", "-"),
    "Degree": ("#eda100", "v", "-"),
    "PageRank": ("#e87ba4", "P", "-"),
    "Entity ablation": ("#6250d6", "X", "-"),
    "Random": ("#8c8c88", "x", ":"),
}
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42,
})


def fig_e1():
    df = pd.read_csv(os.path.join(RES, "e1_coverage_vs_budget.csv"))
    g = df.groupby(["profile", "method", "frac"]).coverage.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    show = ["SPARE (ILP+greedy)", "SPARE-T (A1)", "JustFreq", "Degree", "PageRank",
            "Entity ablation", "Random"]
    for ax, prof, title in zip(axes, ["rdfs", "rl"], ["UB-Synth, RDFS profile (linear)",
                                                       "UB-Synth, OWL 2 RL profile"]):
        for m in show:
            d = g[(g.profile == prof) & (g.method == m)]
            c, mk, ls = STYLE[m]
            ax.plot(d.frac * 100, d["mean"], color=c, marker=mk, ls=ls, label=m)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_xlabel("Budget (% of ABox triples)")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Entailment coverage $f_E(S)/f_E(\\mathcal{A})$")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, "e1_coverage.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "e1_coverage.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    # table at 5% and 10%
    lines = []
    for m in ["SPARE (ILP+greedy)", "SPARE-J (A2)", "SPARE-T (A1)", "JustFreq", "Degree",
              "PageRank", "Entity ablation", "Random"]:
        cells = []
        for prof in ["rdfs", "rl"]:
            for f in [0.02, 0.05, 0.1]:
                d = g[(g.profile == prof) & (g.method == m) & (np.isclose(g.frac, f))]
                cells.append(f"{d['mean'].iloc[0]:.3f}")
        lines.append(m.replace("SPARE", "\\spare{}") + " & " + " & ".join(cells) + r" \\")
    with open(os.path.join(TAB, "e1_table.tex"), "w") as fh:
        fh.write("\n".join(lines))
    return g


def fig_e2():
    df = pd.read_csv(os.path.join(RES, "e2_optimality_gap.csv"))
    s = df[(df.kind == "synth") & (df.p_multi <= 0.5)].copy()
    s["ref"] = s[["ilp", "A1", "A2"]].max(axis=1)
    # p = 0.75: the ILP does not finish; use the HiGHS upper bound (capped objective),
    # which gives a lower bound on the true approximation ratio
    b = pd.read_csv(os.path.join(RES, "e2b_bounds.csv"))
    b = b[b.p_multi == 0.75].copy()
    b["ref"] = b.bound
    b["ilp_optimal"] = b.optimal
    b["max_just"] = df[(df.kind == "synth") & (df.p_multi == 0.75)].set_index("seed").loc[b.seed, "max_just"].values
    b["jf"] = np.nan
    s = pd.concat([s, b[["p_multi", "seed", "max_just", "ilp_optimal", "A1", "A2", "jf", "ref"]]], ignore_index=True)
    s["r1"] = s.A1 / s.ref
    s["r2"] = s.A2 / s.ref
    s["rj"] = s.jf / s.ref
    g = s.groupby("p_multi").agg(r1=("r1", "mean"), r1s=("r1", "std"), r2=("r2", "mean"),
                                 r2s=("r2", "std"), rj=("rj", "mean"), rjs=("rj", "std"),
                                 opt=("ilp_optimal", "mean"), maxj=("max_just", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for col, sd, name in [("r2", "r2s", "SPARE-J (A2)"), ("r1", "r1s", "SPARE-T (A1)"),
                          ("rj", "rjs", "JustFreq")]:
        c, mk, ls = STYLE[name]
        ax.errorbar(g.p_multi, g[col], yerr=g[sd], color=c, marker=mk, ls=ls if ls != "--" else "-",
                    capsize=2, label=name)
    ax.set_xlabel("Share of multi-premise axioms $p$")
    ax.set_ylabel("Greedy / reference")
    ax.set_ylim(0.5, 1.02)
    ax.axvspan(0.62, 0.8, color="#f1f0ec", zorder=0)
    ax.text(0.71, 0.53, "capped / dual bound", ha="center", fontsize=7, color=MUTED)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_gap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "e2_gap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    d = df[df.kind == "dks"].copy()
    d["ref"] = d[["ilp", "A1", "A2"]].max(axis=1)
    dk = d.groupby("k").apply(lambda x: pd.Series(dict(
        A1=(x.A1 / x.ref).mean(), A2=(x.A2 / x.ref).mean(), JF=(x.jf / x.ref).mean(),
        opt=x.ilp_optimal.mean()))).reset_index()
    with open(os.path.join(TAB, "e2_table.tex"), "w") as fh:
        for _, r in g.iterrows():
            jf = "--" if np.isnan(r.rj) else f"{r.rj:.3f}"
            dag = "$^\\dagger$" if r.p_multi == 0.75 else ""
            fh.write(f"SynthJ, $p={r.p_multi:.2f}$ & {r.maxj:.1f} & {r.r1:.3f}{dag} & {r.r2:.3f}{dag} & "
                     f"{jf} & {r.opt*100:.0f}\\% \\\\\n")
        fh.write("\\midrule\n")
        for _, r in dk.iterrows():
            fh.write(f"D$k$S gadget, $k={int(r.k)}$ & 2 & {r.A1:.3f} & {r.A2:.3f} & {r.JF:.3f} & "
                     f"{r.opt*100:.0f}\\% \\\\\n")
    return g, dk


def fig_e3():
    df = pd.read_csv(os.path.join(RES, "e3_robustness.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    st = {"r=1 (ILP, f_E)": ("#2a78d6", "o", "-"), "r=2 (ILP, f_R)": ("#eb6834", "s", "-"),
          "r=1 (greedy)": ("#2a78d6", "o", "--"), "r=2 (greedy)": ("#eb6834", "s", "--"),
          "Degree": STYLE["Degree"]}
    for ax, f in zip(axes, [0.2, 0.3]):
        for m, (c, mk, ls) in st.items():
            d = df[(df.frac == f) & (df.method == m)]
            ax.plot(d.q * 100, d.coverage, color=c, marker=mk, ls=ls, label=m.replace("f_", "$f_") .replace(")", "$)") if "f_" in m else m)
        ax.set_title(f"Budget {int(f*100)}% (OWL 2 RL profile)", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("Retained triples deleted at random (%)")
    axes[0].set_ylabel("Entailment coverage $f_E$ (exact)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "e3_robustness.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "e3_robustness.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def tab_e4():
    df = pd.read_csv(os.path.join(RES, "e4_query_recall.csv"))
    order = ["SPARE-W (ILP, macro-query)", "SPARE (ILP+greedy)", "SPARE-J (A2)", "SPARE-T (A1)",
             "JustFreq", "Entity ablation", "Degree", "PageRank", "Random"]
    with open(os.path.join(TAB, "e4_table.tex"), "w") as fh:
        for m in order:
            d = df[df.method == m].sort_values("frac")
            cells = [f"{v:.3f}" for v in d.macro_recall] + [f"{v:.3f}" for v in d.entail_cov]
            fh.write(m.replace("SPARE", "\\spare{}") + " & " + " & ".join(cells) + " \\\\\n")
    return df


def fig_e5():
    df = pd.read_csv(os.path.join(RES, "e5_scalability.csv"))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for col, name, c, mk in [("build_s", "Materialise + justifications", "#1baf7a", "D"),
                             ("A1_s", "A1 (triple greedy)", "#eb6834", "s"),
                             ("A2_s", "A2 (justification greedy)", "#2a78d6", "^"),
                             ("ilp_solve_s", "A3 HiGHS solve (limit 120 s)", "#6250d6", "X")]:
        if col == "ilp_solve_s":
            ok = df.ilp_optimal.values
            ax.plot(df.n_abox, df[col], color=c, ls="-", label=name)
            ax.plot(df.n_abox[ok], df[col][ok], color=c, marker=mk, ls="none")
            ax.plot(df.n_abox[~ok], df[col][~ok], color=c, marker="o", mfc="white", ls="none")
            for x, y in zip(df.n_abox[~ok], df[col][~ok]):
                ax.annotate("time limit;\nincumbent not\nproven optimal", (x, y), xytext=(-6, -34),
                            textcoords="offset points", ha="right", fontsize=7, color=MUTED)
        else:
            ax.plot(df.n_abox, df[col], color=c, marker=mk, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(df.n_abox); ax.set_xticklabels([f"{v/1000:.1f}k" for v in df.n_abox])
    ax.minorticks_off()
    ax.set_xlabel("ABox size (triples)"); ax.set_ylabel("Wall-clock time (s)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e5_scalability.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "e5_scalability.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(TAB, "e5_table.tex"), "w") as fh:
        for _, r in df.iterrows():
            mark = "" if r.ilp_optimal else "$^\\dagger$"
            fh.write(f"{int(r.n_abox):,} & {int(r.n_targets):,} & {int(r.n_just):,} & {r.build_s:.1f} & "
                     f"{r.A1_s:.1f} & {r.A2_s:.1f} & {r.ilp_build_s:.1f} & {r.ilp_solve_s:.1f}{mark} & "
                     f"{r.A1:.3f} & {r.A2:.3f} & {r.ilp:.3f}{mark} \\\\\n".replace(",", "{,}"))


def tab_e6():
    df = pd.read_csv(os.path.join(RES, "e6_cap_ablation.csv"))
    with open(os.path.join(TAB, "e6_table.tex"), "w") as fh:
        for _, r in df.iterrows():
            fh.write(f"{int(r.m)} & {int(r.n_just):,} & {r.A2:.3f} & {r.ilp:.3f}"
                     f"{'' if r.ilp_optimal else '$^\\dagger$'} \\\\\n".replace(",", "{,}"))


if __name__ == "__main__":
    for fn in [fig_e1, fig_e2, fig_e3, tab_e4, fig_e5, tab_e6]:
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)


# ------------------------------------------------------------------ E8
E8_ORDER = ["DBpedia", "LUBM-1", "OWL2Bench-QL", "OWL2Bench-RL-", "OWL2Bench-EL-"]
E8_LABEL = {"DBpedia": "DBpedia (real)", "LUBM-1": "LUBM(1)", "OWL2Bench-QL": "OWL2Bench QL",
            "OWL2Bench-RL-": "OWL2Bench RL$^-$", "OWL2Bench-EL-": "OWL2Bench EL$^-$"}


def e8_load():
    import glob, json
    dfs, stats = [], {}
    for name in E8_ORDER:
        f = os.path.join(RES, f"e8_{name}.csv")
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
        j = os.path.join(RES, f"e8_datasets_{name}.json")
        if os.path.exists(j):
            stats[name] = json.load(open(j))
    return (pd.concat(dfs) if dfs else pd.DataFrame()), stats


def fig_e8():
    df, stats = e8_load()
    names = [n for n in E8_ORDER if n in set(df.dataset)]
    show = ["SPARE (ILP+greedy)", "SPARE-T (A1)", "JustFreq", "Degree", "PageRank", "Entity ablation", "Random"]
    fig, axes = plt.subplots(1, len(names), figsize=(2.05 * len(names) + 0.4, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        for m in show:
            d = df[(df.dataset == n) & (df.method == m)].sort_values("frac")
            c, mk, ls = STYLE[m]
            ax.plot(d.frac * 100, d.coverage, color=c, marker=mk, ls=ls, label=m, ms=4, lw=1.3)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_title(E8_LABEL[n], fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("Budget (%)", fontsize=8)
    axes[0].set_ylabel("Entailment coverage (exact)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=7.2, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "e8_external.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "e8_external.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    # dataset table
    with open(os.path.join(TAB, "e8_datasets.tex"), "w") as fh:
        for n in names:
            s = stats[n]
            fh.write(f"{E8_LABEL[n]} & {'linear' if s['linear'] else 'multi-premise'} & {s['tbox']:,} & "
                     f"{s['n_abox']:,} & {s['n_targets']:,} & {100*s['multi_share']:.1f} & {s['max_just']} & "
                     f"{s['build_s']:.1f} \\\\\n".replace(",", "{,}"))
    # results at 10 %
    meths = ["SPARE (ILP+greedy)", "SPARE-J (A2)", "SPARE-T (A1)", "JustFreq", "Degree", "PageRank",
             "Entity ablation", "Random"]
    with open(os.path.join(TAB, "e8_table.tex"), "w") as fh:
        for m in meths:
            cells = []
            for n in names:
                d = df[(df.dataset == n) & (df.method == m) & np.isclose(df.frac, 0.1)]
                cells.append(f"{d.coverage.iloc[0]:.3f}" if len(d) else "--")
            fh.write(m.replace("SPARE", "\\spare{}") + " & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\midrule\n")
        cells = []
        for n in names:
            d = df[(df.dataset == n) & (df.method == "SPARE (ILP+greedy)")]
            cells.append(f"{int(d.ilp_capped_optimal.sum())}/{len(d)}")
        fh.write("ILP capped-optimal (budgets) & " + " & ".join(cells) + " \\\\\n")
    return df


if __name__ == "__main__" and os.environ.get("SPARE_E8", "1") == "1":
    try:
        fig_e8(); print("ok fig_e8")
    except Exception as e:
        print("skip fig_e8", e)


# ----------------------------------------------------------------------------- E9 (LLM question answering)
E9_FILE = os.path.join(RES, "e9_kgrag_gemini_gemini-3.8-flash.csv")
E9_ORDER = [("SPARE", "SPARE (ILP+greedy)"), ("JustFreq", "JustFreq"), ("Random", "Random"), ("Degree", "Degree")]


def fig_e9():
    d = pd.read_csv(E9_FILE)
    assert len(d) == 2400 and d.groupby("method").size().eq(600).all()
    names = [n for n, _ in E9_ORDER]
    cols = [STYLE[s][0] for _, s in E9_ORDER]
    acc, ent, yes = [], [], []
    for n in names:
        g = d[d.method == n]
        p, q = g[g.positive], g[~g.positive]
        acc.append(((p.answer == "YES").sum() + (q.answer != "YES").sum()) / len(g))
        ent.append(p.entailed_by_S.mean())
        yes.append((p.answer == "YES").mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.45]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, acc, color=cols, width=0.62)
    for i, v in enumerate(acc):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("accuracy (600 questions)")
    ax.set_title("(a) Accuracy", fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    w = 0.36
    sup, uns = [], []
    for n in names:
        p = d[(d.method == n) & d.positive]
        sup.append(((p.answer == "YES") & p.entailed_by_S).mean())
        uns.append(((p.answer == "YES") & ~p.entailed_by_S).mean())
    ax.bar(x - w / 2, ent, width=w, color=cols, alpha=0.35)
    ax.bar(x + w / 2, sup, width=w, color=cols)
    ax.bar(x + w / 2, uns, width=w, bottom=sup, color="white", edgecolor=cols, hatch="////", linewidth=0.8)
    for i in range(len(names)):
        ax.text(i - w / 2, ent[i] + 0.008, f"{ent[i]:.2f}", ha="center", fontsize=7, color=MUTED)
        ax.text(i + w / 2, sup[i] + uns[i] + 0.008, f"{sup[i] + uns[i]:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("share of 300 positive questions")
    ax.set_title("(b) Positive questions: entailed vs. answered YES", fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    h_ent = tuple(Patch(facecolor=c, alpha=0.35) for c in cols)
    h_sup = tuple(Patch(facecolor=c) for c in cols)
    h_uns = tuple(Patch(facecolor="white", edgecolor=c, hatch="////") for c in cols)
    ax.legend([h_ent, h_sup, h_uns],
              ["type entailed by kept triples", "YES, supported by kept triples", "YES, not supported (model knowledge)"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, handlelength=3.2,
              fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"e9_llm.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__" and os.path.exists(E9_FILE):
    fig_e9(); print("ok fig_e9")
