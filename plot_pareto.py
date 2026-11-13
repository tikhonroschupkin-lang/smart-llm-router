"""Plot test-split cost vs pass-rate curves (Jev routers vs random router) -> results/pareto_test.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from simpleqa import RESULTS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # (file stem, label, color) - categorical slots in fixed order
    ("preroute__ens_p1+p3+p5", "Jev ensemble (p1+p3+p5)", "#2a78d6"),
    ("preroute__p1_choice", "Jev p1: model choice", "#eb6834"),
    ("cascade__c2_correct", "Jev cascade: judge GLM answer", "#1baf7a"),
]

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ref = None
for stem, label, color in SERIES:
    df = pd.read_csv(RESULTS / "curves" / ("test__%s__glm-5.3-flash_guess.csv" % stem))
    ref = df.iloc[0]
    # Pareto frontier of the curve (best pass at each cost)
    d = df.sort_values("cost")
    d = d[d["pass"] >= d["pass"].cummax() - 1e-12]
    ax.plot(d["cost"].values * 1000, d["pass"].values * 100, color=color, lw=2, label=label, solid_capstyle="round")
    end = d.iloc[len(d) // 2]
lo, hi = (ref.cheap_cost, ref.cheap_pass), (ref.exp_cost, ref.exp_pass)
ax.plot([lo[0] * 1000, hi[0] * 1000], [lo[1] * 100, hi[1] * 100], color=INK2, lw=2, ls=(0, (4, 3)), label="Random router")
for (x, y), t in ((lo, "GLM 5.3 Flash (best guess)"), (hi, "Opus 5.5 (best guess)")):
    ax.scatter([x * 1000], [y * 100], s=64, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(t, (x * 1000, y * 100), textcoords="offset points", xytext=(8, -14 if x == lo[0] else 6),
                color=INK, fontsize=9, ha="left" if x == lo[0] else "right")
ax.axhline(0.9 * hi[1] * 100, color=GRID, lw=1, zorder=0)
ax.annotate("90% of Opus pass rate", (lo[0] * 1000, 0.9 * hi[1] * 100), textcoords="offset points", xytext=(4, 4),
            color=INK2, fontsize=8)
ax.set_xlabel("Answer cost per question (USD × 10⁻³)", color=INK2)
ax.set_ylabel("SimpleQA pass rate (%)", color=INK2)
ax.set_title("Jev router vs random router, held-out test split (n=%d)" % int(round(1 / (df.frac_cheap.iloc[1])) ),
             color=INK, loc="left", fontsize=11)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=INK2, length=0)
leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
for t in leg.get_texts():
    t.set_color(INK)
fig.tight_layout()
fig.savefig(RESULTS / "pareto_test.png", facecolor=SURFACE)
print(RESULTS / "pareto_test.png")
