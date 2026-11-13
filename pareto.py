"""Cost / pass-rate Pareto analysis of Jev-based routers vs a random router.

Each router produces a score s(q) = "how safe is it to use the cheap path for q".
Sweeping a threshold traces a curve from all-expensive to all-cheap.
  preroute:  Jev sees only the question. cheap path = GLM answers.          cost = c_glm  | c_opus
  cascade:   GLM answers first, Jev judges that answer. if not safe -> Opus. cost = c_glm  | c_glm + c_opus
Random baseline: send a random fraction p to the cheap model (expected value = straight line).

Prompt versions are defined in PREROUTE_VERSIONS / CASCADE_VERSIONS; Jev calls are cached per
(version, qid) in results/decisions/jev_<version>.jsonl so iterating is cheap.

Usage: python pareto.py [--versions p1 c1 ...]
"""
import argparse
import asyncio
import json

import numpy as np
import pandas as pd

from jev import JevClient
from router import GLM, GLM_GUESS, OPUS, OPUS_GUESS, split_of
from simpleqa import RESULTS, load_dataset, load_results

ANSWER_SPECS = [OPUS, GLM, OPUS_GUESS, GLM_GUESS]
EXPENSIVE = OPUS_GUESS  # dominates plain Opus on both pass rate and cost

# ------------------------------------------------------------------ prompt versions
# Each version: questions dict sent to Jev + a function answers -> score (higher = safer to go cheap)
# preroute state: {"question": ...}; cascade state: {"question": ..., "response": <GLM answer>}

OBSCURITY_LEVELS = [
    "Common knowledge that most adults know",
    "Main, widely reported fact about a famous person, place, work or event",
    "Specific detail about a moderately well-known topic",
    "Obscure detail about a niche person, event, organisation or work",
    "Extremely obscure detail that only appears deep in specialised sources",
]

PREROUTE_VERSIONS = {
    # v1: what router.py uses: direct model choice
    "p1_choice": (
        {"route": {"type": "choice",
                   "instructions": "The `question` is a closed-book factual question (no web access). Two language models can "
                                   "answer it. Which one is more likely to answer it correctly?",
                   "criteria": {"opus": "Claude Opus 5.5: large frontier model with very broad long-tail factual knowledge",
                                "glm": "GLM 5.3 Flash: small fast model, weaker on obscure facts"}}},
        lambda a: a["route"]["probabilities"]["glm"],
    ),
    # v2: a single atomic yes/no on how well-known the fact is
    "p2_wellknown": (
        {"wellknown": {"type": "noul",
                       "instructions": "Is the answer to the `question` a widely known fact, the kind found in the first "
                                       "paragraphs of an encyclopedia article or in common trivia, rather than an obscure detail?"}},
        lambda a: a["wellknown"]["noul"],
    ),
    # v3: graded obscurity rubric (score is probability-weighted level; lower = safer)
    "p3_obscurity": (
        {"obscurity": {"type": "score",
                       "instructions": "How obscure is the fact that the `question` asks for?",
                       "criteria": OBSCURITY_LEVELS}},
        lambda a: -a["obscurity"]["score"],
    ),
    # v4: answer-type classification; type -> risk mapping learned on tune split (LR on type probabilities)
    "p4_type": (
        {"atype": {"type": "choice",
                   "instructions": "What kind of answer does the `question` ask for?",
                   "criteria": {"date": "An exact date, month or year",
                                "number": "A number, count, amount or measurement",
                                "person": "The name of a person",
                                "place": "The name of a place (city, country, river, constellation, ...)",
                                "other": "Anything else: a title, organisation, term, object, event, ..."}}},
        lambda a: [a["atype"]["probabilities"][k] for k in ("date", "number", "person", "place")],
    ),
    # v5: hand-weighted, nothing fitted: hard answer kinds + obscurity
    "p5_hardkind": (
        {"hardkind": {"type": "noul",
                      "instructions": "Does the `question` ask for an exact date or year of a specific event, or for the name of "
                                      "a person who is not famous?"},
         "obscurity": {"type": "score",
                       "instructions": "How obscure is the fact that the `question` asks for?",
                       "criteria": OBSCURITY_LEVELS}},
        lambda a: -(a["hardkind"]["noul"] + a["obscurity"]["score"] / 4.0),
    ),
    # v6: atomic yes/no features combined by logistic regression fitted on the tune split
    "p6_atomic": (
        {"asks_date": {"type": "noul", "instructions": "Does the `question` ask for a date, month or year?"},
         "asks_person": {"type": "noul", "instructions": "Does the `question` ask for the name of a person?"},
         "asks_number": {"type": "noul", "instructions": "Does the `question` ask for a number, count or measurement?"},
         "famous_subject": {"type": "noul",
                            "instructions": "Is the main subject of the `question` (person, work, place or event) famous "
                                            "worldwide?"},
         "minor_detail": {"type": "noul",
                          "instructions": "Does the `question` ask about a minor detail (an award, appointment, episode, "
                                          "degree, exhibition, ...) rather than the main facts of its subject?"},
         "multi_part": {"type": "noul",
                        "instructions": "Does the `question` require several pieces of information to be fully correct "
                                        "(e.g. day, month and year; first, middle and last name; two items)?"}},
        lambda a: [a[k]["noul"] for k in ("asks_date", "asks_person", "asks_number", "famous_subject", "minor_detail",
                                          "multi_part")],
    ),
    # --- iteration 3 (from tune-split error analysis: GLM fails on pop-culture/sports specifics with popular
    #     subjects, succeeds on reference-book facts in history/science/geography)
    "p7_reference": (
        {"reference": {"type": "noul",
                       "instructions": "Is the answer to the `question` the kind of fact recorded in reference works on history, "
                                       "science, geography or scholarship (encyclopedias, textbooks), rather than a detail of "
                                       "entertainment, pop culture, TV, music, games, sports results or recent news?"}},
        lambda a: a["reference"]["noul"],
    ),
    "p8_popdetail": (
        {"popdetail": {"type": "noul",
                       "instructions": "Does the `question` ask for a specific detail of a TV show, film, album, game, reality "
                                       "show, sports competition, product or statistic (e.g. what happened in an episode, who "
                                       "won a minor contest, a percentage in a survey)?"}},
        lambda a: -a["popdetail"]["noul"],
    ),
}

CASCADE_VERSIONS = {
    # c1: abstention only
    "c1_committed": (
        {"committed": {"type": "noul",
                       "instructions": "Does the `response` commit to a specific answer to the `question`?",
                       "criteria": {"true": "It states a concrete answer, even with mild hedging such as 'I believe'.",
                                    "false": "It declines, says it does not know, or gives only vague or partial information."}}},
        lambda a: a["committed"]["noul"],
    ),
    # c2: direct correctness judgement
    "c2_correct": (
        {"correct": {"type": "noul",
                     "instructions": "Is the `response` a factually correct and complete answer to the `question`?"}},
        lambda a: a["correct"]["noul"],
    ),
}


# ------------------------------------------------------------------ Jev calls (cached)

class Cache:
    def __init__(self, version):
        self.path = RESULTS / "decisions" / ("jev_%s.jsonl" % version)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.data[r["key"]] = r

    def put(self, r):
        self.data[r["key"]] = r
        with self.path.open("a") as fh:
            fh.write(json.dumps(r) + "\n")


async def jev_scores(jev, version, questions, score_fn, items):
    """items: list of (key, state). Returns {key: score}."""
    cache = Cache(version)

    async def one(key, state):
        if key not in cache.data:
            r = await jev.ask(state, questions)
            if r["error"]:
                print("  ! jev", version, key, r["error"][:120])
                return
            cache.put({"key": key, "answers": r["answers"], "cost": r["cost"]})

    await asyncio.gather(*[one(k, s) for k, s in items])
    out = {}
    for k, _ in items:
        if k in cache.data:
            try:
                f = score_fn(cache.data[k]["answers"])
                out[k] = [float(x) for x in f] if isinstance(f, list) else float(f)
            except (KeyError, TypeError):
                pass
    return out, sum(cache.data[k]["cost"] for k, _ in items if k in cache.data) / max(1, len(items))


# ------------------------------------------------------------------ learned combiners

def fit_lr(X, y, l2=1.0, iters=3000, lr=0.5):
    """Plain-numpy L2 logistic regression. Returns predict(X) -> probabilities."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (np.asarray(Xn, float) - mu) / sd] @ w)), w


def learned_scores(feats, qids, corr_cheap, corr_exp):
    """Fit on tune split: target = going cheap loses nothing (not (expensive right and cheap wrong))."""
    tune = [q for q in qids if q in feats and split_of(q) == "tune"]
    y = [0 if (corr_exp[q] and not corr_cheap[q]) else 1 for q in tune]
    predict, w = fit_lr([feats[q] for q in tune], y)
    qs = [q for q in qids if q in feats]
    return dict(zip(qs, predict([feats[q] for q in qs]))), w


# ------------------------------------------------------------------ curves

def available():
    A = {s: load_results(s) for s in ANSWER_SPECS}
    qids = sorted(q for q in A[OPUS] if all(A[s].get(q, {}).get("grade") for s in ANSWER_SPECS))
    return A, qids


def curve(qids, score, correct_cheap, correct_exp, cost_cheap, cost_exp, cascade):
    """Sweep threshold. Returns DataFrame of (frac_cheap, cost, pass) points, all-expensive -> all-cheap."""
    s = np.array([score[q] for q in qids], float)
    cc, ce = np.array([correct_cheap[q] for q in qids], float), np.array([correct_exp[q] for q in qids], float)
    kc, ke = np.array([cost_cheap[q] for q in qids]), np.array([cost_exp[q] for q in qids])
    o = np.argsort(-s, kind="stable")  # safest first
    n = len(qids)
    exp_cost = ke + (kc if cascade else 0)
    z = np.zeros(1)
    pas = (np.r_[z, np.cumsum(cc[o])] + ce.sum() - np.r_[z, np.cumsum(ce[o])]) / n
    cost = (np.r_[z, np.cumsum(kc[o])] + exp_cost.sum() - np.r_[z, np.cumsum(exp_cost[o])]) / n
    return pd.DataFrame({"frac_cheap": np.arange(n + 1) / n, "cost": cost, "pass": pas})


def saving_at(df, target_pass, exp_cost):
    """Cost reduction vs the expensive model while keeping pass >= target."""
    ok = df[df["pass"] >= target_pass - 1e-12]
    return 1 - ok["cost"].min() / exp_cost if len(ok) else 0.0


def random_saving(lo, hi, r):
    """Random router: send fraction p to cheap. Keep pass >= r*hi_pass."""
    if hi[1] <= lo[1]:
        return 1 - lo[0] / hi[0]
    p = min(1.0, (1 - r) * hi[1] / (hi[1] - lo[1]))
    return p * (hi[0] - lo[0]) / hi[0]


def interp_pass(df, budget):
    """Best pass rate achievable on this curve with cost <= budget."""
    ok = df[df.cost <= budget + 1e-12]
    return ok["pass"].max() if len(ok) else np.nan


def area_gain(df, random_line):
    """Mean pass-rate gain over the random line, averaged across the cost range [c_cheap, c_exp]."""
    lo, hi = random_line
    grid = np.linspace(lo[0], hi[0], 50)
    rnd = np.interp(grid, [lo[0], hi[0]], [lo[1], hi[1]])
    ours = np.array([interp_pass(df, b) for b in grid])
    return np.nanmean(ours - rnd)


async def main_async(versions):
    A, qids = available()
    df = load_dataset().set_index("qid")
    print("questions with all 4 answer sets: %d (tune %d / test %d)" % (
        len(qids), sum(split_of(q) == "tune" for q in qids), sum(split_of(q) == "test" for q in qids)))
    jev = JevClient("jev-latest", concurrency=16)
    corr = {s: {q: A[s][q]["grade"] == "CORRECT" for q in qids} for s in ANSWER_SPECS}
    cost = {s: {q: A[s][q]["cost"] for q in qids} for s in ANSWER_SPECS}
    rows, curves = [], {}
    try:
        for cheap in (GLM_GUESS, GLM):
            for name, (qs, fn) in PREROUTE_VERSIONS.items():
                if versions and name not in versions:
                    continue
                sc, jc = await jev_scores(jev, name, qs, fn, [(str(q), {"question": df.loc[q, "problem"]}) for q in qids])
                sc = {int(k): v for k, v in sc.items()}
                if sc and isinstance(next(iter(sc.values())), list):
                    sc, w = learned_scores(sc, qids, corr[cheap], corr[EXPENSIVE])
                    print("  %s/%s LR weights (bias first): %s" % (name, cheap.split("/")[-1], np.round(w, 2)))
                curves[("preroute", name, cheap)] = (sc, jc, False)
            for name, (qs, fn) in CASCADE_VERSIONS.items():
                if versions and name not in versions:
                    continue
                items = [("%s|%d" % (cheap, q), {"question": df.loc[q, "problem"], "response": A[cheap][q]["response"] or "(empty)"})
                         for q in qids]
                sc, jc = await jev_scores(jev, name, qs, fn, items)
                sc = {int(k.split("|")[1]): v for k, v in sc.items()}
                curves[("cascade", name, cheap)] = (sc, jc, True)
    finally:
        await jev.close()

    # rank-average ensembles of unfitted preroute prompts (nothing learned -> no overfitting)
    def ranks(sc):
        ks = sorted(sc, key=sc.get)
        return {k: i / max(1, len(ks) - 1) for i, k in enumerate(ks)}
    for cheap in (GLM_GUESS, GLM):
        for ens in (("p1_choice", "p3_obscurity"), ("p1_choice", "p3_obscurity", "p5_hardkind"),
                    ("p1_choice", "p3_obscurity", "p7_reference"), ("p1_choice", "p3_obscurity", "p8_popdetail"),
                    ("p1_choice", "p3_obscurity", "p5_hardkind", "p7_reference", "p8_popdetail")):
            parts = [curves.get(("preroute", e, cheap)) for e in ens]
            if all(parts):
                rk = [ranks(p[0]) for p in parts]
                common = set.intersection(*[set(r) for r in rk])
                curves[("preroute", "ens_" + "+".join(e.split("_")[0] for e in ens), cheap)] = (
                    {q: np.mean([r[q] for r in rk]) for q in common}, sum(p[1] for p in parts), False)

    (RESULTS / "curves").mkdir(exist_ok=True)
    rng = np.random.RandomState(0)
    for split in ("tune", "test", "all"):
        qs = [q for q in qids if split == "all" or split_of(q) == split]
        boots = [rng.choice(len(qs), len(qs)) for _ in range(300)] if split == "test" else []
        for (kind, name, cheap), (sc, jc, casc) in curves.items():
            q2 = [q for q in qs if q in sc]

            def stats(qq):
                c = curve(qq, sc, corr[cheap], corr[EXPENSIVE], cost[cheap], cost[EXPENSIVE], casc)
                lo = (np.mean([cost[cheap][q] for q in qq]), np.mean([corr[cheap][q] for q in qq]))
                hi = (np.mean([cost[EXPENSIVE][q] for q in qq]), np.mean([corr[EXPENSIVE][q] for q in qq]))
                return c, lo, hi
            c, lo, hi = stats(q2)
            row = {"split": split, "router": "%s:%s" % (kind, name), "cheap": cheap.split("/")[-1].replace("glm-5.3-flash", "glm"),
                   "n": len(q2), "gain": area_gain(c, (lo, hi))}
            if boots:
                g = [area_gain(*(lambda c_, l_, h_: (c_, (l_, h_)))(*stats([q2[i] for i in b if i < len(q2)]))) for b in boots]
                row["gain_lo95"], row["gain_hi95"] = np.percentile(g, 2.5), np.percentile(g, 97.5)
            for r in (0.95, 0.90, 0.80):
                row["save@%d" % (r * 100)] = saving_at(c, r * hi[1], hi[0])
                row["rand@%d" % (r * 100)] = random_saving(lo, hi, r)
            row.update({"exp_pass": hi[1], "cheap_pass": lo[1]})
            rows.append(row)
            if split in ("all", "test"):
                c.assign(cheap_cost=lo[0], cheap_pass=lo[1], exp_cost=hi[0], exp_pass=hi[1]).to_csv(
                    RESULTS / "curves" / ("%s__%s__%s__%s.csv" % (split, kind, name, cheap.split("/")[-1].replace("#", "_"))),
                    index=False)
    res = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250, "display.max_columns", 30):
        for split in ("tune", "test"):
            print("\n== %s split. gain = mean pass-rate gain over random router across the cost range;"
                  " save@X = cost cut vs %s keeping X%% of its pass rate (rand@X = random router)" % (split, EXPENSIVE))
            print(res[res.split == split].drop(columns=["split"]).set_index(["router", "cheap"]).sort_values("gain", ascending=False))
    res.to_csv(RESULTS / "pareto_summary.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*")
    asyncio.run(main_async(ap.parse_args().versions))
