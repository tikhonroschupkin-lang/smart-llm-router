"""LLM router for SimpleQA: a cheap decision model routes between Opus 5.5 and GLM 5.3 Flash.

Objective: maximise pass rate (= fraction graded CORRECT). Cost is reported, not optimised.
Decision model: TypeSafe Jev (default) or an LLM via Perplexity (e.g. anthropic/claude-haiku-5-5).
Deciders return probabilities; each strategy has one threshold, tuned on the "tune" split
and reported on the held-out "test" split.

Strategies (all decisions are made by the decision model; downstream answers come from
the two answer models, the decision model never writes the final answer itself):

  always_opus / always_glm / always_opus_guess   baselines
  preroute       decider sees only the question, picks a model
  cascade_glm    Opus answers; decider checks if it committed to an answer; if not -> GLM
  cascade_guess  same, but fallback is Opus re-asked in "best guess" mode
  select2        both models answer; decider picks the more likely correct answer
  select3        Opus, Opus-guess and GLM answer; decider picks
  cascade_select Opus answers; if it abstains -> Opus-guess + GLM, decider picks between them

Usage:
    python router.py                 # dev set of 1000 random questions, Jev decider
    python router.py --decider anthropic/claude-haiku-5-5
    python router.py --limit 0       # full benchmark
"""
import argparse
import asyncio
import hashlib
import json
import random
import re

import pandas as pd

from jev import JevClient
from llm import PerplexityClient
from simpleqa import RESULTS, ensure, load_dataset, load_results, sample_qids, slug

OPUS = "anthropic/claude-opus-5-5"
GLM = "perplexity/glm-5.3-flash"
OPUS_GUESS = OPUS + "#guess"
GLM_GUESS = GLM + "#guess"
DECIDER = "jev-latest"          # TypeSafe Jev; any Perplexity model id also works (LLM decider)

PREROUTE_PROMPT = """You route short closed-book factual questions (no web access) to one of two language models.
Your only goal: maximise the chance that the final answer is correct.

Model A: Claude Opus 5.5 - large frontier model with very broad long-tail knowledge; it sometimes declines when unsure.
Model B: GLM 5.3 Flash - small, fast model; weaker on obscure facts but occasionally knows things A does not.

Question: {question}

Estimate the probability that each model answers this correctly, then choose.
Reply with JSON only: {{"p_A": <0-1>, "p_B": <0-1>, "choice": "A" or "B"}}"""

COMMIT_PROMPT = """Below is a question and a model's response. Did the response commit to a specific answer to the question?
- COMMITTED: it states a concrete answer (even with mild hedging like "I believe").
- ABSTAINED: it declines, says it doesn't know / can't verify, gives only vague or partial information, or asks for clarification without answering.

Question: {question}
Response: {response}

Reply with exactly one word: COMMITTED or ABSTAINED."""

SELECT_PROMPT = """You are judging candidate answers to a closed-book factual question. Exactly one final answer will be submitted; it is scored correct only if it matches the true answer.

Question: {question}

{candidates}

Pick the candidate most likely to be factually correct. Use your own knowledge to check them. Rules:
- A response that declines or gives no specific answer can never be correct, so prefer any plausible specific answer over a refusal.
- If candidates agree on the same answer, that is evidence it is correct.
- Do not prefer a candidate because it is longer or more confident-sounding.

Reply with JSON only: {{"choice": <candidate number>}}"""


def stable_rng(*parts) -> random.Random:
    return random.Random(int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def split_of(qid: int) -> str:
    """Deterministic 40/60 tune/test split."""
    return "tune" if stable_rng("split", qid).random() < 0.4 else "test"


class DecisionCache:
    """Persist decider calls in results/decisions/<kind>__<decider>.jsonl keyed by (qid, key)."""

    def __init__(self, kind: str, decider: str):
        self.path = RESULTS / "decisions" / ("%s__%s.jsonl" % (kind, slug(decider)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.data[r["key"]] = r

    def get(self, key):
        return self.data.get(key)

    def put(self, rec):
        self.data[rec["key"]] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def shuffled(qid, cands):
    order = list(cands)
    stable_rng("order", qid, *[s for s, _ in cands]).shuffle(order)
    return order


class BaseDecider:
    """Three decisions, each returning (probability-ish output, cost):
       p_glm(qid, question)                      P(route to the small model)
       p_committed(qid, spec, question, resp)    P(response commits to an answer)
       select_probs(qid, question, cands)        {spec: P(this candidate is the correct one)}
    """
    def __init__(self, name):
        self.name = name
        self.caches = {k: DecisionCache(k, name) for k in ("preroute", "commit", "select")}

    async def _cached(self, kind, key, compute):
        hit = self.caches[kind].get(key)
        if hit is None:
            hit = await compute()
            hit["key"] = key
            if not hit.get("error"):
                self.caches[kind].put(hit)
        return hit


class LLMDecider(BaseDecider):
    """Generative LLM via Perplexity, prompted to emit JSON."""

    def __init__(self, client: PerplexityClient, model: str):
        super().__init__(model)
        self.client, self.model = client, model

    async def _call(self, kind, key, prompt):
        async def compute():
            r = await self.client.complete(self.model, prompt, max_output_tokens=2048)
            return {"raw": r.text, "cost": r.cost, "latency_s": r.latency_s, "error": r.error}
        return await self._cached(kind, key, compute)

    async def p_glm(self, qid, question):
        h = await self._call("preroute", str(qid), PREROUTE_PROMPT.format(question=question))
        j = parse_json(h["raw"])
        try:
            pa, pb = float(j["p_A"]), float(j["p_B"])
            p = pb / (pa + pb) if pa + pb > 0 else 0.0
        except (KeyError, TypeError, ValueError):
            p = 0.0
        if str(j.get("choice", "")).strip().upper() == "B":
            p = max(p, 0.5 + 1e-6)  # honour the explicit choice at the default threshold
        return p, h["cost"]

    async def p_committed(self, qid, spec, question, response):
        if not response.strip():
            return 0.0, 0.0
        h = await self._call("commit", "%s|%s" % (qid, spec), COMMIT_PROMPT.format(question=question, response=response))
        return (0.0 if "ABSTAIN" in (h["raw"] or "").upper() else 1.0), h["cost"]

    async def select_probs(self, qid, question, cands):
        order = shuffled(qid, cands)
        block = "\n\n".join("Candidate %d:\n%s" % (i + 1, (r or "(empty response)").strip()) for i, (_, r) in enumerate(order))
        h = await self._call("select", "%s|%s" % (qid, ",".join(s for s, _ in order)),
                             SELECT_PROMPT.format(question=question, candidates=block))
        probs = {s: 0.0 for s, _ in cands}
        try:
            probs[order[int(parse_json(h["raw"]).get("choice")) - 1][0]] = 1.0
        except (TypeError, ValueError, IndexError):
            probs[cands[0][0]] = 1.0
        return probs, h["cost"]


JEV_ROUTE_Q = {
    "route": {
        "type": "choice",
        "instructions": "The `question` is a closed-book factual question (no web access). Two language models can answer it. "
                        "Which one is more likely to answer it correctly?",
        "criteria": {
            "opus": "Claude Opus 5.5: large frontier model with very broad long-tail factual knowledge",
            "glm": "GLM 5.3 Flash: small fast model, weaker on obscure facts",
        },
    }
}
JEV_COMMIT_Q = {
    "committed": {
        "type": "noul",
        "instructions": "Does the `response` commit to a specific answer to the `question`?",
        "criteria": {
            "true": "It states a concrete answer, even with mild hedging such as 'I believe'.",
            "false": "It declines, says it does not know or cannot verify, gives only vague or partial information, "
                     "or asks for clarification without answering.",
        },
    }
}


class JevDecider(BaseDecider):
    def __init__(self, jev: JevClient):
        super().__init__(jev.model)
        self.jev = jev

    async def _ask(self, kind, key, state, questions):
        async def compute():
            r = await self.jev.ask(state, questions)
            return {"answers": r["answers"], "model": r.get("model"), "cost": r["cost"],
                    "latency_s": r["latency_s"], "error": r["error"]}
        return await self._cached(kind, key, compute)

    async def p_glm(self, qid, question):
        h = await self._ask("preroute", str(qid), {"question": question}, JEV_ROUTE_Q)
        return float(h["answers"].get("route", {}).get("probabilities", {}).get("glm", 0.0)), h["cost"]

    async def p_committed(self, qid, spec, question, response):
        if not response.strip():
            return 0.0, 0.0
        h = await self._ask("commit", "%s|%s" % (qid, spec), {"question": question, "response": response}, JEV_COMMIT_Q)
        return float(h["answers"].get("committed", {}).get("noul", 1.0)), h["cost"]

    async def select_probs(self, qid, question, cands):
        order = shuffled(qid, cands)
        names = ["candidate_%d" % (i + 1) for i in range(len(order))]
        state = {"question": question}
        state.update({n: (r or "(empty response)").strip() for n, (_, r) in zip(names, order)})
        questions = {
            "best": {
                "type": "choice",
                "instructions": "Which candidate is most likely a factually correct answer to the `question`? "
                                "A candidate that declines or gives no specific answer can never be correct. "
                                "Agreement between candidates is evidence of correctness. "
                                "Do not prefer a candidate for being longer or sounding more confident.",
                "criteria": {n: None for n in names},
            }
        }
        h = await self._ask("select", "%s|%s" % (qid, ",".join(s for s, _ in order)), state, questions)
        p = h["answers"].get("best", {}).get("probabilities", {})
        if not p:
            return {s: float(s == cands[0][0]) for s, _ in cands}, h["cost"]
        return {s: float(p.get(n, 0.0)) for n, (s, _) in zip(names, order)}, h["cost"]


def make_decider(name, client):
    return JevDecider(JevClient(name)) if name.startswith("jev") else LLMDecider(client, name)


# ---------------------------------------------------------------- strategies
# A strategy = (collect, apply, default_t, grid)
#   collect(d, A, q) -> info dict (decider outputs, cached; done once per question)
#   apply(info, t)   -> (final_spec, cost)  (pure; lets us tune the threshold t offline)
# A[spec][qid] are cached graded answers.

def ans_cost(A, spec, qid):
    return A[spec][qid]["cost"]


def always(spec):
    async def collect(d, A, q):
        return {"cost": ans_cost(A, spec, q["qid"])}
    return collect, lambda i, t: (spec, i["cost"]), 0.0, [0.0]


def random_route(a, b, p_b=0.5):
    """Baseline: coin flip per question (seeded by qid, so reproducible). No decider call."""
    async def collect(d, A, q):
        qid = q["qid"]
        return {"b": stable_rng("random", qid).random() < p_b, "ca": ans_cost(A, a, qid), "cb": ans_cost(A, b, qid)}
    return collect, lambda i, t: (b, i["cb"]) if i["b"] else (a, i["ca"]), 0.0, [0.0]


def preroute(opus, glm):
    async def collect(d, A, q):
        p, c = await d.p_glm(q["qid"], q["problem"])
        return {"p": p, "dc": c, "co": ans_cost(A, opus, q["qid"]), "cg": ans_cost(A, glm, q["qid"])}
    def apply(i, t):
        return (glm, i["dc"] + i["cg"]) if i["p"] > t else (opus, i["dc"] + i["co"])
    return collect, apply, 0.5, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]


def cascade(fallback):
    """Opus first; if P(committed) < t, use the fallback answer."""
    async def collect(d, A, q):
        qid = q["qid"]
        p, c = await d.p_committed(qid, OPUS, q["problem"], A[OPUS][qid]["response"])
        return {"p": p, "c0": ans_cost(A, OPUS, qid) + c, "cf": ans_cost(A, fallback, qid)}
    def apply(i, t):
        return (fallback, i["c0"] + i["cf"]) if i["p"] < t else (OPUS, i["c0"])
    return collect, apply, 0.5, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


def pick(probs, primary, margin):
    """Primary candidate unless another beats it by more than `margin`."""
    best = max(probs, key=probs.get)
    return best if best != primary and probs[best] - probs[primary] > margin else primary


SELECT_GRID = [-0.01, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def select(specs):
    """All `specs` answer; decider scores them; specs[0] is primary (wins ties / small margins)."""
    async def collect(d, A, q):
        qid = q["qid"]
        probs, c = await d.select_probs(qid, q["problem"], [(s, A[s][qid]["response"]) for s in specs])
        return {"probs": probs, "cost": c + sum(ans_cost(A, s, qid) for s in specs)}
    def apply(i, t):
        return pick(i["probs"], specs[0], t), i["cost"]
    return collect, apply, 0.0, SELECT_GRID


def cascade_select():
    """Opus first; if it abstains (P<0.5), Opus-guess and GLM answer and the decider picks."""
    async def collect(d, A, q):
        qid = q["qid"]
        p, c = await d.p_committed(qid, OPUS, q["problem"], A[OPUS][qid]["response"])
        info = {"p": p, "c0": ans_cost(A, OPUS, qid) + c}
        if p < 0.5:
            probs, c2 = await d.select_probs(qid, q["problem"], [(s, A[s][qid]["response"]) for s in (OPUS_GUESS, GLM)])
            info.update(probs=probs, c1=c2 + ans_cost(A, OPUS_GUESS, qid) + ans_cost(A, GLM, qid))
        return info
    def apply(i, t):
        if i["p"] >= 0.5:
            return OPUS, i["c0"]
        return pick(i["probs"], OPUS_GUESS, t), i["c0"] + i["c1"]
    return collect, apply, 0.0, SELECT_GRID


STRATEGIES = {
    "always_opus": always(OPUS),
    "always_glm": always(GLM),
    "always_opus_guess": always(OPUS_GUESS),
    "always_glm_guess": always(GLM_GUESS),
    "random": random_route(OPUS, GLM),
    "random_guess": random_route(OPUS_GUESS, GLM_GUESS),
    "preroute": preroute(OPUS, GLM),
    "preroute_guess": preroute(OPUS_GUESS, GLM_GUESS),
    "cascade_glm": cascade(GLM),
    "cascade_guess": cascade(OPUS_GUESS),
    "cascade_select": cascade_select(),
    "select2": select([OPUS, GLM]),
    "select3": select([OPUS, OPUS_GUESS, GLM]),
    "select_guess2": select([OPUS_GUESS, GLM_GUESS]),
}

ANSWER_SPECS = [OPUS, GLM, OPUS_GUESS, GLM_GUESS]


async def evaluate(qids, strategies, decider_name, concurrency):
    client = PerplexityClient(concurrency=concurrency)
    d = make_decider(decider_name, client)
    try:
        await ensure(ANSWER_SPECS, qids, client=client)
        A = {s: load_results(s) for s in ANSWER_SPECS}
        qids = [q for q in qids if all(A[s].get(q, {}).get("grade") for s in ANSWER_SPECS)]
        qs = load_dataset().set_index("qid").loc[qids].reset_index().to_dict("records")
        rows, tuned = [], []
        for name in strategies:
            collect, apply, default_t, grid = STRATEGIES[name]
            infos = await asyncio.gather(*[collect(d, A, q) for q in qs])

            def score(t, split):
                outs = [(q, apply(i, t)) for q, i in zip(qs, infos) if split_of(q["qid"]) == split]
                return sum(A[s][q["qid"]]["grade"] == "CORRECT" for q, (s, _) in outs) / max(1, len(outs)), \
                       sum(c for _, (_, c) in outs) / max(1, len(outs))
            # pick t maximising tune pass rate; ties -> cheaper, then closer to default
            best_t = max(grid, key=lambda t: (round(score(t, "tune")[0], 6), -score(t, "tune")[1], -abs(t - default_t)))
            tuned.append({"strategy": name, "default_t": default_t, "tuned_t": best_t,
                          "pass_tune@default": score(default_t, "tune")[0], "pass_test@default": score(default_t, "test")[0],
                          "pass_tune@tuned": score(best_t, "tune")[0], "pass_test@tuned": score(best_t, "test")[0],
                          "$/q_test@tuned": score(best_t, "test")[1]})
            for q, i in zip(qs, infos):
                spec, cost = apply(i, best_t)
                rows.append({"strategy": name, "qid": q["qid"], "split": split_of(q["qid"]), "spec": spec,
                             "correct": A[spec][q["qid"]]["grade"] == "CORRECT", "cost": cost, "t": best_t})
            print("  done", name, flush=True)
        for name, pool in (("oracle_opus_glm", [OPUS, GLM]), ("oracle_guess2", [OPUS_GUESS, GLM_GUESS]),
                           ("oracle_all4", ANSWER_SPECS)):
            for q in qids:
                rows.append({"strategy": name, "qid": q, "split": split_of(q), "spec": None,
                             "correct": any(A[s][q]["grade"] == "CORRECT" for s in pool), "cost": float("nan")})
        return pd.DataFrame(rows), pd.DataFrame(tuned).set_index("strategy")
    finally:
        await client.close()
        if isinstance(d, JevDecider):
            await d.jev.close()


def print_table(res: pd.DataFrame, tuned: pd.DataFrame):
    order = list(dict.fromkeys(res["strategy"]))
    t = res.groupby(["strategy", "split"]).agg(pass_rate=("correct", "mean"), cost_per_q=("cost", "mean"), n=("qid", "count"))
    t = t.unstack("split")
    t = t[[("pass_rate", "tune"), ("pass_rate", "test"), ("cost_per_q", "test"), ("n", "test")]].loc[order]
    t.columns = ["pass_tune", "pass_test", "$/q_test", "n_test"]
    t = t.join(tuned[["tuned_t", "pass_test@default"]])
    mix = res[res.spec.notna() & (res.split == "test")].groupby("strategy")["spec"].value_counts(normalize=True).unstack(fill_value=0)
    mix.columns = ["%" + c.split("/")[-1] for c in mix.columns]
    t = t.join(mix)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 250, "display.max_columns", 20):
        print(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decider", default=DECIDER)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    res, tuned = asyncio.run(evaluate(sample_qids(a.limit, a.seed), a.strategies, a.decider, a.concurrency))
    out = RESULTS / ("router_eval__%s__n%d.csv" % (slug(a.decider), a.limit))
    res.to_csv(out, index=False)
    print_table(res, tuned)
    print("\nper-question results ->", out)


if __name__ == "__main__":
    main()
