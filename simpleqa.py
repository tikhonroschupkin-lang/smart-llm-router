"""Run SimpleQA (closed-book) for one or more models via Perplexity, grade, and save per-question results.

Results go to results/<model_slug>.jsonl (one line per question, resumable).
These per-question outcomes are the training/eval data for the router.

Usage:
    python simpleqa.py --models anthropic/claude-opus-5-5 perplexity/glm-5.3-flash --limit 50
    python simpleqa.py --report
"""
import argparse
import ast
import asyncio
import json
import re
from pathlib import Path

import pandas as pd

from llm import PerplexityClient

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "simple_qa_test_set.csv"
RESULTS = ROOT / "results"

DEFAULT_MODELS = ["anthropic/claude-opus-5-5", "perplexity/glm-5.3-flash"]
GRADER_MODEL = "openai/gpt-6.1-terra"

# Grader template from openai/simple-evals (simpleqa_eval.py).
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

GRADE_MAP = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    meta = df["metadata"].map(ast.literal_eval)
    df["topic"] = meta.map(lambda m: m.get("topic"))
    df["answer_type"] = meta.map(lambda m: m.get("answer_type"))
    df["qid"] = range(len(df))
    return df[["qid", "problem", "answer", "topic", "answer_type"]]


def load_results(model: str) -> dict:
    path = RESULTS / (slug(model) + ".jsonl")
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except ValueError:  # partially written line from a concurrent run
                    continue
                out[rec["qid"]] = rec
    return out


# A "model spec" is "<perplexity model id>" or "<perplexity model id>#<variant>".
# Variants change the instructions sent with the question.
VARIANTS = {
    "": None,
    "guess": (
        "Answer the question with your single best guess. Never refuse or say you don't know: "
        "if unsure, still commit to the most likely specific answer. Keep the answer short."
    ),
}
# Some models occasionally loop in hidden reasoning until the cap; cap them so it fails fast.
MAX_OUTPUT_TOKENS = {"perplexity/glm-5.3-flash": 8000}


def parse_spec(spec: str):
    model, _, variant = spec.partition("#")
    return model, variant


async def solve_one(client, spec, row, lock, fh):
    model, variant = parse_spec(spec)
    ans = await client.complete(model, row["problem"], instructions=VARIANTS[variant],
                                max_output_tokens=MAX_OUTPUT_TOKENS.get(model, 16000))
    grade, g = None, None
    if ans.error == "status=incomplete" and not ans.text:
        # ran out of output tokens without answering: a real outcome, counts as no answer
        grade, ans.error = "NOT_ATTEMPTED", None
    elif not ans.error:
        g = await client.complete(
            GRADER_MODEL,
            GRADER_TEMPLATE.format(question=row["problem"], target=row["answer"], predicted_answer=ans.text),
            max_output_tokens=1024,
        )
        m = re.search(r"\b([ABC])\b", g.text or "")
        grade = GRADE_MAP[m.group(1)] if m else None
    rec = {
        "qid": int(row["qid"]),
        "problem": row["problem"],
        "gold": row["answer"],
        "topic": row["topic"],
        "answer_type": row["answer_type"],
        "model": spec,
        "response": ans.text,
        "grade": grade,
        "cost": ans.cost,
        "input_tokens": ans.input_tokens,
        "output_tokens": ans.output_tokens,
        "reasoning_tokens": ans.reasoning_tokens,
        "latency_s": round(ans.latency_s, 3),
        "grader_cost": g.cost if g else 0.0,
        "error": ans.error or (g.error if g else None) or (None if grade else "ungradable: %r" % (g.text if g else "")),
    }
    async with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    return rec


def sample_qids(limit: int, seed: int = 0) -> list:
    df = load_dataset()
    if limit:
        df = df.sample(n=limit, random_state=seed)
    return sorted(df["qid"].tolist())


async def ensure(specs, qids, client=None, concurrency=16):
    """Make sure every spec has a graded result for every qid (runs only what's missing)."""
    df = load_dataset()
    df = df[df["qid"].isin(set(qids))]
    RESULTS.mkdir(exist_ok=True)
    # specs run concurrently, each with its own client (own concurrency budget) so a slow
    # model doesn't starve the others; `client` is accepted for API compatibility only.
    async def one(model):
        done = {q for q, r in load_results(model).items() if r.get("grade")}
        todo = df[~df["qid"].isin(done)]
        print("[%s] %d to run (%d already done)" % (model, len(todo), len(done)), flush=True)
        if todo.empty:
            return
        c = PerplexityClient(concurrency=concurrency)
        path = RESULTS / (slug(model) + ".jsonl")
        lock = asyncio.Lock()
        try:
            with path.open("a") as fh:
                tasks = [solve_one(c, model, r, lock, fh) for r in todo.to_dict("records")]
                n = 0
                for fut in asyncio.as_completed(tasks):
                    rec = await fut
                    n += 1
                    if rec["error"]:
                        print("  ! [%s] qid=%d %s" % (model, rec["qid"], rec["error"][:150]), flush=True)
                    if n % 100 == 0:
                        print("  [%s] %d/%d" % (model, n, len(tasks)), flush=True)
        finally:
            await c.close()
        # compact: keep the latest record per qid
        recs = load_results(model)
        path.write_text("".join(json.dumps(recs[q], ensure_ascii=False) + "\n" for q in sorted(recs)))

    await asyncio.gather(*[one(m) for m in specs])


def summarize(recs: list) -> dict:
    n = len(recs)
    c = sum(r["grade"] == "CORRECT" for r in recs)
    i = sum(r["grade"] == "INCORRECT" for r in recs)
    na = sum(r["grade"] == "NOT_ATTEMPTED" for r in recs)
    attempted = c + i
    acc_given_att = c / attempted if attempted else 0.0
    overall = c / n if n else 0.0
    f1 = 2 * overall * acc_given_att / (overall + acc_given_att) if (overall + acc_given_att) else 0.0
    cost = sum(r["cost"] for r in recs)
    return {
        "n": n, "correct": overall, "incorrect": i / n if n else 0, "not_attempted": na / n if n else 0,
        "acc_given_attempted": acc_given_att, "f1": f1,
        "cost_total": cost, "cost_per_q": cost / n if n else 0,
        "latency_mean_s": sum(r["latency_s"] for r in recs) / n if n else 0,
    }


def report(models):
    per = {m: {q: r for q, r in load_results(m).items() if r.get("grade")} for m in models}
    common = sorted(set.intersection(*[set(p) for p in per.values()])) if per else []
    print("Questions graded by all models: %d\n" % len(common))
    rows = []
    for m in models:
        s = summarize([per[m][q] for q in common])
        s["model"] = m
        rows.append(s)
    t = pd.DataFrame(rows).set_index("model")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(t)
    if len(models) == 2 and common:
        a, b = models
        ca = [per[a][q]["grade"] == "CORRECT" for q in common]
        cb = [per[b][q]["grade"] == "CORRECT" for q in common]
        n = len(common)
        both = sum(x and y for x, y in zip(ca, cb))
        only_a = sum(x and not y for x, y in zip(ca, cb))
        only_b = sum(y and not x for x, y in zip(ca, cb))
        print("\nAgreement on CORRECT:")
        print("  both correct      : %.3f" % (both / n))
        print("  only %-12s : %.3f" % (a.split('/')[-1], only_a / n))
        print("  only %-12s : %.3f" % (b.split('/')[-1], only_b / n))
        print("  neither           : %.3f" % ((n - both - only_a - only_b) / n))
        print("  oracle router acc : %.3f" % ((both + only_a + only_b) / n))
        # oracle that picks cheapest correct model
        cost_a = [per[a][q]["cost"] for q in common]
        cost_b = [per[b][q]["cost"] for q in common]
        cheap_is_b = sum(cost_b) < sum(cost_a)
        oracle_cost = 0.0
        for x, y, pa, pb in zip(ca, cb, cost_a, cost_b):
            if cheap_is_b:
                oracle_cost += pb if (y or not x) else pa
            else:
                oracle_cost += pa if (x or not y) else pb
        print("  oracle cost/q     : $%.5f" % (oracle_cost / n))
        topics = pd.DataFrame({"topic": [per[a][q]["topic"] for q in common], a: ca, b: cb})
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print("\nAccuracy by topic:")
            print(topics.groupby("topic").agg(["mean", "count"]).iloc[:, [0, 1, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, default=0, help="random subset size (0 = all 4326)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        asyncio.run(ensure(a.models, sample_qids(a.limit, a.seed), concurrency=a.concurrency))
    report(a.models)


if __name__ == "__main__":
    main()
