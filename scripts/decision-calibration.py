#!/usr/bin/env python3
"""Calibration + option-order study for POST /v1/systemone (issue #710).

Generates a labelled decision set with programmatic ground truth (easy and deliberately hard multi-rule cases), queries a running dotLLM
server, then reports accuracy / NLL / ECE of the raw probabilities, a temperature fitted on one half and scored on the other, and how often
reversing the option order flips the answer (and what averaging the two orders does).

Usage: decision-calibration.py [--url http://localhost:8123] [--n 480] [--out results.json]
"""
import argparse, json, math, random, time, urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


# ---------------------------------------------------------------- dataset
def gen_items(n, seed=7):
    rng = random.Random(seed)
    items = []
    cats = ["electronics", "clothing", "grocery", "furniture", "final_sale"]

    def ret_policy():
        cat = rng.choice(cats)
        days = rng.randint(1, 60)
        opened = rng.random() < 0.5
        limit = {"electronics": 15 if opened else 30, "clothing": 30, "grocery": 3, "furniture": 14, "final_sale": 0}[cat]
        ok = days <= limit and cat != "final_sale"
        state = (f"Policy: electronics 30 days unopened, 15 days if opened; clothing 30 days; grocery 3 days; furniture 14 days; "
                 f"final-sale items are never returnable. Order: a {cat.replace('_', ' ')} item, {'opened' if opened else 'unopened'}, bought {days} days ago.")
        return dict(state=state, q={"type": "noul", "instructions": "Is this item eligible for a return?"}, truth=ok, kind="noul-policy")

    def numeric():
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return dict(state=f"Account A has {a} open tickets. Account B has {b} open tickets.",
                    q={"type": "noul", "instructions": "Does account A have strictly more open tickets than account B?"}, truth=a > b, kind="noul-numeric")

    def route():
        topics = {
            "billing": ["I was charged twice for my subscription.", "My invoice shows the wrong amount.", "Please refund last month's payment."],
            "technical": ["The app crashes when I open settings.", "I cannot log in after the update.", "Uploads fail with error 500."],
            "shipping": ["My parcel has not arrived after two weeks.", "The tracking page shows no updates.", "The courier left it at the wrong address."],
        }
        topic = rng.choice(list(topics))
        text = rng.choice(topics[topic])
        opts = list(topics)
        rng.shuffle(opts)
        return dict(state=f"Customer message: {text}",
                    q={"type": "choice", "instructions": "Which team should handle this message?",
                       "criteria": {o: f"The {o} team." for o in opts}}, truth=topic, kind="choice-route")

    def hard_route():
        # Messages that genuinely mix two topics; ground truth is the stated PRIMARY complaint.
        pairs = [("billing", "technical", "I could not log in, and because of that I got charged for a month I did not use. The charge is what I want fixed."),
                 ("technical", "billing", "The app keeps crashing. Also my last invoice looked a bit odd but that can wait. Please fix the crashes."),
                 ("shipping", "billing", "My parcel never arrived. I have not been refunded yet but the missing parcel is the issue."),
                 ("billing", "shipping", "I was refunded twice by mistake for the late parcel; please correct the double refund.")]
        primary, other, text = rng.choice(pairs)
        opts = [primary, other, "other"]
        rng.shuffle(opts)
        return dict(state=f"Customer message: {text}",
                    q={"type": "choice", "instructions": "Which team should handle the customer's PRIMARY complaint?",
                       "criteria": {o: f"The {o} team." for o in opts}}, truth=primary, kind="choice-hard")

    def thresholds():
        temp = rng.randint(-10, 45)
        band = "cold" if temp < 10 else ("mild" if temp <= 25 else "hot")
        opts = ["cold", "mild", "hot"]
        rng.shuffle(opts)
        return dict(state=f"The temperature is {temp} degrees Celsius. Cold is below 10, mild is 10 to 25 inclusive, hot is above 25.",
                    q={"type": "choice", "instructions": "Which band is the temperature in?", "criteria": {o: f"The {o} band." for o in opts}},
                    truth=band, kind="choice-band")

    gens = [ret_policy, numeric, route, hard_route, thresholds]
    while len(items) < n:
        items.append(rng.choice(gens)())
    return items


def to_request(it, reverse=False):
    q = dict(it["q"])
    if reverse and q["type"] == "choice":
        q["criteria"] = dict(reversed(list(q["criteria"].items())))
    return {"model": "jev-latest", "state": it["state"], "questions": {"q": q}}


def probs_of(ans, it):
    """-> (dict label->p) with labels True/False for noul or the option names for choice."""
    a = ans["answers"]["q"]
    if it["q"]["type"] == "noul":
        return {True: a["noul"], False: 1.0 - a["noul"]}
    return dict(a["probabilities"])


# ---------------------------------------------------------------- metrics
def nll(ps, truths):
    return -sum(math.log(max(p[t], 1e-12)) for p, t in zip(ps, truths)) / len(ps)


def acc(ps, truths):
    return sum(max(p, key=p.get) == t for p, t in zip(ps, truths)) / len(ps)


def ece(ps, truths, bins=10):
    tot = [[0, 0.0, 0.0] for _ in range(bins)]
    for p, t in zip(ps, truths):
        top = max(p, key=p.get)
        conf = p[top]
        b = min(int(conf * bins), bins - 1)
        tot[b][0] += 1
        tot[b][1] += conf
        tot[b][2] += 1.0 if top == t else 0.0
    n = len(ps)
    return sum(c / n * abs(sc / c - sa / c) for c, sc, sa in tot if c)


def temper(p, T):
    lp = {k: math.log(max(v, 1e-12)) / T for k, v in p.items()}
    m = max(lp.values())
    e = {k: math.exp(v - m) for k, v in lp.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def fit_T(ps, truths):
    best, bt = None, 1.0
    for i in range(20, 600):
        T = i / 100
        v = nll([temper(p, T) for p in ps], truths)
        if best is None or v < best:
            best, bt = v, T
    return bt


def report(name, ps, truths):
    return {"set": name, "n": len(ps), "acc": round(acc(ps, truths), 4), "nll": round(nll(ps, truths), 4), "ece": round(ece(ps, truths), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8123")
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--out", default="calibration-results.json")
    a = ap.parse_args()

    items = gen_items(a.n)
    t0 = time.time()
    fwd, rev = [], []
    for it in items:
        fwd.append(probs_of(post(a.url, to_request(it)), it))
        rev.append(probs_of(post(a.url, to_request(it, reverse=True)), it) if it["q"]["type"] == "choice" else None)
    print(f"queried {len(items)} items in {time.time() - t0:.1f}s")
    truths = [it["truth"] for it in items]

    # split: even index = fit, odd = test (item kinds are interleaved by the RNG)
    fit = [i for i in range(len(items)) if i % 2 == 0]
    test = [i for i in range(len(items)) if i % 2 == 1]
    pick = lambda idx, src: [src[i] for i in idx]
    out = {"raw_all": report("all", fwd, truths)}
    for kind in sorted({it["kind"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["kind"] == kind]
        out["raw_" + kind] = report(kind, pick(idx, fwd), pick(idx, truths))
    T = fit_T(pick(fit, fwd), pick(fit, truths))
    out["T_fit"] = T
    out["test_raw"] = report("test raw", pick(test, fwd), pick(test, truths))
    out["test_T"] = report(f"test T={T}", [temper(p, T) for p in pick(test, fwd)], pick(test, truths))
    out["fit_raw"] = report("fit raw", pick(fit, fwd), pick(fit, truths))
    out["fit_T"] = report("fit T", [temper(p, T) for p in pick(fit, fwd)], pick(fit, truths))

    # option-order effect (choice items only)
    ci = [i for i in range(len(items)) if rev[i] is not None]
    flips = sum(max(fwd[i], key=fwd[i].get) != max(rev[i], key=rev[i].get) for i in ci)
    avg = []
    for i in ci:
        keys = fwd[i].keys()
        m = {k: math.sqrt(fwd[i][k] * rev[i][k]) for k in keys}   # geometric mean = average of log-probs
        s = sum(m.values())
        avg.append({k: v / s for k, v in m.items()})
    out["order"] = {"choice_items": len(ci), "argmax_flips_on_reversal": flips,
                    "fwd": report("fwd", pick(ci, fwd), pick(ci, truths)),
                    "rev": report("rev", pick(ci, rev), pick(ci, truths)),
                    "avg_fwd_rev": report("geo-mean(fwd,rev)", avg, pick(ci, truths))}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
