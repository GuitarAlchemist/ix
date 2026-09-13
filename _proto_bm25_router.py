"""Throwaway prototype: BM25 (zero-model, deterministic) intent routing on the
committed 126-prompt held-out router set, vs the two recorded embedding baselines.

Mirrors ix_streeling::search BM25 (Okapi k1=1.2, b=0.75, Lucene IDF) and its
tokenizer (lowercase, split on non-alphanumeric).
"""
import json, math, re, collections, statistics

K1, B = 1.2, 0.75
ROOT = "."


def tokenize(text):
    return [t.lower() for t in re.split(r"[^0-9A-Za-z]+", text) if t]


def idf(n_docs, df):
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class Bm25:
    def __init__(self, docs):
        self.docs = docs                      # list[list[str]]
        self.n = len(docs)
        self.avgdl = max(sum(len(d) for d in docs) / self.n, 1.0)
        self.df = collections.Counter()
        for d in docs:
            for t in set(d):
                self.df[t] += 1
        self.tf = [collections.Counter(d) for d in docs]

    def scores(self, query):
        qt = tokenize(query)
        out = []
        for i, d in enumerate(self.docs):
            dl = len(d)
            s = 0.0
            for term in qt:
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + K1 * (1.0 - B + B * dl / self.avgdl)
                s += idf(self.n, self.df[term]) * (f * (K1 + 1.0)) / denom
            out.append(s)
        return out


held = json.load(open(f"{ROOT}/state/router-spike/heldout-test.json", encoding="utf-8"))
train = json.load(open(f"{ROOT}/state/router-spike/train-set.json", encoding="utf-8"))

# --- Corpora -------------------------------------------------------------
# (a) one doc per TRAIN prompt -> BM25-1NN, the lexical twin of cosine-1NN.
prompt_docs, prompt_labels = [], []
for group in train["inScope"]:
    for p in group["prompts"]:
        prompt_docs.append(tokenize(p))
        prompt_labels.append(group["intentId"])
# (b) one pooled doc per intent.
pooled = collections.OrderedDict()
for group in train["inScope"]:
    pooled[group["intentId"]] = tokenize(" ".join(group["prompts"]))

intents = [g["intentId"] for g in train["inScope"]]
bm_1nn = Bm25(prompt_docs)
bm_pool = Bm25(list(pooled.values()))
pool_labels = list(pooled.keys())

train_oos = train["outOfScope"]


def predict(query, mode):
    if mode == "1nn":
        sc = bm_1nn.scores(query)
        if not sc or max(sc) <= 0.0:
            return None, 0.0
        i = max(range(len(sc)), key=lambda j: sc[j])
        return prompt_labels[i], sc[i]
    sc = bm_pool.scores(query)
    if not sc or max(sc) <= 0.0:
        return None, 0.0
    i = max(range(len(sc)), key=lambda j: sc[j])
    return pool_labels[i], sc[i]


# --- Calibrate the decline threshold on TRAIN ONLY (never on TEST) --------
def calibrate(mode):
    """Pick tau maximizing (in-scope kept + OOS declined) on TRAIN data only.
    In-scope train scores are leave-one-out to avoid a self-match at score max."""
    in_scores = []
    for gi, group in enumerate(train["inScope"]):
        for p in group["prompts"]:
            # leave-one-out: score against every OTHER train prompt
            sc = bm_1nn.scores(p) if mode == "1nn" else bm_pool.scores(p)
            if mode == "1nn":
                order = sorted(range(len(sc)), key=lambda j: -sc[j])
                best = 0.0
                for j in order:
                    if prompt_docs[j] != tokenize(p):
                        best = sc[j]
                        break
                in_scores.append(best)
            else:
                in_scores.append(max(sc))
    oos_scores = [predict(p, mode)[1] for p in train_oos]
    cands = sorted(set(round(x, 4) for x in in_scores + oos_scores))
    best_tau, best_obj = 0.0, -1.0
    for tau in cands:
        keep = sum(1 for s in in_scores if s >= tau) / len(in_scores)
        decl = sum(1 for s in oos_scores if s < tau) / len(oos_scores)
        obj = keep + decl
        if obj > best_obj:
            best_obj, best_tau = obj, tau
    return best_tau


def macro_f1(gold, pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return statistics.mean(f1s), min(f1s)


for mode in ("1nn", "pooled"):
    tau = calibrate(mode)
    in_gold, in_pred_argmax, in_pred_gated = [], [], []
    oos_declined = 0
    oos_total = 0
    for p in held["prompts"]:
        gold = p["expectedIntentId"]
        lab, score = predict(p["prompt"], mode)
        if gold == "__none__":
            oos_total += 1
            if lab is None or score < tau:
                oos_declined += 1
            continue
        in_gold.append(gold)
        in_pred_argmax.append(lab)
        in_pred_gated.append(lab if (lab is not None and score >= tau) else "__declined__")

    argmax_acc = sum(1 for g, p in zip(in_gold, in_pred_argmax) if g == p) / len(in_gold)
    gated_acc = sum(1 for g, p in zip(in_gold, in_pred_gated) if g == p) / len(in_gold)
    mf1, minf1 = macro_f1(in_gold, in_pred_gated, intents)
    print(f"--- BM25 {mode} (tau={tau:.4f}, calibrated on TRAIN only) ---")
    print(f"  in-scope n={len(in_gold)}  argmax acc = {argmax_acc:.3f}")
    print(f"  in-scope gated acc        = {gated_acc:.3f}")
    print(f"  macro-F1 (16 intents)     = {mf1:.3f}   min per-intent F1 = {minf1:.3f}")
    print(f"  OOS decline rate          = {oos_declined}/{oos_total} = {oos_declined/oos_total:.3f}")
    print()

print("Recorded baselines on the SAME held-out set (state/router-spike/RESULTS.md):")
print("  cosine-1NN         : argmax/gated acc 0.809, macro-F1 0.770, OOS-decline 0.500")
print("  production router  : gated acc 0.755, macro-F1 0.745, OOS-decline 0.375")
print("  learned head @0.37 : gated acc 0.818, argmax 0.891, macro-F1 0.817, OOS-decline 0.688")
