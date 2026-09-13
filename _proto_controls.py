"""Controls for the BM25 router measurement.

C1  label-shuffle: shuffle TRAIN intent labels -> argmax accuracy must collapse
    to chance (~1/16 = 0.0625). If it does not, the measurement is not measuring
    what it claims.
C2  query-shuffle: pair each held-out prompt with a RANDOM gold label -> same
    collapse, from the other side.
C3  sensitivity: does the pooled argmax number survive perturbing k1/b?
"""
import json, math, re, collections, random, statistics

K1, B = 1.2, 0.75


def tokenize(text):
    return [t.lower() for t in re.split(r"[^0-9A-Za-z]+", text) if t]


def idf(n, df):
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


class Bm25:
    def __init__(self, docs, k1=K1, b=B):
        self.docs, self.k1, self.b = docs, k1, b
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
            dl, s = len(d), 0.0
            for term in qt:
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                s += idf(self.n, self.df[term]) * (f * (self.k1 + 1.0)) / denom
            out.append(s)
        return out


held = json.load(open("state/router-spike/heldout-test.json", encoding="utf-8"))
train = json.load(open("state/router-spike/train-set.json", encoding="utf-8"))
in_prompts = [p for p in held["prompts"] if p["expectedIntentId"] != "__none__"]


def pooled_argmax_acc(intent_of_group, k1=K1, b=B):
    pooled = collections.OrderedDict()
    for gi, group in enumerate(train["inScope"]):
        pooled[intent_of_group[gi]] = pooled.get(intent_of_group[gi], []) + tokenize(
            " ".join(group["prompts"])
        )
    labels = list(pooled.keys())
    bm = Bm25(list(pooled.values()), k1, b)
    correct = 0
    for p in in_prompts:
        sc = bm.scores(p["prompt"])
        if not sc or max(sc) <= 0.0:
            continue
        i = max(range(len(sc)), key=lambda j: sc[j])
        if labels[i] == p["expectedIntentId"]:
            correct += 1
    return correct / len(in_prompts)


true_labels = [g["intentId"] for g in train["inScope"]]
print(f"C0 baseline pooled argmax          = {pooled_argmax_acc(true_labels):.3f}")

# C1 -- shuffle which intent name each TRAIN group carries.
accs = []
for seed in range(20):
    rng = random.Random(seed)
    shuf = true_labels[:]
    rng.shuffle(shuf)
    accs.append(pooled_argmax_acc(shuf))
print(f"C1 label-shuffle argmax (20 seeds) = mean {statistics.mean(accs):.3f} "
      f"max {max(accs):.3f}   [chance = {1/16:.3f}]")

# C2 -- shuffle the held-out gold labels instead.
gold = [p["expectedIntentId"] for p in in_prompts]
accs2 = []
for seed in range(20):
    rng = random.Random(100 + seed)
    g = gold[:]
    rng.shuffle(g)
    pooled = collections.OrderedDict()
    for group in train["inScope"]:
        pooled[group["intentId"]] = tokenize(" ".join(group["prompts"]))
    labels = list(pooled.keys())
    bm = Bm25(list(pooled.values()))
    correct = 0
    for p, gl in zip(in_prompts, g):
        sc = bm.scores(p["prompt"])
        i = max(range(len(sc)), key=lambda j: sc[j])
        if labels[i] == gl:
            correct += 1
    accs2.append(correct / len(in_prompts))
print(f"C2 gold-shuffle argmax (20 seeds)  = mean {statistics.mean(accs2):.3f} "
      f"max {max(accs2):.3f}   [chance = {1/16:.3f}]")

# C3 -- hyperparameter sensitivity.
print("C3 sensitivity to (k1, b):")
for k1, b in [(1.2, 0.75), (1.2, 0.0), (1.2, 1.0), (0.9, 0.4), (2.0, 0.75), (1.6, 0.6)]:
    print(f"     k1={k1:<4} b={b:<5} -> argmax {pooled_argmax_acc(true_labels, k1, b):.3f}")
