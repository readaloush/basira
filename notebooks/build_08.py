"""Builds notebooks/08_third_seed.ipynb: a third seed for KB and KBF, then mean ± std over three seeds and three-seed ensembles.
Earlier test probabilities come from notebooks 06 (KB seed 42) and 07 (KBF seed 42, KB seed 7, KBF seed 7) attached as inputs."""
import json, nbformat as nbf
src = [''.join(c['source']) for c in json.load(open('notebooks/07_sequences_seeds.ipynb'))['cells']]
setup, hashing, groups, data, run7 = src[1], src[2], src[3], src[4], src[5]
head = run7[:run7.index('P = {}')]
post = run7[run7.index('def decide'):run7.index('R = {}')]
run = head + post + '''
P = {}
for sources, seed, tag in [(["kaggle", "brisc"], 1, "KB_s1"), (["kaggle", "brisc", "feltrin"], 1, "KBF_s1")]:
    P[tag] = run(sources, seed, tag)
def load(name):
    m = glob.glob(f"/kaggle/input/**/{name}", recursive=True); return np.load(m[0]) if m else None
for tag in ["KB_s7", "KBF_s42", "KBF_s7"]:
    P[tag] = (load(f"probs_{tag}.npy"), load(f"probs_fel_{tag}.npy"))
P["KB_s42"] = (load("probs_tf_efficientnetv2_s.in21k_ft_in1k.npy"), None)
for k, (p, pf) in P.items():
    assert p is not None and len(p) == len(test), k
    if pf is not None: assert len(pf) == len(fel_test), k

def evaluate(p, pf):
    T = pick_threshold(p); yh = decide(p, T)
    r = dict(threshold=T, brisc=report(yh[dev], dev), kaggle=report(yh[fin], fin), kaggle_argmax=report(p.argmax(1)[fin], fin))
    if pf is not None: r["feltrin"] = report(decide(pf, T), None, fel_test)
    return r
R = {k: evaluate(*v) for k, v in P.items()}
SUM = {}
for cfg in ["KB", "KBF"]:
    tags = [t for t in R if t.startswith(cfg + "_s")]
    def ms(path):
        v = []
        for t in tags:
            x = R[t]
            for key in path: x = x.get(key) if isinstance(x, dict) else None
            if x is not None: v.append(x)
        return dict(mean=round(float(np.mean(v)), 2), std=round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else None, n=len(v), values=v)
    SUM[cfg] = {"seeds": tags, "kaggle_acc": ms(["kaggle", "acc"]), "kaggle_missed": ms(["kaggle", "missed_tumors"]),
                "kaggle_false_alarms": ms(["kaggle", "false_alarms"]), "kaggle_glioma_other": ms(["kaggle", "glioma_other_sources"]),
                "kaggle_acc_argmax": ms(["kaggle_argmax", "acc"]), "brisc_acc": ms(["brisc", "acc"]), "brisc_missed": ms(["brisc", "missed_tumors"]),
                "feltrin_acc": ms(["feltrin", "acc"])}
    pe = np.mean([P[t][0] for t in tags], 0)
    pfs = [P[t][1] for t in tags if P[t][1] is not None]
    SUM[cfg]["ensemble"] = evaluate(pe, np.mean(pfs, 0) if pfs else None)
    SUM[cfg]["ensemble"]["seeds"] = tags
print(json.dumps(SUM, indent=1, default=str))
json.dump(dict(runs=R, summary=SUM), open(f"{OUT}/results_08.json", "w"), indent=1, default=str)
'''
cells = [nbf.v4.new_markdown_cell("""# 08 · Third seed: mean ± std over three seeds for KB and KBF, and three-seed ensembles
Same protocol as notebook 07. Inputs: the four datasets plus the outputs of notebooks 06 and 07 (saved test probabilities).""")]
cells += [nbf.v4.new_code_cell(c.strip()) for c in [setup, hashing, groups, data, run]]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/08_third_seed.ipynb"); print("ok")
