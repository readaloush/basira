"""Builds notebooks/07_sequences_seeds.ipynb: add multi-sequence data (Feltrin, T1/T1C+/T2; glioma, meningioma, normal),
compare KB vs KBF under the clean protocol with two seeds, and hold out part of Feltrin as a third, sequence-aware test."""
import json, nbformat as nbf
src = [''.join(c['source']) for c in json.load(open('notebooks/06_best_model.ipynb'))['cells']]
setup, hashing, groups, data = src[1], src[2], src[3], src[4]
add_fel = '''
FELK = ["fernando2rad", "brain-tumor-mri-images-17-classes"]
fel_dirs = [d for d in glob.glob("/kaggle/input/**/*NORMAL*", recursive=True) if os.path.isdir(d) and any(k in d for k in FELK)]
FEL = os.path.dirname(sorted(fel_dirs, key=len)[0]); print("feltrin", FEL, sorted(os.listdir(FEL)))
def fel_label(folder):
    f = folder.lower()
    if "normal" in f: return "notumor"
    if f.startswith("meningioma"): return "meningioma"
    if f.startswith("glioma"): return "glioma"
    return None                                     # neurocytoma, schwannoma, other lesions: not in our four classes
for folder in sorted(os.listdir(FEL)):
    lab = fel_label(folder)
    if lab is None or not os.path.isdir(f"{FEL}/{folder}"): continue
    seq = folder.split()[-1].upper()
    for f in sorted(os.listdir(f"{FEL}/{folder}")):
        rows.append(dict(path=f"{FEL}/{folder}/{f}", ds="feltrin", split="train", label=lab, seq=seq))
A = pd.DataFrame(rows); A["seq"] = A.seq.fillna("T1C+")
print(A.groupby(["ds", "split", "label"]).size().unstack()); print(A[A.ds == "feltrin"].groupby(["label", "seq"]).size())
'''
setup = setup.replace('A = pd.DataFrame(rows); print(A.groupby(["ds", "split", "label"]).size().unstack())', add_fel.strip())
assert 'fel_label' in setup
# robust image loading (webp, palette, 16-bit)
hashing = hashing.replace('im = Image.open(p); wh.append(im.size); im = im.convert("L")',
                          'im = Image.open(p); wh.append(im.size); im = im.convert("RGB").convert("L")')
run = '''
# Feltrin: 20% of its near-duplicate groups become a third test set (images from the same patient may still be shared,
# because the dataset has no patient ids; the group split removes only images with a twin)
fel = pool[pool.ds == "feltrin"]
g = np.array(sorted(fel.group.unique())); rng = np.random.default_rng(0); fel_test_groups = set(rng.choice(g, size=len(g) // 5, replace=False))
fel_test = pool[(pool.ds == "feltrin") & pool.group.isin(fel_test_groups)].reset_index(drop=True)
pool_tr = pool[~((pool.ds == "feltrin") & pool.group.isin(fel_test_groups))].reset_index(drop=True)
print("feltrin test", fel_test.groupby(["label", "seq"]).size().to_dict())
test = A.loc[TEST_IDX].reset_index(); y = test.label.map(C2I).values
dev, fin = (test.ds == "brisc").values, (test.ds == "kaggle").values
yf = fel_test.label.map(C2I).values

def run(sources, seed, tag):
    global train_df, val_df, TAG, SEED
    sub = pool_tr[pool_tr.ds.isin(sources)].reset_index(drop=True); sub["strat"] = sub.label + "_" + sub.ds
    tr, va = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42).split(sub, sub.strat, sub.group))
    train_df, val_df = sub.iloc[tr].reset_index(drop=True), sub.iloc[va].reset_index(drop=True)
    TAG, SEED = tag, seed; t0 = time.time()
    m = train_model("tf_efficientnetv2_s.in21k_ft_in1k", epochs=16)
    p, pf = predict(m, test), predict(m, fel_test); del m; torch.cuda.empty_cache()
    np.save(f"{OUT}/probs_{tag}.npy", p); np.save(f"{OUT}/probs_fel_{tag}.npy", pf)
    print(f"{tag}: train {len(train_df)}  BRISC {100*(p.argmax(1)[dev]==y[dev]).mean():.2f}  Kaggle {100*(p.argmax(1)[fin]==y[fin]).mean():.2f}"
          f"  Feltrin {100*(pf.argmax(1)==yf).mean():.2f}  ({(time.time()-t0)/60:.0f} min)")
    return p, pf

P = {}
for sources, seed, tag in [(["kaggle", "brisc", "feltrin"], 42, "KBF_s42"), (["kaggle", "brisc"], 7, "KB_s7"), (["kaggle", "brisc", "feltrin"], 7, "KBF_s7")]:
    P[tag] = run(sources, seed, tag)

def decide(p, t):
    yh = p.argmax(1).copy(); tum = p.copy(); tum[:, NT] = -1; low = (yh == NT) & (p[:, NT] < t)
    yh[low] = tum[low].argmax(1); return yh
def pick_threshold(p):
    rows = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        yh = decide(p, t); rows.append(dict(t=t, acc=100 * (yh[dev] == y[dev]).mean(), missed=int(((yh == NT) & (y != NT) & dev).sum())))
    r = pd.DataFrame(rows); ok = r[r.acc >= r.acc.max() - 0.5]; return float(ok.sort_values(["missed", "t"]).iloc[0].t)
def report(yh, mask, f=None):
    f = (test if f is None else f)[mask] if mask is not None else f
    yy = f.label.map(C2I).values; ok = yh == yy; pred = np.array(CLASSES)[yh]
    r = dict(n=len(f), acc=round(100 * ok.mean(), 2), by_class=(100 * pd.Series(ok).groupby(f.label.values).mean()).round(1).to_dict(),
             missed_tumors=int(((pred == "notumor") & (f.label.values != "notumor")).sum()),
             false_alarms=int(((pred != "notumor") & (f.label.values == "notumor")).sum()))
    if "is512" in f and (f.label == "glioma").any():
        g = (f.label.values == "glioma") & ~f.is512.values; r["glioma_other_sources"] = round(100 * ok[g].mean(), 2) if g.any() else None
    if "seq" in f: r["by_seq"] = (100 * pd.Series(ok).groupby(f.seq.values).mean()).round(1).to_dict()
    return r

R = {}
for tag, (p, pf) in P.items():
    T = pick_threshold(p)
    R[tag] = dict(threshold=T, brisc_dev=report(decide(p, T)[dev], dev), kaggle=report(decide(p, T)[fin], fin),
                  kaggle_argmax=report(p.argmax(1)[fin], fin), feltrin=report(decide(pf, T), None, fel_test))
# two-seed ensembles
for name, tags in [("KBF_2seeds", ["KBF_s42", "KBF_s7"])]:
    p = np.mean([P[t][0] for t in tags], 0); pf = np.mean([P[t][1] for t in tags], 0); T = pick_threshold(p)
    R[name] = dict(threshold=T, brisc_dev=report(decide(p, T)[dev], dev), kaggle=report(decide(p, T)[fin], fin),
                   feltrin=report(decide(pf, T), None, fel_test))
    P[name] = (p, pf)

# abstention on the best configuration by BRISC dev accuracy: confidence threshold chosen on BRISC for 99.5% accuracy
best = max(R, key=lambda k: R[k]["brisc_dev"]["acc"]); p, pf = P[best]; T = R[best]["threshold"]
conf = p.max(1); yh = decide(p, T); ABST = {}
for target in [0.99, 0.995]:
    c_dev = np.sort(conf[dev])[::-1]; ok_dev = (yh[dev] == y[dev])[np.argsort(-conf[dev])]
    cum = np.cumsum(ok_dev) / np.arange(1, dev.sum() + 1); idx = np.nonzero(cum >= target)[0]
    thr = float(c_dev[idx.max()]) if len(idx) else 1.0
    ans = conf >= thr
    ABST[str(target)] = dict(conf_threshold=round(thr, 4),
        kaggle_coverage=round(100 * ans[fin].mean(), 2), kaggle_acc_answered=round(100 * (yh[fin & ans] == y[fin & ans]).mean(), 2),
        kaggle_missed_answered=int(((yh == NT) & (y != NT) & fin & ans).sum()),
        brisc_coverage=round(100 * ans[dev].mean(), 2), brisc_acc_answered=round(100 * (yh[dev & ans] == y[dev & ans]).mean(), 2))
R["best_by_dev"] = best; R["abstention"] = ABST
print(json.dumps(R, indent=1, default=str))
json.dump(R, open(f"{OUT}/results_07.json", "w"), indent=1, default=str)
'''
cells = [nbf.v4.new_markdown_cell("""# 07 · Other MRI sequences, and seed variance
Most remaining errors are gliomas acquired differently (T2/FLAIR, other scanners). Adding gliomas alone would teach "T2 means glioma",
so the added data must cover several classes in each sequence: the Feltrin collection has glioma, meningioma and normal brains in
T1, contrast T1 and T2 (no pituitary class). Pools: KB (Kaggle v2 + BRISC) and KBF (+ Feltrin), same twin removal, 384 px,
EfficientNetV2-S, two seeds. Choices on BRISC test; Kaggle test and a held-out Feltrin part are reported.""")]
cells += [nbf.v4.new_code_cell(c.strip()) for c in [setup, hashing, groups, data, run]]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/07_sequences_seeds.ipynb"); print("ok")
