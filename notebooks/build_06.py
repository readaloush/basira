"""Builds notebooks/06_best_model.ipynb: final model under the clean protocol.
Pool = POOL_SOURCES with test twins removed. Decisions (epochs kept, ensemble, no-tumor threshold) are made on the BRISC test,
and the Kaggle test is reported once at the end, untouched by any choice."""
import json, sys, nbformat as nbf
POOL_SOURCES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["kaggle", "brisc"]
src = [''.join(c['source']) for c in json.load(open('notebooks/05_source_ablation.ipynb'))['cells']]
setup, hashing, groups, data = src[1], src[2], src[3], src[4]
data = data.replace("SIZE = 288", "SIZE = 384")
data = data.replace('def loader(f, train, bs=32):', 'def loader(f, train, bs=24):')
data = data.replace('torch.save(state, f"{OUT}/{name}_{TAG}.pt")', 'torch.save(state, f"{OUT}/{name}_{TAG}.pt")')
assert "SIZE = 384" in data
run = f'''
POOL_SOURCES = {POOL_SOURCES!r}
sub = pool[pool.ds.isin(POOL_SOURCES)].reset_index(drop=True); sub["strat"] = sub.label + "_" + sub.ds
tr, va = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED).split(sub, sub.strat, sub.group))
train_df, val_df = sub.iloc[tr].reset_index(drop=True), sub.iloc[va].reset_index(drop=True)
test = A.loc[TEST_IDX].reset_index(); y = test.label.map(C2I).values
dev, fin = (test.ds == "brisc").values, (test.ds == "kaggle").values          # BRISC test = development, Kaggle test = final
print(len(train_df), len(val_df), train_df.ds.value_counts().to_dict())
P = {{}}
for name in ["tf_efficientnetv2_s.in21k_ft_in1k", "convnext_small.fb_in22k_ft_in1k"]:
    TAG = "final"; t0 = time.time()
    m = train_model(name, epochs=16); P[name] = predict(m, test); del m; torch.cuda.empty_cache()
    ok = P[name].argmax(1) == y
    print(f"{{name}}: dev(BRISC) {{100*ok[dev].mean():.2f}}  final(Kaggle) {{100*ok[fin].mean():.2f}}  ({{(time.time()-t0)/60:.0f}} min)")
    np.save(f"{{OUT}}/probs_{{name}}.npy", P[name])
names = list(P)
cands = {{n: P[n] for n in names}}; cands["ensemble"] = np.mean([P[n] for n in names], 0)
dev_acc = {{k: float((v.argmax(1)[dev] == y[dev]).mean()) for k, v in cands.items()}}
choice = max(dev_acc, key=dev_acc.get); pc = cands[choice]; print("dev accuracy", dev_acc, "-> choose", choice)

# no-tumor threshold: say "no tumor" only if p(no tumor) >= t; otherwise the most likely tumor class.
def decide(p, t):
    yh = p.argmax(1).copy(); tum = p.copy(); tum[:, NT] = -1
    yh[(yh == NT) & (p[:, NT] < t)] = tum[(yh == NT) & (p[:, NT] < t)].argmax(1); return yh
rows = []
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
    yh = decide(pc, t)
    rows.append(dict(t=t, dev_acc=round(100 * (yh[dev] == y[dev]).mean(), 2),
                     dev_missed=int(((yh == NT) & (y != NT) & dev).sum()), dev_false_alarm=int(((yh != NT) & (y == NT) & dev).sum())))
TH = pd.DataFrame(rows); print(TH)
ok_t = TH[TH.dev_acc >= TH.dev_acc.max() - 0.5]                     # within 0.5 point of the best dev accuracy ...
T = float(ok_t.sort_values(["dev_missed", "t"]).iloc[0].t)          # ... the threshold that misses the fewest tumors
print("threshold", T)

def report(yh, mask):
    f = test[mask].assign(pred=[CLASSES[i] for i in yh[mask]], ok=yh[mask] == y[mask])
    kg = f[f.label == "glioma"]; km = f[f.label == "meningioma"]
    return dict(n=int(mask.sum()), acc=round(100 * f.ok.mean(), 2), by_class=(100 * f.groupby("label").ok.mean()).round(1).to_dict(),
                missed_tumors=int(((f.pred == "notumor") & (f.label != "notumor")).sum()),
                false_alarms=int(((f.pred != "notumor") & (f.label == "notumor")).sum()),
                glioma_other_sources=round(100 * kg[~kg.is512].ok.mean(), 2) if (~kg.is512).any() else None,
                meningioma_other_sources=round(100 * km[~km.is512].ok.mean(), 2) if (~km.is512).any() else None,
                confusion=pd.crosstab(f.label, f.pred).to_dict(orient="index"))
yh = decide(pc, T)
FINAL = dict(pool=POOL_SOURCES, train=len(train_df), choice=choice, threshold=T, dev_accuracy=dev_acc,
             thresholds=TH.to_dict(orient="records"),
             kaggle_final=report(yh, fin), kaggle_final_argmax=report(pc.argmax(1), fin), brisc_dev=report(yh, dev))
print(json.dumps(FINAL, indent=1, default=str))
json.dump(FINAL, open(f"{{OUT}}/results_06.json", "w"), indent=1, default=str)
test.assign(pred=[CLASSES[i] for i in yh], p_notumor=pc[:, NT]).drop(columns=["md5"]).to_csv(f"{{OUT}}/test_predictions_06.csv", index=False)
'''
cells = [nbf.v4.new_markdown_cell(f"""# 06 · Final model under the clean protocol
Training pool: {', '.join(POOL_SOURCES)}, with every image that has a twin in either test set removed (as in notebooks 04-05).
384-pixel input; EfficientNetV2-S and ConvNeXt-Small. All choices (single model or ensemble, the no-tumor threshold) are
made on the BRISC test; the Kaggle test is scored once at the end with those choices fixed.""")]
cells += [nbf.v4.new_code_cell(c.strip()) for c in [setup, hashing, groups, data, run]]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/06_best_model.ipynb"); print("ok", POOL_SOURCES)
