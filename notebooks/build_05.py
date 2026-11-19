"""Builds notebooks/05_source_ablation.ipynb from notebook 04's cells: same twin removal, one model per training pool."""
import json, nbformat as nbf
src = [''.join(c['source']) for c in json.load(open('notebooks/04_diverse_training.ipynb'))['cells']]
code_cells = [s for s in src if not s.startswith('#')]
setup, hashing, groups, data, final = [c for c in src if c.startswith(('!pip', '_POP', '# near-dup', 'import torchvision', 'model = train_model'))]
groups = groups.split('pool["strat"]')[0]                      # keep only the grouping part
data = data.replace('for p in pd.concat([train_df.path, val_df.path, test.path]): load(p)', 'for p in pd.concat([pool.path, A.loc[TEST_IDX].path]): load(p)')
data = data.replace('torch.save(state, f"{OUT}/{name}_diverse.pt")', 'torch.save(state, f"{OUT}/{name}_{TAG}.pt")')
run = '''
def run(sources, tag, epochs=16):
    global train_df, val_df, TAG
    TAG = tag
    sub = pool[pool.ds.isin(sources)].reset_index(drop=True); sub["strat"] = sub.label + "_" + sub.ds
    tr, va = next(StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED).split(sub, sub.strat, sub.group))
    train_df, val_df = sub.iloc[tr].reset_index(drop=True), sub.iloc[va].reset_index(drop=True)
    test = A.loc[TEST_IDX].reset_index()
    test["twin"] = best_twin(TEST_IDX, train_df["index"].values); test["tier"] = tiers(test.twin)
    print(tag, len(train_df), len(val_df), "test tiers", test.tier.value_counts().to_dict())
    model = train_model("tf_efficientnetv2_s.in21k_ft_in1k", epochs=epochs)
    pt = predict(model, test); y = test.label.map(C2I).values; yh = pt.argmax(1)
    test["pred"] = [CLASSES[i] for i in yh]; test["conf"] = pt.max(1); test["ok"] = yh == y
    R = {"train": len(train_df), "train_by_source": train_df.ds.value_counts().to_dict()}
    for ds in ["kaggle", "brisc"]:
        f = test[test.ds == ds]
        R[ds] = dict(acc=round(100 * f.ok.mean(), 2), by_class=(100 * f.groupby("label").ok.mean()).round(1).to_dict(),
                     missed_tumors=int(((f.pred == "notumor") & (f.label != "notumor")).sum()),
                     tumor_to_notumor_by_class=f[(f.pred == "notumor") & (f.label != "notumor")].label.value_counts().to_dict())
    kg = test[(test.ds == "kaggle") & (test.label == "glioma")]; km = test[(test.ds == "kaggle") & (test.label == "meningioma")]
    R["kaggle_glioma_other_sources"] = round(100 * kg[~kg.is512].ok.mean(), 2)
    R["kaggle_meningioma_other_sources"] = round(100 * km[~km.is512].ok.mean(), 2)
    print(tag, json.dumps(R, default=str)); test.drop(columns=["md5"]).to_csv(f"{OUT}/test_predictions_{tag}.csv", index=False)
    del model; torch.cuda.empty_cache(); return R

RES = {}
for sources, tag in [(["kaggle"], "K"), (["kaggle", "brisc"], "KB")]:
    RES[tag] = run(sources, tag)
    json.dump(RES, open(f"{OUT}/results_05.json", "w"), indent=1, default=str)
RES["KBS_from_nb04"] = dict(kaggle=dict(acc=93.56, meningioma=87.8, missed_tumors=76), brisc=dict(acc=95.10, meningioma=84.3, missed_tumors=39),
                            kaggle_glioma_other_sources=35.53, kaggle_meningioma_other_sources=62.39)
json.dump(RES, open(f"{OUT}/results_05.json", "w"), indent=1, default=str)
'''
cells = [nbf.v4.new_markdown_cell("""# 05 · Which change did what? Same twin removal, three training pools
Notebook 04 added BRISC and SARTAJ **and** removed 1,817 Kaggle training images that have a twin in a test set.
Here the twin removal is the same, and the pool varies: Kaggle only (K), Kaggle + BRISC (KB). Kaggle + BRISC + SARTAJ is notebook 04.
Same model and recipe (EfficientNetV2-S, 16 epochs).""")]
cells += [nbf.v4.new_code_cell(c.strip()) for c in [setup, hashing, groups, data, run]]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/05_source_ablation.ipynb"); print("ok", len(cells))
