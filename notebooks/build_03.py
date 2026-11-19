"""Builds notebooks/03_errors_and_brisc.ipynb.
Inputs on Kaggle: masoudnickparvar/brain-tumor-mri-dataset (v2), briscdataset/brisc2025,
and the output of notebook 02 (basira-02-strong-models: *.pt, test_predictions.csv). GPU, Internet on."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 03 · Where the strong models fail, and does BRISC 2025 leak too?

1. **Error analysis** of notebook 02 (ensemble, official v2 test): which images are wrong, what they are confused with,
   and whether the errors come from one image source (image size and file properties as a source fingerprint).
2. **BRISC 2025 audit**: train/test near-duplicates inside BRISC, and BRISC images that also appear in the Kaggle dataset.
3. **External test**: the notebook 02 models on BRISC test images that have no twin in the Kaggle training set.
""")

code("""
!pip -q install imagehash
import os, glob, json, hashlib, time, math
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
import imagehash, torch, timm
import matplotlib.pyplot as plt
OUT = "/kaggle/working"; DEV = "cuda"
def first(pattern):
    m = sorted(glob.glob(pattern, recursive=True)); assert m, pattern; return m[0]

KAG = os.path.dirname(first("/kaggle/input/**/Training"))
BRISC = first("/kaggle/input/**/classification_task")
PRED = first("/kaggle/input/**/test_predictions.csv")
PTS = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
print(KAG, BRISC, PRED, PTS, sep="\\n")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]; C2I = {c: i for i, c in enumerate(CLASSES)}
BRISC_MAP = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary", "no_tumor": "notumor", "notumor": "notumor"}
""")

md("## 1 · Error analysis on the Kaggle v2 official test")
code("""
pred = pd.read_csv(PRED)
pred["path"] = [KAG + "/Testing/" + q.split("/Testing/", 1)[1] for q in pred.path]   # input mount path may differ
assert all(os.path.exists(q) for q in pred.path[:20])
pred["ok"] = pred.label == pred.pred
def props(p):
    im = Image.open(p); a = np.asarray(im.convert("L"), np.float32)
    return dict(w=im.width, h=im.height, mode=im.mode, ext=os.path.splitext(p)[1].lower(),
                mean=float(a.mean()), dark_frac=float((a < 10).mean()), kb=round(os.path.getsize(p) / 1024, 1))
pred = pd.concat([pred, pd.DataFrame([props(p) for p in pred.path])], axis=1)
pred["size"] = pred.w.astype(str) + "x" + pred.h.astype(str)
pred["is512"] = (pred.w == 512) & (pred.h == 512)
print("accuracy", round(100 * pred.ok.mean(), 2))
print(pd.crosstab(pred.label, pred.pred, margins=True))
print("\\nerrors by class and tier:\\n", pd.crosstab([pred.label], pred.tier, values=~pred.ok, aggfunc="sum").fillna(0).astype(int))
""")

code("""
# image size as a source fingerprint: figshare slices are 512x512
trn = pd.DataFrame([dict(path=p, label=os.path.basename(os.path.dirname(p))) for p in glob.glob(f"{KAG}/Training/*/*")])
trn = pd.concat([trn, pd.DataFrame([props(p) for p in trn.path])], axis=1); trn["is512"] = (trn.w == 512) & (trn.h == 512)
src = pd.DataFrame({
    "train: share 512x512": trn.groupby("label").is512.mean().round(3),
    "test: share 512x512": pred.groupby("label").is512.mean().round(3)})
print(src)
g = pred[pred.label == "glioma"]
print("\\nglioma test accuracy by 512x512 / other:\\n", g.groupby("is512").ok.agg(["mean", "size"]).round(3))
print("\\nglioma test accuracy by tier and 512x512:\\n", g.pivot_table(index="tier", columns="is512", values="ok", aggfunc=["mean", "size"]).round(3))
print("\\ntop sizes, glioma train:", trn[trn.label == "glioma"]["w"].astype(str).add("x").add(trn[trn.label == "glioma"]["h"].astype(str)).value_counts().head(5).to_dict())
print("top sizes, glioma test:", g["size"].value_counts().head(5).to_dict())
print("missed gliomas predicted as:", g[~g.ok].pred.value_counts().to_dict())
""")

code("""
# figure: 18 missed gliomas (with prediction and confidence) next to 6 correctly classified training gliomas
wrong = pred[(pred.label == "glioma") & ~pred.ok].sort_values("conf", ascending=False).head(18)
ref = trn[trn.label == "glioma"].sample(6, random_state=0)
fig, ax = plt.subplots(4, 6, figsize=(16, 11))
for k, (_, r) in enumerate(wrong.iterrows()):
    a = ax.flat[k]; a.imshow(Image.open(r.path).convert("L"), cmap="gray"); a.axis("off")
    a.set_title(f"test glioma -> {r.pred} ({r.conf:.2f})\\n{r['size']}", fontsize=8, color="#b00")
for k, (_, r) in enumerate(ref.iterrows()):
    a = ax.flat[18 + k]; a.imshow(Image.open(r.path).convert("L"), cmap="gray"); a.axis("off")
    a.set_title(f"training glioma\\n{r.w}x{r.h}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/missed_gliomas.png", dpi=110); plt.show()
""")

md("## 2 · BRISC 2025: internal leakage and overlap with the Kaggle dataset")
code("""
rows = []
for split in ["train", "test"]:
    for cls in sorted(os.listdir(f"{BRISC}/{split}")):
        for f in sorted(os.listdir(f"{BRISC}/{split}/{cls}")):
            rows.append(dict(path=f"{BRISC}/{split}/{cls}/{f}", ds="brisc", split=split, label=BRISC_MAP[cls]))
for split in ["Training", "Testing"]:
    for cls in sorted(os.listdir(f"{KAG}/{split}")):
        for f in sorted(os.listdir(f"{KAG}/{split}/{cls}")):
            rows.append(dict(path=f"{KAG}/{split}/{cls}/{f}", ds="kaggle", split=split.lower().replace("ing", ""), label=cls))
A = pd.DataFrame(rows); print(A.groupby(["ds", "split", "label"]).size().unstack())
_POP = np.array([bin(k).count("1") for k in range(256)], np.uint8)
popcount = lambda x: _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)
t0 = time.time(); md5s, ph, th = [], [], []
for p in A.path:
    md5s.append(hashlib.md5(open(p, "rb").read()).hexdigest())
    im = Image.open(p).convert("L"); ph.append(int(str(imagehash.phash(im)), 16))
    a = np.asarray(im.resize((64, 64), Image.BILINEAR), np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6)
    th.append(a / np.sqrt(len(a)))
A["md5"] = md5s; ph = np.array(ph, np.uint64); th = np.stack(th); print(f"hashed {len(A)} in {time.time()-t0:.0f}s")

def best_twin(q_idx, ref_idx):
    # best correlation of each query image with any reference image (pHash Hamming <= 10, NCC >= 0.90)
    ref_idx = np.asarray(ref_idx); best = np.zeros(len(q_idx)); exact = np.zeros(len(q_idx), bool)
    ref_md5 = set(A.md5.values[ref_idx])
    for k, i in enumerate(q_idx):
        d = popcount(ph[i] ^ ph[ref_idx]); cand = ref_idx[d <= 10]
        if len(cand): c = th[cand] @ th[i]; best[k] = max(0, c.max()) if c.max() >= 0.90 else 0
        exact[k] = A.md5.values[i] in ref_md5
    return best, exact

def tiers(b):
    return pd.cut(pd.Series(b), [-1, 1e-4, 0.95, 0.98, 2], right=False, labels=["no twin", "similar", "very similar", "near-identical"])

ix = lambda **kw: np.nonzero(np.logical_and.reduce([A[k].values == v for k, v in kw.items()]))[0]
AUD = {}
for name, q, r in [("brisc_test_vs_brisc_train", ix(ds="brisc", split="test"), ix(ds="brisc", split="train")),
                   ("brisc_test_vs_kaggle_train", ix(ds="brisc", split="test"), ix(ds="kaggle", split="train")),
                   ("brisc_train_vs_kaggle_test", ix(ds="brisc", split="train"), ix(ds="kaggle", split="test"))]:
    b, e = best_twin(q, r); tt = tiers(b)
    AUD[name] = dict(n=len(q), exact=int(e.sum()), tiers=tt.value_counts().sort_index().to_dict(),
                     tiers_by_class=pd.crosstab(A.label.values[q], tt.values).to_dict(orient="index"))
    if name == "brisc_test_vs_brisc_train": A.loc[q, "twin_brisc_train"] = b
    if name == "brisc_test_vs_kaggle_train": A.loc[q, "twin_kaggle_train"] = b
    print(name, json.dumps(AUD[name], indent=1, default=str))
""")

md("## 3 · External test: notebook 02 models on BRISC test images")
code("""
def crop_brain(img, thr=12, pad=6):
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2))); ys, xs = np.nonzero(g > thr)
    if len(xs) < 100: return img
    return img.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, g.shape[1]), min(ys.max() + pad, g.shape[0])))
def prep(p, size=288):
    im = crop_brain(Image.open(p).convert("RGB")); w, h = im.size; s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    c = Image.new("RGB", (size, size)); c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    x = (np.asarray(c, np.float32) / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32)

@torch.no_grad()
def predict(model, paths, bs=64):
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([prep(p) for p in paths[i:i + bs]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            pr = (model(x).float().softmax(1) + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        out.append(pr.cpu())
    return torch.cat(out).numpy()

bt = A[(A.ds == "brisc") & (A.split == "test")].reset_index(drop=True)
P = {}
for pt in PTS:
    name = os.path.basename(pt)[:-3]
    m = timm.create_model(name, pretrained=False, num_classes=4).to(DEV); m.load_state_dict(torch.load(pt, map_location=DEV))
    P[name] = predict(m, list(bt.path)); del m; torch.cuda.empty_cache()
P["ensemble"] = np.mean([P[k] for k in list(P)], 0)
y = bt.label.map(C2I).values; bt["tier_kaggle"] = tiers(bt.twin_kaggle_train.fillna(0).values).values
EXT = {}
for k, p in P.items():
    ok = p.argmax(1) == y
    EXT[k] = dict(acc=round(100 * ok.mean(), 2),
                  by_tier=(100 * pd.Series(ok).groupby(bt.tier_kaggle.values, observed=False).mean()).round(2).to_dict(),
                  by_class_no_twin=(100 * pd.Series(ok[bt.tier_kaggle == "no twin"]).groupby(bt.label[bt.tier_kaggle == "no twin"].values).mean()).round(1).to_dict(),
                  missed_tumors=int(((p.argmax(1) == C2I["notumor"]) & (y != C2I["notumor"])).sum()))
    print(k, json.dumps(EXT[k], default=str))
bt.assign(pred=[CLASSES[i] for i in P["ensemble"].argmax(1)], conf=P["ensemble"].max(1)).drop(columns=["md5"]).to_csv(f"{OUT}/brisc_test_predictions.csv", index=False)
json.dump(dict(brisc_audit=AUD, external=EXT, kaggle_confusion=pd.crosstab(pred.label, pred.pred).to_dict()),
          open(f"{OUT}/results_03.json", "w"), indent=1, default=str)
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/03_errors_and_brisc.ipynb"); print("ok", len(cells))
