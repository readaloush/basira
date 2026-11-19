"""Builds notebooks/04_diverse_training.ipynb.
Inputs on Kaggle: masoudnickparvar/brain-tumor-mri-dataset (v2), briscdataset/brisc2025,
sartajbhuvaji/brain-tumor-classification-mri. GPU T4, Internet on."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 04 · Break the source shortcut: train on several sources, without leaking the test sets

Notebook 03 found that every training glioma in the Kaggle set is a 512x512 figshare slice. The model learned the source:
test gliomas from other sources were 22% correct. Here the training pool adds BRISC train and SARTAJ (all four classes,
several scanners and sequences), so no class is tied to one source.

Leakage control: any added image with an exact copy or a twin (pHash <= 10 and correlation >= 0.90) in the Kaggle test
or the BRISC test is removed from the pool. Test images are then re-tiered against the whole pool.
Model: EfficientNetV2-S, same recipe as notebook 02, so the only change is the data.
""")

code("""
!pip -q install imagehash
import os, glob, json, hashlib, time, random, math
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
import imagehash, torch, timm
from sklearn.model_selection import StratifiedGroupKFold
SEED = 42; OUT = "/kaggle/working"; DEV = "cuda"
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]; C2I = {c: i for i, c in enumerate(CLASSES)}; NT = C2I["notumor"]
NORM = {"glioma": "glioma", "glioma_tumor": "glioma", "meningioma": "meningioma", "meningioma_tumor": "meningioma",
        "notumor": "notumor", "no_tumor": "notumor", "pituitary": "pituitary", "pituitary_tumor": "pituitary"}
def find_dir(keys, name):
    m = [d for d in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if any(k in d for k in keys) and os.path.isdir(d)]
    assert m, (keys, name); return sorted(m, key=len)[0]
KAGK, BRK, SARK = ["masoudnickparvar", "brain-tumor-mri-dataset"], ["brisc"], ["sartaj", "brain-tumor-classification-mri"]
KAG_TR, KAG_TE = find_dir(KAGK, "Training"), find_dir(KAGK, "Testing")
BR = find_dir(BRK, "classification_task")
SAR_TR, SAR_TE = find_dir(SARK, "Training"), find_dir(SARK, "Testing")
assert len({KAG_TR, SAR_TR}) == 2
print(KAG_TR, BR, SAR_TR, sep="\\n")
rows = []
def add(root, ds, split):
    for cls in sorted(os.listdir(root)):
        if cls not in NORM: continue
        for f in sorted(os.listdir(f"{root}/{cls}")):
            rows.append(dict(path=f"{root}/{cls}/{f}", ds=ds, split=split, label=NORM[cls]))
add(KAG_TR, "kaggle", "train"); add(KAG_TE, "kaggle", "test")
add(f"{BR}/train", "brisc", "train"); add(f"{BR}/test", "brisc", "test")
add(SAR_TR, "sartaj", "train"); add(SAR_TE, "sartaj", "train")      # SARTAJ is used for training only
A = pd.DataFrame(rows); print(A.groupby(["ds", "split", "label"]).size().unstack())
""")

code("""
_POP = np.array([bin(k).count("1") for k in range(256)], np.uint8)
popcount = lambda x: _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)
t0 = time.time(); md5s, ph, th, wh = [], [], [], []
for p in A.path:
    md5s.append(hashlib.md5(open(p, "rb").read()).hexdigest())
    im = Image.open(p); wh.append(im.size); im = im.convert("L"); ph.append(int(str(imagehash.phash(im)), 16))
    a = np.asarray(im.resize((64, 64), Image.BILINEAR), np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6)
    th.append(a / np.sqrt(len(a)))
A["md5"] = md5s; A["is512"] = [w == 512 and h == 512 for w, h in wh]
ph = np.array(ph, np.uint64); th = np.stack(th); print(f"hashed {len(A)} in {time.time()-t0:.0f}s")

def best_twin(q_idx, ref_idx):
    ref_idx = np.asarray(ref_idx); ref_md5 = set(A.md5.values[ref_idx]); best = np.zeros(len(q_idx))
    for k, i in enumerate(q_idx):
        if A.md5.values[i] in ref_md5: best[k] = 1.0; continue
        cand = ref_idx[popcount(ph[i] ^ ph[ref_idx]) <= 10]
        if len(cand):
            c = float((th[cand] @ th[i]).max()); best[k] = c if c >= 0.90 else 0
    return best
tiers = lambda b: pd.cut(pd.Series(b), [-1, 1e-4, 0.95, 0.98, 2], right=False,
                         labels=["no twin", "similar", "very similar", "near-identical"]).values
TEST_IDX = np.nonzero((A.split == "test").values)[0]
cand_idx = np.nonzero((A.split == "train").values)[0]
# remove from the pool every training image that has a twin in either test set (checked from the training side)
b = best_twin(cand_idx, TEST_IDX)
removed = pd.crosstab(A.ds.values[cand_idx], b > 0)
print("pool images removed for having a twin in a test set:\\n", removed)
POOL = cand_idx[b == 0]
pool = A.loc[POOL].reset_index()            # 'index' keeps the row in A
print(pool.groupby(["ds", "label"]).size().unstack())
""")

code("""
# near-duplicate groups inside the pool, for a group-split validation set
n = len(POOL); pairs = []
for k in range(n - 1):
    i = POOL[k]; d = popcount(ph[i] ^ ph[POOL[k + 1:]])
    for j in np.nonzero(d <= 10)[0] + k + 1:
        if float(th[POOL[j]] @ th[i]) >= 0.90 or md5s[i] == md5s[POOL[j]]: pairs.append((k, int(j)))
parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j in pairs: parent[find(i)] = find(j)
pool["group"] = [find(i) for i in range(n)]
pool["strat"] = pool.label + "_" + pool.ds
sgkf = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED)
tr, va = next(sgkf.split(pool, pool.strat, pool.group))
train_df, val_df = pool.iloc[tr].reset_index(drop=True), pool.iloc[va].reset_index(drop=True)
print(len(train_df), len(val_df))
# test images re-tiered against the training part of the pool
test = A.loc[TEST_IDX].reset_index()
test["twin"] = best_twin(TEST_IDX, train_df["index"].values); test["tier"] = tiers(test.twin)
print(pd.crosstab([test.ds, test.label], test.tier))
""")

code("""
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
SIZE = 288
def crop_brain(img, thr=12, pad=6):
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2))); ys, xs = np.nonzero(g > thr)
    if len(xs) < 100: return img
    return img.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, g.shape[1]), min(ys.max() + pad, g.shape[0])))
CACHE = {}
def load(p):
    if p not in CACHE:
        im = crop_brain(Image.open(p).convert("RGB")); w, h = im.size; s = SIZE / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        c = Image.new("RGB", (SIZE, SIZE)); c.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2)); CACHE[p] = c
    return CACHE[p]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TF_TR = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(), T.RandomRotation(12),
                   T.ColorJitter(brightness=0.25, contrast=0.25), T.RandomGrayscale(0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
TF_TE = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
class MRI(Dataset):
    def __init__(self, f, tf): self.p, self.y, self.tf = list(f.path), [C2I[l] for l in f.label], tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(load(self.p[i])), self.y[i]
def loader(f, train, bs=32):
    return DataLoader(MRI(f, TF_TR if train else TF_TE), batch_size=bs if train else 64, shuffle=train,
                      num_workers=4, pin_memory=True, drop_last=train)
t0 = time.time()
for p in pd.concat([train_df.path, val_df.path, test.path]): load(p)
print("cached", len(CACHE), f"in {time.time()-t0:.0f}s")

@torch.no_grad()
def predict(model, f, tta=True):
    model.eval(); out = []
    for x, _ in loader(f, False):
        x = x.to(DEV)
        with autocast("cuda", dtype=torch.float16):
            p = model(x).float().softmax(1)
            if tta: p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()

def train_model(name, epochs=16, lr=3e-4):
    seed_all(SEED)
    model = timm.create_model(name, pretrained=True, num_classes=4, drop_path_rate=0.1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    tl = loader(train_df, True); steps = epochs * len(tl); warm = len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1); scaler = GradScaler(); best, state = -1, None
    yv = val_df.label.map(C2I).values
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in tl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with autocast("cuda", dtype=torch.float16): loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        pv = predict(model, val_df, tta=False); acc = (pv.argmax(1) == yv).mean()
        # macro accuracy over sources: the smallest source must not be drowned by the largest
        src_acc = pd.Series(pv.argmax(1) == yv).groupby(val_df.ds.values).mean()
        print(f"ep{ep:2d} loss {tot/len(tl):.3f} val {100*acc:.2f}%  by source {dict((100*src_acc).round(1))} ({time.time()-t0:.0f}s)")
        score = src_acc.mean()
        if score > best: best, state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); torch.save(state, f"{OUT}/{name}_diverse.pt"); return model
""")

code("""
model = train_model("tf_efficientnetv2_s.in21k_ft_in1k")
pt = predict(model, test); y = test.label.map(C2I).values; yh = pt.argmax(1); test["pred"] = [CLASSES[i] for i in yh]; test["conf"] = pt.max(1)
test["ok"] = yh == y
def summary(f):
    return dict(n=len(f), acc=round(100 * f.ok.mean(), 2),
                no_twin=round(100 * f[f.tier == "no twin"].ok.mean(), 2), n_no_twin=int((f.tier == "no twin").sum()),
                missed_tumors=int(((f.pred == "notumor") & (f.label != "notumor")).sum()))
R = {}
for ds in ["kaggle", "brisc"]:
    f = test[test.ds == ds]; R[ds] = summary(f)
    R[ds]["by_class"] = (100 * f.groupby("label").ok.mean()).round(1).to_dict()
    R[ds]["by_class_no_twin"] = (100 * f[f.tier == "no twin"].groupby("label").ok.mean()).round(1).to_dict()
kg = test[(test.ds == "kaggle") & (test.label == "glioma")]
R["kaggle_glioma_512"] = round(100 * kg[kg.is512].ok.mean(), 2); R["kaggle_glioma_other_sources"] = round(100 * kg[~kg.is512].ok.mean(), 2)
R["kaggle_glioma_other_sources_n"] = int((~kg.is512).sum())
R["kaggle_glioma_other_sources_by_tier"] = kg[~kg.is512].groupby("tier", observed=False).ok.agg(["mean", "size"]).round(3).to_dict(orient="index")
km = test[(test.ds == "kaggle") & (test.label == "meningioma")]
R["kaggle_meningioma_other_sources"] = round(100 * km[~km.is512].ok.mean(), 2)
R["kaggle_confusion"] = pd.crosstab(test[test.ds == "kaggle"].label, test[test.ds == "kaggle"].pred).to_dict()
R["previous_model_nb02"] = dict(kaggle_acc=95.88, kaggle_no_twin=90.17, glioma_other_sources=22.4, brisc_acc=99.2)
print(json.dumps(R, indent=1, default=str))
test.drop(columns=["md5"]).to_csv(f"{OUT}/test_predictions_04.csv", index=False)
json.dump(dict(results=R, pool=pool.groupby(["ds", "label"]).size().unstack().to_dict(),
               removed=removed.to_dict()), open(f"{OUT}/results_04.json", "w"), indent=1, default=str)
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/04_diverse_training.ipynb"); print("ok", len(cells))
