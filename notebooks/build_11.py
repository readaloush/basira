"""Builds notebooks/11_tumor_outline.ipynb: a small U-Net that outlines the tumor, trained on the BRISC 2025 radiologist masks
under the same clean protocol (no training image with a twin in any test set; validation split by near-duplicate group;
the BRISC segmentation test is scored once at the end). Exports an fp16-weight ONNX model for the browser demo and packs it
as base64 text parts. Inputs on Kaggle: briscdataset/brisc2025, masoudnickparvar/brain-tumor-mri-dataset (v2). GPU, Internet on."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 11 · Outlining the tumor

The classifier says *which* tumor; its heat map only shows where the evidence was, not the tumor's border.
This notebook trains a U-Net (EfficientNet-B0 encoder, timm) on the BRISC 2025 segmentation masks drawn by radiologists, so that the demo
can draw the tumor's outline and a box around it.

* Same clean protocol as the classifier: training images with a near-identical twin (correlation ≥ 0.90) in any test set are removed,
  validation is split by near-duplicate group, and the BRISC segmentation test is scored once at the end.
* No-tumor slices with empty masks are mixed into training, so the model learns to draw nothing on a normal brain.
* Same preprocessing as the demo: crop the head, keep the aspect ratio, pad to a square.
""")

code("""
!pip -q install imagehash segmentation-models-pytorch onnx onnxruntime
import os, glob, json, hashlib, time, random, math, base64
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
import imagehash, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupKFold
SEED = 42; OUT = "/kaggle/working"; DEV = "cuda"; SIZE = 320
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
def find_dirs(keys, name):
    return sorted([d for d in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isdir(d) and any(k in d.lower() for k in keys)], key=len)
SEG = find_dirs(["brisc"], "segmentation_task")[0]
CLS = find_dirs(["brisc"], "classification_task")[0]
KAG_TE = find_dirs(["masoudnickparvar", "brain-tumor-mri-dataset"], "Testing")[0]
print(SEG, CLS, KAG_TE, sep="\\n")
for root, dirs, files in os.walk(SEG):
    if files: print(root, len(files), files[:2])
""")

code("""
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
stem = lambda p: os.path.splitext(os.path.basename(p))[0]
def seg_split(split):
    base = f"{SEG}/{split}"
    imgs = [p for p in glob.glob(f"{base}/**/*", recursive=True) if p.lower().endswith(IMG_EXT) and "/images" in p.lower()]
    msks = [p for p in glob.glob(f"{base}/**/*", recursive=True) if p.lower().endswith(IMG_EXT) and "/masks" in p.lower()]
    mk = {}
    for p in msks: mk[stem(p).replace("_mask", "").replace("-mask", "")] = p
    rows = [dict(path=p, mask=mk.get(stem(p)), ds="brisc_seg", split=split) for p in sorted(imgs)]
    df = pd.DataFrame(rows); print(split, len(df), "images,", df["mask"].notna().sum(), "with a mask"); return df[df["mask"].notna()]
seg_tr, seg_te = seg_split("train"), seg_split("test")
# tumor type from the BRISC file name (brisc2025_train_00001_gl_ax_t1.jpg -> gl)
TYPE = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}
def ttype(p):
    for part in stem(p).split("_"):
        if part in TYPE: return TYPE[part]
    return "unknown"
for d in (seg_tr, seg_te): d["label"] = [ttype(p) for p in d.path]
print(seg_tr.label.value_counts().to_dict(), seg_te.label.value_counts().to_dict())
# no-tumor slices (empty masks) from the BRISC classification training split
nt = sorted(glob.glob(f"{CLS}/train/no_tumor/*") + glob.glob(f"{CLS}/train/notumor/*"))
norm_tr = pd.DataFrame(dict(path=nt, mask=None, ds="brisc_cls", split="train", label="notumor"))
# test sets used for the twin check, and no-tumor test slices for the false-outline check
cls_te = pd.DataFrame([dict(path=p, mask=None, ds="brisc_cls", split="test", label=os.path.basename(os.path.dirname(p)).replace("no_tumor", "notumor"))
                       for p in sorted(glob.glob(f"{CLS}/test/*/*"))])
kag_te = pd.DataFrame([dict(path=p, mask=None, ds="kaggle", split="test", label=os.path.basename(os.path.dirname(p)))
                       for p in sorted(glob.glob(f"{KAG_TE}/*/*"))])
A = pd.concat([seg_tr, norm_tr, seg_te, cls_te, kag_te], ignore_index=True)
A["role"] = ["train" if (r.split == "train") else "test" for r in A.itertuples()]
print(A.groupby(["ds", "split", "label"]).size().unstack(fill_value=0))
""")

md("## Twin removal (same rule as notebooks 04–08)")
code("""
_POP = np.array([bin(k).count("1") for k in range(256)], np.uint8)
popcount = lambda x: _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)
t0 = time.time(); md5s, ph, th = [], [], []
for p in A.path:
    md5s.append(hashlib.md5(open(p, "rb").read()).hexdigest())
    im = Image.open(p).convert("L"); ph.append(int(str(imagehash.phash(im)), 16))
    a = np.asarray(im.resize((64, 64), Image.BILINEAR), np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6); th.append(a / np.sqrt(len(a)))
A["md5"] = md5s; ph = np.array(ph, np.uint64); th = np.stack(th); print(f"hashed {len(A)} in {time.time()-t0:.0f}s")
def best_twin(q_idx, ref_idx):
    ref_idx = np.asarray(ref_idx); ref_md5 = set(A.md5.values[ref_idx]); best = np.zeros(len(q_idx))
    for k, i in enumerate(q_idx):
        if A.md5.values[i] in ref_md5: best[k] = 1.0; continue
        cand = ref_idx[popcount(ph[i] ^ ph[ref_idx]) <= 10]
        if len(cand):
            c = float((th[cand] @ th[i]).max()); best[k] = c if c >= 0.90 else 0
    return best
TEST_IDX = np.nonzero((A.role == "test").values)[0]; TR_IDX = np.nonzero((A.role == "train").values)[0]
b = best_twin(TR_IDX, TEST_IDX)
print("training images removed for a twin in a test set:\\n", pd.crosstab(A.ds.values[TR_IDX], b > 0))
POOL = TR_IDX[b == 0]; pool = A.loc[POOL].reset_index()
# near-duplicate groups inside the pool -> group split for validation
n = len(POOL); parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for k in range(n - 1):
    i = POOL[k]; d = popcount(ph[i] ^ ph[POOL[k + 1:]])
    for j in np.nonzero(d <= 10)[0] + k + 1:
        if float(th[POOL[j]] @ th[i]) >= 0.90 or md5s[i] == md5s[POOL[j]]: parent[find(k)] = find(int(j))
pool["group"] = [find(i) for i in range(n)]
tr_i, va_i = next(GroupKFold(n_splits=8).split(pool, groups=pool.group))
train_df, val_df = pool.iloc[tr_i].reset_index(drop=True), pool.iloc[va_i].reset_index(drop=True)
print("train", train_df.label.value_counts().to_dict(), "\\nval", val_df.label.value_counts().to_dict())
""")

md("## Data: same crop as the demo, image and mask transformed together")
code("""
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T2
from torchvision import tv_tensors
from torch.amp import autocast, GradScaler
def crop_box(img, thr=12, pad=6):
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2))); ys, xs = np.nonzero(g > thr)
    if len(xs) < 100: return (0, 0, img.width, img.height)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, g.shape[1]), min(ys.max() + pad, g.shape[0]))
def square(im, box, resample):
    im = im.crop(box); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), resample)
    c = Image.new(im.mode, (SIZE, SIZE)); c.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2)); return c
CACHE = {}
def load(p, mp):
    if p not in CACHE:
        im = Image.open(p).convert("RGB"); box = crop_box(im)
        x = np.asarray(square(im, box, Image.BICUBIC), np.uint8)
        if mp is None or (isinstance(mp, float) and math.isnan(mp)): m = np.zeros((SIZE, SIZE), np.uint8)
        else:
            mi = Image.open(mp).convert("L")
            if mi.size != im.size: mi = mi.resize(im.size, Image.NEAREST)
            m = (np.asarray(square(mi, box, Image.NEAREST)) > 127).astype(np.uint8)
        CACHE[p] = (x, m)
    return CACHE[p]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
AUG = T2.Compose([T2.RandomHorizontalFlip(), T2.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                  T2.RandomApply([T2.ColorJitter(brightness=0.25, contrast=0.25)], p=0.7)])
class Seg(Dataset):
    def __init__(self, f, train): self.f, self.train = f.reset_index(drop=True), train
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]; x, m = load(r.path, r["mask"])
        x = tv_tensors.Image(torch.from_numpy(x).permute(2, 0, 1)); m = tv_tensors.Mask(torch.from_numpy(m)[None])
        if self.train: x, m = AUG(x, m)
        x = (x.float() / 255 - MEAN) / STD
        return x, m.float()
def loader(f, train, bs=16):
    return DataLoader(Seg(f, train), batch_size=bs if train else 32, shuffle=train, num_workers=4, pin_memory=True, drop_last=train)
t0 = time.time()
for r in pd.concat([pool, A.loc[TEST_IDX]]).itertuples(): load(r.path, r.mask)
print("cached", len(CACHE), f"in {time.time()-t0:.0f}s")
""")

md("## Train")
code("""
@torch.no_grad()
def predict(model, f):
    model.eval(); out = []
    for x, _ in loader(f, False):
        x = x.to(DEV)
        with autocast("cuda", dtype=torch.float16):
            p = (model(x).float().sigmoid() + model(torch.flip(x, dims=[3])).float().sigmoid().flip(dims=[3])) / 2
        out.append(p[:, 0].cpu().numpy().astype(np.float16))
    return np.concatenate(out)
def masks(f): return np.stack([load(r.path, r.mask)[1] for r in f.itertuples()])
def dice(p, m, t=0.5):
    a = p >= t; inter = (a & (m > 0)).sum((1, 2)); s = a.sum((1, 2)) + (m > 0).sum((1, 2))
    return np.where(s == 0, 1.0, 2 * inter / np.maximum(s, 1))
def iou(p, m, t=0.5):
    a = p >= t; b = m > 0; inter = (a & b).sum((1, 2)); u = (a | b).sum((1, 2)); return np.where(u == 0, 1.0, inter / np.maximum(u, 1))

def train_seg(epochs=40, lr=5e-4):
    seed_all(SEED)
    model = smp.Unet("tu-efficientnet_b0", encoder_weights="imagenet", in_channels=3, classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tl = loader(train_df, True); steps = epochs * len(tl); warm = len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    dl = smp.losses.DiceLoss("binary", from_logits=True); bce = torch.nn.BCEWithLogitsLoss(); scaler = GradScaler()
    mv = masks(val_df); tum = mv.reshape(len(mv), -1).any(1); best, state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, m in tl:
            x, m = x.to(DEV, non_blocking=True), m.to(DEV, non_blocking=True)
            with autocast("cuda", dtype=torch.float16): o = model(x); loss = dl(o, m) + bce(o, m)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        pv = predict(model, val_df); d = dice(pv, mv)
        score = d[tum].mean()                                         # Dice on tumor slices ...
        fp = float(((pv >= 0.5).reshape(len(pv), -1).sum(1)[~tum] > 0).mean()) if (~tum).any() else 0.0   # ... and normal slices with any outline
        hist.append(dict(ep=ep, loss=tot / len(tl), val_dice_tumor=round(float(score), 4), val_normal_with_outline=round(fp, 4)))
        print(hist[-1], f"{time.time()-t0:.0f}s")
        if score > best: best, state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); return model, hist
model, HIST = train_seg()
torch.save(model.state_dict(), f"{OUT}/unet_effb0_{SIZE}.pt")
""")

md("## Choose the outline rule on validation (threshold, smallest region), then score the BRISC test once")
code("""
from scipy import ndimage
def regions(p, t, min_frac):
    a = p >= t; lab, k = ndimage.label(a)
    if k == 0: return a
    sizes = ndimage.sum(a, lab, range(1, k + 1)); keep = np.zeros(k + 1, bool); keep[1:] = sizes >= min_frac * a.size
    return keep[lab]
def evaluate(P, M, t, min_frac):
    out = np.stack([regions(p.astype(np.float32), t, min_frac) for p in P]); tum = M.reshape(len(M), -1).any(1)
    inter = (out & (M > 0)).sum((1, 2)); s = out.sum((1, 2)) + (M > 0).sum((1, 2)); u = (out | (M > 0)).sum((1, 2))
    d = np.where(s == 0, 1.0, 2 * inter / np.maximum(s, 1)); j = np.where(u == 0, 1.0, inter / np.maximum(u, 1))
    found = out.reshape(len(out), -1).any(1)
    return dict(dice_tumor=round(float(d[tum].mean()), 4), iou_tumor=round(float(j[tum].mean()), 4),
                tumor_found=round(float(found[tum].mean()), 4), normal_with_outline=round(float(found[~tum].mean()), 4) if (~tum).any() else None)
PV, MV = predict(model, val_df), masks(val_df)
grid = [dict(t=t, min_frac=mf, **evaluate(PV, MV, t, mf)) for t in [0.4, 0.5, 0.6] for mf in [0, 0.0005, 0.001, 0.002]]
G = pd.DataFrame(grid); print(G)
ok = G[G.tumor_found >= G.tumor_found.max() - 0.01]
RULE = ok.sort_values(["normal_with_outline", "dice_tumor"], ascending=[True, False]).iloc[0]
T_OUT, MIN_FRAC = float(RULE.t), float(RULE.min_frac); print("rule: threshold", T_OUT, "smallest region", MIN_FRAC)

te = A.loc[TEST_IDX]; seg_test = te[te.ds == "brisc_seg"].reset_index(drop=True)
PT, MT = predict(model, seg_test), masks(seg_test)
res = {"brisc_seg_test_all": evaluate(PT, MT, T_OUT, MIN_FRAC)}
for c in ["glioma", "meningioma", "pituitary"]:
    k = (seg_test.label == c).values
    if k.any(): res[c] = dict(n=int(k.sum()), **evaluate(PT[k], MT[k], T_OUT, MIN_FRAC))
# false outlines on no-tumor test slices that the model never saw (BRISC and Kaggle)
for name, f in [("brisc_test_notumor", te[(te.ds == "brisc_cls") & (te.label == "notumor")]), ("kaggle_test_notumor", te[(te.ds == "kaggle") & (te.label == "notumor")])]:
    P0 = predict(model, f.reset_index(drop=True)); found = np.array([regions(p.astype(np.float32), T_OUT, MIN_FRAC).any() for p in P0])
    res[name] = dict(n=len(f), with_outline=round(float(found.mean()), 4))
RESULTS = dict(train=len(train_df), val=len(val_df), size=SIZE, rule=dict(threshold=T_OUT, min_region_fraction=MIN_FRAC),
               val_grid=grid, history=HIST, test=res)
print(json.dumps(res, indent=1)); json.dump(RESULTS, open(f"{OUT}/results_11.json", "w"), indent=1)
""")

code("""
# figure: test slices with the radiologist's outline (green) and the model's (red)
import matplotlib.pyplot as plt
rng = np.random.default_rng(0); pick = []
for c in ["glioma", "meningioma", "pituitary"]: pick += list(rng.choice(np.nonzero((seg_test.label == c).values)[0], 4, replace=False))
fig, ax = plt.subplots(3, 4, figsize=(14, 11))
for a, i in zip(ax.flat, pick):
    x, m = load(seg_test.path[i], seg_test["mask"][i]); o = regions(PT[i].astype(np.float32), T_OUT, MIN_FRAC)
    a.imshow(x); a.contour(m, [0.5], colors="#3ddc84", linewidths=1.5); a.contour(o, [0.5], colors="#ff3b3b", linewidths=1.5)
    a.set_title(f"{seg_test.label[i]}  Dice {dice(PT[i:i+1], m[None])[0]:.2f}", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/outline_examples.png", dpi=110); plt.show()
""")

md("## Export for the browser: input 320×320 RGB in [0, 1], output the tumor probability per pixel")
code("""
import onnx, onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto
class Demo(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m; self.register_buffer("mean", MEAN[None]); self.register_buffer("std", STD[None])
    def forward(self, x): return self.m((x - self.mean) / self.std).sigmoid()
D = Demo(model.float().cpu()).eval(); x = torch.rand(1, 3, SIZE, SIZE)
torch.onnx.export(D, x, f"{OUT}/outline_fp32.onnx", input_names=["image"], output_names=["mask"], opset_version=17, dynamo=False)
g = onnx.load(f"{OUT}/outline_fp32.onnx"); casts, keep = [], []
for t in g.graph.initializer:
    a = numpy_helper.to_array(t)
    if a.dtype == np.float32 and a.size >= 1024:
        keep.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_h"))
        casts.append(helper.make_node("Cast", [t.name + "_h"], [t.name], to=TensorProto.FLOAT))
    else: keep.append(t)
del g.graph.initializer[:]; g.graph.initializer.extend(keep)
nodes = list(g.graph.node); del g.graph.node[:]; g.graph.node.extend(casts + nodes); onnx.save(g, f"{OUT}/outline_fp16.onnx")
for k in ["fp32", "fp16"]: print(k, round(os.path.getsize(f"{OUT}/outline_{k}.onnx") / 1e6, 1), "MB")
# the browser model must give the same outlines as the trained one
s = ort.InferenceSession(f"{OUT}/outline_fp16.onnx"); P16 = []
for r in seg_test.itertuples():
    xi = load(r.path, r.mask)[0].astype(np.float32).transpose(2, 0, 1)[None] / 255
    P16.append(s.run(None, {"image": xi})[0][0, 0])
P16 = np.stack(P16); e16 = evaluate(P16, MT, T_OUT, MIN_FRAC)
print("fp16 ONNX (no flip TTA) on BRISC seg test:", e16); RESULTS["onnx_fp16_no_tta"] = e16
json.dump(RESULTS, open(f"{OUT}/results_11.json", "w"), indent=1)
""")

md("## Pack as base64 text parts (moved to GitHub through a browser)")
code("""
b = open(f"{OUT}/outline_fp16.onnx", "rb").read(); n = 2; k = -(-len(b) // n)
meta = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), parts=[], rule=RESULTS["rule"], size=SIZE)
for i in range(n):
    part = b[i * k:(i + 1) * k]
    open(f"{OUT}/opart{i+1}.txt", "w").write(base64.b64encode(part).decode())
    meta["parts"].append(dict(name=f"outline_fp16.part{i+1}", bytes=len(part), sha256=hashlib.sha256(part).hexdigest()))
json.dump(meta, open(f"{OUT}/outline_meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/11_tumor_outline.ipynb"); print("ok")
