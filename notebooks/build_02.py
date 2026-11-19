"""Builds notebooks/02_strong_models.ipynb (self-contained; Kaggle, GPU T4, Internet on, dataset v2 attached)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 02 · Stronger models, judged on the images that have no twin in training

Same official split as notebook 01 (dataset v2), so the MobileNetV2 baseline is directly comparable:
92.0% on the full official test, **83.6% on the 590 test images with no twin in training**, 66.1% on gliomas without a twin.

Changes, one at a time where it matters:
1. Validation is split by near-duplicate group, so model selection never sees a training image's twin.
2. Brain cropping (remove the black margin), larger input (288 px), MRI-appropriate augmentation.
3. Modern pretrained backbones (ConvNeXt-Tiny, EfficientNetV2-S) fully fine-tuned with AdamW, cosine schedule, label smoothing, mixed precision.
4. Horizontal-flip test-time augmentation and a two-model ensemble.
5. Abstention: a confidence threshold chosen on validation; report accuracy on the images the model does answer.
""")

code("""
!pip -q install imagehash
import os, glob, json, hashlib, time, random, math
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
import imagehash, torch, timm
from sklearn.model_selection import StratifiedGroupKFold
SEED = 42; OUT = "/kaggle/working"
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
DEV = "cuda"; print(torch.__version__, timm.__version__, torch.cuda.get_device_name(0))

roots = sorted({os.path.dirname(r) for r in glob.glob("/kaggle/input/**/Training", recursive=True)})
BASE = roots[0]; rows = []
for split in ["Training", "Testing"]:
    for cls in sorted(os.listdir(f"{BASE}/{split}")):
        for f in sorted(os.listdir(f"{BASE}/{split}/{cls}")):
            rows.append(dict(path=f"{BASE}/{split}/{cls}/{f}", split=split, label=cls))
df = pd.DataFrame(rows); print(df.groupby(["split", "label"]).size().unstack())
CLASSES = sorted(df.label.unique()); C2I = {c: i for i, c in enumerate(CLASSES)}; NT = C2I["notumor"]
""")

md("## Near-duplicate groups and direct-twin tiers (same procedure as notebook 01)")
code("""
_POP = np.array([bin(k).count("1") for k in range(256)], np.uint8)
popcount = lambda x: _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)
md5s, ph, th = [], [], []
for p in df.path:
    md5s.append(hashlib.md5(open(p, "rb").read()).hexdigest())
    im = Image.open(p).convert("L"); ph.append(int(str(imagehash.phash(im)), 16))
    a = np.asarray(im.resize((64, 64), Image.BILINEAR), np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6)
    th.append(a / np.sqrt(len(a)))
df["md5"] = md5s; ph = np.array(ph, np.uint64); th = np.stack(th); n = len(df)
pairs = []
for i in range(n - 1):
    d = popcount(ph[i] ^ ph[i + 1:])
    for j in np.nonzero(d <= 10)[0] + i + 1:
        c = float(th[i] @ th[j])
        if c >= 0.90 or md5s[i] == md5s[j]: pairs.append((i, int(j), c))
parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j, _ in pairs: parent[find(i)] = find(j)
df["group"] = [find(i) for i in range(n)]
is_tr = (df.split == "Training").values; best = np.zeros(n)
for i, j, c in pairs:
    if is_tr[i] != is_tr[j]:
        t_ = j if is_tr[i] else i; best[t_] = max(best[t_], c)
df["twin_ncc"] = best
df["tier"] = pd.cut(df.twin_ncc, [-1, 1e-4, 0.95, 0.98, 2], right=False,
                    labels=["no twin", "similar", "very similar", "near-identical"])
test = df[df.split == "Testing"].reset_index(drop=True)
print(len(pairs), "pairs;", test.tier.value_counts().sort_index().to_dict())
""")

code("""
# validation = 15% of Training, split by group so no validation image has a twin in the training part
tr_all = df[df.split == "Training"].reset_index(drop=True)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
tr_idx, va_idx = next(sgkf.split(tr_all, tr_all.label, tr_all.group))
train_df, val_df = tr_all.iloc[tr_idx].reset_index(drop=True), tr_all.iloc[va_idx].reset_index(drop=True)
assert not set(train_df.group) & set(val_df.group)
print(len(train_df), len(val_df), val_df.label.value_counts().to_dict())
""")

md("## Data: crop the brain, resize, augment")
code("""
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

def crop_brain(img, thr=12, pad=6):
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2)))
    ys, xs = np.nonzero(g > thr)
    if len(xs) < 100: return img
    x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad, g.shape[1]), max(ys.min() - pad, 0), min(ys.max() + pad, g.shape[0])
    return img.crop((x0, y0, x1, y1))

CACHE = {}
def load(p, size):
    k = (p, size)
    if k not in CACHE:
        im = crop_brain(Image.open(p).convert("RGB"))
        w, h = im.size; s = size / max(w, h)           # keep aspect ratio, pad to square
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size)); canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        CACHE[k] = canvas
    return CACHE[k]

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
def tfms(size, train):
    if not train: return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(), T.RandomRotation(12),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize(MEAN, STD)])

class MRI(Dataset):
    def __init__(self, frame, size, train): self.f, self.size, self.t = frame.reset_index(drop=True), size, tfms(size, train)
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]; return self.t(load(r.path, self.size)), C2I[r.label]

def loader(frame, size, train, bs=32):
    return DataLoader(MRI(frame, size, train), batch_size=bs, shuffle=train, num_workers=4, pin_memory=True, drop_last=train)

t0 = time.time()
for p_ in df.path: load(p_, 288)        # cache once in the main process; workers share it after fork
print("cached", len(CACHE), f"images in {time.time()-t0:.0f}s")
""")

md("## Training")
code("""
from torch.amp import autocast, GradScaler

@torch.no_grad()
def predict(model, frame, size, tta=True):
    model.eval(); out = []
    for x, _ in loader(frame, size, False, bs=64):
        x = x.to(DEV)
        with autocast("cuda", dtype=torch.float16):
            p = model(x).float().softmax(1)
            if tta: p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()

def train_model(name, size=288, epochs=16, lr=3e-4, bs=32, seed=SEED):
    seed_all(seed)
    model = timm.create_model(name, pretrained=True, num_classes=len(CLASSES), drop_path_rate=0.1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    tl = loader(train_df, size, True, bs); steps = epochs * len(tl); warm = len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1); scaler = GradScaler()
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in tl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with autocast("cuda", dtype=torch.float16): loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        pv = predict(model, val_df, size, tta=False); acc = (pv.argmax(1) == val_df.label.map(C2I).values).mean()
        hist.append((ep, tot / len(tl), acc)); print(f"{name} ep{ep:2d} loss {tot/len(tl):.3f} val {100*acc:.2f}% ({time.time()-t0:.0f}s)")
        if acc > best: best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save(best_state, f"{OUT}/{name}.pt")
    return model, best, hist
""")

md("## Scoring: overall, by twin tier, by class, missed tumors")
code("""
y_test = test.label.map(C2I).values; y_val = val_df.label.map(C2I).values
def report(p, tag):
    yh = p.argmax(1); ok = yh == y_test; t = test.assign(ok=ok)
    r = dict(acc=round(100 * ok.mean(), 2),
             acc_no_twin=round(100 * ok[(test.tier == "no twin").values].mean(), 2),
             glioma_no_twin=round(100 * ok[((test.tier == "no twin") & (test.label == "glioma")).values].mean(), 2),
             missed_tumors=int(((yh == NT) & (y_test != NT)).sum()), false_alarms=int(((yh != NT) & (y_test == NT)).sum()),
             by_tier=(100 * t.groupby("tier", observed=False).ok.mean()).round(2).to_dict(),
             by_class_no_twin=(100 * t[t.tier == "no twin"].groupby("label").ok.mean()).round(1).to_dict())
    print(f"{tag:34s} all {r['acc']:6.2f}  no-twin {r['acc_no_twin']:6.2f}  glioma no-twin {r['glioma_no_twin']:6.2f}  missed {r['missed_tumors']}")
    return r
RES, PV, PT = {}, {}, {}
""")

code("""
for name in ["convnext_tiny.fb_in22k_ft_in1k", "tf_efficientnetv2_s.in21k_ft_in1k"]:
    t0 = time.time(); m, vbest, hist = train_model(name)
    PV[name], PT[name] = predict(m, val_df, 288), predict(m, test, 288)
    RES[name] = report(PT[name], name); RES[name]["val_best"] = round(100 * float(vbest), 2)
    RES[name]["minutes"] = round((time.time() - t0) / 60, 1)
    del m; torch.cuda.empty_cache()
""")

code("""
# ensemble = mean of the two models' probabilities
pv_e = np.mean([PV[k] for k in PV], 0); pt_e = np.mean([PT[k] for k in PT], 0)
RES["ensemble"] = report(pt_e, "ensemble (2 models, flip TTA)")
""")

md("## Abstention: answer only when confident; threshold chosen on validation")
code("""
def risk_coverage(p, y):
    conf = p.max(1); order = np.argsort(-conf); ok = (p.argmax(1) == y)[order]
    cov = np.arange(1, len(y) + 1) / len(y); acc = np.cumsum(ok) / np.arange(1, len(y) + 1)
    return conf[order], cov, acc
ABST = {}
cv, covv, accv = risk_coverage(pv_e, y_val)
for target in [0.99, 0.995]:
    ok_idx = np.nonzero(accv >= target)[0]
    thr = float(cv[ok_idx.max()]) if len(ok_idx) else 1.0           # lowest confidence that keeps val accuracy >= target
    ans = pt_e.max(1) >= thr; yh = pt_e.argmax(1)
    nt_mask = (test.tier == "no twin").values
    ABST[str(target)] = dict(threshold=round(thr, 4),
        coverage=round(100 * ans.mean(), 1), acc_answered=round(100 * (yh[ans] == y_test[ans]).mean(), 2),
        coverage_no_twin=round(100 * ans[nt_mask].mean(), 1),
        acc_answered_no_twin=round(100 * (yh[ans & nt_mask] == y_test[ans & nt_mask]).mean(), 2),
        missed_tumors_answered=int(((yh == NT) & (y_test != NT) & ans).sum()))
    print(target, ABST[str(target)])
""")

code("""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pt_e.argmax(1)).tolist()
test.assign(pred=[CLASSES[i] for i in pt_e.argmax(1)], conf=pt_e.max(1)).drop(columns=["md5"]).to_csv(f"{OUT}/test_predictions.csv", index=False)
json.dump(dict(results=RES, abstention=ABST, confusion=cm, classes=CLASSES,
               baseline_mobilenet=dict(acc=92.0, acc_no_twin=83.56, glioma_no_twin=66.1)),
          open(f"{OUT}/results_02.json", "w"), indent=1, default=str)
print(json.dumps(RES["ensemble"], indent=1, default=str))
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/02_strong_models.ipynb"); print("ok", len(cells))
