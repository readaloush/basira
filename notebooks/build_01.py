"""Builds notebooks/01_leakage_audit.ipynb (self-contained, runs on Kaggle with a T4)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 01 · How much of the Kaggle brain-tumor test set is already in training?

Dataset: *Brain Tumor MRI Dataset* (Nickparvar). Attach version 1 (5,712 + 1,311 images, the one most
"99%" notebooks used) and version 2 (5,600 + 1,600, described as de-duplicated) as inputs.

1. Find exact duplicates (MD5) and near-duplicates (perceptual hash, confirmed by pixel correlation).
2. Count official test images that have a twin in the official training set, and twins that carry two different labels.
3. Build a clean split: near-duplicates are grouped, and a whole group goes to one side only.
4. Train the original MobileNetV2 recipe (Keras, as in `brain-tumor-classification-cnn`) and measure it on
   the official test set, split into *leaked* and *clean* images, and on the clean split.
""")

code("""
!pip -q install imagehash
import os, glob, json, hashlib, time, random
import numpy as np, pandas as pd
from PIL import Image
import imagehash
SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUT = "/kaggle/working"

# version 2 is attached as input; version 1 (the one most "99%" notebooks used) is downloaded here
try:
    import kagglehub
    V1 = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset/versions/1")
except Exception as e:
    V1 = None; print("could not download version 1:", e)
found = glob.glob("/kaggle/input/**/Training", recursive=True) + (glob.glob(f"{V1}/**/Training", recursive=True) if V1 else [])
roots = sorted({os.path.dirname(r) for r in found})
assert roots, "Add the 'Brain Tumor MRI Dataset' (masoudnickparvar) as input; versions 1 and 2 can both be attached"
rows, seen = [], set()
for base in roots:
    n_train = sum(len(fs) for _, _, fs in os.walk(f"{base}/Training"))
    ver = "v2" if n_train == 5600 else "v1"           # v1: 5,712 training images; v2: 5,600
    if ver in seen: continue                          # same version found twice
    seen.add(ver); print(ver, n_train, base)
    for split in ["Training", "Testing"]:
        for cls in sorted(os.listdir(f"{base}/{split}")):
            for f in sorted(os.listdir(f"{base}/{split}/{cls}")):
                rows.append(dict(path=f"{base}/{split}/{cls}/{f}", version=ver, split=split, label=cls))
ALL = pd.DataFrame(rows)
print(ALL.groupby(["version", "split", "label"]).size().unstack())
VERSIONS = sorted(ALL.version.unique()); print(roots, VERSIONS)
""")

code("""
t = time.time()
md5s, ph, thumbs = [], [], []
for p in ALL.path:
    b = open(p, "rb").read(); md5s.append(hashlib.md5(b).hexdigest())
    im = Image.open(p).convert("L")
    ph.append(int(str(imagehash.phash(im, hash_size=8)), 16))
    a = np.asarray(im.resize((64, 64), Image.BILINEAR), np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6); thumbs.append(a / np.sqrt(len(a)))
ALL["md5"] = md5s
ALL["ph"] = np.array(ph, dtype=np.uint64); THUMBS = np.stack(thumbs)
print(f"hashed {len(ALL)} images in {time.time()-t:.0f}s")
""")

code("""
HAM, NCC = 10, 0.90
_POP = np.array([bin(k).count("1") for k in range(256)], np.uint8)
def popcount(x):  # works on numpy 1.x and 2.x
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)

def audit_version(ver):
    idx = ALL.index[ALL.version == ver].values
    df = ALL.loc[idx].reset_index(drop=True)
    ph, th = df.ph.values.astype(np.uint64), THUMBS[idx]
    n = len(df); pairs = []
    for i in range(n - 1):
        d = popcount(ph[i] ^ ph[i + 1:])
        for j in np.nonzero(d <= HAM)[0] + i + 1:
            c = float(th[i] @ th[j])
            if c >= NCC or df.md5[i] == df.md5[j]:
                pairs.append((i, int(j), int(d[j - i - 1]), c))
    pairs = pd.DataFrame(pairs, columns=["i", "j", "ham", "ncc"])
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(pairs.i, pairs.j): parent[find(i)] = find(j)
    df["group"] = [find(i) for i in range(n)]
    g = df.groupby("group")
    train_groups = set(df[df.split == "Training"].group)
    test = df[df.split == "Testing"].copy(); test["leaked"] = test.group.isin(train_groups)
    # direct twin in training (no chaining): best correlation among pairs that link this test image to a training image
    is_tr = (df.split == "Training").values; best = np.zeros(n)
    for i, j, c in zip(pairs.i, pairs.j, pairs.ncc):
        if is_tr[i] != is_tr[j]:
            t_ = j if is_tr[i] else i; best[t_] = max(best[t_], c)
    test["twin_ncc"] = best[test.index]
    test["tier"] = pd.cut(test.twin_ncc, [-1, 0.0001, 0.95, 0.98, 2], right=False,
                          labels=["no twin", "similar (0.90-0.95)", "very similar (0.95-0.98)", "near-identical (>=0.98)"])
    test["exact_leak"] = test.md5.isin(set(df[df.split == "Training"].md5))
    conflict = {k for k, v in g.label.nunique().items() if v > 1}
    a = dict(version=ver, images=n, train=int((df.split == "Training").sum()), test=len(test),
             images_with_exact_copy=int(df.duplicated("md5", keep=False).sum()),
             near_dup_pairs=len(pairs),
             images_in_near_dup_group=int((g.group.transform("size") > 1).sum()),
             test_exact_copy_in_train=int(test.exact_leak.sum()),
             test_near_dup_in_train=int(test.leaked.sum()),
             test_near_dup_in_train_pct=round(100 * test.leaked.mean(), 2),
             groups_with_two_labels=len(conflict),
             images_in_label_conflict=int(df.group.isin(conflict).sum()),
             leaked_pct_by_class=(100 * test.groupby("label").leaked.mean()).round(1).to_dict(),
             direct_twin_tiers=test.tier.value_counts().sort_index().to_dict(),
             direct_twin_tiers_by_class=pd.crosstab(test.label, test.tier).to_dict(orient="index"))
    return df, test, pairs, conflict, a

R = {}
for v in VERSIONS:
    t = time.time(); R[v] = audit_version(v)
    print(f"{v}: {time.time()-t:.0f}s"); print(json.dumps(R[v][4], indent=1, default=str))
""")

code("""
# figure: example pairs (official test image | its twin in training)
import matplotlib.pyplot as plt
FIGV = max(R, key=lambda v: R[v][4]["test_near_dup_in_train"])
df, test = R[FIGV][0], R[FIGV][1]
ex = []
for _, r in test[test.leaked].sample(frac=1, random_state=SEED).iterrows():
    tw = df[(df.group == r.group) & (df.split == "Training")].iloc[0]
    ex.append((r, tw))
    if len(ex) == 6: break
fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
for k, (a, b) in enumerate(ex):
    for row, r in enumerate((a, b)):
        ax[row, k].imshow(Image.open(r.path).convert("L"), cmap="gray"); ax[row, k].axis("off")
        ax[row, k].set_title(("test: " if row == 0 else "train: ") + r.label, fontsize=9)
plt.suptitle(f"{FIGV}: official test images (top) and their twins in the training folder (bottom)")
plt.tight_layout(); plt.savefig(f"{OUT}/leak_examples.png", dpi=120); plt.show()
""")

code("""
# clean split 70/15/15: stratified by label, grouped by near-duplicate group; label-conflict groups dropped
from sklearn.model_selection import StratifiedGroupKFold
CV = VERSIONS[-1]                                    # newest attached version
df, _, pairs, conflict_groups, _ = R[CV]
clean = df[~df.group.isin(conflict_groups)].copy()
# one representative per group keeps the test set free of within-test copies too
clean = clean.drop_duplicates("md5")
sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
fold = np.zeros(len(clean), int)
for k, (_, idx) in enumerate(sgkf.split(clean, clean.label, clean.group)): fold[idx] = k
clean["cv"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
assert not (set(clean[clean.cv == "train"].group) & set(clean[clean.cv == "test"].group))
assert not (set(clean[clean.cv == "val"].group) & set(clean[clean.cv == "test"].group))
print(clean.groupby(["cv", "label"]).size().unstack())
ALL.drop(columns=["ph"]).to_csv(f"{OUT}/images.csv", index=False)
clean[["path", "label", "group", "cv"]].to_csv(f"{OUT}/clean_split.csv", index=False)
for v in R: R[v][2].to_csv(f"{OUT}/near_dup_pairs_{v}.csv", index=False)
""")

md("""
## The original recipe: MobileNetV2, last 30 layers unfrozen, Adam 1e-4, 10 epochs, early stopping
Same settings as `brain_tumor_best_model.py`. Run A trains on the official Training folder (20% held out for
validation, as in the original); run B trains on the clean split.
""")

code("""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from sklearn.metrics import confusion_matrix
tf.keras.utils.set_random_seed(SEED)
IMG, BS = 224, 32
CLASSES = sorted(ALL.label.unique()); C2I = {c: i for i, c in enumerate(CLASSES)}; print(CLASSES)

def load(path, y):
    x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    x = tf.image.resize(x, (IMG, IMG)) / 255.0
    return x, tf.one_hot(y, len(CLASSES))

def ds(frame, shuffle=False):
    d = tf.data.Dataset.from_tensor_slices((frame.path.values, frame.label.map(C2I).values))
    if shuffle: d = d.shuffle(len(frame), seed=SEED)
    return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BS).prefetch(tf.data.AUTOTUNE)

def build():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
    for l in base.layers[:-30]: l.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.5)(layers.Dense(128, activation="relu")(x))
    m = Model(base.input, layers.Dense(len(CLASSES), activation="softmax")(x))
    m.compile(tf.keras.optimizers.Adam(1e-4), "categorical_crossentropy", metrics=["accuracy"])
    return m

def train(tr, va, name):
    m = build()
    cb = [tf.keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True)]
    h = m.fit(ds(tr, True), validation_data=ds(va), epochs=10, callbacks=cb, verbose=2)
    m.save(f"{OUT}/{name}.keras"); return m

def score(m, frame):
    p = m.predict(ds(frame), verbose=0); y = frame.label.map(C2I).values; yh = p.argmax(1)
    nt = C2I["notumor"]; tumor = y != nt
    return dict(n=len(frame), acc=round(100 * float((yh == y).mean()), 2),
                missed_tumors=int(((yh == nt) & tumor).sum()), tumors=int(tumor.sum()),
                false_alarms=int(((yh != nt) & ~tumor).sum()),
                cm=confusion_matrix(y, yh, labels=range(len(CLASSES))).tolist()), p
""")

code("""
# run A: each version's official split, original validation hold-out (random 20% of Training)
res = {}
for v in VERSIONS:
    df, test = R[v][0], R[v][1]
    tr_off = df[df.split == "Training"].sample(frac=1, random_state=SEED)
    cut = int(0.8 * len(tr_off)); A = train(tr_off[:cut], tr_off[cut:], f"mobilenet_official_{v}")
    res[f"A_{v}_official_test"], _ = score(A, test)
    if test.leaked.any(): res[f"A_{v}_official_test_leaked"], _ = score(A, test[test.leaked])
    res[f"A_{v}_official_test_clean"], _ = score(A, test[~test.leaked])
    # accuracy by direct-twin tier, overall and per class
    yh = pA.argmax(1) if (pA := A.predict(ds(test), verbose=0)) is not None else None
    t2 = test.assign(ok=(yh == test.label.map(C2I).values))
    res[f"A_{v}_by_tier"] = (100 * t2.groupby("tier", observed=False).ok.mean()).round(2).to_dict()
    res[f"A_{v}_by_tier_n"] = t2.groupby("tier", observed=False).size().to_dict()
    res[f"A_{v}_by_class_and_tier"] = (100 * t2.pivot_table(index="label", columns="tier", values="ok", aggfunc="mean", observed=False)).round(1).to_dict(orient="index")
    res[f"A_{v}_by_class_leaked_vs_not"] = (100 * t2.pivot_table(index="label", columns="leaked", values="ok", aggfunc="mean")).round(1).rename(columns={True: "has_twin_group", False: "no_twin_group"}).to_dict(orient="index")
    print(v, json.dumps({k: res[k] for k in res if k.startswith(f"A_{v}_by")}, indent=1, default=str))
    tf.keras.backend.clear_session()
""")

code("""
# run B: clean split
B = train(clean[clean.cv == "train"], clean[clean.cv == "val"], "mobilenet_clean")
res["B_clean_split_test"], pB = score(B, clean[clean.cv == "test"])
for k, v in res.items():
    if isinstance(v, dict) and "n" in v: print(f"{k:28s} n={v['n']:5d}  acc={v['acc']:6.2f}%  missed tumors {v['missed_tumors']}/{v['tumors']}  false alarms {v['false_alarms']}")
json.dump(dict(audit={v: R[v][4] for v in R}, results=res, classes=CLASSES, ham=HAM, ncc=NCC),
          open(f"{OUT}/results_01_{'_'.join(VERSIONS)}.json", "w"), indent=1, default=str)
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/01_leakage_audit.ipynb")
print("ok", len(cells))
