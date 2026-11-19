"""Builds notebooks/09_export_onnx.ipynb: export the demo model (EfficientNetV2-S, KBF seed 7) to ONNX with a class-activation map
output, quantize, and check that accuracy on the Kaggle and BRISC tests is unchanged. Inputs: Kaggle v2, BRISC, notebook 07 output."""
import nbformat as nbf
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md("""# 09 · Export the demo model to ONNX (browser inference)
EfficientNetV2-S trained on Kaggle v2 + BRISC + Feltrin (seed 7, notebook 07). The exported graph takes a 384x384 RGB image
in [0, 1], normalises it, and returns class probabilities and a class-activation map (classifier weights applied to the last
feature map, so no gradients are needed in the browser). fp32 and int8 versions are scored on both tests.""")
code("""
!pip -q install onnx onnxruntime onnxconverter-common
import os, glob, json, time, numpy as np, pandas as pd, torch, timm, onnx, onnxruntime as ort
from PIL import Image, ImageFilter
OUT = "/kaggle/working"; CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]; C2I = {c: i for i, c in enumerate(CLASSES)}
NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
pt = [p for p in glob.glob("/kaggle/input/**/*.pt", recursive=True) if p.endswith("KBF_s7.pt")][0]; print(pt)
m = timm.create_model(NAME, pretrained=False, num_classes=4); m.load_state_dict(torch.load(pt, map_location="cpu")); m.eval()
class Demo(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):                      # x: (1, 3, 384, 384), values in [0, 1]
        f = self.m.forward_features((x - self.mean) / self.std)
        probs = self.m.forward_head(f).softmax(1)
        W, b = self.m.classifier.weight, self.m.classifier.bias
        cam = torch.nn.functional.conv2d(f, W[:, :, None, None], b)     # classifier applied at every location
        return probs, cam
D = Demo(m).eval(); x = torch.rand(1, 3, 384, 384)
with torch.no_grad(): p, c = D(x)
print(p.shape, c.shape)
torch.onnx.export(D, x, f"{OUT}/basira_fp32.onnx", input_names=["image"], output_names=["probs", "cam"], opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(f"{OUT}/basira_fp32.onnx", f"{OUT}/basira_int8.onnx", weight_type=QuantType.QUInt8, per_channel=True)
# fp16 weights, fp32 compute: large initializers stored as float16 and cast back to float32 when the model loads
from onnx import numpy_helper, helper, TensorProto
g = onnx.load(f"{OUT}/basira_fp32.onnx"); casts, keep = [], []
for t in g.graph.initializer:
    a = numpy_helper.to_array(t)
    if a.dtype == np.float32 and a.size >= 1024:
        keep.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_h"))
        casts.append(helper.make_node("Cast", [t.name + "_h"], [t.name], to=TensorProto.FLOAT))
    else: keep.append(t)
del g.graph.initializer[:]; g.graph.initializer.extend(keep)
nodes = list(g.graph.node); del g.graph.node[:]; g.graph.node.extend(casts + nodes); onnx.save(g, f"{OUT}/basira_fp16.onnx")
for k in ["fp32", "int8", "fp16"]: print(k, round(os.path.getsize(f"{OUT}/basira_{k}.onnx") / 1e6, 1), "MB")
""")
code("""
def crop_brain(img, thr=12, pad=6):
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(2))); ys, xs = np.nonzero(g > thr)
    if len(xs) < 100: return img
    return img.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, g.shape[1]), min(ys.max() + pad, g.shape[0])))
def prep(p, size=384):
    im = crop_brain(Image.open(p).convert("RGB")); w, h = im.size; s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    c = Image.new("RGB", (size, size)); c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return (np.asarray(c, np.float32) / 255).transpose(2, 0, 1)[None]
def find_dir(keys, name):
    m_ = [d for d in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if any(k in d for k in keys) and os.path.isdir(d)]
    return sorted(m_, key=len)[0]
KT = find_dir(["masoudnickparvar", "brain-tumor-mri-dataset"], "Testing"); BR = find_dir(["brisc"], "classification_task") + "/test"
MAP = {"no_tumor": "notumor"}
rows = [dict(path=f"{KT}/{c}/{f}", ds="kaggle", label=c) for c in sorted(os.listdir(KT)) for f in sorted(os.listdir(f"{KT}/{c}"))]
rows += [dict(path=f"{BR}/{c}/{f}", ds="brisc", label=MAP.get(c, c)) for c in sorted(os.listdir(BR)) for f in sorted(os.listdir(f"{BR}/{c}"))]
T = pd.DataFrame(rows); y = T.label.map(C2I).values; print(T.groupby(["ds", "label"]).size().unstack())
X = np.concatenate([prep(p) for p in T.path]); print(X.shape)
with torch.no_grad():
    PT = np.concatenate([D(torch.from_numpy(X[i:i + 32]))[0].numpy() for i in range(0, len(X), 32)])
def decide(p, t=0.9):
    yh = p.argmax(1).copy(); tum = p.copy(); tum[:, 2] = -1; low = (yh == 2) & (p[:, 2] < t); yh[low] = tum[low].argmax(1); return yh
def score(p, tag):
    r = {}
    for ds in ["kaggle", "brisc"]:
        k = (T.ds == ds).values; yh = decide(p)[k]; yy = y[k]
        r[ds] = dict(acc=round(100 * (yh == yy).mean(), 2), missed=int(((yh == 2) & (yy != 2)).sum()), false_alarms=int(((yh != 2) & (yy == 2)).sum()))
    print(tag, r); return r
R = {"torch": score(PT, "torch")}
for k in ["fp32", "int8", "fp16"]:
    s = ort.InferenceSession(f"{OUT}/basira_{k}.onnx", providers=["CPUExecutionProvider"]); t0 = time.time()
    P = np.concatenate([s.run(["probs"], {"image": X[i:i + 1]})[0] for i in range(len(X))])
    R[k] = score(P, k); R[k]["ms_per_image_cpu"] = round(1000 * (time.time() - t0) / len(X), 1)
    R[k]["max_abs_prob_diff_vs_torch"] = float(np.abs(P - PT).max()); R[k]["size_mb"] = round(os.path.getsize(f"{OUT}/basira_{k}.onnx") / 1e6, 1)
    R[k]["argmax_agreement_with_torch"] = round(100 * float((P.argmax(1) == PT.argmax(1)).mean()), 2)
json.dump(R, open(f"{OUT}/export_check.json", "w"), indent=1); print(json.dumps(R, indent=1))
""")
code("""
# demo examples: two correctly classified Kaggle test images per class, plus the probabilities and CAM the page should reproduce
import shutil
os.makedirs(f"{OUT}/examples", exist_ok=True); ex = []
yh = decide(PT)
for c in CLASSES:
    idx = [i for i in np.nonzero((T.ds == "kaggle").values & (y == C2I[c]) & (yh == y))[0]][:40]
    for i in np.random.default_rng(1).choice(idx, 2, replace=False):
        name = f"{c}_{os.path.basename(T.path[i])}"; shutil.copy(T.path[i], f"{OUT}/examples/{name}")
        ex.append(dict(file=name, label=c, probs=[round(float(v), 4) for v in PT[i]]))
json.dump(ex, open(f"{OUT}/examples/examples.json", "w"), indent=1); print(ex)
""")
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/09_export_onnx.ipynb"); print("ok")
