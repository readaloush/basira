"""Builds notebooks/10_pack_for_web.ipynb: split the fp16-weight ONNX model into 4 parts and write each, and the example images,
as base64 text, so that the files can be moved to the GitHub repository through a browser."""
import nbformat as nbf
code = r'''
import glob, os, base64, json, hashlib
OUT = "/kaggle/working"
src = [p for p in glob.glob("/kaggle/input/**/basira_fp16.onnx", recursive=True)][0]
b = open(src, "rb").read(); n = 4; k = -(-len(b) // n)
meta = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), parts=[])
for i in range(n):
    part = b[i * k:(i + 1) * k]
    open(f"{OUT}/part{i+1}.txt", "w").write(base64.b64encode(part).decode())
    meta["parts"].append(dict(name=f"basira_fp16.part{i+1}", bytes=len(part), sha256=hashlib.sha256(part).hexdigest()))
ex_dir = os.path.dirname([p for p in glob.glob("/kaggle/input/**/examples/examples.json", recursive=True)][0])
files = {f: base64.b64encode(open(f"{ex_dir}/{f}", "rb").read()).decode() for f in sorted(os.listdir(ex_dir))}
open(f"{OUT}/examples.txt", "w").write(json.dumps(files))
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1); print(json.dumps(meta, indent=1)); print(list(files))
'''
nb = nbf.v4.new_notebook(); nb.cells = [nbf.v4.new_markdown_cell("# 10 · Pack the demo model and examples as base64 text"), nbf.v4.new_code_cell(code.strip())]
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/10_pack_for_web.ipynb"); print("ok")
