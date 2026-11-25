# Compare out_<tag>/*.npz (LAMMPS fix becfield/kk) with the Python references. Runs in the EGCNN env (CPU):
#   Zpy  = efmlp_lammps.BECFieldForce.predict_becs  (e3nn model.py, float32, ASR)   <- the production reference
#   Z64  = tools/egcnn_ref.py on the exported JSON, float64 (exact export of the same network)
import sys, glob, os, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
sys.path.insert(0, "/homenfsxx3/ynwang/EFMLP_method_compare/MD/common")
sys.path.insert(0, "/homenfsxx3/ynwang/symmetrix_test/egcnn/tools")
from efmlp_lammps import BECFieldForce, load_egcnn, ZNUM
from egcnn_ref import EGCNNRef
tag = sys.argv[1]
ZERO = tag.endswith("_zero")
names = sys.argv[2:]
M = {"bto": "BaTiO3", "hbn": "hBN_v2"}
nets, refs = {}, {}
print(f"{'struct':10s} {'prec':8s} {'N':>5s} | max|Z-Zpy32| max|Z-Z64|  (e) | field   max|dF-eZ64| max|dF| (meV/A) | |sum dF| (meV/A)")
for name in names:
    d = np.load(f"struct/{name}.npz"); s = str(d["system"])
    if s not in nets:
        bf = BECFieldForce.__new__(BECFieldForce)
        bf.device = torch.device("cpu"); bf.net = load_egcnn(f"/homenfsxx3/ynwang/EFMLP_method_compare/BEC_PRED/{M[s]}", "cpu")
        bf.type_to_Z = np.asarray(ZNUM[s]); bf.cutoff = 4.5
        nets[s] = bf; refs[s] = EGCNNRef(f"/homenfsxx3/ynwang/symmetrix_test/egcnn/models/egcnn_{s}.json")
    types = np.array([ZNUM[s].index(z) + 1 for z in d["numbers"]])
    Zpy = nets[s].predict_becs(d["positions"], types, d["cell"]).reshape(-1, 9)
    Z64 = refs[s].becs(d["numbers"], d["positions"], d["cell"]).reshape(-1, 9)
    for prec in ("float32", "float64"):
        p = f"out_{tag}/{name}_{prec}.npz"
        if not os.path.exists(p): continue
        r = np.load(p)
        line = f"{name:10s} {prec:8s} {len(Z64):5d} | {np.abs(r['Z']-Zpy).max():10.2e} {np.abs(r['Z']-Z64).max():10.2e}      |"
        for k in (1, 2):
            eps = r["fields"][k]
            dF = r["F"][k] - (0 if ZERO else r["F"][0])
            ref = np.einsum("a,iab->ib", eps, Z64.reshape(-1, 3, 3))
            line += f" f{k} {np.abs(dF-ref).max()*1e3:9.2e} {np.abs(ref).max()*1e3:7.1f}  {np.abs(dF.sum(0)).max()*1e3:8.1e} |"
        print(line, flush=True)
    print(f"{'':10s} {'py32-64':8s} {'':5s} | {np.abs(Zpy-Z64).max():10.2e}   (float32 e3nn vs float64 reference)", flush=True)
