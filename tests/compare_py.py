# out_py_<tag> vs float64 reference (tools/egcnn_ref.py) and vs LAMMPS fix zfile results (out_final_sm120_zero)
import sys, glob, os, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/homenfsxx3/ynwang/symmetrix_test/egcnn/tools")
from egcnn_ref import EGCNNRef
from ase import Atoms
from ase.neighborlist import neighbor_list
tag = sys.argv[1]
refs = {s: EGCNNRef(f"/homenfsxx3/ynwang/symmetrix_test/egcnn/models/egcnn_{s}.json") for s in ("bto", "hbn")}
print(f"{'case':16s} {'N':>5s} | float32: max|Z-Z64|  vs LAMMPS fix | float64: max|Z-Z64|  vs LAMMPS fix   (e)")
for name in ("bto40_f0", "hbn36_f0", "bto2560", "hbn8960", "bto40_raw", "bto_cluster", "hbn_slab_pbc110"):
    r = {p: np.load(f"out_py_{tag}/{name}_{p}.npz") for p in ("float32", "float64")}
    d = r["float64"]; s = "hbn" if 5 in d["numbers"] else "bto"
    at = Atoms(numbers=d["numbers"], positions=d["positions"], cell=d["cell"], pbc=d["pbc"])
    i, j, D = neighbor_list("ijD", at, 4.5)
    ref = refs[s]
    Z64 = ref.forward(torch.as_tensor(at.numbers), torch.as_tensor(i), torch.as_tensor(j), ref.t(D)).numpy()
    line = f"{name:16s} {len(at):5d} |"
    for p in ("float32", "float64"):
        lf = f"out_final_sm120_zero/{name}_{p}.npz"
        lam = f"{np.abs(r[p]['Z'] - np.load(lf)['Z']).max():9.2e}" if os.path.exists(lf) else "      -  "
        line += f"   {np.abs(r[p]['Z'] - Z64).max():9.2e}     {lam}   |"
    print(line, flush=True)
