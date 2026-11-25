# Python replay of BECFieldForce's polarization accumulation (efmlp_lammps._callback logic) on the LAMMPS
# trajectory out_<tag>/md_<struct>/traj.dump, with Z from predict_becs (float32 e3nn). EGCNN env, CPU.
import sys, os, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
sys.path.insert(0, "/homenfsxx3/ynwang/EFMLP_method_compare/MD/common")
from efmlp_lammps import BECFieldForce, load_egcnn, ZNUM
tag, name = sys.argv[1], sys.argv[2]
M = {"bto": "BaTiO3", "hbn": "hBN_v2"}
d = np.load(f"struct/{name}.npz"); s = str(d["system"]); cell = d["cell"]
o = f"out_{tag}/md_{name}"
bf = BECFieldForce.__new__(BECFieldForce)
bf.device = torch.device("cpu"); bf.net = load_egcnn(f"/homenfsxx3/ynwang/EFMLP_method_compare/BEC_PRED/{M[s]}", "cpu")
bf.type_to_Z = np.asarray(ZNUM[s]); bf.cutoff = 4.5
types = np.array([ZNUM[s].index(z) + 1 for z in d["numbers"]])
# frames
frames, lines = {}, open(f"{o}/traj.dump").read().split("ITEM: TIMESTEP\n")[1:]
for blk in lines:
    L = blk.splitlines(); step = int(L[0]); nat = int(L[2])
    a = np.array([[float(v) for v in l.split()] for l in L[-nat:]])
    frames[step] = a[np.argsort(a[:, 0])][:, 1:]
# LAMMPS P trace (last value per step)
Pl = {}
for l in open(f"{o}/A.log"):
    p = l.split()
    if len(p) == 9 and p[0].isdigit():
        Pl[int(p[0])] = np.array([float(v) for v in p[3:6]])
P, prev, err = np.zeros(3), None, []
for step in sorted(frames):
    x = frames[step]
    becs = bf.predict_becs(x, types, cell)
    frac = np.linalg.solve(cell.T, x.T).T
    if prev is not None:
        f0, b0 = prev
        ds = frac - f0; ds -= np.round(ds)
        P += np.einsum("iab,ib->a", 0.5 * (becs + b0), ds @ cell)
    prev = (frac, becs)
    err.append(np.abs(P - Pl[step]).max())
    if step % 50 == 0:
        print(f"step {step:4d}  P_python {P.round(6)}  P_lammps {Pl[step].round(6)}  |diff| {err[-1]:.2e}", flush=True)
print(f"{name}: max |P_lammps - P_python| over {len(frames)} steps = {max(err):.2e} e*A ; |P| final {np.abs(P).max():.4f}")
