# Python module bornfield on the GPU -> out_py_<tag>/<case>_<prec>.npz ; timing; ASE MD smoke test
import os, sys, time, numpy as np
from ase import Atoms
from ase.io import read
from bornfield import BornField, PolarizationTracker
tag = sys.argv[1]
o = f"out_py_{tag}"; os.makedirs(o, exist_ok=True)
M = "/homenfsxx3/ynwang/symmetrix_test/egcnn/models/egcnn_{}.json"
R = "/homenfsxx3/ynwang/EFMLP_method_compare/"
cases = {}
for name in ("bto40_f0", "hbn36_f0", "bto2560", "hbn8960"):
    d = np.load(f"struct/{name}.npz")
    cases[name] = (str(d["system"]), Atoms(numbers=d["numbers"], positions=d["positions"], cell=d["cell"], pbc=True))
a = read(R + "data/BaTiO3/MLP_dataset/BaTiO3_all_test_bec.extxyz", 1)       # original (unrotated) triclinic frame
cases["bto40_raw"] = ("bto", Atoms(a.numbers, a.positions + 3.7 * a.cell[0], cell=a.cell, pbc=True))  # unwrapped
c = read(R + "data/BaTiO3/MLP_dataset/BaTiO3_all_test_bec.extxyz", 2) * (2, 2, 2)
c = c[np.linalg.norm(c.positions - c.positions.mean(0), axis=1) < 6.0]
cases["bto_cluster"] = ("bto", Atoms(c.numbers, c.positions, pbc=False))                     # non-periodic, no cell
s = read(R + "data/hBN/MLP_dataset/hBN_all_test_bec.extxyz", 1)
cases["hbn_slab_pbc110"] = ("hbn", Atoms(s.numbers, s.positions, cell=s.cell, pbc=[True, True, False]))
calcs = {}
for name, (sy, at) in cases.items():
    for prec in ("float32", "float64"):
        k = (sy, prec)
        if k not in calcs: calcs[k] = BornField(M.format(sy), precision=prec)
        Z = calcs[k].becs(at)
        np.savez(f"{o}/{name}_{prec}.npz", Z=Z.reshape(-1, 9), numbers=at.numbers, positions=at.positions,
                 cell=at.cell.array, pbc=at.pbc)
    print(name, len(at), "edges", calcs[(sy, "float32")].model.edges, flush=True)
# rotation equivariance: Z(R r) = R Z R^T
rng = np.random.default_rng(0)
q, _ = np.linalg.qr(rng.normal(size=(3, 3)));  q *= np.sign(np.linalg.det(q))
sy, at = cases["bto40_raw"]
b = at.copy(); b.positions = at.positions @ q.T; b.cell = at.cell.array @ q.T
for prec in ("float32", "float64"):
    Z0 = calcs[(sy, prec)].becs(at); Z1 = calcs[(sy, prec)].becs(b)
    print(f"rotation {prec}: max |Z(Rr) - R Z R^T| = {np.abs(Z1 - np.einsum('ab,ibc,dc->iad', q, Z0, q)).max():.2e} e")
# timing
for name in ("bto2560", "hbn8960"):
    sy, at = cases[name]; bf = calcs[(sy, "float32")]
    for _ in range(3): bf.becs(at)
    t = time.perf_counter(); n = 20
    for _ in range(n): bf.becs(at)
    print(f"timing {name}: {(time.perf_counter() - t) / n * 1e3:.2f} ms per call (incl. host neighbour binning, copies)")
# ASE MD: zero-field MACE (symmetrix) + BornField, PolarizationTracker
from ase.calculators.mixing import SumCalculator
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from symmetrix.experimental import MACEFoundation
sy, at = cases["bto40_raw"]; at = at * (2, 2, 2)
bf = BornField(M.format("bto"), electric_field=(0, 0, 0.03))
at.calc = SumCalculator([MACEFoundation(R + "MD/models_lammps/NON_FIELD__mace_bto.json", dtype="float32"), bf])
MaxwellBoltzmannDistribution(at, temperature_K=300, rng=np.random.default_rng(1))
tr = PolarizationTracker()
dyn = VelocityVerlet(at, 1.0 * units.fs)
at.get_forces(); tr.update(at)
dyn.attach(lambda: tr.update(at), interval=1)
dyn.run(50)
F = at.get_forces(); Ff = bf.results["forces"]
print(f"ASE MD 320 atoms 50 steps ok: P = {tr.P.round(5)} e*A, |sum F_field| = {np.abs(Ff.sum(0)).max():.1e} eV/A")
