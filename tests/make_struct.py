# test structures -> struct/*.data (+ .npz with numbers/positions/cell in tag order). Runs in any env with ase.
import numpy as np
from ase.io import read, write
R = "/homenfsxx3/ynwang/EFMLP_method_compare/"
spec = {"bto": ["Ba", "Ti", "O"], "hbn": ["B", "N"]}
out = {}
for k in (0, 1, 2):
    out[f"bto40_f{k}"] = ("bto", read(R + "data/BaTiO3/MLP_dataset/BaTiO3_all_test_bec.extxyz", k))
    out[f"hbn36_f{k}"] = ("hbn", read(R + "data/hBN/MLP_dataset/hBN_all_test_bec.extxyz", k))
a = read(R + "data/BaTiO3/MLP_dataset/BaTiO3_all_test_bec.extxyz", 0) * (4, 4, 4)
a.rattle(0.03, seed=1)
out["bto2560"] = ("bto", a)
b = read(R + "data/BaTiO3/MLP_dataset/BaTiO3_all_test_bec.extxyz", 0) * (2, 2, 2); b.rattle(0.03, seed=2)
out["bto320"] = ("bto", b)
b = read(R + "data/hBN/MLP_dataset/hBN_all_test_bec.extxyz", 0) * (3, 3, 1); b.rattle(0.03, seed=3)
out["hbn324"] = ("hbn", b)
a = read(R + "MD/hBN/H3_domain_wall/m3/relaxed.data", format="lammps-data", atom_style="atomic",
         Z_of_type={1: 5, 2: 7})
out["hbn8960"] = ("hbn", a)
for name, (sys_, a) in out.items():
    a.wrap()
    write(f"struct/{name}.data", a, format="lammps-data", specorder=spec[sys_], masses=True)
    zt = {i + 1: z for i, z in enumerate(__import__("ase.data", fromlist=["x"]).atomic_numbers[e] for e in spec[sys_])}
    b = read(f"struct/{name}.data", format="lammps-data", atom_style="atomic", Z_of_type=zt)   # LAMMPS frame (rotated)
    np.savez(f"struct/{name}.npz", numbers=b.numbers, positions=b.positions, cell=b.cell.array, system=sys_)
    print(name, len(a), a.cell.cellpar().round(3))
