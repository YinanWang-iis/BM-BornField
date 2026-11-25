# Single points through LAMMPS (pair symmetrix MACE zero field + fix becfield/kk) -> out_<tag>/<struct>_<prec>.npz
#   Z (tag order, from zfile), F for fields FIELDS, P vector.   usage: python sp_bec.py TAG struct1 struct2 ...
import os, sys, numpy as np
from lammps import lammps
E = "/homenfsxx3/ynwang/symmetrix_test/egcnn"
MACE = {"bto": "/homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD__mace_bto.json",
        "hbn": "/homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD_v2__mace_hbn.json"}
EL = {"bto": "Ba Ti O", "hbn": "B N"}
FIELDS = [(0, 0, 0), (0, 0, 0.03), (0.02, -0.01, 0.03)]
tag = sys.argv[1]
ZERO = os.environ.get("PAIR") == "zero"   # pair_style zero: forces = field force only
os.makedirs(f"out_{tag}", exist_ok=True)
for name in sys.argv[2:]:
    s = str(np.load(f"struct/{name}.npz")["system"])
    for prec in ("float32", "float64"):
        zf = f"out_{tag}/{name}_{prec}.z"
        if os.path.exists(zf): os.remove(zf)
        lmp = lammps(cmdargs=["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half",
                              "-log", f"out_{tag}/{name}_{prec}.log", "-screen", "none"])
        lmp.commands_string(f"""
units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data {E}/tests/struct/{name}.data
{"pair_style zero 6.0" if ZERO else "pair_style symmetrix/mace/foundation/float32"}
{"pair_coeff * *" if ZERO else "pair_coeff * * " + MACE[s] + " " + EL[s]}
variable ex equal 0
variable ey equal 0
variable ez equal 0
fix ef all becfield {E}/models/egcnn_{s}.json {EL[s]} efield v_ex v_ey v_ez precision {prec} zfile {zf} 1 {os.environ.get("EXTRA", "")}
""")
        Fs, Ps = [], []
        for fld in FIELDS:
            for k, v in zip("xyz", fld):
                lmp.command(f"variable e{k} equal {v}")
            lmp.command("run 0 post no")
            ids = np.asarray(lmp.numpy.extract_atom("id")[:lmp.get_natoms()])
            f = np.asarray(lmp.numpy.extract_atom("f")[:lmp.get_natoms()], float)
            Fs.append(f[np.argsort(ids)].copy())
            Ps.append([lmp.numpy.extract_fix("ef", 0, 1, k) for k in range(6)])
        lmp.close()
        blocks = open(zf).read().split("# step")[1:]
        Z = np.array([[float(v) for v in l.split()[1:]] for l in blocks[-1].strip().splitlines()[1:]])
        np.savez(f"out_{tag}/{name}_{prec}.npz", Z=Z, F=np.array(Fs), P=np.array(Ps), fields=np.array(FIELDS))
        print(name, prec, "ok", Z.shape, flush=True)
