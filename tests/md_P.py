# Short NVE MD with a time-dependent field: LAMMPS fix becfield/kk P(t) + per-step dump for the Python replay,
# plus restart / p0 continuation checks.   usage: python md_P.py TAG struct [nsteps]
import os, sys, numpy as np
from lammps import lammps
E = "/homenfsxx3/ynwang/symmetrix_test/egcnn"
MACE = {"bto": "/homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD__mace_bto.json",
        "hbn": "/homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD_v2__mace_hbn.json"}
EL = {"bto": "Ba Ti O", "hbn": "B N"}
tag, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
h = n // 2
s = str(np.load(f"struct/{name}.npz")["system"])
o = f"out_{tag}/md_{name}"
os.makedirs(o, exist_ok=True)
KK = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half", "-screen", "none"]
HEAD = f"""
units metal
atom_style atomic
atom_modify map yes
boundary p p p
"""
POT = f"""
pair_style symmetrix/mace/foundation/float32
pair_coeff * * {MACE[s]} {EL[s]}
variable ex equal 0.01*sin(0.05*step)
variable ey equal -0.005
variable ez equal 0.0006*step-0.03
timestep 0.001
"""
FIX = f"fix ef all becfield {E}/models/egcnn_{s}.json {EL[s]} efield v_ex v_ey v_ez"
TH = "thermo_style custom step temp pe f_ef[1] f_ef[2] f_ef[3] f_ef[4] f_ef[5] f_ef[6]\nthermo_modify format float %.10g\nthermo 1\n"

def P(lmp): return np.array([lmp.numpy.extract_fix("ef", 0, 1, k) for k in range(3)])

# A: continuous run, dump every step
lmp = lammps(cmdargs=KK + ["-log", f"{o}/A.log"])
lmp.commands_string(HEAD + f"read_data {E}/tests/struct/{name}.data\n" + POT + FIX + f"""
velocity all create 600 4928459 mom yes rot no
fix 1 all nve
dump d all custom 1 {o}/traj.dump id x y z
dump_modify d sort id format float %.12f
{TH}
run {h}
write_restart {o}/half.restart
write_data {o}/half.data
run {n - h}
""")
PA = P(lmp); lmp.close()
# B: read_restart (P restored from the restart file)
lmp = lammps(cmdargs=KK + ["-log", f"{o}/B.log"])
lmp.commands_string(HEAD + f"read_restart {o}/half.restart\n" + POT + FIX + f"\nfix 1 all nve\n{TH}\nrun {n - h}\n")
PB = P(lmp); lmp.close()
# C: write_data + p0 (what a checkpoint-based driver does)
rows = [l.split() for l in open(f"{o}/A.log") if l.split()[:1] == [str(h)] and len(l.split()) == 9]
p0 = rows[0][3:6]
lmp = lammps(cmdargs=KK + ["-log", f"{o}/C.log"])
lmp.commands_string(HEAD + f"read_data {o}/half.data\nreset_timestep {h}\n" + POT + FIX + f" p0 {' '.join(p0)}\nfix 1 all nve\n{TH}\nrun {n - h}\n")
PC = P(lmp); lmp.close()
np.savez(f"{o}/P.npz", PA=PA, PB=PB, PC=PC)
print(f"{name}: P(A, continuous)       = {PA}")
print(f"{name}: P(B, read_restart)     = {PB}   diff {np.abs(PB-PA).max():.2e}")
print(f"{name}: P(C, write_data + p0)  = {PC}   diff {np.abs(PC-PA).max():.2e}")
