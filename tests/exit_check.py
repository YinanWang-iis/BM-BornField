import sys
from lammps import lammps
mode = sys.argv[1]
lmp = lammps(cmdargs=["-k","on","g","1","-sf","kk","-pk","kokkos","newton","on","neigh","half","-screen","none","-log","none"])
lmp.commands_string("""units metal
atom_style atomic
atom_modify map yes
read_data struct/bto40_f0.data
pair_style symmetrix/mace/foundation/float32
pair_coeff * * /homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD__mace_bto.json Ba Ti O
""" + ("fix ef all becfield ../models/egcnn_bto.json Ba Ti O efield 0 0 0.03\n" if "fix" in mode else "") + "run 0")
lmp.close()
if "fin" in mode: lmp.finalize()
print("done", mode, flush=True)
