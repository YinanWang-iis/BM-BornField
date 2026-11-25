# minimize with fix becfield present (field on): must run, add no field force, leave P untouched; then MD continues
import numpy as np
from lammps import lammps
lmp = lammps(cmdargs=["-k","on","g","1","-sf","kk","-pk","kokkos","newton","on","neigh","half","-screen","none","-log","out_min.log"])
lmp.commands_string("""units metal
atom_style atomic
atom_modify map yes
read_data struct/bto320.data
pair_style symmetrix/mace/foundation/float32
pair_coeff * * /homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps/NON_FIELD__mace_bto.json Ba Ti O
variable ez equal 0.03
fix ef all becfield ../models/egcnn_bto.json Ba Ti O efield 0 0 v_ez
thermo_style custom step pe fnorm f_ef[1] f_ef[2] f_ef[3]
velocity all create 300 1 mom yes rot no
fix 1 all nve
run 20
unfix 1
min_style cg
minimize 0.0 1.0e-3 200 10000
fix 1 all nve
run 20
""")
print("minimize + MD ok, P =", [lmp.numpy.extract_fix("ef", 0, 1, k) for k in range(3)])
lmp.close(); lmp.finalize()
