# Source this file:  source /homenfsxx3/ynwang/symmetrix_test/egcnn/env.sh   (on a GPU node)
# Same as field/env.sh (python venv per GPU architecture), but LAMMPS = egcnn/lammps_<arch>, which has
#   pair_style symmetrix/mace[/foundation]...  (identical to field/lammps*)  +  fix becfield/kk (EGCNN BEC field force)
# Also exports the LAMMPS python module (lammps-src/python) + liblammps for driving LAMMPS from Python,
# and the bornfield python module (EGCNN Born charges on the GPU, ASE calculator) for this architecture.
# SYMX_ARCH=sm120|sm80 before sourcing forces an architecture (e.g. on a node without a visible GPU).
source /homenfsxx3/ynwang/symmetrix_test/field/env.sh
if [ -n "${SYMMETRIX_ARCH:-}" ]; then
  _E=/homenfsxx3/ynwang/symmetrix_test/egcnn
  export SYMMETRIX_LMP=$_E/lammps_$SYMMETRIX_ARCH/bin/lmp
  export PATH=$_E/lammps_$SYMMETRIX_ARCH/bin:$PATH
  export LD_LIBRARY_PATH=$_E/lammps_$SYMMETRIX_ARCH/lib64:$_E/lammps_$SYMMETRIX_ARCH/lib:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=$_E/python_$SYMMETRIX_ARCH:$_E/lammps-src/python${PYTHONPATH:+:$PYTHONPATH}   # bornfield + lammps modules
  export EGCNN_MODELS=$_E/models
  echo "[egcnn/env.sh] LAMMPS with fix becfield/kk: $SYMMETRIX_LMP" >&2
  unset _E
fi
