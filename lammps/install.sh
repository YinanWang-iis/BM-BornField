#!/bin/bash
# Link fix becfield (EGCNN BEC field force, Kokkos/CUDA) into a LAMMPS source tree that already has pair_symmetrix:
#   bash install.sh path/to/lammps
set -e
lammps=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
for f in fix_becfield.h fix_becfield.cpp; do ln -sfn $here/$f $lammps/src/$f; done
for f in fix_becfield_kokkos.h fix_becfield_kokkos.cpp egcnn_generated.hpp; do ln -sfn $here/$f $lammps/src/KOKKOS/$f; done
echo "fix becfield linked into $lammps"
