#!/bin/bash
# pair_style symmetrix MACE-Field single points with the egcnn LAMMPS build (same set as field/tests/archcheck/sp.sh)
#   -> sp_pair_<tag>/ ; compare with: python compare_pair.py <tag>
LMP=/homenfsxx3/ynwang/symmetrix_test/egcnn/bin/lmp
J=/homenfsxx3/ynwang/symmetrix_test/field/models/macefield_bto_ffpmlin-8-22-56.json
D=/homenfsxx3/ynwang/symmetrix_test/field/tests/lammps
tag=${1:-sm120}
cd /homenfsxx3/ynwang/symmetrix_test/egcnn/tests; mkdir -p sp_pair_$tag
KK="-k on g 1 -sf kk -pk kokkos newton on neigh half"
for s in bto40 bto320; do
 for p in "symmetrix/mace/foundation f64" "symmetrix/mace/foundation/float32 f32" "symmetrix/mace/foundation/float32-mixed mix"; do
  set -- $p; ps=$1; t=$2; fi=0
  for f in "0 0 0" "0 0 0.03" "0 0 -0.03" "0.02 -0.01 0.03"; do
   set -- $f
   $LMP $KK -in $D/in.sp -log sp_pair_$tag/${s}_${t}_f${fi}.log -screen none -var data $D/$s.data -var ps $ps -var json $J \
        -var ex $1 -var ey $2 -var ez $3 -var out sp_pair_$tag/${s}_${t}_f${fi}
   fi=$((fi+1))
  done
 done
done
echo pair_same done
