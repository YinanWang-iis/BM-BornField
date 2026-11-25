#!/bin/bash
# usage: bench.sh TAG   (egcnn/env.sh must be sourced, on a GPU)
tag=$1; cd /homenfsxx3/ynwang/symmetrix_test/egcnn/tests; mkdir -p out_$tag
M=/homenfsxx3/ynwang/EFMLP_method_compare/MD/models_lammps
for c in "bto2560 $M/NON_FIELD__mace_bto.json Ba_Ti_O bto" "hbn8960 $M/NON_FIELD_v2__mace_hbn.json B_N hbn"; do
  set -- $c; el=${3//_/ }
  for fx in 0 1; do
    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.bench -log out_$tag/bench_$1_fix$fx.log -screen none \
        -var data struct/$1.data -var mace $2 -var el "$el" -var model ../models/egcnn_$4.json -var fix $fx
    echo "$1 fix=$fx: $(grep '^Loop time' out_$tag/bench_$1_fix$fx.log | tail -1)  $(grep 'becfield/kk:.*calls' out_$tag/bench_$1_fix$fx.log | tail -1)"
  done
done
