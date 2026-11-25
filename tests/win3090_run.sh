#!/bin/bash
# sm80 checks on win3090 (RTX 3090, WSL)
cd /homenfsxx3/ynwang/symmetrix_test/egcnn/tests
source ../env.sh
echo "arch=$SYMMETRIX_ARCH lmp=$SYMMETRIX_LMP"
PAIR=zero python sp_bec.py final_sm80_zero bto40_f0 hbn36_f0 bto2560 hbn8960 2>&1 | grep -c ok
python md_P.py final_sm80 bto320 200 2>&1 | grep "P("
bash bench.sh final_sm80 2>&1 | grep fix=
