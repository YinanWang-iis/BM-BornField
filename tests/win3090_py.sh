#!/bin/bash
cd /homenfsxx3/ynwang/symmetrix_test/egcnn/tests
source ../env.sh
python py_bornfield.py sm80 2>&1 | grep -E "rotation|timing|ASE MD|edges"
PAIR=zero python sp_bec.py refac_sm80_zero bto40_f0 hbn36_f0 2>&1 | grep -c ok
