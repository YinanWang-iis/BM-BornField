#!/bin/bash
# Python module bornfield (EGCNN Born effective charges on the GPU + ASE calculator) -> egcnn/python_<ARCH>/bornfield
#   ARCH=sm120 srun -p ix14-23 -w ix19 -c 8 --mem=16G -t 1:00:00 bash build_python.sh > build_python_sm120.log 2>&1
set -eo pipefail
ARCH=${ARCH:?set ARCH=sm120 or sm80}
SW=/homenfsxx3/ynwang/symmetrix_test/software
F=/homenfsxx3/ynwang/symmetrix_test/field
E=/homenfsxx3/ynwang/symmetrix_test/egcnn
source $SW/env.sh
case $ARCH in sm120) CUARCH=120 ;; sm80) CUARCH=80 ;; *) echo bad ARCH; exit 1 ;; esac
B=$E/python/build_$ARCH
rm -rf $B
cmake -S $E/python -B $B -G Ninja -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=$F/tools/portable-cc/x86_64-conda-linux-gnu-g++ \
  -D CMAKE_CUDA_COMPILER=$CUDACXX -D CMAKE_CUDA_HOST_COMPILER=$F/tools/portable-cc/x86_64-conda-linux-gnu-g++ \
  -D CMAKE_CUDA_ARCHITECTURES=$CUARCH -D Python_EXECUTABLE=$SW/env/bin/python \
  -D pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -D SYMMETRIX_SOURCE=$F/symmetrix/libsymmetrix/source \
  -D CMAKE_SHARED_LINKER_FLAGS=-Wl,--allow-shlib-undefined -D CMAKE_INSTALL_RPATH=$SW/env/lib \
  -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build $B -j ${SLURM_CPUS_PER_TASK:-8}
D=$E/python_$ARCH/bornfield
mkdir -p $D
cp $E/python/bornfield/*.py $D/
cp $B/_bornfield*.so $D/
echo BUILD_DONE
