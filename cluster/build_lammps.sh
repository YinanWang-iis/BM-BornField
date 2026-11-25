#!/bin/bash
# LAMMPS (stable) + pair_symmetrix (MACE / MACE-Field, from field/symmetrix) + fix becfield/kk (EGCNN), Kokkos-CUDA.
#   ARCH=sm120 (Blackwell) or ARCH=sm80 (A100/3090/4090/6000 Ada).  Compile on a CPU node:
#   ARCH=sm120 srun -p ix14-23 -w ix19 -c 16 --mem=32G -t 3:00:00 bash build_lammps.sh > build_lammps_sm120.log 2>&1
set -eo pipefail
ARCH=${ARCH:?set ARCH=sm120 or sm80}
SW=/homenfsxx3/ynwang/symmetrix_test/software
F=/homenfsxx3/ynwang/symmetrix_test/field
E=/homenfsxx3/ynwang/symmetrix_test/egcnn
source $SW/env.sh
# -march=native is hard-coded by pair_symmetrix/libsymmetrix CMake: the wrapper rewrites it to x86-64-v3 (portable)
export CC=$F/tools/portable-cc/x86_64-conda-linux-gnu-gcc
export CXX=$F/tools/portable-cc/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
case $ARCH in
  sm120) CUARCH=120; KARCH=Kokkos_ARCH_BLACKWELL120 ;;
  sm80)  CUARCH=80;  KARCH=Kokkos_ARCH_AMPERE80 ;;
  *) echo "bad ARCH"; exit 1 ;;
esac
echo "host: $(hostname) ARCH=$ARCH"; nvcc --version | tail -1
bash $F/symmetrix/pair_symmetrix/install.sh $E/lammps-src
bash $E/lammps/install.sh $E/lammps-src
cd $E/lammps-src
B=build_$ARCH
[ "${CLEAN:-0}" = 1 ] && rm -rf $B
cmake -S cmake -B $B -G Ninja \
  -D CMAKE_INSTALL_PREFIX=$E/lammps_$ARCH \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER=$CC \
  -D CMAKE_CXX_COMPILER=$CXX \
  -D CMAKE_CUDA_COMPILER=$CUDACXX \
  -D CMAKE_CUDA_HOST_COMPILER=$CXX \
  -D CMAKE_CUDA_ARCHITECTURES=$CUARCH \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_MPI=OFF \
  -D BUILD_OMP=ON \
  -D WITH_ZLIB=OFF \
  -D PKG_KOKKOS=ON \
  -D PKG_MANYBODY=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D $KARCH=ON \
  -D Kokkos_ARCH_NATIVE=OFF \
  -D SPHERICART_ARCH_NATIVE=OFF \
  -D SYMMETRIX_KOKKOS=ON \
  -D SYMMETRIX_SPHERICART_CUDA=ON \
  -D KokkosKernels_ENABLE_TPL_CUBLAS=ON \
  -D KokkosKernels_ENABLE_TPL_CUSPARSE=OFF \
  -D KokkosKernels_ENABLE_TPL_CUSOLVER=OFF \
  -D CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
  -D CMAKE_INSTALL_RPATH=$E/lammps_$ARCH/lib64:$E/lammps_$ARCH/lib:$SW/env/lib
cmake --build $B -j ${SLURM_CPUS_PER_TASK:-16}
cmake --install $B
echo BUILD_DONE
