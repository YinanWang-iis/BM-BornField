/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   This software is distributed under the GNU General Public License.

   fix becfield/kk: EGCNN Born effective charges (EFMLP method 2) on the GPU.  See fix_becfield_kokkos.h.

   Network (BEC_PRED/<system>/model.py, EquivarBECModel, exported by tools/export_egcnn.py):
     node features x (N x 768) = 64x0e + 64x1o + 64x1e + 64x2e, e3nn layout [block][channel u][m]
     edges src -> dst with vector v = r_dst - r_src (here: receiver i, neighbour j, v = x_i - x_j)
     per layer: w_e = cut(r) * (W2 silu(W1 rbf(r) + b1) + b2)   (17 paths x 64 channels)
                agg_i = sum_e TP_uvu(x_src, Y(v/|v|), w_e);  x += agg L
     Z* = x Mout (output linear + CartesianTensor), minus the mean over all atoms.
   Kernels: one thread block per receiver atom (64 channels x G edge groups), radial MLP via cuBLAS GEMMs.
------------------------------------------------------------------------- */

#ifdef SYMMETRIX_CUDA

#include "fix_becfield_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "variable.h"

#include "egcnn_generated.hpp"
#include "nlohmann_json.hpp"

#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

using namespace LAMMPS_NS;
using namespace FixConst;
using AT = ArrayTypes<LMPDeviceType>;

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess)                                                                       \
      throw std::runtime_error(std::string("fix becfield/kk CUDA error: ") +                     \
                               cudaGetErrorString(e_) + " at " + __FILE__ + ":" +                \
                               std::to_string(__LINE__));                                        \
  } while (0)
#define CB(x)                                                                                    \
  do {                                                                                           \
    cublasStatus_t s_ = (x);                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                             \
      throw std::runtime_error("fix becfield/kk cuBLAS error " + std::to_string((int) s_) +      \
                               " at line " + std::to_string(__LINE__));                          \
  } while (0)

namespace becfield {

constexpr int F = 64;              // channels
constexpr int D = 768;             // node feature dim
constexpr int NR = 32;             // radial basis
constexpr int NW = egcnn::NPATH * F;    // 1088 tensor-product weights per edge
constexpr int R1 = NR + 1;         // rbf + constant 1 (folds b1 into the GEMM)
constexpr int H1 = F + 1;          // hidden + constant 1 (folds b2 into the GEMM)
constexpr int G = 4;               // edge groups per block in the TP kernel

struct Geo {
  double means[NR];
  double gamma, rc;
};

// ---------------- cuBLAS row-major helpers: C (m x n) = A (m x k) . B^T (n x k) or A . B (k x n)
inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                           int k, const float *al, const float *A, int lda, const float *B, int ldb,
                           const float *be, float *C, int ldc)
{
  return cublasSgemm(h, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}
inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                           int k, const double *al, const double *A, int lda, const double *B, int ldb,
                           const double *be, double *C, int ldc)
{
  return cublasDgemm(h, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}
// row-major C[rows x n] (ldc) = A[rows x k] (lda) . W^T, W row-major [n x k] (ldw)
template <class T>
void gemm_ABt(cublasHandle_t h, int rows, int n, int k, const T *A, int lda, const T *W, int ldw, T beta,
              T *C, int ldc)
{
  const T one = 1;
  CB(gemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, W, ldw, A, lda, &beta, C, ldc));
}
// row-major C[rows x n] = A[rows x k] . B, B row-major [k x n]
template <class T>
void gemm_AB(cublasHandle_t h, int rows, int n, int k, const T *A, int lda, const T *B, int ldb, T beta,
             T *C, int ldc)
{
  const T one = 1;
  CB(gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k, &one, B, ldb, A, lda, &beta, C, ldc));
}

// ---------------- kernels
// edges: one warp per receiver atom, order-preserving compaction of the (cutoff-filtered) LAMMPS full list
__device__ __forceinline__ bool edge_ok(AT::t_kkfloat_1d_3_lr x, AT::t_neighbors_2d neighbors, int i, int jj,
                                        double rc2, int &j, double &vx, double &vy, double &vz)
{
  j = neighbors(i, jj) & NEIGHMASK;
  vx = x(i, 0) - x(j, 0);    // r_dst - r_src (receiver i, sender j)
  vy = x(i, 1) - x(j, 1);
  vz = x(i, 2) - x(j, 2);
  return vx * vx + vy * vy + vz * vz < rc2;
}

__global__ void k_count(int n, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d numneigh,
                        AT::t_neighbors_2d neighbors, double rc2, int *cnt)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (i > n) return;
  if (i == n) {
    if (lane == 0) cnt[n] = 0;
    return;
  }
  int c = 0;
  const int nn = numneigh(i);
  for (int jj = lane; jj < nn; jj += 32) {
    int j;
    double vx, vy, vz;
    c += edge_ok(x, neighbors, i, jj, rc2, j, vx, vy, vz);
  }
  for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(~0u, c, o);
  if (lane == 0) cnt[i] = c;
}

__global__ void k_fill(int n, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors,
                       Kokkos::View<int *> owner, const int *rowptr, double rc2, int *src, double *vec, int *err)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (i >= n) return;
  int base = rowptr[i];
  const int nn = numneigh(i);
  for (int j0 = 0; j0 < nn; j0 += 32) {
    const int jj = j0 + lane;
    int j = 0;
    double vx = 0, vy = 0, vz = 0;
    const bool ok = jj < nn && edge_ok(x, neighbors, i, jj, rc2, j, vx, vy, vz);
    const unsigned m = __ballot_sync(~0u, ok);
    if (ok) {
      const int e = base + __popc(m & ((1u << lane) - 1u));
      const int s = owner(j);
      if (s < 0 || s >= n) atomicOr(err, 1);
      src[e] = s < 0 ? 0 : s;
      vec[3L * e] = vx;
      vec[3L * e + 1] = vy;
      vec[3L * e + 2] = vz;
    }
    base += __popc(m);
  }
}

// per-edge geometry: Y (component SH of v/(|v|+1e-8)), r; for the exact radial path also rbf|1 and cut(r)
template <class T>
__global__ void k_geo(long E, const double *vec, Geo g, bool exact, T *Y, T *R, T *rbf1, T *cut)
{
  const long e = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const double vx = vec[3 * e], vy = vec[3 * e + 1], vz = vec[3 * e + 2];
  const double d = sqrt(vx * vx + vy * vy + vz * vz);
  const double inv = 1.0 / (d + 1e-8);
  double Yd[9];
  egcnn::sh<double>(vx * inv, vy * inv, vz * inv, Yd);
  for (int k = 0; k < 9; ++k) Y[e * 9 + k] = T(Yd[k]);
  R[e] = T(d);
  if (exact) {
    for (int k = 0; k < NR; ++k) {
      const double t = d - g.means[k];
      rbf1[e * R1 + k] = T(exp(-g.gamma * t * t));
    }
    rbf1[e * R1 + NR] = T(1);
    const double xc = d / g.rc;
    cut[e] = T(xc < 1.0 ? exp(-xc * xc / (1.0 - xc * xc + 1e-8)) : 0.0);
  }
}

template <class T>
__global__ void k_embed(int n, AT::t_int_1d type, const T *x0t, T *x)
{
  const long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long) n * D) return;
  const int i = t / D, k = t % D;
  x[t] = x0t[(long) (type(i) - 1) * D + k];
}

template <class T> __global__ void k_silu(long ne, T *H)
{
  const long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ne * H1) return;
  const int k = t % H1;
  if (k == F) {
    H[t] = T(1);
  } else {
    const T v = H[t];
    H[t] = v / (T(1) + exp(-v));
  }
}

struct Spline {
  const void *tab;    // [NG+1][2][NW]: value, derivative (d/dr) at r_k = k*h; cut(r) included
  int ng;
  double h;
};

template <class T, bool SPL>
__global__ void __launch_bounds__(F *G)
    k_tp(int a0, const int *rowptr, int ebase, const int *src, const T *Y, const T *R, const T *cut, const T *W,
         Spline sp, const T *x, T *agg)
{
  const int c = a0 + blockIdx.x;
  const int u = threadIdx.x % F, g = threadIdx.x / F;
  T acc[12];
#pragma unroll
  for (int k = 0; k < 12; ++k) acc[k] = 0;
  const int e1 = rowptr[c + 1];
  const T hh = T(sp.h), hinv = T(1.0 / sp.h);
  for (int e = rowptr[c] + g; e < e1; e += G) {
    const T *xs = x + (long) src[e] * D;
    T xa[12], y[9], w[egcnn::NPATH];
    xa[0] = xs[u];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
      xa[1 + m] = xs[F + 3 * u + m];
      xa[4 + m] = xs[4 * F + 3 * u + m];
    }
#pragma unroll
    for (int m = 0; m < 5; ++m) xa[7 + m] = xs[7 * F + 5 * u + m];
#pragma unroll
    for (int k = 0; k < 9; ++k) y[k] = Y[(long) e * 9 + k];
    if constexpr (SPL) {
      const T s = R[e] * hinv;
      int k = int(s);
      k = k < 0 ? 0 : (k > sp.ng - 1 ? sp.ng - 1 : k);
      const T t = s - T(k), t1 = T(1) - t;
      const T h00 = (T(1) + T(2) * t) * t1 * t1, h10 = t * t1 * t1 * hh, h01 = t * t * (T(3) - T(2) * t),
              h11 = -t * t * t1 * hh;
      const T *r0 = static_cast<const T *>(sp.tab) + (long) k * 2 * NW + u;
      const T *r1 = r0 + 2 * NW;
#pragma unroll
      for (int p = 0; p < egcnn::NPATH; ++p)
        w[p] = h00 * r0[p * F] + h10 * r0[NW + p * F] + h01 * r1[p * F] + h11 * r1[NW + p * F];
    } else {
      const T ce = cut[e];
      const T *we = W + (long) (e - ebase) * NW + u;
#pragma unroll
      for (int p = 0; p < egcnn::NPATH; ++p) w[p] = we[p * F] * ce;
    }
    egcnn::tp<T>(xa, y, w, acc);
  }
  if (G > 1) {
    __shared__ T red[G][12][F];
#pragma unroll
    for (int k = 0; k < 12; ++k) red[g][k][u] = acc[k];
    __syncthreads();
    if (g != 0) return;
    for (int h = 1; h < G; ++h)
#pragma unroll
      for (int k = 0; k < 12; ++k) acc[k] += red[h][k][u];
  }
  T *o = agg + (long) c * D;
  o[u] = acc[0];
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    o[F + 3 * u + m] = acc[1 + m];
    o[4 * F + 3 * u + m] = acc[4 + m];
  }
#pragma unroll
  for (int m = 0; m < 5; ++m) o[7 * F + 5 * u + m] = acc[7 + m];
}


// x += agg L with L = blockdiag over irreps (0e,1o,1e,2e) of Wk (64x64) (x) I_(2l+1); Wb = 4 x [64][64]
template <class T> __global__ void __launch_bounds__(256) k_lin(const T *agg, const T *Wb, T *x)
{
  __shared__ T a[D];
  const long n = blockIdx.x;
  for (int k = threadIdx.x; k < D; k += 256) a[k] = agg[n * D + k];
  __syncthreads();
  for (int o = threadIdx.x; o < D; o += 256) {
    int b, off, d;
    if (o < F) { b = 0; off = 0; d = 1; }
    else if (o < 4 * F) { b = 1; off = F; d = 3; }
    else if (o < 7 * F) { b = 2; off = 4 * F; d = 3; }
    else { b = 3; off = 7 * F; d = 5; }
    const int up = (o - off) / d, m = (o - off) % d;
    const T *Wk = Wb + b * F * F + up;
    T s = 0;
#pragma unroll 8
    for (int u = 0; u < F; ++u) s += Wk[u * F] * a[off + u * d + m];
    x[n * D + o] += s;
  }
}

template <class T> __global__ void k_sum(int n, const T *Z, double *zsum)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double v[9];
  for (int k = 0; k < 9; ++k) v[k] = i < n ? double(Z[(long) i * 9 + k]) : 0.0;
  for (int k = 0; k < 9; ++k)
    for (int o = 16; o > 0; o >>= 1) v[k] += __shfl_xor_sync(~0u, v[k], o);
  if ((threadIdx.x & 31) == 0)
    for (int k = 0; k < 9; ++k) atomicAdd(&zsum[k], v[k]);
}

struct Box {
  double eps[3], boxlo[3], hinv[6], cell[3][3], cellavg[3][3];
  int have_prev;
};

template <class T>
__global__ void k_final(int n, const T *Zr, const double *zsum, Box b, AT::t_kkfloat_1d_3_lr x,
                        AT::t_tagint_1d tag, AT::t_kkacc_1d_3 f, double *prevfrac, double *prevZ,
                        double *dP)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double p[3] = {0, 0, 0};
  if (i < n) {
    double Z[9];
    for (int k = 0; k < 9; ++k) Z[k] = double(Zr[(long) i * 9 + k]) - zsum[k] / n;
    for (int bb = 0; bb < 3; ++bb)
      f(i, bb) += b.eps[0] * Z[bb] + b.eps[1] * Z[3 + bb] + b.eps[2] * Z[6 + bb];
    const double dx = x(i, 0) - b.boxlo[0], dy = x(i, 1) - b.boxlo[1], dz = x(i, 2) - b.boxlo[2];
    const double s[3] = {b.hinv[0] * dx + b.hinv[5] * dy + b.hinv[4] * dz, b.hinv[1] * dy + b.hinv[3] * dz,
                         b.hinv[2] * dz};
    const long t = tag(i) - 1;
    if (b.have_prev) {
      double ds[3], dr[3];
      for (int k = 0; k < 3; ++k) {
        ds[k] = s[k] - prevfrac[t * 3 + k];
        ds[k] -= rint(ds[k]);
      }
      for (int bb = 0; bb < 3; ++bb)
        dr[bb] = ds[0] * b.cellavg[0][bb] + ds[1] * b.cellavg[1][bb] + ds[2] * b.cellavg[2][bb];
      for (int a = 0; a < 3; ++a)
        for (int bb = 0; bb < 3; ++bb) p[a] += 0.5 * (Z[a * 3 + bb] + prevZ[t * 9 + a * 3 + bb]) * dr[bb];
    }
    for (int k = 0; k < 3; ++k) prevfrac[t * 3 + k] = s[k];
    for (int k = 0; k < 9; ++k) prevZ[t * 9 + k] = Z[k];
  }
  for (int a = 0; a < 3; ++a)
    for (int o = 16; o > 0; o >>= 1) p[a] += __shfl_xor_sync(~0u, p[a], o);
  if ((threadIdx.x & 31) == 0)
    for (int a = 0; a < 3; ++a) atomicAdd(&dP[a], p[a]);
}

// ---------------- evaluator
template <class T> struct DevBuf {
  T *p = nullptr;
  size_t n = 0;
  void need(size_t m)
  {
    if (m <= n) return;
    if (p) cudaFree(p);
    m = m + m / 4 + 64;
    CK(cudaMalloc(&p, m * sizeof(T)));
    n = m;
  }
  void set(const std::vector<T> &v)
  {
    need(v.size());
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~DevBuf()
  {
    if (p) cudaFree(p);
  }
};

struct EvaluatorBase {
  virtual ~EvaluatorBase() = default;
  // returns dP of this call (added to the fix's accumulator) ; writes Z(by tag) on request
  virtual void compute(int n, int natoms, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d type, AT::t_tagint_1d tag,
                       AT::t_kkacc_1d_3 f, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors,
                       Kokkos::View<int *> owner, const Box &box, double dP[3]) = 0;
  virtual void get_z(std::vector<double> &z, int natoms) = 0;
  virtual long edges() const = 0;
  double rc = 0;
};

template <class T> struct Evaluator : EvaluatorBase {
  long max_edges;
  Geo geo{};
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
  DevBuf<T> x0t, Mout, x, agg, Zr, rbf1, Y, R, cut, H, W;
  std::vector<DevBuf<T>> W1, W2, L, Wb, tab;
  DevBuf<int> cnt, rowptr, src, err;
  DevBuf<double> vec;
  bool exact = false, blockdiag = true;
  int ng = 1024;
  DevBuf<char> scan_tmp;
  DevBuf<double> zsum, prevfrac, prevZ, dP;
  std::vector<int> h_rowptr;
  long E = 0;
  // optional phase timing (env BECFIELD_PROFILE=1), printed to stderr when the fix is deleted
  bool prof = getenv("BECFIELD_PROFILE") != nullptr;
  std::vector<cudaEvent_t> evs;
  std::vector<std::string> evn;
  std::map<std::string, double> acc;
  long ncalls = 0;
  void mark(const char *name)
  {
    if (!prof) return;
    if (evn.size() == evs.size()) {
      cudaEvent_t e;
      CK(cudaEventCreate(&e));
      evs.push_back(e);
    }
    evn.emplace_back(name);
    CK(cudaEventRecord(evs[evn.size() - 1], stream));
  }

  Evaluator(const nlohmann::json &j, const std::vector<int> &type_Z, long max_edges_, bool exact_, int ng_) :
      max_edges(max_edges_), W1(6), W2(6), L(6), Wb(6), tab(6), exact(exact_), ng(ng_)
  {
    const auto &ar = j.at("arch");
    if (j.at("format") != "egcnn-bec-1" || ar.at("num_layers") != 6 || ar.at("feature_size") != F ||
        ar.at("num_radial") != NR || ar.at("edge_sh_lmax") != 2 || ar.at("equivariant_cart") != true)
      throw std::runtime_error("unsupported EGCNN model file (format/architecture)");
    rc = geo.rc = ar.at("cutoff").get<double>();
    geo.gamma = j.at("rbf_gamma").get<double>();
    auto means = j.at("rbf_means").get<std::vector<double>>();
    for (int k = 0; k < NR; ++k) geo.means[k] = means.at(k);
    // compiled-in tensor-product coefficients must match the file
    const double *cc = egcnn::coef_check();
    const int *pc = egcnn::path_check();
    int m = 0;
    const auto &tp = j.at("tp");
    if ((int) tp.size() != egcnn::NPATH) throw std::runtime_error("EGCNN model: wrong number of TP paths");
    for (int p = 0; p < egcnn::NPATH; ++p) {
      const auto &P = tp[p];
      if (P.at("i1") != pc[3 * p] || P.at("i2") != pc[3 * p + 1] || P.at("io") != pc[3 * p + 2])
        throw std::runtime_error("EGCNN model: TP path layout differs from the compiled kernels");
      for (const auto &a : P.at("C"))
        for (const auto &b : a)
          for (const auto &c : b)
            if (std::fabs(c.get<double>() - cc[m++]) > 1e-10)
              throw std::runtime_error("EGCNN model: TP coefficients differ from the compiled kernels");
    }
    if (m != egcnn::NCOEF) throw std::runtime_error("EGCNN model: TP coefficient count mismatch");

    auto flat = [](const nlohmann::json &a) {
      std::vector<T> v;
      for (const auto &r : a)
        if (r.is_array())
          for (const auto &c : r) v.push_back(T(c.get<double>()));
        else
          v.push_back(T(r.get<double>()));
      return v;
    };
    const auto &x0 = j.at("x0");
    std::vector<T> xt;
    for (int Z : type_Z) {
      const auto &row = x0.at(Z);
      for (const auto &c : row) xt.push_back(T(c.get<double>()));
    }
    x0t.set(xt);
    Mout.set(flat(j.at("Mout")));
    const auto &layers = j.at("layers");
    const int boff[4] = {0, F, 4 * F, 7 * F}, bdim[4] = {1, 3, 3, 5};
    for (int l = 0; l < 6; ++l) {
      const auto &Ly = layers.at(l);
      auto jd = [](const nlohmann::json &a) {
        std::vector<double> v;
        for (const auto &r : a)
          if (r.is_array())
            for (const auto &c : r) v.push_back(c.get<double>());
          else
            v.push_back(r.get<double>());
        return v;
      };
      auto w1 = jd(Ly.at("W1")), b1 = jd(Ly.at("b1")), w2 = jd(Ly.at("W2")), b2 = jd(Ly.at("b2"));
      auto Ld = jd(Ly.at("L"));
      // block-diagonal structure of the e3nn Linear: L[off+u*d+m][off+u'*d+m'] = Wk[u][u'] delta_mm'
      std::vector<T> wb(4 * F * F);
      double big = 0, viol = 0;
      for (double v : Ld) big = std::max(big, std::fabs(v));
      std::vector<char> inblk((size_t) D * D, 0);
      for (int b = 0; b < 4; ++b)
        for (int u = 0; u < F; ++u)
          for (int up = 0; up < F; ++up) {
            const double ref = Ld[(size_t) (boff[b] + u * bdim[b]) * D + boff[b] + up * bdim[b]];
            wb[b * F * F + u * F + up] = T(ref);
            for (int m = 0; m < bdim[b]; ++m)
              for (int mp = 0; mp < bdim[b]; ++mp) {
                const size_t ix = (size_t) (boff[b] + u * bdim[b] + m) * D + boff[b] + up * bdim[b] + mp;
                inblk[ix] = 1;
                viol = std::max(viol, std::fabs(Ld[ix] - (m == mp ? ref : 0.0)));
              }
          }
      for (size_t ix = 0; ix < inblk.size(); ++ix)
        if (!inblk[ix]) viol = std::max(viol, std::fabs(Ld[ix]));
      if (viol > 1e-12 * big) blockdiag = false;
      Wb[l].set(wb);
      std::vector<T> Lt(Ld.begin(), Ld.end());
      L[l].set(Lt);
      if (exact) {
        std::vector<T> a1(F * R1), a2((size_t) NW * H1);    // [W | b] row-major
        for (int r = 0; r < F; ++r) {
          for (int k = 0; k < NR; ++k) a1[r * R1 + k] = T(w1[r * NR + k]);
          a1[r * R1 + NR] = T(b1[r]);
        }
        for (int r = 0; r < NW; ++r) {
          for (int k = 0; k < F; ++k) a2[(size_t) r * H1 + k] = T(w2[(size_t) r * F + k]);
          a2[(size_t) r * H1 + F] = T(b2[r]);
        }
        W1[l].set(a1);
        W2[l].set(a2);
      } else {
        // cubic-Hermite table of w(r) = cut(r) (W2 silu(W1 rbf(r) + b1) + b2) and dw/dr on r_k = k h, h = rc/ng
        const double h = geo.rc / ng;
        std::vector<T> tb((size_t) (ng + 1) * 2 * NW);
        std::vector<double> rb(NR), drb(NR), z(F), hv(F), dh(F);
        for (int k = 0; k <= ng; ++k) {
          const double r = k * h, xc = r / geo.rc, den = 1.0 - xc * xc + 1e-8;
          const double c = xc < 1.0 ? std::exp(-xc * xc / den) : 0.0;
          const double dc = xc < 1.0 ? c * (-2.0 * xc * (1.0 + 1e-8)) / (den * den) / geo.rc : 0.0;
          for (int q = 0; q < NR; ++q) {
            const double t = r - geo.means[q];
            rb[q] = std::exp(-geo.gamma * t * t);
            drb[q] = -2.0 * geo.gamma * t * rb[q];
          }
          for (int a = 0; a < F; ++a) {
            double zz = b1[a], dz = 0;
            for (int q = 0; q < NR; ++q) {
              zz += w1[a * NR + q] * rb[q];
              dz += w1[a * NR + q] * drb[q];
            }
            const double sg = 1.0 / (1.0 + std::exp(-zz));
            hv[a] = zz * sg;
            dh[a] = (sg * (1.0 + zz * (1.0 - sg))) * dz;
          }
          for (int o = 0; o < NW; ++o) {
            double wv = b2[o], dw = 0;
            for (int a = 0; a < F; ++a) {
              wv += w2[(size_t) o * F + a] * hv[a];
              dw += w2[(size_t) o * F + a] * dh[a];
            }
            tb[((size_t) k * 2) * NW + o] = T(c * wv);
            tb[((size_t) k * 2 + 1) * NW + o] = T(dc * wv + c * dw);
          }
        }
        tab[l].set(tb);
      }
    }
    CB(cublasCreate(&blas));
    CB(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));    // plain fp32/fp64, no TF32
    zsum.need(9);
    dP.need(3);
    err.need(1);
  }
  ~Evaluator() override
  {
    if (prof && ncalls) {
      double tot = 0;
      for (auto &kv : acc) tot += kv.second;
      fprintf(stderr, "[becfield profile] %ld calls, %.3f ms/call:", ncalls, tot / ncalls);
      for (auto &kv : acc) fprintf(stderr, " %s %.3f", kv.first.c_str(), kv.second / ncalls);
      fprintf(stderr, " (ms/call)\n");
    }
    for (auto e : evs) cudaEventDestroy(e);
    if (blas) cublasDestroy(blas);
  }
  long edges() const override { return E; }

  void compute(int n, int natoms, AT::t_kkfloat_1d_3_lr xv, AT::t_int_1d type, AT::t_tagint_1d tag,
               AT::t_kkacc_1d_3 f, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors,
               Kokkos::View<int *> owner, const Box &box, double dPh[3]) override
  {
    stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    CB(cublasSetStream(blas, stream));
    const int TB = 128;
    const int nb = (n + 1 + TB - 1) / TB;
    evn.clear();
    mark("start");
    // ---- edges (CSR by receiver)
    cnt.need(n + 1);
    rowptr.need(n + 1);
    const int nw = ((n + 1) * 32 + TB - 1) / TB;    // one warp per atom
    k_count<<<nw, TB, 0, stream>>>(n, xv, numneigh, neighbors, rc * rc, cnt.p);
    size_t bytes = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, cnt.p, rowptr.p, n + 1, stream));
    scan_tmp.need(bytes);
    CK(cub::DeviceScan::ExclusiveSum(scan_tmp.p, bytes, cnt.p, rowptr.p, n + 1, stream));
    h_rowptr.resize(n + 1);
    CK(cudaMemcpyAsync(h_rowptr.data(), rowptr.p, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    E = h_rowptr[n];
    src.need(E + 1);
    vec.need(3 * (E + 1));
    Y.need((E + 1) * 9);
    R.need(E + 1);
    if (exact) {
      rbf1.need((E + 1) * R1);
      cut.need(E + 1);
    }
    CK(cudaMemsetAsync(err.p, 0, sizeof(int), stream));
    k_fill<<<nw, TB, 0, stream>>>(n, xv, numneigh, neighbors, owner, rowptr.p, rc * rc, src.p, vec.p, err.p);
    if (E > 0)
      k_geo<T><<<(E + 255) / 256, 256, 0, stream>>>(E, vec.p, geo, exact, Y.p, R.p, rbf1.p, cut.p);
    // ---- chunks of receivers with at most max_edges edges
    std::vector<int> cuts{0};
    while (cuts.back() < n) {
      int a = cuts.back(), b = a + 1;
      while (b < n && (long) h_rowptr[b + 1] - h_rowptr[a] <= max_edges) ++b;
      cuts.push_back(b);
    }
    if (!exact) cuts = {0, n};    // no per-edge weight buffer -> no chunking needed
    long emax = 1;
    for (size_t c = 1; c < cuts.size(); ++c)
      emax = std::max(emax, (long) h_rowptr[cuts[c]] - h_rowptr[cuts[c - 1]]);
    if (exact) {
      H.need(emax * H1);
      W.need(emax * NW);
    }
    x.need((size_t) n * D);
    agg.need((size_t) n * D);
    Zr.need((size_t) n * 9);
    k_embed<T><<<((long) n * D + 255) / 256, 256, 0, stream>>>(n, type, x0t.p, x.p);
    mark("edges");
    for (int l = 0; l < 6; ++l) {
      for (size_t c = 1; c < cuts.size(); ++c) {
        const int a0 = cuts[c - 1], a1 = cuts[c];
        const long eb = h_rowptr[a0], ne = (long) h_rowptr[a1] - eb;
        if (exact && ne > 0) {
          gemm_ABt<T>(blas, ne, F, R1, rbf1.p + eb * R1, R1, W1[l].p, R1, T(0), H.p, H1);
          k_silu<T><<<(ne * H1 + 255) / 256, 256, 0, stream>>>(ne, H.p);
          gemm_ABt<T>(blas, ne, NW, H1, H.p, H1, W2[l].p, H1, T(0), W.p, NW);
        }
        mark("radial");
        const Spline sp{tab[l].p, ng, rc / ng};
        if (a1 > a0) {
          if (exact)
            k_tp<T, false><<<a1 - a0, F * G, 0, stream>>>(a0, rowptr.p, eb, src.p, Y.p, R.p, cut.p, W.p, sp, x.p,
                                                          agg.p);
          else
            k_tp<T, true><<<a1 - a0, F * G, 0, stream>>>(a0, rowptr.p, eb, src.p, Y.p, R.p, cut.p, W.p, sp, x.p,
                                                         agg.p);
        }
        mark("tp");
      }
      if (blockdiag)
        k_lin<T><<<n, 256, 0, stream>>>(agg.p, Wb[l].p, x.p);    // x += agg L (block-diagonal)
      else
        gemm_AB<T>(blas, n, D, D, agg.p, D, L[l].p, D, T(1), x.p, D);
      mark("linear");
    }
    gemm_AB<T>(blas, n, 9, D, x.p, D, Mout.p, 9, T(0), Zr.p, 9);
    // ---- ASR, forces, polarization
    prevfrac.need((size_t) natoms * 3);
    prevZ.need((size_t) natoms * 9);
    CK(cudaMemsetAsync(zsum.p, 0, 9 * sizeof(double), stream));
    CK(cudaMemsetAsync(dP.p, 0, 3 * sizeof(double), stream));
    k_sum<T><<<nb, TB, 0, stream>>>(n, Zr.p, zsum.p);
    k_final<T><<<nb, TB, 0, stream>>>(n, Zr.p, zsum.p, box, xv, tag, f, prevfrac.p, prevZ.p, dP.p);
    int herr = 0;
    CK(cudaMemcpyAsync(dPh, dP.p, 3 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CK(cudaMemcpyAsync(&herr, err.p, sizeof(int), cudaMemcpyDeviceToHost, stream));
    mark("output");
    CK(cudaStreamSynchronize(stream));
    CK(cudaGetLastError());
    if (prof) {
      for (size_t k = 1; k < evn.size(); ++k) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, evs[k - 1], evs[k]));
        acc[evn[k]] += ms;
      }
      ++ncalls;
    }
    if (herr) throw std::runtime_error("a neighbour has no owned atom (atom map / ghost inconsistency)");
  }

  void get_z(std::vector<double> &z, int natoms) override
  {
    z.resize((size_t) natoms * 9);
    CK(cudaMemcpy(z.data(), prevZ.p, z.size() * sizeof(double), cudaMemcpyDeviceToHost));
  }
};

}    // namespace becfield

/* ---------------------------------------------------------------------- */

static const char *const PT[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl",
    "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm"};

FixBECFieldKokkos::FixBECFieldKokkos(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = Device;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = F_MASK;

  restart_global = 1;
  vector_flag = 1;
  size_vector = 6;
  global_freq = 1;
  extvector = 0;
  thermo_energy = 0;
  energy_global_flag = 0;
  virial_global_flag = 0;

  if (!lmp->kokkos || lmp->kokkos->ngpus < 1)
    error->all(FLERR, "fix becfield/kk requires KOKKOS with a CUDA GPU ('-k on g 1 -sf kk')");
  const int ntypes = atom->ntypes;
  if (narg < 4 + ntypes)
    error->all(FLERR, "Usage: fix ID group becfield/kk model.json elem1 ... elemN(ntypes) efield Ex Ey Ez "
                      "[p0 Px Py Pz] [precision float32|float64] [radial spline|exact] [grid N] [chunk Nedges] "
                      "[zfile file every] "
                      "[timing yes|no]");
  model_file = arg[3];
  for (int t = 0; t < ntypes; ++t) elements.emplace_back(arg[4 + t]);
  bool have_field = false;
  int iarg = 4 + ntypes;
  while (iarg < narg) {
    const std::string kw = arg[iarg];
    auto need = [&](int k) {
      if (iarg + k >= narg) error->all(FLERR, "fix becfield/kk: missing value(s) after {}", kw);
    };
    if (kw == "efield") {
      need(3);
      for (int d = 0; d < 3; ++d) {
        efield_str[d] = arg[iarg + 1 + d];
        if (!utils::strmatch(efield_str[d], "^v_"))
          efield_val[d] = utils::numeric(FLERR, efield_str[d], false, lmp);
      }
      have_field = true;
      iarg += 4;
    } else if (kw == "p0") {
      need(3);
      for (int d = 0; d < 3; ++d) P[d] = utils::numeric(FLERR, arg[iarg + 1 + d], false, lmp);
      iarg += 4;
    } else if (kw == "precision") {
      need(1);
      const std::string v = arg[iarg + 1];
      if (v == "float32") use_double = false;
      else if (v == "float64") use_double = true;
      else error->all(FLERR, "fix becfield/kk precision must be float32 or float64");
      iarg += 2;
    } else if (kw == "radial") {
      need(1);
      const std::string v = arg[iarg + 1];
      if (v == "spline") radial_exact = false;
      else if (v == "exact") radial_exact = true;
      else error->all(FLERR, "fix becfield/kk radial must be spline or exact");
      iarg += 2;
    } else if (kw == "grid") {
      need(1);
      grid = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (grid < 64) error->all(FLERR, "fix becfield/kk grid must be >= 64");
      iarg += 2;
    } else if (kw == "chunk") {
      need(1);
      max_edges = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp);
      if (max_edges < 1000) error->all(FLERR, "fix becfield/kk chunk must be >= 1000 edges");
      iarg += 2;
    } else if (kw == "zfile") {
      need(2);
      zfile = arg[iarg + 1];
      zevery = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
      if (zevery < 1) error->all(FLERR, "fix becfield/kk zfile interval must be >= 1");
      iarg += 3;
    } else if (kw == "timing") {
      need(1);
      timing = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "fix becfield/kk: unknown keyword {}", kw);
    }
  }
  if (!have_field) error->all(FLERR, "fix becfield/kk requires the efield keyword");

  std::vector<int> Z;
  for (auto &e : elements) {
    int z = -1;
    for (int k = 0; k < 100; ++k)
      if (e == PT[k]) z = k + 1;
    if (z < 0) error->all(FLERR, "fix becfield/kk: unknown element {}", e);
    Z.push_back(z);
  }
  try {
    std::ifstream in(model_file);
    if (!in) error->all(FLERR, "fix becfield/kk: cannot open {}", model_file);
    auto j = nlohmann::json::parse(in);
    if (use_double)
      ev = std::make_unique<becfield::Evaluator<double>>(j, Z, max_edges, radial_exact, grid);
    else
      ev = std::make_unique<becfield::Evaluator<float>>(j, Z, max_edges, radial_exact, grid);
  } catch (std::exception &e) {
    error->all(FLERR, "fix becfield/kk: {}", e.what());
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "fix becfield/kk: EGCNN {} ({}, radial {}), cutoff {} A, elements:{}\n", model_file,
                   use_double ? "float64" : "float32",
                   radial_exact ? "exact (GEMM)" : fmt::format("spline, {} intervals", grid), ev->rc, [&] {
                     std::string s;
                     for (auto &e : elements) s += " " + e;
                     return s;
                   }());
}

FixBECFieldKokkos::~FixBECFieldKokkos() = default;

int FixBECFieldKokkos::setmask()
{
  return POST_FORCE;    // nothing during minimize: no field force, no polarization update
}

void FixBECFieldKokkos::init()
{
  if (comm->nprocs > 1) error->all(FLERR, "fix becfield/kk supports a single MPI rank only");
  if (igroup != group->find("all")) error->all(FLERR, "fix becfield/kk must be applied to group all");
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "fix becfield/kk requires 'atom_modify map [yes|array|hash]'");
  if (!atom->tag_enable || atom->natoms != (bigint) atom->map_tag_max)
    error->all(FLERR, "fix becfield/kk requires consecutive atom IDs 1..N");
  if (!force->pair || force->pair->cutforce < ev->rc)
    error->all(FLERR, "fix becfield/kk needs a pair style with cutoff >= {} A (the EGCNN cutoff)", ev->rc);
  for (int d = 0; d < 3; ++d) {
    efield_var[d] = -1;
    if (utils::strmatch(efield_str[d], "^v_")) {
      efield_var[d] = input->variable->find(efield_str[d].c_str() + 2);
      if (efield_var[d] < 0)
        error->all(FLERR, "Variable {} for fix becfield/kk does not exist", efield_str[d].substr(2));
      if (!input->variable->equalstyle(efield_var[d]))
        error->all(FLERR, "Variable {} for fix becfield/kk must be equal-style", efield_str[d].substr(2));
    }
  }
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(0);
  request->set_kokkos_device(1);
}

void FixBECFieldKokkos::init_list(int, NeighList *ptr)
{
  list = ptr;
}

void FixBECFieldKokkos::setup(int vflag)
{
  post_force(vflag);
}

void FixBECFieldKokkos::post_force(int)
{
  const auto t0 = std::chrono::steady_clock::now();
  for (int d = 0; d < 3; ++d) {
    field[d] = efield_var[d] >= 0 ? input->variable->compute_equal(efield_var[d]) : efield_val[d];
    if (!std::isfinite(field[d])) error->all(FLERR, "fix becfield/kk efield component {} is not finite", d + 1);
  }
  atomKK->sync(execution_space, datamask_read);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto f = atomKK->k_f.view<LMPDeviceType>();
  auto type = atomKK->k_type.view<LMPDeviceType>();
  auto tag = atomKK->k_tag.view<LMPDeviceType>();
  auto k_list = static_cast<NeighListKokkos<LMPDeviceType> *>(list);
  const int nlocal = atom->nlocal, nall = atom->nlocal + atom->nghost;

  if ((int) owner.extent(0) < nall) Kokkos::realloc(owner, nall);
  {
    auto own = owner;
    auto map_style = atom->map_style;
    auto k_map_array = atomKK->k_map_array;
    auto k_map_hash = atomKK->k_map_hash;
    k_map_array.template sync<LMPDeviceType>();
    Kokkos::parallel_for(
        "FixBECFieldKokkos::owner", nall, KOKKOS_LAMBDA(const int i) {
          own(i) = AtomKokkos::map_kokkos<LMPDeviceType>(tag(i), map_style, k_map_array, k_map_hash);
        });
  }

  becfield::Box b{};
  const double *h = domain->h, *hi = domain->h_inv;
  for (int d = 0; d < 3; ++d) {
    b.eps[d] = field[d];
    b.boxlo[d] = domain->boxlo[d];
  }
  for (int k = 0; k < 6; ++k) b.hinv[k] = hi[k];
  auto cellof = [](const double *hh, double c[3][3]) {
    const double m[3][3] = {{hh[0], 0, 0}, {hh[5], hh[1], 0}, {hh[4], hh[3], hh[2]}};
    std::memcpy(c, m, sizeof(m));
  };
  double cnow[3][3], cprev[3][3];
  cellof(h, cnow);
  cellof(hprev, cprev);
  std::memcpy(b.cell, cnow, sizeof(cnow));
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) b.cellavg[r][c] = 0.5 * (cnow[r][c] + cprev[r][c]);
  b.have_prev = have_prev ? 1 : 0;

  double dP[3] = {0, 0, 0};
  try {
    ev->compute(nlocal, (int) atom->natoms, x, type, tag, f, k_list->d_numneigh, k_list->d_neighbors, owner, b,
                dP);
  } catch (std::exception &e) {
    error->one(FLERR, "fix becfield/kk: {}", e.what());
  }
  for (int d = 0; d < 3; ++d) P[d] += dP[d];
  for (int k = 0; k < 6; ++k) hprev[k] = h[k];
  have_prev = true;
  atomKK->modified(execution_space, datamask_modify);

  if (zevery > 0 && update->ntimestep % zevery == 0) write_z();
  if (timing) {
    t_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++n_calls;
  }
}

void FixBECFieldKokkos::write_z()
{
  std::vector<double> z;
  ev->get_z(z, (int) atom->natoms);
  if (comm->me != 0) return;
  FILE *fp = fopen(zfile.c_str(), "a");
  if (!fp) error->one(FLERR, "fix becfield/kk: cannot open {}", zfile);
  utils::print(fp, "# step {} natoms {} field {:.10g} {:.10g} {:.10g} P {:.12g} {:.12g} {:.12g}\n",
             update->ntimestep, atom->natoms, field[0], field[1], field[2], P[0], P[1], P[2]);
  for (bigint t = 0; t < atom->natoms; ++t) {
    utils::print(fp, "{}", t + 1);
    for (int k = 0; k < 9; ++k) utils::print(fp, " {:.9e}", z[t * 9 + k]);
    fputc('\n', fp);
  }
  fclose(fp);
}

void FixBECFieldKokkos::post_run()
{
  if (timing && n_calls > 0 && comm->me == 0)
    utils::logmesg(lmp, "fix becfield/kk: {} calls, {:.3f} ms per call (incl. sync), {} edges\n", n_calls,
                   1e3 * t_total / n_calls, ev->edges());
  t_total = 0;
  n_calls = 0;
}

double FixBECFieldKokkos::compute_vector(int n)
{
  return n < 3 ? P[n] : field[n - 3];
}

void FixBECFieldKokkos::write_restart(FILE *fp)
{
  if (comm->me == 0) {
    int size = 3 * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(P, sizeof(double), 3, fp);
  }
}

void FixBECFieldKokkos::restart(char *buf)
{
  auto *list_ = (double *) buf;
  for (int d = 0; d < 3; ++d) P[d] = list_[d];
  have_prev = false;
}

#endif
