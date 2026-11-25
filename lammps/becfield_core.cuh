/* ----------------------------------------------------------------------
   BM-BornField: CUDA core of the EGCNN Born-effective-charge network (shared by the LAMMPS
   fix becfield/kk and the Python module bornfield).  This software is distributed under the
   GNU General Public License (version 2), like LAMMPS.

   Network (BEC_PRED/<system>/model.py, EquivarBECModel; Kutana et al., Sci. Rep. 15, 16719 (2025)),
   exported by tools/export_egcnn.py:
     node features x (N x 768) = 64x0e + 64x1o + 64x1e + 64x2e, e3nn layout [block][channel u][m]
     edges src -> dst with vector v = r_dst - r_src
     per layer: w_e = cut(r) * (W2 silu(W1 rbf(r) + b1) + b2)   (17 paths x 64 channels)
                agg_i = sum_e TP_uvu(x_src, Y(v/|v|), w_e);  x += agg L
     Z* = x Mout (output linear + CartesianTensor), minus the mean over all atoms (acoustic sum rule).
   Usage: count edges per receiver into count_buffer(n), scan(n) -> E, fill src()/vec() in CSR order
   (rowptr()), then run(n, type) -> device pointer to Z* (double, N x 9, ASR applied).
------------------------------------------------------------------------- */
#pragma once

#include "egcnn_generated.hpp"
#include "nlohmann_json.hpp"

#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess)                                                                       \
      throw std::runtime_error(std::string("becfield CUDA error: ") + cudaGetErrorString(e_) +  \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));              \
  } while (0)
#define CB(x)                                                                                    \
  do {                                                                                           \
    cublasStatus_t s_ = (x);                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                             \
      throw std::runtime_error("becfield cuBLAS error " + std::to_string((int) s_) +             \
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
__global__ void k_embed(int n, const int *type, const T *x0t, T *x)    // type: 1-based
{
  const long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long) n * D) return;
  const int i = t / D, k = t % D;
  x[t] = x0t[(long) (type[i] - 1) * D + k];
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

template <class T> __global__ void k_asr(int n, const T *Zr, const double *zsum, double *Z)
{
  const long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long) n * 9) return;
  Z[t] = double(Zr[t]) - zsum[t % 9] / n;
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

struct CoreBase {
  virtual ~CoreBase() = default;
  virtual void set_stream(cudaStream_t s) = 0;
  virtual int *count_buffer(int n) = 0;           // n+1 ints; caller writes per-receiver edge counts
  virtual long scan(int n) = 0;                   // -> E; allocates src/vec; host rowptr available
  virtual int *rowptr() = 0;
  virtual int *src() = 0;                         // E: sender atom index (0..n-1)
  virtual double *vec() = 0;                      // 3E: r_dst - r_src
  virtual int *errflag() = 0;
  virtual const double *run(int n, const int *type) = 0;    // Z* (double, n x 9, ASR), on the device
  virtual void mark(const char *name) = 0;
  virtual long edges() const = 0;
  virtual void accumulate_profile() = 0;    // after the caller's final synchronisation
  double rc = 0;
};

template <class T> struct Core : CoreBase {
  long max_edges;
  Geo geo{};
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
  DevBuf<T> x0t, Mout, x, agg, Zr, rbf1, Y, R, cut, H, W;
  std::vector<DevBuf<T>> W1, W2, L, Wb, tab;
  DevBuf<int> cnt, rowptr_, src_, err;
  DevBuf<double> vec_;
  bool exact = false, blockdiag = true;
  int ng = 1024;
  DevBuf<char> scan_tmp;
  DevBuf<double> zsum, Zd;
  std::vector<int> h_rowptr;
  long E = 0;
  // optional phase timing (env BECFIELD_PROFILE=1), printed to stderr at destruction
  bool prof = getenv("BECFIELD_PROFILE") != nullptr;
  std::vector<cudaEvent_t> evs;
  std::vector<std::string> evn;
  std::map<std::string, double> acc;
  long ncalls = 0;
  Core(const nlohmann::json &j, const std::vector<int> &type_Z, long max_edges_, bool exact_, int ng_) :
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
    rc = geo.rc;
    err.need(1);
  }
  ~Core() override
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
  void set_stream(cudaStream_t s) override
  {
    stream = s;
    CB(cublasSetStream(blas, stream));
  }
  void mark(const char *name) override
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
  int *count_buffer(int n) override
  {
    evn.clear();
    mark("start");
    cnt.need(n + 1);
    rowptr_.need(n + 1);
    return cnt.p;
  }
  long scan(int n) override
  {
    size_t bytes = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, cnt.p, rowptr_.p, n + 1, stream));
    scan_tmp.need(bytes);
    CK(cub::DeviceScan::ExclusiveSum(scan_tmp.p, bytes, cnt.p, rowptr_.p, n + 1, stream));
    h_rowptr.resize(n + 1);
    CK(cudaMemcpyAsync(h_rowptr.data(), rowptr_.p, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    E = h_rowptr[n];
    src_.need(E + 1);
    vec_.need(3 * (E + 1));
    CK(cudaMemsetAsync(err.p, 0, sizeof(int), stream));
    return E;
  }
  int *rowptr() override { return rowptr_.p; }
  int *src() override { return src_.p; }
  double *vec() override { return vec_.p; }
  int *errflag() override { return err.p; }

  const double *run(int n, const int *type) override
  {
    Y.need((E + 1) * 9);
    R.need(E + 1);
    if (exact) {
      rbf1.need((E + 1) * R1);
      cut.need(E + 1);
    }
    if (E > 0) k_geo<T><<<(E + 255) / 256, 256, 0, stream>>>(E, vec_.p, geo, exact, Y.p, R.p, rbf1.p, cut.p);
    // chunks of receivers with at most max_edges edges (exact radial path only)
    std::vector<int> cuts{0};
    if (exact) {
      while (cuts.back() < n) {
        int a = cuts.back(), b = a + 1;
        while (b < n && (long) h_rowptr[b + 1] - h_rowptr[a] <= max_edges) ++b;
        cuts.push_back(b);
      }
    } else {
      cuts.push_back(n);
    }
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
    Zd.need((size_t) n * 9);
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
            k_tp<T, false><<<a1 - a0, F * G, 0, stream>>>(a0, rowptr_.p, eb, src_.p, Y.p, R.p, cut.p, W.p, sp,
                                                          x.p, agg.p);
          else
            k_tp<T, true><<<a1 - a0, F * G, 0, stream>>>(a0, rowptr_.p, eb, src_.p, Y.p, R.p, cut.p, W.p, sp,
                                                         x.p, agg.p);
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
    CK(cudaMemsetAsync(zsum.p, 0, 9 * sizeof(double), stream));
    k_sum<T><<<(n + 127) / 128, 128, 0, stream>>>(n, Zr.p, zsum.p);
    k_asr<T><<<((long) n * 9 + 255) / 256, 256, 0, stream>>>(n, Zr.p, zsum.p, Zd.p);
    return Zd.p;
  }
  void accumulate_profile() override
  {
    if (!prof) return;
    for (size_t k = 1; k < evn.size(); ++k) {
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, evs[k - 1], evs[k]));
      acc[evn[k]] += ms;
    }
    ++ncalls;
  }
};

// factory: precision "float32"/"float64", radial exact (GEMM) or spline table with ng intervals
inline CoreBase *make_core(const nlohmann::json &j, const std::vector<int> &type_Z, bool use_double, long max_edges,
                           bool exact, int ng)
{
  if (use_double) return new Core<double>(j, type_Z, max_edges, exact, ng);
  return new Core<float>(j, type_Z, max_edges, exact, ng);
}

}    // namespace becfield
