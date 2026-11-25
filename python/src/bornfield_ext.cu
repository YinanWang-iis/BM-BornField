// BM-BornField Python extension: EGCNN Born effective charges on the GPU with its own cell-list neighbour list.
// The network is the same CUDA core as the LAMMPS fix becfield/kk (lammps/becfield_core.cuh).
// GPL-2.0 (the core is shared with the LAMMPS fix).
#include "becfield_core.cuh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <fstream>
#include <memory>

namespace py = pybind11;
using namespace becfield;

namespace {

struct Grid {
  double cell[3][3];    // rows a, b, c
  int nb[3];            // bins per direction
  int m[3];             // neighbour-bin offset range per direction
  int pbc[3];
};

// atoms sorted by bin: order[k] = atom, bstart[b] .. bstart[b+1]; bin of atom i: bin_of[i] (flattened)
__device__ __forceinline__ bool nb_bin(const Grid &g, const int *bi, int o0, int o1, int o2, int &b, double sh[3])
{
  int img[3], c[3];
  const int o[3] = {o0, o1, o2};
  for (int d = 0; d < 3; ++d) {
    int t = bi[d] + o[d];
    if (g.pbc[d]) {
      img[d] = (int) floor((double) t / g.nb[d]);
      c[d] = t - img[d] * g.nb[d];
    } else {
      if (t < 0 || t >= g.nb[d]) return false;
      img[d] = 0;
      c[d] = t;
    }
  }
  for (int k = 0; k < 3; ++k) sh[k] = img[0] * g.cell[0][k] + img[1] * g.cell[1][k] + img[2] * g.cell[2][k];
  b = (c[0] * g.nb[1] + c[1]) * g.nb[2] + c[2];
  return true;
}

// one warp per receiver i; edges (i <- j + shift) with |r_i - r_j - shift| < rc, excluding the atom itself
template <bool FILL>
__global__ void k_nl(int n, const double *pos, const int *binc, const int *order, const int *bstart, Grid g,
                     double rc2, int *cnt, const int *rowptr, int *src, double *vec)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (i > n) return;
  if (i == n) {
    if (!FILL && lane == 0) cnt[n] = 0;
    return;
  }
  const double xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
  const int *bi = binc + 3 * i;
  int c = 0, base = FILL ? rowptr[i] : 0;
  for (int o0 = -g.m[0]; o0 <= g.m[0]; ++o0)
    for (int o1 = -g.m[1]; o1 <= g.m[1]; ++o1)
      for (int o2 = -g.m[2]; o2 <= g.m[2]; ++o2) {
        int b;
        double sh[3];
        if (!nb_bin(g, bi, o0, o1, o2, b, sh)) continue;
        const int s0 = bstart[b], s1 = bstart[b + 1];
        for (int k0 = s0; k0 < s1; k0 += 32) {
          const int k = k0 + lane;
          bool ok = false;
          int j = 0;
          double vx = 0, vy = 0, vz = 0;
          if (k < s1) {
            j = order[k];
            vx = xi - (pos[3 * j] + sh[0]);
            vy = yi - (pos[3 * j + 1] + sh[1]);
            vz = zi - (pos[3 * j + 2] + sh[2]);
            const double r2 = vx * vx + vy * vy + vz * vz;
            ok = r2 < rc2 && r2 > 1e-20;
          }
          if (FILL) {
            const unsigned mk = __ballot_sync(~0u, ok);
            if (ok) {
              const int e = base + __popc(mk & ((1u << lane) - 1u));
              src[e] = j;
              vec[3L * e] = vx;
              vec[3L * e + 1] = vy;
              vec[3L * e + 2] = vz;
            }
            base += __popc(mk);
          } else {
            c += ok;
          }
        }
      }
  if (!FILL) {
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(~0u, c, o);
    if (lane == 0) cnt[i] = c;
  }
}

class BornFieldGPU {
 public:
  BornFieldGPU(const std::string &model, const std::string &precision, const std::string &radial, int grid,
               long chunk)
  {
    if (precision != "float32" && precision != "float64")
      throw std::invalid_argument("precision must be float32 or float64");
    if (radial != "spline" && radial != "exact") throw std::invalid_argument("radial must be spline or exact");
    std::ifstream in(model);
    if (!in) throw std::runtime_error("cannot open " + model);
    auto j = nlohmann::json::parse(in);
    std::vector<int> all;    // "type" = atomic number: x0 rows for Z = 1..99
    for (int z = 1; z < 100; ++z) all.push_back(z);
    core.reset(make_core(j, all, precision == "float64", chunk, radial == "exact", grid));
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    core->set_stream(stream);
    rc = core->rc;
  }
  ~BornFieldGPU()
  {
    core.reset();
    if (stream) cudaStreamDestroy(stream);
  }

  // positions (N,3) Angstrom, numbers (N,), cell (3,3) rows, pbc (3,) -> Z* (N,9), row-major Z[a,b], ASR applied
  py::array_t<double> compute(py::array_t<double, py::array::c_style | py::array::forcecast> positions,
                              py::array_t<int, py::array::c_style | py::array::forcecast> numbers,
                              py::array_t<double, py::array::c_style | py::array::forcecast> cell,
                              std::vector<bool> pbc)
  {
    const int n = (int) numbers.size();
    if (positions.ndim() != 2 || positions.shape(0) != n || positions.shape(1) != 3)
      throw std::invalid_argument("positions must be (N,3)");
    if (cell.size() != 9 || pbc.size() != 3) throw std::invalid_argument("cell must be 3x3 and pbc length 3");
    const double *P = positions.data();
    const int *Zn = numbers.data();
    for (int i = 0; i < n; ++i)
      if (Zn[i] < 1 || Zn[i] > 99) throw std::invalid_argument("atomic numbers must be 1..99");
    // ---- host: wrap, bins
    Grid g{};
    double C[3][3];
    std::copy(cell.data(), cell.data() + 9, &C[0][0]);
    // non-periodic directions: replace by a box around the atoms (vector orthogonal to the others is not needed:
    // the binning only has to cover the atoms)
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], P[3 * i + d]);
        hi[d] = std::max(hi[d], P[3 * i + d]);
      }
    bool any_np = false;
    for (int d = 0; d < 3; ++d) {
      g.pbc[d] = pbc[d] ? 1 : 0;
      any_np |= !pbc[d];
    }
    if (any_np) {
      // require: non-periodic directions must be along Cartesian axes (as for ASE molecules) or given by a valid cell
      for (int d = 0; d < 3; ++d)
        if (!pbc[d]) {
          double len = 0;
          for (int k = 0; k < 3; ++k) len += C[d][k] * C[d][k];
          if (len < 1e-12) {    // zero cell vector: use the Cartesian axis d spanning the atoms
            for (int k = 0; k < 3; ++k) C[d][k] = 0;
            C[d][d] = (hi[d] - lo[d]) + 2 * rc + 1.0;
          }
        }
    }
    double inv[3][3];
    {
      const double det = C[0][0] * (C[1][1] * C[2][2] - C[1][2] * C[2][1]) -
          C[0][1] * (C[1][0] * C[2][2] - C[1][2] * C[2][0]) + C[0][2] * (C[1][0] * C[2][1] - C[1][1] * C[2][0]);
      if (std::fabs(det) < 1e-12) throw std::invalid_argument("singular cell");
      inv[0][0] = (C[1][1] * C[2][2] - C[1][2] * C[2][1]) / det;
      inv[0][1] = (C[0][2] * C[2][1] - C[0][1] * C[2][2]) / det;
      inv[0][2] = (C[0][1] * C[1][2] - C[0][2] * C[1][1]) / det;
      inv[1][0] = (C[1][2] * C[2][0] - C[1][0] * C[2][2]) / det;
      inv[1][1] = (C[0][0] * C[2][2] - C[0][2] * C[2][0]) / det;
      inv[1][2] = (C[0][2] * C[1][0] - C[0][0] * C[1][2]) / det;
      inv[2][0] = (C[1][0] * C[2][1] - C[1][1] * C[2][0]) / det;
      inv[2][1] = (C[0][1] * C[2][0] - C[0][0] * C[2][1]) / det;
      inv[2][2] = (C[0][0] * C[1][1] - C[0][1] * C[1][0]) / det;
    }
    // fractional coords s = r . inv (r = s . C); perpendicular width of direction d = 1/|column d of inv|
    std::vector<double> frac(3 * (size_t) n), pos(3 * (size_t) n);
    double smin[3] = {1e300, 1e300, 1e300}, smax[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < 3; ++d) {
        double s = P[3 * i] * inv[0][d] + P[3 * i + 1] * inv[1][d] + P[3 * i + 2] * inv[2][d];
        if (pbc[d]) s -= std::floor(s);
        frac[3 * i + d] = s;
        smin[d] = std::min(smin[d], s);
        smax[d] = std::max(smax[d], s);
      }
      for (int k = 0; k < 3; ++k)
        pos[3 * i + k] = frac[3 * i] * C[0][k] + frac[3 * i + 1] * C[1][k] + frac[3 * i + 2] * C[2][k];
    }
    double s0[3], s1[3];
    for (int d = 0; d < 3; ++d) {
      double w = 0;
      for (int k = 0; k < 3; ++k) w += inv[k][d] * inv[k][d];
      const double width = 1.0 / std::sqrt(w);    // perpendicular extent of one cell in direction d
      s0[d] = pbc[d] ? 0.0 : smin[d];
      s1[d] = pbc[d] ? 1.0 : std::max(smax[d], smin[d] + 1e-9);
      const double span = (s1[d] - s0[d]) * width;
      g.nb[d] = std::max(1, std::min(256, (int) std::floor(span / rc)));
      const double binw = span / g.nb[d];
      g.m[d] = (int) std::ceil(rc / binw);
      if (!pbc[d]) g.m[d] = std::min(g.m[d], g.nb[d]);
      for (int k = 0; k < 3; ++k) g.cell[d][k] = C[d][k];
    }
    const int nbin = g.nb[0] * g.nb[1] * g.nb[2];
    std::vector<int> binc(3 * (size_t) n), bin(n), order(n), bstart(nbin + 1, 0);
    for (int i = 0; i < n; ++i) {
      int b3[3];
      for (int d = 0; d < 3; ++d) {
        int t = (int) std::floor((frac[3 * i + d] - s0[d]) / (s1[d] - s0[d]) * g.nb[d]);
        b3[d] = std::min(std::max(t, 0), g.nb[d] - 1);
        binc[3 * i + d] = b3[d];
      }
      bin[i] = (b3[0] * g.nb[1] + b3[1]) * g.nb[2] + b3[2];
      ++bstart[bin[i] + 1];
    }
    for (int b = 0; b < nbin; ++b) bstart[b + 1] += bstart[b];
    {
      std::vector<int> fillp(bstart.begin(), bstart.end() - 1);
      for (int i = 0; i < n; ++i) order[fillp[bin[i]]++] = i;
    }
    // ---- device
    d_pos.need(3 * (size_t) n);
    d_binc.need(3 * (size_t) n);
    d_order.need(n);
    d_bstart.need(nbin + 1);
    d_type.need(n);
    CK(cudaMemcpyAsync(d_pos.p, pos.data(), pos.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_binc.p, binc.data(), binc.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_order.p, order.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_bstart.p, bstart.data(), (nbin + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_type.p, Zn, n * sizeof(int), cudaMemcpyHostToDevice, stream));
    const int TB = 128, nw = ((n + 1) * 32 + TB - 1) / TB;
    int *cnt = core->count_buffer(n);
    k_nl<false><<<nw, TB, 0, stream>>>(n, d_pos.p, d_binc.p, d_order.p, d_bstart.p, g, rc * rc, cnt, nullptr,
                                       nullptr, nullptr);
    core->scan(n);
    k_nl<true><<<nw, TB, 0, stream>>>(n, d_pos.p, d_binc.p, d_order.p, d_bstart.p, g, rc * rc, nullptr,
                                      core->rowptr(), core->src(), core->vec());
    const double *Zd = core->run(n, d_type.p);
    py::array_t<double> out({(py::ssize_t) n, (py::ssize_t) 9});
    CK(cudaMemcpyAsync(out.mutable_data(), Zd, (size_t) n * 9 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    core->mark("output");
    CK(cudaStreamSynchronize(stream));
    CK(cudaGetLastError());
    core->accumulate_profile();
    return out;
  }
  long edges() const { return core->edges(); }
  double cutoff() const { return rc; }

 private:
  std::unique_ptr<CoreBase> core;
  cudaStream_t stream = nullptr;
  double rc = 0;
  DevBuf<double> d_pos;
  DevBuf<int> d_binc, d_order, d_bstart, d_type;
};

}    // namespace

PYBIND11_MODULE(_bornfield, m)
{
  m.doc() = "BM-BornField: EGCNN Born effective charges on the GPU (CUDA core shared with LAMMPS fix becfield/kk)";
  py::class_<BornFieldGPU>(m, "BornFieldGPU")
      .def(py::init<const std::string &, const std::string &, const std::string &, int, long>(), py::arg("model"),
           py::arg("precision") = "float32", py::arg("radial") = "spline", py::arg("grid") = 1024,
           py::arg("chunk") = 1L << 18)
      .def("compute", &BornFieldGPU::compute, py::arg("positions"), py::arg("numbers"), py::arg("cell"),
           py::arg("pbc"), "Z* (N,9) row-major Z[a,b] (a: field/polarization, b: displacement), ASR applied")
      .def_property_readonly("edges", &BornFieldGPU::edges)
      .def_property_readonly("cutoff", &BornFieldGPU::cutoff);
}
